"""
Diagnostic-only check for Wan2.2 Group Offloading support.

This script does not download or instantiate a Wan2.2 checkpoint and does not
run model inference. It restricts CUDA visibility to GPU 0 for this process.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import os
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        try:
            module = importlib.import_module(package_name)
            return str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            return f"not available ({type(exc).__name__}: {exc})"
    except Exception as exc:
        return f"could not be determined ({type(exc).__name__}: {exc})"


def print_signature(label: str, callable_object: object) -> None:
    try:
        print(f"{label}: {inspect.signature(callable_object)}")
    except (TypeError, ValueError) as exc:
        print(f"{label}: signature unavailable ({type(exc).__name__}: {exc})")


def signature_supports(signature: inspect.Signature, parameter: str) -> str:
    if parameter in signature.parameters:
        parameter_object = signature.parameters[parameter]
        default = parameter_object.default
        default_text = "required" if default is inspect.Parameter.empty else repr(default)
        return f"YES ({default_text})"
    return "NO"


def main() -> int:
    print("Wan2.2 Group Offloading diagnostic")
    print(f"Python: {sys.version.split()[0]}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")

    print_section("Package versions")
    print(f"PyTorch: {package_version('torch')}")
    print(f"Diffusers: {package_version('diffusers')}")
    print(f"Transformers: {package_version('transformers')}")
    print(f"Accelerate: {package_version('accelerate')}")

    print_section("CUDA and GPU")
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        print(f"PyTorch import failed: {type(exc).__name__}: {exc}")
        torch = None

    if torch is None:
        cuda_available = False
        visible_gpu_count = 0
    else:
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:
            cuda_available = False
            print(f"CUDA availability check failed: {type(exc).__name__}: {exc}")

        try:
            visible_gpu_count = torch.cuda.device_count()
        except Exception as exc:
            visible_gpu_count = 0
            print(f"GPU count check failed: {type(exc).__name__}: {exc}")

        print(f"CUDA available: {'YES' if cuda_available else 'NO'}")
        print(f"Visible GPU count: {visible_gpu_count}")

        if not cuda_available or visible_gpu_count == 0:
            print("GPU 0: not accessible")
        else:
            device_index = 0
            try:
                name = torch.cuda.get_device_name(device_index)
                properties = torch.cuda.get_device_properties(device_index)
                total_vram_bytes = int(properties.total_memory)
                total_vram_gib = total_vram_bytes / (1024 ** 3)
                print(f"GPU 0 name: {name}")
                print(
                    f"GPU 0 VRAM: {total_vram_gib:.2f} GiB "
                    f"({total_vram_bytes} bytes)"
                )
            except Exception as exc:
                print(
                    f"GPU 0 properties unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )

            try:
                bf16_supported = bool(torch.cuda.is_bf16_supported())
                print(
                    "BF16 supported by torch.cuda.is_bf16_supported(): "
                    f"{'YES' if bf16_supported else 'NO'}"
                )
            except Exception as exc:
                print(
                    f"BF16 support check failed: "
                    f"{type(exc).__name__}: {exc}"
                )

    print_section("WanTransformer3DModel Group Offloading API")
    wan_transformer_class = None
    try:
        diffusers = importlib.import_module("diffusers")
        wan_transformer_class = getattr(diffusers, "WanTransformer3DModel", None)
        if wan_transformer_class is None:
            from diffusers.models.transformers.transformer_wan import (
                WanTransformer3DModel,
            )

            wan_transformer_class = WanTransformer3DModel
    except Exception as exc:
        print(
            f"WanTransformer3DModel import failed: "
            f"{type(exc).__name__}: {exc}"
        )

    enable_method = None
    enable_signature = None
    if wan_transformer_class is not None:
        print(f"WanTransformer3DModel: {wan_transformer_class}")
        enable_method = getattr(wan_transformer_class, "enable_group_offload", None)
        print(
            "enable_group_offload present: "
            f"{'YES' if enable_method is not None else 'NO'}"
        )

        if enable_method is not None:
            print_signature(
                "enable_group_offload signature",
                enable_method,
            )
            try:
                enable_signature = inspect.signature(enable_method)
            except (TypeError, ValueError) as exc:
                print(
                    f"enable_group_offload signature could not be inspected: "
                    f"{type(exc).__name__}: {exc}"
                )

    print_section("apply_group_offloading API")
    apply_group_offloading = None
    apply_signature = None
    try:
        hooks = importlib.import_module("diffusers.hooks")
        apply_group_offloading = getattr(hooks, "apply_group_offloading", None)
    except Exception as exc:
        print(
            f"diffusers.hooks import failed: "
            f"{type(exc).__name__}: {exc}"
        )

    print(
        "apply_group_offloading present: "
        f"{'YES' if apply_group_offloading is not None else 'NO'}"
    )
    if apply_group_offloading is not None:
        print_signature(
            "apply_group_offloading signature",
            apply_group_offloading,
        )
        try:
            apply_signature = inspect.signature(apply_group_offloading)
        except (TypeError, ValueError) as exc:
            print(
                f"apply_group_offloading signature could not be inspected: "
                f"{type(exc).__name__}: {exc}"
            )

    print_section("Offloading parameter support")

    signatures = [
        (
            "enable_group_offload",
            enable_signature,
        ),
        (
            "apply_group_offloading",
            apply_signature,
        ),
    ]
    parameters = (
        "offload_type",
        "num_blocks_per_group",
        "use_stream",
        "non_blocking",
    )

    for signature_name, signature in signatures:
        print(f"{signature_name}:")
        if signature is None:
            print("  Signature unavailable; parameter support is NOT CONFIRMED.")
            continue

        for parameter in parameters:
            print(
                f"  {parameter}: "
                f"{signature_supports(signature, parameter)}"
            )

    print_section("Unconfirmed capabilities")
    print(
        "- Actual block execution and lazy prefetch: "
        "NOT CONFIRMED until a GPU forward is run"
    )
    print(
        "- Wan2.2 checkpoint loading compatibility: "
        "NOT CONFIRMED; no checkpoint was loaded"
    )
    print(
        "- CUDA stream overlap with computation: "
        "NOT CONFIRMED; no inference was run"
    )
    print(
        "- Peak VRAM and CPU RAM usage: "
        "NOT CONFIRMED; no model was loaded"
    )
    print(
        "- Correct block grouping for Wan2.2: "
        "NOT CONFIRMED; no transformer model was instantiated"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
