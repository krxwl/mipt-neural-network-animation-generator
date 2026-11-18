"""
Wan2.2 I2V-A14B transformer Group Offloading smoke test.

The script loads exactly one transformer, applies Diffusers Group Offloading,
and does not run a forward pass or video generation.
"""

from __future__ import annotations

import inspect
import os
import sys
import time
import gc
import shutil
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    import torch
    from diffusers import WanTransformer3DModel
    from diffusers.hooks import apply_group_offloading
except (ImportError, ModuleNotFoundError) as exc:
    print(f"ERROR: required package import failed: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc


MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
SUBFOLDER = "transformer"
DTYPE = torch.bfloat16
ONLOAD_DEVICE = torch.device("cuda:0")
OFFLOAD_DEVICE = torch.device("cpu")
REQUIRED_PARAMETERS = (
    "offload_type",
    "num_blocks_per_group",
    "use_stream",
    "non_blocking",
)


class SmokeTestError(RuntimeError):
    """Raised when the environment cannot run this smoke test."""


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def format_gib(value: int | float) -> str:
    return f"{float(value) / (1024 ** 3):.2f} GiB"


def get_existing_disk_usage(path: str) -> tuple[str, Any]:
    resolved_path = os.path.abspath(os.path.expanduser(path))
    check_path = resolved_path
    while not os.path.exists(check_path):
        parent = os.path.dirname(check_path)
        if parent == check_path:
            break
        check_path = parent
    return check_path, shutil.disk_usage(check_path)


def print_system_resource_checks() -> None:
    print_section("System resources")

    try:
        psutil = __import__("psutil")
    except ImportError:
        print("psutil: unavailable; system RAM check was NOT PERFORMED")
    else:
        try:
            virtual_memory = psutil.virtual_memory()
            print(
                f"System RAM total: {format_gib(virtual_memory.total)}"
            )
            print(
                f"System RAM available: {format_gib(virtual_memory.available)}"
            )
            print(
                f"System RAM used: {format_gib(virtual_memory.used)} "
                f"({virtual_memory.percent:.1f}%)"
            )
        except Exception as exc:
            print(
                "System RAM check failed: "
                f"{type(exc).__name__}: {exc}"
            )

    try:
        from huggingface_hub import constants

        hub_cache = constants.HF_HUB_CACHE
    except ImportError:
        hub_cache = os.environ.get("HF_HUB_CACHE")
        if not hub_cache:
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                hub_cache = os.path.join(hf_home, "hub")
            else:
                hub_cache = os.path.join(
                    os.path.expanduser("~"),
                    ".cache",
                    "huggingface",
                    "hub",
                )

    print(f"Hugging Face hub cache: {os.path.abspath(hub_cache)}")
    try:
        existing_path, disk_usage = get_existing_disk_usage(hub_cache)
        if os.path.abspath(existing_path) != os.path.abspath(hub_cache):
            print(
                f"Hugging Face cache directory not present yet; "
                f"disk check uses existing parent: {existing_path}"
            )
        print(f"Hugging Face cache disk total: {format_gib(disk_usage.total)}")
        print(f"Hugging Face cache disk free: {format_gib(disk_usage.free)}")
        print(f"Hugging Face cache disk used: {format_gib(disk_usage.used)}")
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        print(
            "Hugging Face cache disk check failed: "
            f"{type(exc).__name__}: {exc}"
        )

    print(
        "WARNING: Loading one full Wan2.2 I2V-A14B Transformer "
        "may require significant CPU RAM and Hugging Face cache disk space."
    )


def print_versions() -> None:
    import accelerate
    import diffusers
    import transformers

    print_section("Environment")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Diffusers: {diffusers.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Accelerate: {accelerate.__version__}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")


def validate_environment() -> None:
    print_section("CUDA and BF16 checks")

    if not torch.cuda.is_available():
        raise SmokeTestError("CUDA is unavailable; the transformer will not be loaded.")

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count != 1:
        raise SmokeTestError(
            "Expected exactly one visible GPU after CUDA_VISIBLE_DEVICES=0, "
            f"but found {visible_gpu_count}."
        )

    gpu_name = torch.cuda.get_device_name(0)
    properties = torch.cuda.get_device_properties(0)
    print(f"Visible GPU count: {visible_gpu_count}")
    print(f"GPU 0: {gpu_name}")
    print(f"GPU 0 total VRAM: {format_gib(properties.total_memory)}")

    if not torch.cuda.is_bf16_supported():
        raise SmokeTestError(
            "torch.cuda.is_bf16_supported() returned False; "
            "the required BF16 experiment will not be started."
        )

    print("BF16 supported: YES")


def validate_api() -> tuple[inspect.Signature, inspect.Signature]:
    print_section("Diffusers API checks")

    enable_group_offload = getattr(WanTransformer3DModel, "enable_group_offload", None)
    if enable_group_offload is None:
        raise SmokeTestError(
            "WanTransformer3DModel.enable_group_offload is unavailable."
        )

    enable_signature = inspect.signature(enable_group_offload)
    apply_signature = inspect.signature(apply_group_offloading)
    print(f"enable_group_offload: {enable_signature}")
    print(f"apply_group_offloading: {apply_signature}")

    for function_name, signature in (
        ("enable_group_offload", enable_signature),
        ("apply_group_offloading", apply_signature),
    ):
        missing = [
            parameter
            for parameter in REQUIRED_PARAMETERS
            if parameter not in signature.parameters
        ]
        if missing:
            raise SmokeTestError(
                f"{function_name} does not support required parameters: "
                f"{', '.join(missing)}."
            )

    return enable_signature, apply_signature


def print_configuration() -> None:
    print_section("Experiment configuration")
    print(f"Checkpoint: {MODEL_ID}")
    print(f"Subfolder: {SUBFOLDER}")
    print(f"Dtype: {DTYPE}")
    print(f"Onload device: {ONLOAD_DEVICE}")
    print(f"Offload device: {OFFLOAD_DEVICE}")
    print("Offload type: block_level")
    print("Blocks per group: 1")
    print("Use stream: True")
    print("Non-blocking transfers: True")
    print("Forward/inference: NOT RUN")
    print("Second transformer: NOT LOADED")


def memory_snapshot() -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(ONLOAD_DEVICE)
    return {
        "free": int(free_bytes),
        "total": int(total_bytes),
        "used": int(total_bytes - free_bytes),
        "allocated": int(torch.cuda.memory_allocated(ONLOAD_DEVICE)),
        "reserved": int(torch.cuda.memory_reserved(ONLOAD_DEVICE)),
        "peak_allocated": int(torch.cuda.max_memory_allocated(ONLOAD_DEVICE)),
        "peak_reserved": int(torch.cuda.max_memory_reserved(ONLOAD_DEVICE)),
    }


def print_memory_snapshot(label: str, snapshot: dict[str, int]) -> None:
    print(f"{label}:")
    print(f"  available VRAM: {format_gib(snapshot['free'])}")
    print(f"  used VRAM: {format_gib(snapshot['used'])}")
    print(f"  allocated by PyTorch: {format_gib(snapshot['allocated'])}")
    print(f"  reserved by PyTorch: {format_gib(snapshot['reserved'])}")
    print(f"  peak allocated by PyTorch: {format_gib(snapshot['peak_allocated'])}")
    print(f"  peak reserved by PyTorch: {format_gib(snapshot['peak_reserved'])}")


def load_and_configure_transformer() -> tuple[Any, float, float]:
    torch.cuda.synchronize(ONLOAD_DEVICE)
    torch.cuda.reset_peak_memory_stats(ONLOAD_DEVICE)
    before = memory_snapshot()

    load_started = time.perf_counter()
    try:
        transformer = WanTransformer3DModel.from_pretrained(
            MODEL_ID,
            subfolder=SUBFOLDER,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        raise SmokeTestError(
            "Wan2.2 transformer loading failed before Group Offloading "
            f"could be configured: {type(exc).__name__}: {exc}"
        ) from exc
    torch.cuda.synchronize(ONLOAD_DEVICE)
    load_seconds = time.perf_counter() - load_started

    torch.cuda.synchronize(ONLOAD_DEVICE)
    offload_started = time.perf_counter()
    try:
        transformer.enable_group_offload(
            onload_device=ONLOAD_DEVICE,
            offload_device=OFFLOAD_DEVICE,
            offload_type="block_level",
            num_blocks_per_group=1,
            use_stream=True,
            non_blocking=True,
        )
    except Exception as exc:
        del transformer
        torch.cuda.empty_cache()
        gc.collect()
        raise SmokeTestError(
            "Group Offloading configuration failed for the single Wan2.2 "
            f"transformer: {type(exc).__name__}: {exc}"
        ) from exc
    torch.cuda.synchronize(ONLOAD_DEVICE)
    offload_seconds = time.perf_counter() - offload_started

    after = memory_snapshot()
    print_memory_snapshot("Before loading", before)
    print_memory_snapshot("After loading and Group Offloading setup", after)
    return transformer, load_seconds, offload_seconds


def main() -> int:
    transformer = None
    try:
        print_versions()
        validate_environment()
        print_system_resource_checks()
        validate_api()
        print_configuration()
        transformer, load_seconds, offload_seconds = load_and_configure_transformer()

        print_section("Results")
        print(f"Model load time: {load_seconds:.3f} s")
        print(f"Group Offloading setup time: {offload_seconds:.3f} s")
        print("Transformer loaded: YES")
        print("Group Offloading API applied: YES")
        print("Prefetch execution: NOT CONFIRMED (no forward was run)")
        print("Video generation: NOT RUN")
        return 0
    except SmokeTestError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if transformer is not None:
            del transformer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
