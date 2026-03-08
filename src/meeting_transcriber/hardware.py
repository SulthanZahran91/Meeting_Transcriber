from __future__ import annotations

from dataclasses import dataclass
import os
import platform
import shutil
import subprocess
from typing import Optional

import psutil
from rich.console import Console
from rich.panel import Panel


@dataclass(frozen=True)
class HardwareProfile:
    device: str
    gpu_name: Optional[str]
    gpu_vram_gb: Optional[float]
    ram_gb: float
    cpu_name: str
    cpu_cores: int


@dataclass(frozen=True)
class ModelRecommendation:
    whisper_model: str
    device: str
    compute_type: str
    beam_size: int
    reason: str


def detect_hardware() -> HardwareProfile:
    device = "cpu"
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None

    cuda_details = _detect_cuda_with_torch() or _detect_cuda_with_nvidia_smi()
    if cuda_details is not None:
        device = "cuda"
        gpu_name, gpu_vram_gb = cuda_details

    ram_gb = psutil.virtual_memory().total / (1024**3)
    cpu_name = platform.processor() or "Unknown CPU"
    cpu_cores = os.cpu_count() or 1

    return HardwareProfile(
        device=device,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        ram_gb=ram_gb,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
    )


def _detect_cuda_with_torch() -> tuple[str, float] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        props = torch.cuda.get_device_properties(0)
        return props.name, props.total_memory / (1024**3)
    except Exception:
        # Torch is optional and hardware detection should never crash startup.
        return None


def _detect_cuda_with_nvidia_smi() -> tuple[str, float] | None:
    if shutil.which("nvidia-smi") is None:
        return None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    first_line = next(
        (line.strip() for line in result.stdout.splitlines() if line.strip()),
        None,
    )
    if first_line is None:
        return None

    try:
        gpu_name, vram_mb = [part.strip() for part in first_line.split(",", maxsplit=1)]
        return gpu_name, float(vram_mb) / 1024.0
    except (TypeError, ValueError):
        return None


def recommend_config(
    hw: HardwareProfile,
    force_model: Optional[str] = None,
    force_device: Optional[str] = None,
) -> ModelRecommendation:
    target_device = force_device or hw.device
    if target_device not in {"cpu", "cuda"}:
        target_device = hw.device

    model, compute_type, beam_size = _select_auto_profile(
        target_device, hw.gpu_vram_gb, hw.ram_gb
    )
    reason_bits = [_selection_reason(target_device, hw.gpu_vram_gb, hw.ram_gb)]

    if force_model:
        model = force_model
        reason_bits.append(f"Model forced to '{force_model}'.")

    if force_device:
        reason_bits.append(f"Device forced to '{force_device}'.")

    warning = _risk_warning(model, target_device, hw.gpu_vram_gb, hw.ram_gb)
    if warning:
        reason_bits.append(f"WARNING: {warning}")

    return ModelRecommendation(
        whisper_model=model,
        device=target_device,
        compute_type=compute_type,
        beam_size=beam_size,
        reason=" ".join(reason_bits).strip(),
    )


def print_hardware_summary(
    hw: HardwareProfile, recommendation: ModelRecommendation, console: Optional[Console] = None
) -> None:
    console = console or Console()
    gpu_line = (
        f"{hw.gpu_name} ({hw.gpu_vram_gb:.1f} GB VRAM)"
        if hw.gpu_name and hw.gpu_vram_gb is not None
        else "Not detected"
    )
    body = "\n".join(
        [
            f"[bold]Detected Hardware[/bold]",
            f"- Device: {hw.device}",
            f"- CPU: {hw.cpu_name} ({hw.cpu_cores} cores)",
            f"- RAM: {hw.ram_gb:.1f} GB",
            f"- GPU: {gpu_line}",
            "",
            f"[bold]Recommended Runtime[/bold]",
            f"- Whisper model: {recommendation.whisper_model}",
            f"- Device: {recommendation.device}",
            f"- Compute type: {recommendation.compute_type}",
            f"- Beam size: {recommendation.beam_size}",
            "",
            recommendation.reason,
        ]
    )
    console.print(Panel(body, title="Hardware + Recommendation", expand=False))


def _select_auto_profile(
    device: str, gpu_vram_gb: Optional[float], ram_gb: float
) -> tuple[str, str, int]:
    if device == "cuda":
        vram = gpu_vram_gb or 0.0
        if vram >= 10:
            return "large-v3", "float16", 5
        if vram >= 6:
            return "medium", "float16", 5
        if vram >= 4:
            return "small", "float16", 5
        return "small", "int8", 3

    if ram_gb >= 24:
        return "large-v3", "int8", 5
    if ram_gb >= 12:
        return "medium", "int8", 5
    if ram_gb >= 8:
        return "small", "int8", 3
    return "tiny", "int8", 1


def _selection_reason(device: str, gpu_vram_gb: Optional[float], ram_gb: float) -> str:
    if device == "cuda":
        vram = gpu_vram_gb or 0.0
        return f"Auto-selected from CUDA profile (VRAM={vram:.1f} GB)."
    return f"Auto-selected from CPU profile (RAM={ram_gb:.1f} GB)."


def _risk_warning(
    model: str, device: str, gpu_vram_gb: Optional[float], ram_gb: float
) -> Optional[str]:
    if device == "cuda":
        vram = gpu_vram_gb or 0.0
        needed = {"large-v3": 10.0, "medium": 6.0, "small": 4.0, "tiny": 2.0}
        threshold = needed.get(model)
        if threshold is not None and vram < threshold:
            return (
                f"'{model}' may exceed available VRAM ({vram:.1f} GB). "
                "Consider --model small/tiny or --device cpu."
            )
        return None

    needed_ram = {"large-v3": 24.0, "medium": 12.0, "small": 8.0, "tiny": 4.0}
    threshold = needed_ram.get(model)
    if threshold is not None and ram_gb < threshold:
        return (
            f"'{model}' may exceed available RAM ({ram_gb:.1f} GB). "
            "Consider --model small/tiny."
        )
    return None
