import sys
import subprocess
import types

import pytest

from meeting_transcriber.hardware import (
    HardwareProfile,
    detect_hardware,
    recommend_config,
)


def test_cpu_recommendation_matrix_boundaries() -> None:
    tiny = HardwareProfile("cpu", None, None, 7.9, "cpu", 4)
    small = HardwareProfile("cpu", None, None, 8.0, "cpu", 4)
    medium = HardwareProfile("cpu", None, None, 12.0, "cpu", 4)
    large = HardwareProfile("cpu", None, None, 24.0, "cpu", 4)

    assert recommend_config(tiny).whisper_model == "tiny"
    assert recommend_config(small).whisper_model == "small"
    assert recommend_config(medium).whisper_model == "medium"
    assert recommend_config(large).whisper_model == "large-v3"


def test_cuda_recommendation_matrix_boundaries() -> None:
    low = HardwareProfile("cuda", "GPU", 3.9, 32.0, "cpu", 4)
    small = HardwareProfile("cuda", "GPU", 4.0, 32.0, "cpu", 4)
    medium = HardwareProfile("cuda", "GPU", 6.0, 32.0, "cpu", 4)
    large = HardwareProfile("cuda", "GPU", 10.0, 32.0, "cpu", 4)

    assert recommend_config(low).whisper_model == "small"
    assert recommend_config(low).compute_type == "int8"
    assert recommend_config(small).whisper_model == "small"
    assert recommend_config(small).compute_type == "float16"
    assert recommend_config(medium).whisper_model == "medium"
    assert recommend_config(large).whisper_model == "large-v3"


def test_force_model_kept_with_risk_warning() -> None:
    hw = HardwareProfile("cpu", None, None, 8.0, "cpu", 4)
    rec = recommend_config(hw, force_model="large-v3")

    assert rec.whisper_model == "large-v3"
    assert "WARNING:" in rec.reason


def test_force_device_cpu_on_cuda_machine() -> None:
    hw = HardwareProfile("cuda", "GPU", 12.0, 32.0, "cpu", 4)
    rec = recommend_config(hw, force_device="cpu")

    assert rec.device == "cpu"
    assert rec.compute_type == "int8"


def test_detect_hardware_without_torch(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr("meeting_transcriber.hardware.shutil.which", lambda _name: None)
    hw = detect_hardware()
    assert hw.device == "cpu"


def test_detect_hardware_uses_nvidia_smi_when_torch_missing(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr("meeting_transcriber.hardware.shutil.which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        "meeting_transcriber.hardware.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="NVIDIA GeForce RTX 4090, 24564\n",
            stderr="",
        ),
    )

    hw = detect_hardware()

    assert hw.device == "cuda"
    assert hw.gpu_name == "NVIDIA GeForce RTX 4090"
    assert hw.gpu_vram_gb == pytest.approx(24564 / 1024.0)


def test_detect_hardware_uses_nvidia_smi_when_torch_has_no_cuda(monkeypatch) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("meeting_transcriber.hardware.shutil.which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        "meeting_transcriber.hardware.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="NVIDIA RTX 4070 Laptop GPU, 8188\n",
            stderr="",
        ),
    )

    hw = detect_hardware()

    assert hw.device == "cuda"
    assert hw.gpu_name == "NVIDIA RTX 4070 Laptop GPU"
    assert hw.gpu_vram_gb == pytest.approx(8188 / 1024.0)


def test_detect_hardware_ignores_bad_nvidia_smi_output(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr("meeting_transcriber.hardware.shutil.which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        "meeting_transcriber.hardware.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="malformed output\n",
            stderr="",
        ),
    )

    hw = detect_hardware()

    assert hw.device == "cpu"
    assert hw.gpu_name is None
    assert hw.gpu_vram_gb is None
