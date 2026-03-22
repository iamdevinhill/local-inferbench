"""Hardware monitoring for GPU (via pynvml) and CPU/RAM (via psutil).

Runs in a background thread during benchmark execution. Gracefully degrades
if pynvml is not installed or no NVIDIA GPU is present.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import psutil

logger = logging.getLogger(__name__)

# Try to import nvidia-ml-py (preferred) or pynvml, but don't fail if unavailable
try:
    import pynvml  # noqa: F401 — works with both nvidia-ml-py and pynvml packages

    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


@dataclass
class HardwareSnapshot:
    """A single point-in-time hardware reading."""

    timestamp: float
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_utilization: float | None = None
    vram_used_gb: float | None = None
    vram_total_gb: float | None = None
    gpu_temperature: float | None = None
    gpu_power_watts: float | None = None


@dataclass
class HardwareSummary:
    """Summary statistics from a series of hardware snapshots."""

    peak_vram_gb: float | None
    avg_gpu_utilization: float | None
    peak_gpu_temperature: float | None
    avg_cpu_percent: float
    peak_ram_used_gb: float
    ram_total_gb: float
    gpu_name: str | None
    snapshot_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "peak_vram_gb": self.peak_vram_gb,
            "avg_gpu_utilization": self.avg_gpu_utilization,
            "peak_gpu_temperature": self.peak_gpu_temperature,
            "avg_cpu_percent": self.avg_cpu_percent,
            "peak_ram_used_gb": self.peak_ram_used_gb,
            "ram_total_gb": self.ram_total_gb,
            "gpu_name": self.gpu_name,
            "snapshot_count": self.snapshot_count,
        }


@dataclass
class HardwareMonitor:
    """Polls CPU/RAM and GPU metrics in a background thread."""

    poll_interval: float = 0.5
    gpu_device: int | None = None
    snapshots: list[HardwareSnapshot] = field(default_factory=list)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _nvml_initialized: bool = field(default=False, repr=False)
    _gpu_handle: object | None = field(default=None, repr=False)
    _gpu_name: str | None = field(default=None, repr=False)

    def _init_nvml(self) -> None:
        if not _PYNVML_AVAILABLE:
            logger.info("pynvml not installed — GPU monitoring disabled")
            return
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            device_index = self.gpu_device if self.gpu_device is not None else 0
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._gpu_name = pynvml.nvmlDeviceGetName(self._gpu_handle)
            if isinstance(self._gpu_name, bytes):
                self._gpu_name = self._gpu_name.decode("utf-8")
        except Exception as e:
            logger.warning("Failed to initialize NVML: %s", e)
            self._nvml_initialized = False

    def _shutdown_nvml(self) -> None:
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False

    def _take_snapshot(self) -> HardwareSnapshot:
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_used_gb = mem.used / (1024**3)
        ram_total_gb = mem.total / (1024**3)

        gpu_util = None
        vram_used = None
        vram_total = None
        gpu_temp = None
        gpu_power = None

        if self._nvml_initialized and self._gpu_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu_util = float(util.gpu)

                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                vram_used = mem_info.used / (1024**3)
                vram_total = mem_info.total / (1024**3)

                gpu_temp = float(
                    pynvml.nvmlDeviceGetTemperature(
                        self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                )

                gpu_power = (
                    pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle) / 1000.0
                )
            except Exception as e:
                logger.debug("GPU poll error: %s", e)

        return HardwareSnapshot(
            timestamp=time.perf_counter(),
            cpu_percent=cpu_percent,
            ram_used_gb=ram_used_gb,
            ram_total_gb=ram_total_gb,
            gpu_utilization=gpu_util,
            vram_used_gb=vram_used,
            vram_total_gb=vram_total,
            gpu_temperature=gpu_temp,
            gpu_power_watts=gpu_power,
        )

    def _poll_loop(self) -> None:
        self._init_nvml()
        try:
            while not self._stop_event.is_set():
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                self._stop_event.wait(timeout=self.poll_interval)
        finally:
            self._shutdown_nvml()

    def start(self) -> None:
        """Start the background monitoring thread."""
        self._stop_event.clear()
        self.snapshots.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def summarize(self) -> HardwareSummary:
        """Compute summary statistics from collected snapshots."""
        if not self.snapshots:
            return HardwareSummary(
                peak_vram_gb=None,
                avg_gpu_utilization=None,
                peak_gpu_temperature=None,
                avg_cpu_percent=0.0,
                peak_ram_used_gb=0.0,
                ram_total_gb=0.0,
                gpu_name=self._gpu_name,
                snapshot_count=0,
            )

        cpu_values = [s.cpu_percent for s in self.snapshots]
        ram_values = [s.ram_used_gb for s in self.snapshots]

        gpu_utils = [s.gpu_utilization for s in self.snapshots if s.gpu_utilization is not None]
        vram_values = [s.vram_used_gb for s in self.snapshots if s.vram_used_gb is not None]
        gpu_temps = [s.gpu_temperature for s in self.snapshots if s.gpu_temperature is not None]

        return HardwareSummary(
            peak_vram_gb=max(vram_values) if vram_values else None,
            avg_gpu_utilization=(sum(gpu_utils) / len(gpu_utils)) if gpu_utils else None,
            peak_gpu_temperature=max(gpu_temps) if gpu_temps else None,
            avg_cpu_percent=sum(cpu_values) / len(cpu_values),
            peak_ram_used_gb=max(ram_values),
            ram_total_gb=self.snapshots[0].ram_total_gb,
            gpu_name=self._gpu_name,
            snapshot_count=len(self.snapshots),
        )


def detect_hardware() -> dict[str, object]:
    """Detect available hardware and return a summary dict."""
    info: dict[str, object] = {
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpus": [],
    }

    if not _PYNVML_AVAILABLE:
        info["gpu_monitoring"] = "pynvml not installed"
        return info

    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append({
                "index": i,
                "name": name,
                "vram_total_gb": round(mem.total / (1024**3), 2),
            })
        info["gpus"] = gpus
        pynvml.nvmlShutdown()
    except Exception as e:
        info["gpu_monitoring"] = f"NVML error: {e}"

    return info
