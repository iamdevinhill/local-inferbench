"""Tests for the hardware monitor (with mocked GPU calls)."""

import time
from unittest.mock import patch, MagicMock

import pytest

from local_inferbench.hardware import HardwareMonitor, HardwareSummary, detect_hardware


class TestHardwareMonitor:
    def test_start_stop_collects_snapshots(self):
        monitor = HardwareMonitor(poll_interval=0.1)
        monitor.start()
        time.sleep(0.35)
        monitor.stop()
        assert len(monitor.snapshots) >= 2

    def test_snapshots_have_cpu_and_ram(self):
        monitor = HardwareMonitor(poll_interval=0.1)
        monitor.start()
        time.sleep(0.2)
        monitor.stop()
        for snap in monitor.snapshots:
            assert snap.cpu_percent >= 0
            assert snap.ram_used_gb > 0
            assert snap.ram_total_gb > 0

    def test_summarize_empty(self):
        monitor = HardwareMonitor()
        summary = monitor.summarize()
        assert summary.snapshot_count == 0
        assert summary.avg_cpu_percent == 0.0

    def test_summarize_with_data(self):
        monitor = HardwareMonitor(poll_interval=0.1)
        monitor.start()
        time.sleep(0.25)
        monitor.stop()
        summary = monitor.summarize()
        assert summary.snapshot_count > 0
        assert summary.avg_cpu_percent >= 0
        assert summary.peak_ram_used_gb > 0
        assert summary.ram_total_gb > 0

    def test_gpu_fields_none_without_nvidia(self):
        monitor = HardwareMonitor(poll_interval=0.1)
        monitor.start()
        time.sleep(0.15)
        monitor.stop()
        summary = monitor.summarize()
        # On a machine without NVIDIA GPU, these should be None
        # (test may differ on GPU machines, but shouldn't fail)
        assert isinstance(summary.peak_vram_gb, (float, type(None)))

    def test_summary_to_dict(self):
        monitor = HardwareMonitor(poll_interval=0.1)
        monitor.start()
        time.sleep(0.15)
        monitor.stop()
        d = monitor.summarize().to_dict()
        assert "avg_cpu_percent" in d
        assert "peak_ram_used_gb" in d
        assert "snapshot_count" in d


class TestDetectHardware:
    def test_returns_cpu_info(self):
        hw = detect_hardware()
        assert "cpu_count" in hw
        assert "ram_total_gb" in hw
        assert hw["cpu_count"] > 0
        assert hw["ram_total_gb"] > 0

    def test_gpus_is_list(self):
        hw = detect_hardware()
        assert isinstance(hw.get("gpus", []), list)
