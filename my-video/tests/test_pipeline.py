#!/usr/bin/env python3
"""
Digital Human Pipeline - Standardized End-to-End Test

Runs the full pipeline: Douyin Download → Voice TTS → Digital Human → Subtitle → BGM
Monitors CPU/GPU usage, timing, and generates a detailed test report.

Usage:
    python tests/test_pipeline.py --material-dir /path/to/materials [--douyin-url URL]
    python tests/test_pipeline.py --douyin-url "https://v.douyin.com/xxx"
    python tests/test_pipeline.py --material-dir /path --douyin-url URL --output-dir /path/to/output
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline_test")


# ---------------------------------------------------------------------------
# Data classes for test results
# ---------------------------------------------------------------------------

@dataclass
class ResourceSnapshot:
    timestamp: float = 0.0
    cpu_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    gpu_util_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_temp_celsius: Optional[float] = None


@dataclass
class StepResult:
    name: str
    status: str = "pending"  # pending / running / success / failed / skipped
    start_time: float = 0.0
    end_time: float = 0.0
    elapsed_seconds: float = 0.0
    output_files: list[str] = field(default_factory=list)
    output_sizes_mb: list[float] = field(default_factory=list)
    error_message: str = ""
    details: dict = field(default_factory=dict)
    resource_samples: list[dict] = field(default_factory=list)
    peak_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    peak_gpu_util_percent: Optional[float] = None
    peak_gpu_memory_mb: Optional[float] = None
    avg_cpu_percent: float = 0.0
    avg_gpu_util_percent: Optional[float] = None


@dataclass
class PipelineTestReport:
    test_id: str = ""
    start_time: str = ""
    end_time: str = ""
    total_elapsed_seconds: float = 0.0
    platform_info: dict = field(default_factory=dict)
    gpu_info: dict = field(default_factory=dict)
    input_config: dict = field(default_factory=dict)
    steps: list[dict] = field(default_factory=list)
    overall_status: str = "pending"
    summary: str = ""


# ---------------------------------------------------------------------------
# Resource monitoring
# ---------------------------------------------------------------------------

def _get_gpu_info() -> dict:
    """Get GPU information via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,cuda_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 4:
                return {
                    "name": parts[0],
                    "memory_total_mb": float(parts[1]),
                    "driver_version": parts[2],
                    "cuda_version": parts[3],
                }
    except Exception:
        pass
    return {}


def _sample_resources() -> ResourceSnapshot:
    """Take a single resource usage snapshot."""
    snap = ResourceSnapshot(timestamp=time.time())

    try:
        import psutil
        snap.cpu_percent = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()
        snap.memory_used_mb = mem.used / (1024 ** 2)
        snap.memory_total_mb = mem.total / (1024 ** 2)
    except ImportError:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 4:
                snap.gpu_util_percent = float(parts[0])
                snap.gpu_memory_used_mb = float(parts[1])
                snap.gpu_memory_total_mb = float(parts[2])
                snap.gpu_temp_celsius = float(parts[3])
    except Exception:
        pass

    return snap


class ResourceMonitor:
    """Background thread that samples system resources at fixed intervals."""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.samples: list[ResourceSnapshot] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._stop_event.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[ResourceSnapshot]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.samples

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self.samples.append(_sample_resources())
            except Exception:
                pass
            self._stop_event.wait(self.interval)


def _compute_resource_stats(samples: list[ResourceSnapshot]) -> dict:
    """Compute peak and average from resource samples."""
    if not samples:
        return {}
    stats = {
        "peak_cpu_percent": max(s.cpu_percent for s in samples),
        "avg_cpu_percent": round(sum(s.cpu_percent for s in samples) / len(samples), 1),
        "peak_memory_mb": round(max(s.memory_used_mb for s in samples), 1),
    }
    gpu_utils = [s.gpu_util_percent for s in samples if s.gpu_util_percent is not None]
    gpu_mems = [s.gpu_memory_used_mb for s in samples if s.gpu_memory_used_mb is not None]
    if gpu_utils:
        stats["peak_gpu_util_percent"] = max(gpu_utils)
        stats["avg_gpu_util_percent"] = round(sum(gpu_utils) / len(gpu_utils), 1)
    if gpu_mems:
        stats["peak_gpu_memory_mb"] = round(max(gpu_mems), 1)
    return stats


def _file_size_mb(path: Path | str) -> float:
    p = Path(path)
    if p.exists():
        return round(p.stat().st_size / (1024 ** 2), 2)
    return 0.0


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_douyin_download(douyin_url: str, output_dir: Path) -> StepResult:
    """Step 1 (optional): Download video from Douyin."""
    step = StepResult(name="douyin_download")
    monitor = ResourceMonitor()
    monitor.start()
    step.start_time = time.time()
    step.status = "running"

    try:
        from scraper.engine import (
            extract_douyin_share_link,
            resolve_douyin_page_url,
            download_video_ytdlp,
        )

        share_link = extract_douyin_share_link(douyin_url)
        if not share_link:
            raise RuntimeError(f"Cannot extract share link from: {douyin_url}")

        logger.info(f"[Douyin] Resolved share link: {share_link}")
        page_url = resolve_douyin_page_url(share_link)
        logger.info(f"[Douyin] Page URL: {page_url}")

        download_dir = output_dir / "douyin_downloads"
        video_path, title = download_video_ytdlp(page_url, download_dir=download_dir)

        step.output_files = [str(video_path)]
        step.output_sizes_mb = [_file_size_mb(video_path)]
        step.details = {"title": title, "page_url": page_url}
        step.status = "success"
        logger.info(f"[Douyin] Downloaded: {video_path} ({step.output_sizes_mb[0]:.1f} MB)")

    except Exception as e:
        step.status = "failed"
        step.error_message = f"{type(e).__name__}: {e}"
        logger.error(f"[Douyin] Failed: {e}")
        traceback.print_exc()

    step.end_time = time.time()
    step.elapsed_seconds = round(step.end_time - step.start_time, 2)
    samples = monitor.stop()
    step.resource_samples = [asdict(s) for s in samples]
    stats = _compute_resource_stats(samples)
    for k, v in stats.items():
        setattr(step, k, v)
    return step


def step_douyin_transcribe(video_path: str, output_dir: Path) -> StepResult:
    """Step 2 (optional): Transcribe Douyin video to get text."""
    step = StepResult(name="douyin_transcribe")
    monitor = ResourceMonitor()
    monitor.start()
    step.start_time = time.time()
    step.status = "running"

    try:
        from scraper.engine import transcribe_video
        transcript = transcribe_video(video_path, save_text_file=True)
        step.details = {
            "transcript_length": len(transcript),
            "transcript_preview": transcript[:500],
        }
        step.status = "success"
        logger.info(f"[Douyin Transcribe] Got {len(transcript)} chars")
    except Exception as e:
        step.status = "failed"
        step.error_message = f"{type(e).__name__}: {e}"
        logger.error(f"[Douyin Transcribe] Failed: {e}")
        traceback.print_exc()

    step.end_time = time.time()
    step.elapsed_seconds = round(step.end_time - step.start_time, 2)
    samples = monitor.stop()
    step.resource_samples = [asdict(s) for s in samples]
    stats = _compute_resource_stats(samples)
    for k, v in stats.items():
        setattr(step, k, v)
    return step


def step_voice_tts(text: str, speaker: str, output_dir: Path, speed: float = 1.0) -> StepResult:
    """Step 3: Generate speech audio via CosyVoice2 TTS."""
    step = StepResult(name="voice_tts")
    monitor = ResourceMonitor()
    monitor.start()
    step.start_time = time.time()
    step.status = "running"

    try:
        from tts.engine import VoiceEngine
        engine = VoiceEngine()

        available_speakers = engine.list_speakers()
        logger.info(f"[Voice] Available speakers: {available_speakers}")
        step.details["available_speakers"] = available_speakers

        if speaker not in available_speakers:
            if available_speakers:
                speaker = available_speakers[0]
                logger.warning(f"[Voice] Requested speaker not found, using: {speaker}")
            else:
                raise RuntimeError("No speakers available")

        audio_path = output_dir / "tts_output.wav"
        result_path = engine.synthesize_to_file(
            text=text,
            speaker=speaker,
            output_path=audio_path,
            speed=speed,
        )

        step.output_files = [str(result_path)]
        step.output_sizes_mb = [_file_size_mb(result_path)]
        step.details["speaker"] = speaker
        step.details["text_length"] = len(text)
        step.details["text_preview"] = text[:200]
        step.details["speed"] = speed
        step.status = "success"
        logger.info(f"[Voice] Generated audio: {result_path} ({step.output_sizes_mb[0]:.1f} MB)")

    except Exception as e:
        step.status = "failed"
        step.error_message = f"{type(e).__name__}: {e}"
        logger.error(f"[Voice] Failed: {e}")
        traceback.print_exc()

    step.end_time = time.time()
    step.elapsed_seconds = round(step.end_time - step.start_time, 2)
    samples = monitor.stop()
    step.resource_samples = [asdict(s) for s in samples]
    stats = _compute_resource_stats(samples)
    for k, v in stats.items():
        setattr(step, k, v)
    return step


def step_digital_human(audio_path: str, face_video: str, output_dir: Path) -> StepResult:
    """Step 4: Generate lip-synced digital human video."""
    step = StepResult(name="digital_human")
    monitor = ResourceMonitor(interval=3.0)
    monitor.start()
    step.start_time = time.time()
    step.status = "running"

    try:
        from avatar.engine import DigitalHumanEngine
        engine = DigitalHumanEngine()

        video_output = output_dir / "digital_human_raw.mp4"
        result = engine.generate(
            audio=audio_path,
            video=face_video,
            output_path=video_output,
        )

        step.output_files = [str(result.output_path)]
        step.output_sizes_mb = [_file_size_mb(result.output_path)]
        step.details = {
            "runtime": result.runtime,
            "runtime_description": result.runtime_description,
            "engine_elapsed_seconds": result.elapsed_seconds,
            "reference_video": str(result.reference_video),
        }
        step.status = "success"
        logger.info(
            f"[DigitalHuman] Generated: {result.output_path} "
            f"({step.output_sizes_mb[0]:.1f} MB, {result.elapsed_seconds:.1f}s, {result.runtime})"
        )

    except Exception as e:
        step.status = "failed"
        step.error_message = f"{type(e).__name__}: {e}"
        logger.error(f"[DigitalHuman] Failed: {e}")
        traceback.print_exc()

    step.end_time = time.time()
    step.elapsed_seconds = round(step.end_time - step.start_time, 2)
    samples = monitor.stop()
    step.resource_samples = [asdict(s) for s in samples]
    stats = _compute_resource_stats(samples)
    for k, v in stats.items():
        setattr(step, k, v)
    return step


def step_subtitle(audio_path: str, video_path: str, output_dir: Path) -> StepResult:
    """Step 5: Generate subtitles and burn into video."""
    step = StepResult(name="subtitle")
    monitor = ResourceMonitor()
    monitor.start()
    step.start_time = time.time()
    step.status = "running"

    try:
        from subtitle.engine import SubtitleEngine
        engine = SubtitleEngine()

        srt_path = output_dir / "subtitles.srt"
        srt_result = engine.generate_srt(
            input_path=audio_path,
            output_path=str(srt_path),
            language="zh",
        )

        step.details["srt_entries"] = srt_result.entries_count
        step.details["detected_language"] = srt_result.detected_language
        logger.info(f"[Subtitle] SRT generated: {srt_result.srt_path} ({srt_result.entries_count} entries)")

        burned_output = output_dir / "digital_human_subtitled.mp4"
        burn_result = engine.burn_subtitles(
            video_path=video_path,
            subtitle_path=str(srt_result.srt_path),
            output_path=str(burned_output),
        )

        step.output_files = [str(srt_result.srt_path), str(burn_result.output_path)]
        step.output_sizes_mb = [
            _file_size_mb(srt_result.srt_path),
            _file_size_mb(burn_result.output_path),
        ]
        step.status = "success"
        logger.info(f"[Subtitle] Burned: {burn_result.output_path} ({step.output_sizes_mb[1]:.1f} MB)")

    except Exception as e:
        step.status = "failed"
        step.error_message = f"{type(e).__name__}: {e}"
        logger.error(f"[Subtitle] Failed: {e}")
        traceback.print_exc()

    step.end_time = time.time()
    step.elapsed_seconds = round(step.end_time - step.start_time, 2)
    samples = monitor.stop()
    step.resource_samples = [asdict(s) for s in samples]
    stats = _compute_resource_stats(samples)
    for k, v in stats.items():
        setattr(step, k, v)
    return step


def step_bgm(video_path: str, output_dir: Path, bgm_name: Optional[str] = None) -> StepResult:
    """Step 6: Mix background music into final video."""
    step = StepResult(name="bgm_mix")
    monitor = ResourceMonitor()
    monitor.start()
    step.start_time = time.time()
    step.status = "running"

    try:
        from audio_mixer.engine import BgmEngine
        engine = BgmEngine()

        tracks = engine.list_library_tracks()
        step.details["available_tracks"] = [t.name for t in tracks]
        logger.info(f"[BGM] Available tracks: {[t.name for t in tracks]}")

        if not tracks:
            step.status = "skipped"
            step.details["reason"] = "No BGM tracks available in library"
            logger.warning("[BGM] No tracks available, skipping")
        else:
            final_output = output_dir / "final_output.mp4"
            result = engine.mix(
                video_path=video_path,
                bgm_name=bgm_name,
                output_path=str(final_output),
                random_choice=bgm_name is None,
                bgm_volume=0.35,
            )

            step.output_files = [str(result.output_path)]
            step.output_sizes_mb = [_file_size_mb(result.output_path)]
            step.details["track_used"] = result.track_name
            step.details["volume"] = result.volume
            step.details["had_original_audio"] = result.had_original_audio
            step.status = "success"
            logger.info(
                f"[BGM] Mixed: {result.output_path} "
                f"({step.output_sizes_mb[0]:.1f} MB, track={result.track_name})"
            )

    except Exception as e:
        step.status = "failed"
        step.error_message = f"{type(e).__name__}: {e}"
        logger.error(f"[BGM] Failed: {e}")
        traceback.print_exc()

    step.end_time = time.time()
    step.elapsed_seconds = round(step.end_time - step.start_time, 2)
    samples = monitor.stop()
    step.resource_samples = [asdict(s) for s in samples]
    stats = _compute_resource_stats(samples)
    for k, v in stats.items():
        setattr(step, k, v)
    return step


def step_workflow_e2e(
    text: str,
    speaker: str,
    face_video: str,
    output_dir: Path,
    speed: float = 1.0,
    bgm_random: bool = True,
) -> StepResult:
    """Alternative: Run the full workflow engine end-to-end as a single step."""
    step = StepResult(name="workflow_e2e")
    monitor = ResourceMonitor(interval=3.0)
    monitor.start()
    step.start_time = time.time()
    step.status = "running"

    try:
        from pipeline.engine import WorkflowEngine
        engine = WorkflowEngine()

        result = engine.run(
            text=text,
            speaker=speaker,
            video=face_video,
            output=str(output_dir / "workflow_final.mp4"),
            with_subtitles=True,
            bgm_random=bgm_random,
            speed=speed,
        )

        step.output_files = [str(result.final_video_path)]
        step.output_sizes_mb = [_file_size_mb(result.final_video_path)]
        if result.audio_path:
            step.output_files.append(str(result.audio_path))
            step.output_sizes_mb.append(_file_size_mb(result.audio_path))
        if result.subtitle_path:
            step.output_files.append(str(result.subtitle_path))
            step.output_sizes_mb.append(_file_size_mb(result.subtitle_path))

        step.details = {
            "audio_generated": result.audio_generated,
            "subtitle_generated": result.subtitle_generated,
            "subtitle_burned": result.subtitle_burned,
            "bgm_applied": result.bgm_applied,
            "video_runtime": result.video_runtime,
            "video_runtime_description": result.video_runtime_description,
            "video_elapsed_seconds": result.video_elapsed_seconds,
        }
        step.status = "success"
        logger.info(f"[Workflow E2E] Final: {result.final_video_path}")

    except Exception as e:
        step.status = "failed"
        step.error_message = f"{type(e).__name__}: {e}"
        logger.error(f"[Workflow E2E] Failed: {e}")
        traceback.print_exc()

    step.end_time = time.time()
    step.elapsed_seconds = round(step.end_time - step.start_time, 2)
    samples = monitor.stop()
    step.resource_samples = [asdict(s) for s in samples]
    stats = _compute_resource_stats(samples)
    for k, v in stats.items():
        setattr(step, k, v)
    return step


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _get_platform_info() -> dict:
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
    }
    try:
        import psutil
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        mem = psutil.virtual_memory()
        info["memory_total_gb"] = round(mem.total / (1024 ** 3), 1)
    except ImportError:
        pass

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    return info


def generate_report(
    steps: list[StepResult],
    input_config: dict,
    test_start: float,
    test_end: float,
) -> PipelineTestReport:
    """Build a PipelineTestReport from collected step results."""
    report = PipelineTestReport()
    report.test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report.start_time = datetime.fromtimestamp(test_start).isoformat()
    report.end_time = datetime.fromtimestamp(test_end).isoformat()
    report.total_elapsed_seconds = round(test_end - test_start, 2)
    report.platform_info = _get_platform_info()
    report.gpu_info = _get_gpu_info()
    report.input_config = input_config
    report.steps = [asdict(s) for s in steps]

    failed = [s for s in steps if s.status == "failed"]
    skipped = [s for s in steps if s.status == "skipped"]
    success = [s for s in steps if s.status == "success"]

    if failed:
        report.overall_status = "failed"
    elif len(success) + len(skipped) == len(steps):
        report.overall_status = "success"
    else:
        report.overall_status = "partial"

    report.summary = (
        f"Total: {len(steps)} steps | "
        f"Success: {len(success)} | Failed: {len(failed)} | Skipped: {len(skipped)} | "
        f"Time: {report.total_elapsed_seconds:.1f}s"
    )
    return report


def format_report_markdown(report: PipelineTestReport) -> str:
    """Format the report as a readable markdown string."""
    lines = []
    lines.append("# Digital Human Pipeline Test Report")
    lines.append("")
    lines.append(f"**Test ID:** {report.test_id}")
    lines.append(f"**Status:** {report.overall_status.upper()}")
    lines.append(f"**Duration:** {report.total_elapsed_seconds:.1f}s")
    lines.append(f"**Start:** {report.start_time}")
    lines.append(f"**End:** {report.end_time}")
    lines.append("")

    # Platform
    lines.append("## Platform")
    lines.append("")
    pi = report.platform_info
    lines.append(f"| Item | Value |")
    lines.append(f"|------|-------|")
    for k, v in pi.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # GPU
    if report.gpu_info:
        lines.append("## GPU")
        lines.append("")
        lines.append(f"| Item | Value |")
        lines.append(f"|------|-------|")
        for k, v in report.gpu_info.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    # Input Config
    lines.append("## Input Configuration")
    lines.append("")
    lines.append("```json")
    safe_config = {k: str(v) if isinstance(v, Path) else v for k, v in report.input_config.items()}
    lines.append(json.dumps(safe_config, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    # Steps
    lines.append("## Pipeline Steps")
    lines.append("")

    for step_data in report.steps:
        status_icon = {"success": "OK", "failed": "FAIL", "skipped": "SKIP"}.get(
            step_data["status"], "?"
        )
        lines.append(f"### [{status_icon}] {step_data['name']}")
        lines.append("")
        lines.append(f"- **Status:** {step_data['status']}")
        lines.append(f"- **Duration:** {step_data['elapsed_seconds']:.2f}s")
        lines.append(f"- **Peak CPU:** {step_data.get('peak_cpu_percent', 0):.1f}%")
        lines.append(f"- **Avg CPU:** {step_data.get('avg_cpu_percent', 0):.1f}%")
        lines.append(f"- **Peak Memory:** {step_data.get('peak_memory_mb', 0):.0f} MB")

        if step_data.get("peak_gpu_util_percent") is not None:
            lines.append(f"- **Peak GPU Util:** {step_data['peak_gpu_util_percent']:.1f}%")
        if step_data.get("avg_gpu_util_percent") is not None:
            lines.append(f"- **Avg GPU Util:** {step_data['avg_gpu_util_percent']:.1f}%")
        if step_data.get("peak_gpu_memory_mb") is not None:
            lines.append(f"- **Peak GPU Memory:** {step_data['peak_gpu_memory_mb']:.0f} MB")

        if step_data.get("error_message"):
            lines.append(f"- **Error:** `{step_data['error_message']}`")

        if step_data.get("output_files"):
            lines.append("- **Output Files:**")
            for f, s in zip(step_data["output_files"], step_data.get("output_sizes_mb", [])):
                lines.append(f"  - `{f}` ({s:.2f} MB)")

        if step_data.get("details"):
            lines.append("- **Details:**")
            for k, v in step_data["details"].items():
                val_str = str(v)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "..."
                lines.append(f"  - {k}: {val_str}")

        lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(report.summary)
    lines.append("")

    # Timing breakdown
    lines.append("### Timing Breakdown")
    lines.append("")
    lines.append("| Step | Duration (s) | Status |")
    lines.append("|------|-------------|--------|")
    for step_data in report.steps:
        lines.append(
            f"| {step_data['name']} | {step_data['elapsed_seconds']:.2f} | {step_data['status']} |"
        )
    lines.append(f"| **TOTAL** | **{report.total_elapsed_seconds:.2f}** | **{report.overall_status}** |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

def find_face_video(material_dir: Path) -> Optional[str]:
    """Find a usable face reference video in the material directory."""
    for ext in (".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"):
        for f in sorted(material_dir.glob(f"*{ext}")):
            if f.stat().st_size > 0 and not f.name.startswith("."):
                return str(f)
    return None


def run_pipeline(
    material_dir: Optional[str] = None,
    douyin_url: Optional[str] = None,
    output_dir: Optional[str] = None,
    text: Optional[str] = None,
    speaker: str = "",
    speed: float = 1.0,
    face_video: Optional[str] = None,
    bgm_name: Optional[str] = None,
    mode: str = "step_by_step",  # "step_by_step" or "workflow_e2e"
) -> PipelineTestReport:
    """Run the full pipeline test.

    Args:
        material_dir: Directory with face reference videos
        douyin_url: Douyin video URL to download and transcribe for TTS text
        output_dir: Where to store outputs and report
        text: Text for TTS (if not using Douyin transcription)
        speaker: TTS speaker name
        speed: TTS speed
        face_video: Explicit face video path (overrides material_dir scan)
        bgm_name: Specific BGM track name
        mode: "step_by_step" runs each engine individually, "workflow_e2e" uses WorkflowEngine
    """
    test_start = time.time()
    steps: list[StepResult] = []

    # Resolve output dir
    if output_dir:
        out_dir = Path(output_dir)
    elif material_dir:
        out_dir = Path(material_dir) / "test_output"
    else:
        out_dir = PROJECT_ROOT / "test_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve face video
    resolved_face = face_video
    if not resolved_face and material_dir:
        resolved_face = find_face_video(Path(material_dir))
    if not resolved_face:
        # Fallback to built-in face
        builtin_face = PROJECT_ROOT / "avatar" / "faces" / "test-video.mp4"
        if builtin_face.exists():
            resolved_face = str(builtin_face)

    # Collect input config
    input_config = {
        "material_dir": material_dir,
        "douyin_url": douyin_url,
        "output_dir": str(out_dir),
        "text": text,
        "speaker": speaker,
        "speed": speed,
        "face_video": resolved_face,
        "bgm_name": bgm_name,
        "mode": mode,
    }

    logger.info("=" * 70)
    logger.info("DIGITAL HUMAN PIPELINE TEST")
    logger.info("=" * 70)
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Face video: {resolved_face}")
    logger.info(f"Mode: {mode}")

    tts_text = text
    douyin_video_path = None

    # Step 1: Douyin Download (optional)
    if douyin_url:
        logger.info("-" * 50)
        logger.info("STEP 1: Douyin Download")
        logger.info("-" * 50)
        dl_result = step_douyin_download(douyin_url, out_dir)
        steps.append(dl_result)

        if dl_result.status == "success" and dl_result.output_files:
            douyin_video_path = dl_result.output_files[0]

            # Step 2: Douyin Transcribe
            logger.info("-" * 50)
            logger.info("STEP 2: Douyin Transcribe")
            logger.info("-" * 50)
            tr_result = step_douyin_transcribe(douyin_video_path, out_dir)
            steps.append(tr_result)

            if tr_result.status == "success" and not tts_text:
                tts_text = tr_result.details.get("transcript_preview", "")
                logger.info(f"Using Douyin transcript as TTS text: {tts_text[:100]}...")

    # Ensure we have text
    if not tts_text:
        tts_text = (
            "大家好，欢迎来到我的频道。今天我们来聊一聊人工智能在视频制作中的应用。"
            "通过数字人技术，我们可以快速生成高质量的视频内容，大大提升了内容创作的效率。"
        )
        logger.info(f"Using default TTS text: {tts_text}")

    if not resolved_face:
        logger.error("No face video found! Cannot proceed with digital human generation.")
        test_end = time.time()
        report = generate_report(steps, input_config, test_start, test_end)
        report.overall_status = "failed"
        report.summary += " | ERROR: No face reference video found"
        return report

    if mode == "workflow_e2e":
        # Run entire workflow as single step
        logger.info("-" * 50)
        logger.info("STEP: Workflow End-to-End")
        logger.info("-" * 50)
        wf_result = step_workflow_e2e(
            text=tts_text,
            speaker=speaker,
            face_video=resolved_face,
            output_dir=out_dir,
            speed=speed,
            bgm_random=bgm_name is None,
        )
        steps.append(wf_result)
    else:
        # Step-by-step mode
        # Step 3: Voice TTS
        logger.info("-" * 50)
        logger.info("STEP 3: Voice TTS")
        logger.info("-" * 50)
        voice_result = step_voice_tts(tts_text, speaker, out_dir, speed)
        steps.append(voice_result)

        if voice_result.status != "success":
            logger.error("Voice TTS failed, cannot continue pipeline")
            test_end = time.time()
            return generate_report(steps, input_config, test_start, test_end)

        audio_path = voice_result.output_files[0]

        # Step 4: Digital Human
        logger.info("-" * 50)
        logger.info("STEP 4: Digital Human Generation")
        logger.info("-" * 50)
        dh_result = step_digital_human(audio_path, resolved_face, out_dir)
        steps.append(dh_result)

        if dh_result.status != "success":
            logger.error("Digital Human failed, cannot continue pipeline")
            test_end = time.time()
            return generate_report(steps, input_config, test_start, test_end)

        raw_video_path = dh_result.output_files[0]

        # Step 5: Subtitle
        logger.info("-" * 50)
        logger.info("STEP 5: Subtitle Generation & Burning")
        logger.info("-" * 50)
        sub_result = step_subtitle(audio_path, raw_video_path, out_dir)
        steps.append(sub_result)

        # Use subtitled video if available, otherwise raw
        current_video = raw_video_path
        if sub_result.status == "success" and len(sub_result.output_files) >= 2:
            current_video = sub_result.output_files[1]  # burned video

        # Step 6: BGM
        logger.info("-" * 50)
        logger.info("STEP 6: BGM Mixing")
        logger.info("-" * 50)
        bgm_result = step_bgm(current_video, out_dir, bgm_name)
        steps.append(bgm_result)

    test_end = time.time()
    report = generate_report(steps, input_config, test_start, test_end)

    # Save report
    report_json_path = out_dir / f"test_report_{report.test_id}.json"
    report_md_path = out_dir / f"test_report_{report.test_id}.md"

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False, default=str)

    md_content = format_report_markdown(report)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("=" * 70)
    logger.info(f"TEST COMPLETE: {report.overall_status.upper()}")
    logger.info(f"Total time: {report.total_elapsed_seconds:.1f}s")
    logger.info(f"JSON report: {report_json_path}")
    logger.info(f"MD report:   {report_md_path}")
    logger.info("=" * 70)

    # Print markdown report to stdout
    print("\n" + md_content)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Digital Human Pipeline - End-to-End Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--material-dir", "-d",
        help="Directory containing face reference videos (MP4/MOV)",
    )
    parser.add_argument(
        "--douyin-url", "-u",
        help="Douyin share URL to download and transcribe",
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for test results (default: <material-dir>/test_output)",
    )
    parser.add_argument(
        "--text", "-t",
        help="TTS text (overrides Douyin transcription)",
    )
    parser.add_argument(
        "--speaker", "-s",
        default="",
        help="TTS speaker name",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="TTS speed (default: 1.0)",
    )
    parser.add_argument(
        "--face-video", "-f",
        help="Explicit face reference video path",
    )
    parser.add_argument(
        "--bgm-name",
        help="Specific BGM track name",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["step_by_step", "workflow_e2e"],
        default="step_by_step",
        help="Test mode: step_by_step (individual engines) or workflow_e2e (WorkflowEngine)",
    )

    args = parser.parse_args()

    if not args.material_dir and not args.douyin_url:
        parser.error("At least one of --material-dir or --douyin-url is required")

    report = run_pipeline(
        material_dir=args.material_dir,
        douyin_url=args.douyin_url,
        output_dir=args.output_dir,
        text=args.text,
        speaker=args.speaker,
        speed=args.speed,
        face_video=args.face_video,
        bgm_name=args.bgm_name,
        mode=args.mode,
    )

    sys.exit(0 if report.overall_status == "success" else 1)


if __name__ == "__main__":
    main()
