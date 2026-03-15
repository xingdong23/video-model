from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("api.dependencies")


class EngineManager:
    """Singleton manager: holds engine instances.

    Concurrency is managed by TaskManager queues, not per-engine locks.
    """

    _instance: Optional["EngineManager"] = None

    def __init__(self):
        self._voice_engine = None
        self._digital_human_engine = None
        self._subtitle_engine = None
        self._bgm_engine = None
        self._rewrite_engine = None
        self._workflow_engine = None

    # ── Voice ──

    def init_voice(self):
        from .config import get_settings
        settings = get_settings()
        from tts.engine import VoiceEngine
        logger.info("Loading VoiceEngine (model_dir=%s, fp16=%s) ...", settings.voice_model_dir, settings.voice_fp16)
        self._voice_engine = VoiceEngine(model_dir=settings.voice_model_dir, fp16=settings.voice_fp16)
        logger.info("VoiceEngine ready (%d speakers)", len(self._voice_engine.list_speakers()))

    @property
    def voice_engine(self):
        return self._voice_engine

    # ── Digital Human ──

    def init_digital_human(self):
        from .config import get_settings
        settings = get_settings()
        from avatar.engine import DigitalHumanEngine
        logger.info("Loading DigitalHumanEngine ...")
        self._digital_human_engine = DigitalHumanEngine(
            tuilionnx_dir=settings.digital_human_tuilionnx_dir,
            ffmpeg_bin=settings.digital_human_ffmpeg_bin,
            runtime=settings.digital_human_runtime,
        )
        if settings.digital_human_warmup:
            logger.info("Preparing DigitalHumanEngine runtime ...")
            self._digital_human_engine.prepare_runtime()
        logger.info("DigitalHumanEngine ready")

    @property
    def digital_human_engine(self):
        return self._digital_human_engine

    # ── Subtitle ──

    def init_subtitle(self):
        from .config import get_settings
        settings = get_settings()
        from subtitle.engine import SubtitleEngine
        logger.info("Loading SubtitleEngine ...")
        self._subtitle_engine = SubtitleEngine(
            model_name=settings.subtitle_model_name,
            device=settings.subtitle_device,
            compute_type=settings.subtitle_compute_type,
            ffmpeg_bin=settings.subtitle_ffmpeg_bin,
        )
        logger.info("SubtitleEngine ready")

    @property
    def subtitle_engine(self):
        return self._subtitle_engine

    # ── BGM ──

    def init_bgm(self):
        from .config import get_settings
        settings = get_settings()
        from audio_mixer.engine import BgmEngine
        logger.info("Loading BgmEngine ...")
        self._bgm_engine = BgmEngine(
            library_dir=settings.bgm_library_dir,
            ffmpeg_bin=settings.bgm_ffmpeg_bin,
        )
        logger.info("BgmEngine ready")

    @property
    def bgm_engine(self):
        return self._bgm_engine

    # ── Rewrite ──

    def init_rewrite(self):
        from .config import get_settings
        settings = get_settings()
        from copywriter.engine import RewriteEngine
        logger.info("Loading RewriteEngine ...")
        self._rewrite_engine = RewriteEngine(
            api_key=settings.rewrite_api_key,
            base_url=settings.rewrite_base_url,
            model=settings.rewrite_model,
        )
        logger.info("RewriteEngine ready")

    @property
    def rewrite_engine(self):
        return self._rewrite_engine

    # ── Workflow (reuses sub-engines) ──

    @property
    def workflow_engine(self):
        if self._workflow_engine is None:
            from .config import get_settings
            settings = get_settings()
            from pipeline.engine import WorkflowEngine
            self._workflow_engine = WorkflowEngine(
                voice_model_dir=settings.voice_model_dir,
                tuilionnx_dir=settings.digital_human_tuilionnx_dir,
                ffmpeg_bin=settings.digital_human_ffmpeg_bin,
                runtime=settings.digital_human_runtime,
                subtitle_model_name=settings.subtitle_model_name,
                subtitle_device=settings.subtitle_device,
                subtitle_compute_type=settings.subtitle_compute_type,
                bgm_library_dir=settings.bgm_library_dir,
            )
            # Inject pre-loaded engines if available
            if self._voice_engine is not None:
                self._workflow_engine._voice_engine = self._voice_engine
            if self._digital_human_engine is not None:
                self._workflow_engine._digital_human_engine = self._digital_human_engine
            if self._subtitle_engine is not None:
                self._workflow_engine._subtitle_engine = self._subtitle_engine
            if self._bgm_engine is not None:
                self._workflow_engine._bgm_engine = self._bgm_engine
        return self._workflow_engine

    @classmethod
    def get_instance(cls) -> "EngineManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def get_engine_manager() -> EngineManager:
    return EngineManager.get_instance()
