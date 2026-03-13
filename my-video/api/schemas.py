from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


# ── Unified response wrapper ──


class ErrorDetail(BaseModel):
    code: str
    message: str


class APIResponse(BaseModel):
    success: bool
    data: Any = None
    error: Optional[ErrorDetail] = None


# ── Voice ──


class VoiceSynthesizeRequest(BaseModel):
    text: str
    speaker: str
    speed: float = 1.0
    callback_url: Optional[str] = None


class VoiceZeroShotRequest(BaseModel):
    text: str
    prompt_text: str
    speed: float = 1.0


class VoiceExportRequest(BaseModel):
    voice_name: str
    prompt_text: str


class SpeakerItem(BaseModel):
    name: str
    vid: int


# ── Digital Human ──


class DigitalHumanGenerateRequest(BaseModel):
    face: Optional[str] = None
    video_file_id: Optional[str] = None
    audio_file_id: str
    batch_size: int = 4
    sync_offset: int = 0
    scale_h: float = 1.6
    scale_w: float = 3.6
    compress_inference: bool = False
    beautify_teeth: bool = False
    runtime: Optional[str] = None
    callback_url: Optional[str] = None


# ── Subtitle ──


class SubtitleGenerateSrtRequest(BaseModel):
    audio_file_id: str
    language: Optional[str] = "zh"
    max_chars: int = 20
    beam_size: int = 10
    best_of: int = 5
    vad_filter: bool = True
    vad_min_silence_ms: int = 1000
    speech_pad_ms: int = 300
    apply_correction: bool = False
    correction_api_key: Optional[str] = None
    correction_api_base: Optional[str] = None
    correction_model_name: Optional[str] = None
    correction_timeout: Optional[int] = None
    callback_url: Optional[str] = None


class SubtitleCorrectRequest(BaseModel):
    srt_file_id: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model_name: Optional[str] = None
    request_timeout: Optional[int] = None


class SubtitleBurnRequest(BaseModel):
    video_file_id: str
    srt_file_id: str
    font_path: Optional[str] = None
    font_name: Optional[str] = None
    font_index: int = 0
    font_size: int = 24
    font_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline: int = 1
    wrap_style: int = 2
    bottom_margin: int = 30
    callback_url: Optional[str] = None


# ── BGM ──


class BgmMixRequest(BaseModel):
    video_file_id: str
    bgm_path: Optional[str] = None
    bgm_name: Optional[str] = None
    random_choice: bool = False
    bgm_volume: float = 0.35
    original_volume: float = 1.0
    loop_bgm: bool = True
    fade_out_seconds: float = 0.0
    callback_url: Optional[str] = None


class BgmTrackItem(BaseModel):
    name: str
    relative_path: str


# ── Rewrite ──


class RewriteAutoRequest(BaseModel):
    text: str


class RewriteInstructionRequest(BaseModel):
    text: str
    instruction: str


# ── Douyin ──


class DouyinTranscribeRequest(BaseModel):
    share_link: str
    callback_url: Optional[str] = None


class DouyinDownloadRequest(BaseModel):
    share_link: str
    callback_url: Optional[str] = None


# ── Workflow (grouped config) ──


class AudioSourceConfig(BaseModel):
    file_id: Optional[str] = None
    text: Optional[str] = None
    speaker: Optional[str] = None
    prompt_text: Optional[str] = None
    prompt_audio_file_id: Optional[str] = None
    speed: float = 1.0


class DigitalHumanConfig(BaseModel):
    face: Optional[str] = None
    video_file_id: Optional[str] = None
    batch_size: int = 4
    sync_offset: int = 0
    scale_h: float = 1.6
    scale_w: float = 3.6
    compress_inference: bool = False
    beautify_teeth: bool = False
    runtime: Optional[str] = None


class SubtitleGenerateConfig(BaseModel):
    language: str = "zh"
    max_chars: int = 20
    beam_size: int = 10
    best_of: int = 5
    vad_filter: bool = True
    vad_min_silence_ms: int = 1000
    speech_pad_ms: int = 300
    correct: bool = False
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    llm_model: Optional[str] = None
    request_timeout: Optional[int] = None


class SubtitleStyleConfig(BaseModel):
    font_path: Optional[str] = None
    font_name: Optional[str] = None
    font_index: int = 0
    font_size: int = 24
    font_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline: int = 1
    wrap_style: int = 2
    bottom_margin: int = 30


class BgmConfig(BaseModel):
    path: Optional[str] = None
    name: Optional[str] = None
    random: bool = False
    volume: float = 0.35
    original_volume: float = 1.0
    fade_out: float = 0.0
    loop: bool = True


class WorkflowRunRequest(BaseModel):
    audio: AudioSourceConfig
    digital_human: DigitalHumanConfig = Field(default_factory=DigitalHumanConfig)
    subtitle: Optional[SubtitleGenerateConfig] = None
    subtitle_style: Optional[SubtitleStyleConfig] = None
    bgm: Optional[BgmConfig] = None


# ── File info ──


class FileInfo(BaseModel):
    file_id: str
    original_name: str
    category: str
    size_bytes: int
    created_at: float
    download_url: str
