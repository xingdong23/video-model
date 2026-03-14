from __future__ import annotations

import io
import logging
import os
import sys
import wave
from pathlib import Path
from typing import Iterable, List

from .config import get_paths, resolve_model_dir

logger = logging.getLogger(__name__)


def _bootstrap_vendor_paths() -> None:
    paths = get_paths()
    extra_paths = [
        paths.vendor_dir,
        paths.third_party_dir / "AcademiCodec",
        paths.third_party_dir / "Matcha-TTS",
    ]
    for path in reversed(extra_paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_vendor_paths()

from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: E402
from cosyvoice.utils.file_utils import load_wav  # noqa: E402


CUSTOM_VOICE_KEYS = (
    "flow_embedding",
    "llm_embedding",
    "llm_prompt_speech_token",
    "llm_prompt_speech_token_len",
    "flow_prompt_speech_token",
    "flow_prompt_speech_token_len",
    "prompt_speech_feat",
    "prompt_speech_feat_len",
    "prompt_text",
    "prompt_text_len",
)


class VoiceEngine:
    def __init__(self, model_dir: str | os.PathLike | None = None, fp16: bool = True):
        self.paths = get_paths()
        self.model_dir = resolve_model_dir(model_dir)
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        os.environ["COSYVOICE_CUSTOM_VOICE_DIR"] = str(self.paths.speakers_dir)
        self._model = CosyVoice2(str(self.model_dir), fp16=fp16)

    @property
    def sample_rate(self) -> int:
        return int(self._model.sample_rate)

    def list_builtin_speakers(self) -> List[str]:
        return list(self._model.list_available_spks())

    def list_custom_speakers(self) -> List[str]:
        if not self.paths.speakers_dir.exists():
            return []
        return sorted(path.stem for path in self.paths.speakers_dir.glob("*.pt"))

    def list_speakers(self) -> List[str]:
        ordered = []
        for speaker in self.list_builtin_speakers() + self.list_custom_speakers():
            if speaker not in ordered:
                ordered.append(speaker)
        return ordered

    def synthesize_tensor(self, text: str, speaker: str, speed: float = 1.0):
        if not text or not text.strip():
            raise ValueError("text is required")
        if not speaker:
            raise ValueError("speaker is required")
        try:
            return self._collect_tts_outputs(
                self._model.inference_sft(
                    text.strip(),
                    speaker,
                    stream=False,
                    speed=float(speed),
                )
            )
        finally:
            self._clear_cuda_cache()

    def synthesize_to_wav_bytes(self, text: str, speaker: str, speed: float = 1.0) -> bytes:
        waveform = self.synthesize_tensor(text=text, speaker=speaker, speed=speed)
        return tensor_to_wav_bytes(waveform, sample_rate=self.sample_rate)

    def synthesize_to_file(
        self,
        text: str,
        speaker: str,
        output_path: str | os.PathLike,
        speed: float = 1.0,
    ) -> Path:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        wav_bytes = self.synthesize_to_wav_bytes(text=text, speaker=speaker, speed=speed)
        output.write_bytes(wav_bytes)
        return output

    def synthesize_zero_shot_tensor(
        self,
        text: str,
        prompt_text: str,
        prompt_audio_path: str | os.PathLike,
        speed: float = 1.0,
    ):
        if not text or not text.strip():
            raise ValueError("text is required")
        if not prompt_text or not prompt_text.strip():
            raise ValueError("prompt_text is required")
        prompt_speech_16k = self._load_prompt_audio(prompt_audio_path)
        normalized_prompt_text = self._model.frontend.text_normalize(
            prompt_text.strip(),
            split=False,
            text_frontend=True,
        )
        try:
            return self._collect_tts_outputs(
                self._iter_zero_shot_outputs(
                    text=text.strip(),
                    normalized_prompt_text=normalized_prompt_text,
                    prompt_speech_16k=prompt_speech_16k,
                    speed=float(speed),
                )
            )
        finally:
            self._clear_cuda_cache()

    def synthesize_zero_shot_to_wav_bytes(
        self,
        text: str,
        prompt_text: str,
        prompt_audio_path: str | os.PathLike,
        speed: float = 1.0,
    ) -> bytes:
        waveform = self.synthesize_zero_shot_tensor(
            text=text,
            prompt_text=prompt_text,
            prompt_audio_path=prompt_audio_path,
            speed=speed,
        )
        return tensor_to_wav_bytes(waveform, sample_rate=self.sample_rate)

    def synthesize_zero_shot_to_file(
        self,
        text: str,
        prompt_text: str,
        prompt_audio_path: str | os.PathLike,
        output_path: str | os.PathLike,
        speed: float = 1.0,
    ) -> Path:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        wav_bytes = self.synthesize_zero_shot_to_wav_bytes(
            text=text,
            prompt_text=prompt_text,
            prompt_audio_path=prompt_audio_path,
            speed=speed,
        )
        output.write_bytes(wav_bytes)
        return output

    def export_custom_voice(
        self,
        voice_name: str,
        prompt_text: str,
        prompt_audio_path: str | os.PathLike,
        output_path: str | os.PathLike | None = None,
    ) -> Path:
        import torch

        normalized_name = voice_name.strip()
        if not normalized_name:
            raise ValueError("voice_name is required")
        if not prompt_text or not prompt_text.strip():
            raise ValueError("prompt_text is required")
        prompt_speech_16k = self._load_prompt_audio(prompt_audio_path)
        normalized_prompt_text = self._model.frontend.text_normalize(
            prompt_text.strip(),
            split=False,
            text_frontend=True,
        )
        try:
            model_input = self._model.frontend.frontend_zero_shot(
                "你好",
                normalized_prompt_text,
                prompt_speech_16k,
                self.sample_rate,
            )
            voice_payload = {}
            for key in CUSTOM_VOICE_KEYS:
                value = model_input[key]
                if hasattr(value, "detach"):
                    value = value.detach().cpu()
                voice_payload[key] = value
        finally:
            self._clear_cuda_cache()

        destination = (
            Path(output_path).expanduser().resolve()
            if output_path
            else (self.paths.speakers_dir / f"{normalized_name}.pt").resolve()
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(voice_payload, destination)
        return destination

    def _iter_zero_shot_outputs(
        self,
        text: str,
        normalized_prompt_text: str,
        prompt_speech_16k,
        speed: float,
    ) -> Iterable[dict]:
        text_segments = list(
            self._model.frontend.text_normalize(
                text,
                split=True,
                text_frontend=True,
            )
        )
        for segment in text_segments:
            model_input = self._model.frontend.frontend_zero_shot(
                segment,
                normalized_prompt_text,
                prompt_speech_16k,
                self.sample_rate,
            )
            yield from self._model.model.tts(
                **model_input,
                stream=False,
                speed=speed,
            )

    @staticmethod
    def _collect_tts_outputs(outputs: Iterable[dict]):
        import torch

        chunks = []
        for output in outputs:
            chunk = output["tts_speech"]
            if hasattr(chunk, "detach"):
                chunk = chunk.detach().cpu()
            chunks.append(chunk)
        if not chunks:
            raise RuntimeError("no audio generated")
        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks, dim=1)

    @staticmethod
    def _load_prompt_audio(prompt_audio_path: str | os.PathLike):
        prompt_path = Path(prompt_audio_path).expanduser().resolve()
        if not prompt_path.exists():
            raise FileNotFoundError(f"prompt audio not found: {prompt_path}")
        return load_wav(str(prompt_path), 16000)

    @staticmethod
    def _clear_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("CUDA cache clear failed", exc_info=True)


def tensor_to_wav_bytes(waveform, sample_rate: int) -> bytes:
    import numpy as np

    array = waveform.detach().cpu()
    if array.ndim == 2:
        array = array.squeeze(0)
    array = array.clamp(-1, 1).numpy()
    pcm = np.int16(array * 32767)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def synthesize_to_file(
    text: str,
    speaker: str,
    output_path: str | os.PathLike,
    speed: float = 1.0,
    model_dir: str | os.PathLike | None = None,
) -> Path:
    engine = VoiceEngine(model_dir=model_dir)
    return engine.synthesize_to_file(
        text=text,
        speaker=speaker,
        output_path=output_path,
        speed=speed,
    )


def synthesize_zero_shot_to_file(
    text: str,
    prompt_text: str,
    prompt_audio_path: str | os.PathLike,
    output_path: str | os.PathLike,
    speed: float = 1.0,
    model_dir: str | os.PathLike | None = None,
) -> Path:
    engine = VoiceEngine(model_dir=model_dir)
    return engine.synthesize_zero_shot_to_file(
        text=text,
        prompt_text=prompt_text,
        prompt_audio_path=prompt_audio_path,
        output_path=output_path,
        speed=speed,
    )


def export_custom_voice(
    voice_name: str,
    prompt_text: str,
    prompt_audio_path: str | os.PathLike,
    output_path: str | os.PathLike | None = None,
    model_dir: str | os.PathLike | None = None,
) -> Path:
    engine = VoiceEngine(model_dir=model_dir)
    return engine.export_custom_voice(
        voice_name=voice_name,
        prompt_text=prompt_text,
        prompt_audio_path=prompt_audio_path,
        output_path=output_path,
    )
