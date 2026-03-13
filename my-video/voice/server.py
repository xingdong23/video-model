from __future__ import annotations

import argparse
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

if TYPE_CHECKING:
    from .engine import VoiceEngine


def _load_voice_engine():
    from .engine import VoiceEngine

    return VoiceEngine


def create_app(model_dir: str | None = None) -> Flask:
    app = Flask(__name__)
    CORS(app, origins="*")

    @lru_cache(maxsize=1)
    def get_engine() -> "VoiceEngine":
        VoiceEngine = _load_voice_engine()
        return VoiceEngine(model_dir=model_dir)

    @app.get("/health")
    def health():
        engine = get_engine()
        return jsonify(
            {
                "status": "ok",
                "sample_rate": engine.sample_rate,
                "speakers": len(engine.list_speakers()),
            }
        )

    @app.get("/speakers")
    def speakers():
        engine = get_engine()
        items = [{"name": speaker, "vid": index + 1} for index, speaker in enumerate(engine.list_builtin_speakers())]
        return jsonify(items)

    @app.get("/speakers_list")
    def speakers_list():
        return jsonify(get_engine().list_speakers())

    @app.post("/tts_to_audio")
    def tts_to_audio():
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text", "")).strip()
        speaker = str(payload.get("speaker", "")).strip()
        if not text or not speaker:
            return jsonify({"error": "text and speaker are required"}), 400
        try:
            speed = float(payload.get("speed", 1.0))
        except (TypeError, ValueError):
            return jsonify({"error": "speed must be a number"}), 400
        wav_bytes = get_engine().synthesize_to_wav_bytes(
            text=text,
            speaker=speaker,
            speed=speed,
        )
        return Response(wav_bytes, mimetype="audio/wav")

    @app.route("/inference_zero_shot", methods=["GET", "POST"])
    def inference_zero_shot():
        text = str(request.values.get("tts_text", "")).strip()
        prompt_text = str(request.values.get("prompt_text", "")).strip()
        prompt_wav = request.files.get("prompt_wav")
        if not text or not prompt_text or prompt_wav is None:
            return jsonify({"error": "tts_text, prompt_text and prompt_wav are required"}), 400
        try:
            speed = float(request.values.get("speed", 1.0))
        except (TypeError, ValueError):
            return jsonify({"error": "speed must be a number"}), 400
        temp_path = _persist_upload(prompt_wav)
        try:
            wav_bytes = get_engine().synthesize_zero_shot_to_wav_bytes(
                text=text,
                prompt_text=prompt_text,
                prompt_audio_path=temp_path,
                speed=speed,
            )
            return Response(wav_bytes, mimetype="audio/wav")
        finally:
            _cleanup_temp_file(temp_path)

    @app.post("/voices/export_pt")
    def export_pt():
        voice_name = str(request.values.get("voice_name", "")).strip()
        prompt_text = str(request.values.get("prompt_text", "")).strip()
        prompt_wav = request.files.get("prompt_wav")
        if not voice_name or not prompt_text or prompt_wav is None:
            return jsonify({"error": "voice_name, prompt_text and prompt_wav are required"}), 400
        temp_path = _persist_upload(prompt_wav)
        try:
            saved_path = get_engine().export_custom_voice(
                voice_name=voice_name,
                prompt_text=prompt_text,
                prompt_audio_path=temp_path,
            )
            return jsonify(
                {
                    "status": "ok",
                    "voice_name": voice_name,
                    "path": str(saved_path),
                }
            )
        finally:
            _cleanup_temp_file(temp_path)

    return app


def _persist_upload(upload) -> str:
    suffix = Path(upload.filename or "prompt.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        upload.save(temp_file)
        return temp_file.name


def _cleanup_temp_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local CosyVoice HTTP service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9880)
    parser.add_argument("--model-dir")
    args = parser.parse_args()
    app = create_app(model_dir=args.model_dir)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
