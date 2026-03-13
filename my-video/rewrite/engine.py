from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import get_paths, resolve_settings
from .prompts import AUTO_MODE, AUTO_REWRITE_PROMPT, DEFAULT_SYSTEM_ROLE, normalize_mode


class RewriteError(RuntimeError):
    """Raised when rewrite execution fails."""


class RewriteEngine:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        config_file: str | os.PathLike | None = None,
        temperature: float = 0.7,
    ):
        self.paths = get_paths()
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.settings = resolve_settings(
            api_key=api_key,
            base_url=base_url,
            model=model,
            config_file=config_file,
        )
        self.temperature = float(temperature)
        self._client: Any | None = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ModuleNotFoundError as exc:
                raise RewriteError("缺少依赖 openai，请先安装 rewrite/requirements.txt") from exc

            self._client = OpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
            )
        return self._client

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            raise RewriteError("text is required")
        return normalized

    def _chat(self, messages: list[dict[str, str]], *, error_prefix: str) -> str:
        try:
            completion = self._get_client().chat.completions.create(
                model=self.settings.model,
                messages=messages,
                temperature=self.temperature,
            )
        except Exception as exc:
            raise RewriteError(f"{error_prefix}: {exc}") from exc

        try:
            content = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise RewriteError(f"{error_prefix}: 返回格式不符合预期") from exc

        text = str(content or "").strip()
        if not text:
            raise RewriteError(f"{error_prefix}: 返回内容为空")
        return text

    def auto_rewrite(self, text: str) -> str:
        source_text = self._normalize_text(text)
        if not self.settings.api_key:
            raise RewriteError("请提供有效的API Key进行自动仿写")

        return self._chat(
            [
                {"role": "system", "content": DEFAULT_SYSTEM_ROLE},
                {
                    "role": "user",
                    "content": f"{AUTO_REWRITE_PROMPT}\n\n原文案：{source_text}",
                },
            ],
            error_prefix="自动仿写失败",
        )

    def rewrite_with_instruction(self, text: str, instruction: str) -> str:
        source_text = self._normalize_text(text)
        prompt_text = str(instruction or "").strip()
        if not prompt_text:
            raise RewriteError("请输入改写指令后再点击按钮")
        if not self.settings.api_key:
            raise RewriteError("请提供有效的API Key进行文案改写")

        return self._chat(
            [
                {"role": "system", "content": DEFAULT_SYSTEM_ROLE},
                {
                    "role": "user",
                    "content": f"{prompt_text}\n\n原文案：{source_text}",
                },
            ],
            error_prefix="根据指令仿写失败",
        )

    def execute_rewrite(self, text: str, mode: str, prompt: str = "") -> str:
        normalized_mode = normalize_mode(mode)
        if normalized_mode == AUTO_MODE:
            return self.auto_rewrite(text)
        return self.rewrite_with_instruction(text, prompt)

    def save_output(self, content: str, output_path: str | os.PathLike | None = None) -> Path:
        text = str(content or "").strip()
        if not text:
            raise RewriteError("content is required")

        output = (
            Path(output_path).expanduser().resolve()
            if output_path
            else self.paths.output_dir / "output_txt.txt"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        return output


def execute_rewrite(
    text: str,
    mode: str,
    prompt: str = "",
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    config_file: str | os.PathLike | None = None,
    temperature: float = 0.7,
) -> str:
    engine = RewriteEngine(
        api_key=api_key,
        base_url=base_url,
        model=model,
        config_file=config_file,
        temperature=temperature,
    )
    return engine.execute_rewrite(text=text, mode=mode, prompt=prompt)
