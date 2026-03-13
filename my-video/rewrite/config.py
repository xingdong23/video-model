from __future__ import annotations

import configparser
import json
import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "deepseek-v3"


@dataclass(frozen=True)
class RewritePaths:
    root: Path = PACKAGE_ROOT
    output_dir: Path = PACKAGE_ROOT / "output"
    local_config_file: Path = PACKAGE_ROOT / "config.ini"
    project_config_file: Path = PROJECT_ROOT / "config.ini"
    repo_config_file: Path = REPO_ROOT / "config.ini"


@dataclass(frozen=True)
class RewriteSettings:
    api_key: str
    base_url: str
    model: str
    config_file: Path | None


def get_paths() -> RewritePaths:
    return RewritePaths()


def clean_api_key(api_key_input: str | None) -> str:
    if not api_key_input:
        return ""

    cleaned = str(api_key_input).strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        try:
            key_list = json.loads(cleaned)
        except json.JSONDecodeError:
            cleaned = cleaned.strip("[]\"' ")
        else:
            if isinstance(key_list, list):
                for item in key_list:
                    candidate = str(item).strip()
                    if candidate:
                        cleaned = candidate
                        break

    if "," in cleaned:
        for item in cleaned.split(","):
            candidate = item.strip()
            if candidate:
                cleaned = candidate
                break

    return cleaned.strip()


def _load_config_file(config_file: str | os.PathLike | None = None) -> tuple[configparser.ConfigParser | None, Path | None]:
    candidates = []
    if config_file:
        candidates.append(Path(config_file).expanduser().resolve())

    paths = get_paths()
    candidates.extend(
        [
            paths.local_config_file,
            paths.project_config_file,
            paths.repo_config_file,
        ]
    )

    seen = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        parser = configparser.ConfigParser()
        parser.read(candidate, encoding="utf-8")
        if parser.sections():
            return parser, candidate
    return None, None


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def resolve_settings(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    config_file: str | os.PathLike | None = None,
) -> RewriteSettings:
    parser, loaded_config = _load_config_file(config_file)

    config_api_key = ""
    config_base_url = ""
    config_model = ""
    if parser:
        config_api_key = clean_api_key(parser.get("deepseek_apikey", "key", fallback=""))
        config_base_url = parser.get("openai", "openai_api_base", fallback="").strip()
        config_model = parser.get("openai", "default_model", fallback="").strip()

    resolved_api_key = clean_api_key(
        api_key
        or _first_env("REWRITE_API_KEY", "DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY")
        or config_api_key
    )
    resolved_base_url = (
        str(
            base_url
            or _first_env("REWRITE_API_BASE", "DASHSCOPE_API_BASE", "OPENAI_API_BASE")
            or config_base_url
            or DEFAULT_API_BASE
        )
        .strip()
    )
    resolved_model = (
        str(
            model
            or _first_env("REWRITE_MODEL", "DEEPSEEK_MODEL", "OPENAI_MODEL")
            or config_model
            or DEFAULT_MODEL
        )
        .strip()
    )

    return RewriteSettings(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        model=resolved_model,
        config_file=loaded_config,
    )
