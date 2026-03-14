from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from copywriter.engine import RewriteEngine, RewriteError  # type: ignore
else:
    from .engine import RewriteEngine, RewriteError


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--text", help="直接传入要仿写的文案")
    source_group.add_argument("--input-file", help="从文本文件读取要仿写的文案")
    parser.add_argument("--output", help="将结果写入指定文件")
    parser.add_argument("--api-key", help="覆盖 API Key")
    parser.add_argument("--base-url", help="覆盖 API Base URL")
    parser.add_argument("--model", help="覆盖模型名")
    parser.add_argument("--config-file", help="显式指定配置文件路径")
    parser.add_argument("--temperature", type=float, default=0.7)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reusable copywriting rewrite module for DigiHuman")
    subparsers = parser.add_subparsers(dest="command", required=True)

    auto = subparsers.add_parser("auto", help="使用原项目自动仿写 prompt")
    _add_shared_args(auto)

    prompt = subparsers.add_parser("prompt", help="根据自定义指令仿写")
    _add_shared_args(prompt)
    prompt.add_argument("--instruction", required=True, help="自定义仿写指令")

    execute = subparsers.add_parser("execute", help="按原项目模式名执行仿写")
    _add_shared_args(execute)
    execute.add_argument("--mode", required=True, help="仿写模式，如 AI自动仿写 / 根据指令仿写")
    execute.add_argument("--prompt", default="", help="mode 为指令仿写时使用")

    return parser


def _read_source_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    input_path = Path(args.input_file).expanduser().resolve()
    return input_path.read_text(encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        text = _read_source_text(args)
        engine = RewriteEngine(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            config_file=args.config_file,
            temperature=args.temperature,
        )

        if args.command == "auto":
            result = engine.auto_rewrite(text)
        elif args.command == "prompt":
            result = engine.rewrite_with_instruction(text, args.instruction)
        else:
            result = engine.execute_rewrite(text, mode=args.mode, prompt=args.prompt)

        if args.output:
            output = engine.save_output(result, args.output)
            print(output)
        else:
            print(result)
    except (OSError, RewriteError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
