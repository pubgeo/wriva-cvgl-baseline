from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "inference_configs" / "infer_tiles.jsonc"


def add_config_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str = "Path to inference config file",
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=help_text,
    )
    return parser


def resolve_config_path(config_arg: str | Path) -> Path:
    raw_path = Path(config_arg).expanduser()
    candidates = [raw_path] if raw_path.is_absolute() else [
        Path.cwd() / raw_path,
        SCRIPT_DIR / raw_path,
    ]

    seen = set()
    deduped_candidates = []
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        deduped_candidates.append(candidate)
        if candidate.is_file():
            return candidate

    tried_paths = "\n".join(f"- {candidate}" for candidate in deduped_candidates)
    raise FileNotFoundError(
        f"Could not find config file {str(config_arg)!r}. Tried:\n{tried_paths}"
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _strip_json_comments(text: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False
    in_line_comment = False
    in_block_comment = False
    i = 0

    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                out.append(ch)
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            if ch == "\n":
                out.append(ch)
            i += 1
            continue

        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "\"":
                in_string = False
            i += 1
            continue

        if ch == "\"":
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def load_json_with_comments(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return json.loads(_strip_json_comments(text))


def load_pipeline_config(config_arg: str | Path) -> Dict[str, Any]:
    config = load_json_with_comments(resolve_config_path(config_arg))
    if not isinstance(config, dict):
        raise RuntimeError(
            f"Expected top-level JSON object in {config_arg!r}, got {type(config)}"
        )
    return config


def require_config_section(
    config: Mapping[str, Any],
    section_name: str,
) -> Dict[str, Any]:
    section = config.get(section_name)
    if not isinstance(section, dict):
        raise KeyError(
            f"Missing required config section {section_name!r} in the loaded config file"
        )
    return dict(section)


def get_global_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    return require_config_section(config, "global")


def require_config_value(
    config: Mapping[str, Any],
    key: str,
) -> Any:
    if key not in config:
        raise KeyError(f"Missing required config key {key!r}")
    return config[key]
