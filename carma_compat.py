"""Compatibility helpers for Carmageddon Max Pack (CARMA95 renderer stack)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CompatibilityChange:
    path: Path
    updated: bool


def _patch_ddraw_ini(path: Path, renderer: str, windowed: bool) -> CompatibilityChange:
    if not path.exists():
        return CompatibilityChange(path=path, updated=False)

    text = path.read_text(encoding="utf-8", errors="ignore")
    original = text

    # Keep patch conservative: only touch keys known to influence old wrapper behavior.
    text = _set_ini_value(text, "renderer", renderer)
    text = _set_ini_value(text, "windowed", "true" if windowed else "false")

    if text != original:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8", errors="ignore")
        path.write_text(text, encoding="utf-8")
        return CompatibilityChange(path=path, updated=True)

    return CompatibilityChange(path=path, updated=False)


def _set_ini_value(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    found = False
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith(f"{key.lower()}="):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)

    if not found:
        out.append(f"{key}={value}")

    return "\n".join(out) + "\n"


def prepare_renderer_compat(game_root: str, renderer: str = "opengl", windowed: bool = False) -> list[CompatibilityChange]:
    """Patch known ddraw.ini files for Win95 renderer reliability.

    Known layout for Max Pack installs often includes both CARMA and CARSPLAT folders.
    """
    root = Path(game_root).expanduser().resolve()

    candidates = [
        root / "CARMA" / "ddraw.ini",
        root / "CARSPLAT" / "ddraw.ini",
        root / "ddraw.ini",
    ]

    return [_patch_ddraw_ini(path, renderer=renderer, windowed=windowed) for path in candidates]
