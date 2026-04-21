import argparse
import shlex
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw

import daianna_intro
import depth_module
from carma_compat import prepare_renderer_compat


CAPTURE_SIZE = (640, 480)
DEFAULT_WINDOW_CANDIDATES = ("CARMA95.exe", "Carmageddon", "dethrace")


def grab_window(window_titles: tuple[str, ...], size: tuple[int, int] = CAPTURE_SIZE):
    for title in window_titles:
        windows = gw.getWindowsWithTitle(title)
        if windows:
            window = windows[0]
            left, top = window.left, window.top
            width, height = size
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            return frame, title
    return None, None


def launch_game(executable: str, launch_args: str = "", working_dir: str | None = None):
    exe_path = Path(executable).expanduser().resolve()
    if not exe_path.exists():
        raise FileNotFoundError(f"Game executable not found: {exe_path}")

    cwd = working_dir or str(exe_path.parent)
    cmd = [str(exe_path), *shlex.split(launch_args)]
    print(f"Launching game: {' '.join(cmd)} (cwd={cwd})")
    return subprocess.Popen(cmd, cwd=cwd)


def launch_injector(injector_executable: str, working_dir: str | None = None):
    injector = Path(injector_executable).expanduser().resolve()
    if not injector.exists():
        raise FileNotFoundError(f"Injector executable not found: {injector}")

    cwd = working_dir or str(injector.parent)
    print(f"Launching injector UI: {injector}")
    return subprocess.Popen([str(injector)], cwd=cwd)


def run_live(
    window_titles: tuple[str, ...],
    depth_method: str = "fast",
    launch_path: str | None = None,
    launch_args: str = "",
    working_dir: str | None = None,
):
    print(f"Starting live capture for {window_titles} using '{depth_method}' depth mode...")
    launched = False

    while True:
        frame, matched_title = grab_window(window_titles)
        if frame is None:
            if launch_path and not launched:
                launch_game(launch_path, launch_args=launch_args, working_dir=working_dir)
                launched = True
                time.sleep(2.0)
                continue

            print(f"Waiting for game window matching: {window_titles}")
            time.sleep(0.5)
            continue

        result = depth_module.reconstruct_environment(frame, depth_method=depth_method)
        depth_module.visualize_result(result)

        annotated = frame.copy()
        hint = result.driving_hint
        cv2.putText(
            annotated,
            f"Hint: {hint.command} ({hint.confidence:.2f})",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(f"CARMA Capture ({matched_title})", annotated)
        cv2.imshow("Depth (raw)", result.depth_map)
        cv2.imshow("Depth (perspective-corrected)", result.corrected_depth_map)
        cv2.imshow("Occupancy", result.occupancy_grid * 255)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="dAIanna runtime for CARMA95.exe")
    parser.add_argument(
        "--window",
        default=",".join(DEFAULT_WINDOW_CANDIDATES),
        help="Comma-separated window-title candidates (first match is used).",
    )
    parser.add_argument(
        "--mode",
        choices=["live", "video"],
        default="live",
        help="Run against a live game window or a video file.",
    )
    parser.add_argument("--video", default="sample_input.mkv", help="Video path when --mode video is used.")
    parser.add_argument(
        "--depth",
        choices=["fast", "midas"],
        default="fast",
        help="Depth estimation mode.",
    )
    parser.add_argument(
        "--launch-game",
        default=None,
        help="Optional path to CARMA executable; auto-launched when no matching window is found.",
    )
    parser.add_argument("--launch-args", default="", help="Optional CLI args forwarded to --launch-game.")
    parser.add_argument("--working-dir", default=None, help="Optional working dir for launching executables.")
    parser.add_argument(
        "--injector",
        default=None,
        help="Optional path to dAIannaInjector.exe; launched once at startup to assist DLL hooking.",
    )
    parser.add_argument(
        "--game-dir",
        default=None,
        help="Root folder for CARMA/CARSPLAT; used to patch ddraw.ini compatibility settings.",
    )
    parser.add_argument(
        "--prepare-compat",
        action="store_true",
        help="Patch ddraw.ini files for old renderer compatibility before launch/capture.",
    )
    parser.add_argument(
        "--renderer",
        choices=["opengl", "gdi"],
        default="opengl",
        help="Renderer written to ddraw.ini when --prepare-compat is set.",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="If set with --prepare-compat, enforce windowed=true in ddraw.ini for easier capture stability.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    daianna_intro.play_intro()

    if args.depth == "midas":
        depth_module.initialize_midas()

    if args.prepare_compat:
        if not args.game_dir:
            raise ValueError("--prepare-compat requires --game-dir")
        changes = prepare_renderer_compat(
            game_root=args.game_dir,
            renderer=args.renderer,
            windowed=args.windowed,
        )
        for change in changes:
            state = "updated" if change.updated else "unchanged/missing"
            print(f"compat: {change.path} -> {state}")

    if args.injector:
        launch_injector(args.injector, working_dir=args.working_dir)

    if args.mode == "video":
        depth_module.process_video(args.video, depth_method=args.depth)
    else:
        window_titles = tuple(part.strip() for part in args.window.split(",") if part.strip())
        run_live(
            window_titles=window_titles,
            depth_method=args.depth,
            launch_path=args.launch_game,
            launch_args=args.launch_args,
            working_dir=args.working_dir,
        )


if __name__ == "__main__":
    main()
