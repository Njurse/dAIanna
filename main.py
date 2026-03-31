import argparse
import time

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw

import daianna_intro
import depth_module


CAPTURE_SIZE = (640, 480)


def grab_window(window_title: str, size: tuple[int, int] = CAPTURE_SIZE):
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        return None

    window = windows[0]
    left, top = window.left, window.top
    width, height = size

    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return frame


def run_live(window_title: str, depth_method: str = "fast"):
    print(f"Starting live capture for '{window_title}' using '{depth_method}' depth mode...")
    while True:
        frame = grab_window(window_title)
        if frame is None:
            print(f"Waiting for window: {window_title}")
            time.sleep(0.5)
            continue

        result = depth_module.reconstruct_environment(frame, depth_method=depth_method)
        depth_module.visualize_result(result)

        cv2.imshow("CARMA95 Capture", frame)
        cv2.imshow("Depth (raw)", result.depth_map)
        cv2.imshow("Depth (perspective-corrected)", result.corrected_depth_map)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="dAIanna runtime for CARMA95.exe")
    parser.add_argument("--window", default="CARMA95.exe", help="Window title to capture.")
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
    return parser.parse_args()


def main():
    args = parse_args()
    daianna_intro.play_intro()

    if args.depth == "midas":
        depth_module.initialize_midas()

    if args.mode == "video":
        depth_module.process_video(args.video, depth_method=args.depth)
    else:
        run_live(args.window, depth_method=args.depth)


if __name__ == "__main__":
    main()
