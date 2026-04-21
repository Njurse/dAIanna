"""Fast scene reconstruction helpers for CARMA95.exe captures."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torchvision.transforms as transforms
except Exception:  # Optional dependency for MiDaS mode.
    torch = None
    transforms = None


@dataclass(slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(slots=True)
class DrivingHint:
    command: Literal["forward", "left", "right", "brake"]
    confidence: float
    sector_risk: dict[str, float]


@dataclass(slots=True)
class ReconstructionResult:
    depth_map: np.ndarray
    corrected_depth_map: np.ndarray
    occupancy_grid: np.ndarray
    point_cloud: np.ndarray
    corrected_point_cloud: np.ndarray
    horizon_y: int
    driving_hint: DrivingHint
    processing_ms: float


_midas_model = None
_midas_device = None
_midas_transform = None


def initialize_midas() -> bool:
    """Attempt to initialize MiDaS. Returns True when available."""
    global _midas_model, _midas_device, _midas_transform
    if torch is None or transforms is None:
        print("MiDaS unavailable: torch/torchvision not installed. Using fast mode.")
        return False

    if _midas_model is not None:
        return True

    print("Initializing MiDaS model...")
    _midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    _midas_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _midas_model.to(_midas_device)
    _midas_model.eval()
    _midas_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
        ]
    )
    print(f"MiDaS initialized on {_midas_device}.")
    return True


def _estimate_depth_fast(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    inv_luma = 255 - gray
    h, _ = gray.shape
    row_bias = np.linspace(0.3, 1.0, h, dtype=np.float32)[:, None]

    sobel = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel = cv2.normalize(np.abs(sobel), None, 0.0, 1.0, cv2.NORM_MINMAX)

    depth = (inv_luma.astype(np.float32) * row_bias) * (1.0 - 0.25 * sobel)
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    return depth.astype(np.uint8)


def _estimate_depth_midas(frame_bgr: np.ndarray) -> np.ndarray:
    if _midas_model is None:
        raise RuntimeError("MiDaS not initialized. Call initialize_midas() first.")

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = _midas_transform(image_rgb).unsqueeze(0).to(_midas_device)

    with torch.no_grad():
        prediction = _midas_model(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=frame_bgr.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth = prediction.cpu().numpy()
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    return depth.astype(np.uint8)


def estimate_depth(frame_bgr: np.ndarray, method: Literal["fast", "midas"] = "fast") -> np.ndarray:
    if method == "midas" and _midas_model is not None:
        return _estimate_depth_midas(frame_bgr)
    return _estimate_depth_fast(frame_bgr)


def estimate_horizon(depth_map: np.ndarray) -> int:
    far_threshold = np.percentile(depth_map, 70)
    row_density = (depth_map >= far_threshold).mean(axis=1)
    candidates = np.where(row_density > np.percentile(row_density, 75))[0]
    return int(candidates[0]) if len(candidates) else depth_map.shape[0] // 2


def perspective_warp_correction(depth_map: np.ndarray, horizon_y: int, strength: float = 0.35) -> np.ndarray:
    h, _ = depth_map.shape
    rows = np.arange(h, dtype=np.float32)
    distance = np.clip(rows - float(horizon_y), 0.0, None)
    scale = 1.0 + strength * np.tanh((distance - (h * 0.25)) / (h * 0.25))
    warped = depth_map.astype(np.float32) * scale[:, None]
    warped = cv2.normalize(warped, None, 0, 255, cv2.NORM_MINMAX)
    return warped.astype(np.uint8)


def build_occupancy_grid(corrected_depth_map: np.ndarray, near_percentile: float = 65.0) -> np.ndarray:
    """Create a binary near-obstacle map in image space."""
    threshold = np.percentile(corrected_depth_map, near_percentile)
    near = corrected_depth_map <= threshold
    # Ignore sky/top region; driving hazards live near the lower 2/3.
    h = corrected_depth_map.shape[0]
    near[: h // 3, :] = False
    return near.astype(np.uint8)


def derive_driving_hint(occupancy_grid: np.ndarray) -> DrivingHint:
    """Estimate risk in left/center/right sectors and emit a basic control hint."""
    h, w = occupancy_grid.shape
    bottom = occupancy_grid[h // 2 :, :]
    thirds = [0, w // 3, (2 * w) // 3, w]

    left_risk = float(bottom[:, thirds[0] : thirds[1]].mean())
    center_risk = float(bottom[:, thirds[1] : thirds[2]].mean())
    right_risk = float(bottom[:, thirds[2] : thirds[3]].mean())

    risk = {"left": left_risk, "center": center_risk, "right": right_risk}
    safest = min(risk, key=risk.get)

    if center_risk < 0.12:
        return DrivingHint(command="forward", confidence=float(1.0 - center_risk), sector_risk=risk)
    if min(risk.values()) > 0.28:
        return DrivingHint(command="brake", confidence=float(min(risk.values())), sector_risk=risk)

    command = "left" if safest == "left" else "right" if safest == "right" else "forward"
    confidence = float(1.0 - risk[safest])
    return DrivingHint(command=command, confidence=confidence, sector_risk=risk)


def build_intrinsics(width: int, height: int, fov_degrees: float = 74.0) -> CameraIntrinsics:
    f = (width * 0.5) / np.tan(np.deg2rad(fov_degrees * 0.5))
    return CameraIntrinsics(fx=float(f), fy=float(f), cx=width / 2.0, cy=height / 2.0)


def depth_to_point_cloud(depth_map: np.ndarray, intr: CameraIntrinsics, z_scale: float = 0.04) -> np.ndarray:
    h, w = depth_map.shape
    y, x = np.mgrid[0:h, 0:w]
    z = depth_map.astype(np.float32) * z_scale

    x3d = (x - intr.cx) * z / intr.fx
    y3d = (y - intr.cy) * z / intr.fy
    return np.stack((x3d, -y3d, z), axis=-1).reshape(-1, 3)


def reconstruct_environment(
    frame_bgr: np.ndarray,
    depth_method: Literal["fast", "midas"] = "fast",
    target_size: tuple[int, int] = (320, 240),
) -> ReconstructionResult:
    t0 = time.perf_counter()

    frame = cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_AREA)
    depth = estimate_depth(frame, method=depth_method)
    horizon = estimate_horizon(depth)
    corrected_depth = perspective_warp_correction(depth, horizon)
    occupancy = build_occupancy_grid(corrected_depth)
    hint = derive_driving_hint(occupancy)

    intr = build_intrinsics(width=target_size[0], height=target_size[1])
    points = depth_to_point_cloud(depth, intr)
    corrected_points = depth_to_point_cloud(corrected_depth, intr)

    return ReconstructionResult(
        depth_map=depth,
        corrected_depth_map=corrected_depth,
        occupancy_grid=occupancy,
        point_cloud=points,
        corrected_point_cloud=corrected_points,
        horizon_y=horizon,
        driving_hint=hint,
        processing_ms=(time.perf_counter() - t0) * 1000.0,
    )


_figure = None
_axis = None


def _ensure_plot():
    global _figure, _axis
    if _figure is None:
        _figure = plt.figure("dAIanna Visualizer", figsize=(11, 7))
        _axis = _figure.add_subplot(111, projection="3d")
        plt.ion()


def visualize_result(result: ReconstructionResult, subsample: int = 8) -> None:
    _ensure_plot()
    _axis.clear()

    pts = result.corrected_point_cloud[::subsample]
    _axis.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap="inferno", s=1)
    _axis.set_title(
        "3D Environment | "
        f"{result.processing_ms:.1f} ms | horizon={result.horizon_y}px | "
        f"hint={result.driving_hint.command} ({result.driving_hint.confidence:.2f})"
    )
    _axis.set_xlabel("X")
    _axis.set_ylabel("Y")
    _axis.set_zlabel("Z")
    plt.draw()
    plt.pause(0.001)


def process_video(video_path: str, depth_method: Literal["fast", "midas"] = "fast") -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = reconstruct_environment(frame, depth_method=depth_method)
        visualize_result(result)

        cv2.imshow("Depth (raw)", result.depth_map)
        cv2.imshow("Depth (perspective-corrected)", result.corrected_depth_map)
        cv2.imshow("Occupancy", result.occupancy_grid * 255)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    pass
