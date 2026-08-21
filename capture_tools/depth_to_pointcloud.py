#!/usr/bin/env python3
"""Convert recorded DepthAI uint16 depth PNGs to binary little-endian PLY.

The generated XYZ coordinates are expressed in meters in the rectified-left
camera convention: +X right, +Y down, +Z forward.  This is a per-frame
conversion; point clouds from different frames are not registered together.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


CAM_B_SOCKET = 1
INVALID_DEPTH = 0
SATURATED_DEPTH = int(np.iinfo(np.uint16).max)


@dataclass(frozen=True)
class Intrinsics:
    matrix: np.ndarray
    width: int
    height: int
    source: str

    @property
    def fx(self) -> float:
        return float(self.matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.matrix[1, 2])


@dataclass(frozen=True)
class FrameResult:
    depth_path: Path
    output_path: Path
    points: int
    sampled_pixels: int
    colorized: bool
    elapsed_seconds: float


def warning(message: str) -> None:
    print(f"[警告] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "把 collect_stereo_dataset.py 保存的 uint16 毫米深度 PNG "
            "离线转换为二进制小端 PLY 点云。"
        )
    )
    parser.add_argument("session", type=Path, help="采集会话目录（包含 depth_mm）")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--frame",
        type=int,
        default=0,
        help="转换指定帧编号（默认：%(default)s）",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="转换 depth_mm 中的全部帧",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="使用 --all 时每隔多少帧转换一帧（默认：%(default)s）",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="像素采样步长；2 表示横纵方向各取每 2 个像素（默认：%(default)s）",
    )
    parser.add_argument(
        "--min-depth-mm",
        type=int,
        default=1,
        help="保留的最小深度，毫米，含边界（默认：%(default)s）",
    )
    parser.add_argument(
        "--max-depth-mm",
        type=int,
        default=SATURATED_DEPTH - 1,
        help="保留的最大深度，毫米，含边界（默认：%(default)s）",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="读取 left_rectified 同名灰度图，并作为 PLY 的 RGB 灰度颜色",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="输出目录（默认：会话目录下的 pointcloud_ply）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的同名 PLY",
    )
    args = parser.parse_args()

    if args.frame < 0:
        parser.error("--frame 不能小于 0")
    if args.step <= 0:
        parser.error("--step 必须大于 0")
    if args.step != 1 and not args.all:
        parser.error("--step 仅能与 --all 一起使用")
    if args.stride <= 0:
        parser.error("--stride 必须大于 0")
    if not 1 <= args.min_depth_mm < SATURATED_DEPTH:
        parser.error(f"--min-depth-mm 必须在 1..{SATURATED_DEPTH - 1} 内")
    if not 1 <= args.max_depth_mm < SATURATED_DEPTH:
        parser.error(f"--max-depth-mm 必须在 1..{SATURATED_DEPTH - 1} 内")
    if args.min_depth_mm > args.max_depth_mm:
        parser.error("--min-depth-mm 不能大于 --max-depth-mm")
    return args


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise FileNotFoundError(f"缺少文件：{path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 格式错误：{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return data


def validate_matrix(value: object, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 不是有效数值矩阵") from exc
    if matrix.shape != (3, 3):
        raise ValueError(f"{label} 必须是 3x3，实际为 {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} 包含非有限数值")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError(f"{label} 的 fx/fy 必须大于 0")
    if not np.isclose(matrix[2, 2], 1.0):
        raise ValueError(f"{label} 的 K[2,2] 应为 1")
    return matrix


def positive_dimension(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} 必须是正整数")
    try:
        dimension = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是正整数") from exc
    if dimension <= 0 or dimension != value:
        raise ValueError(f"{label} 必须是正整数")
    return dimension


def scale_intrinsics(
    matrix: np.ndarray,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    label: str,
) -> np.ndarray:
    if source_width == target_width and source_height == target_height:
        return matrix.copy()

    source_ratio = source_width / source_height
    target_ratio = target_width / target_height
    if not np.isclose(source_ratio, target_ratio, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"{label} 分辨率为 {source_width}x{source_height}，深度图为 "
            f"{target_width}x{target_height}，宽高比不同，不能可靠地自动缩放内参"
        )

    scaled = matrix.copy()
    scaled[0, :] *= target_width / source_width
    scaled[1, :] *= target_height / source_height
    warning(
        f"{label} 已从 {source_width}x{source_height} 按比例缩放到 "
        f"{target_width}x{target_height}"
    )
    return scaled


def load_effective_intrinsics(
    session_dir: Path,
    image_width: int,
    image_height: int,
) -> Intrinsics | None:
    session_path = session_dir / "session.json"
    if not session_path.is_file():
        return None

    session = load_json(session_path)
    geometry = session.get("geometry")
    if not isinstance(geometry, dict):
        return None
    depth_camera = geometry.get("depth_camera")
    if not isinstance(depth_camera, dict):
        return None
    if "effective_intrinsics" not in depth_camera:
        return None

    label = "session.json: geometry.depth_camera.effective_intrinsics"
    matrix = validate_matrix(depth_camera["effective_intrinsics"], label)
    source_width = positive_dimension(depth_camera.get("width"), f"{label} 的 width")
    source_height = positive_dimension(depth_camera.get("height"), f"{label} 的 height")
    matrix = scale_intrinsics(
        matrix,
        source_width,
        source_height,
        image_width,
        image_height,
        label,
    )

    coordinate_system = depth_camera.get("coordinate_system")
    if coordinate_system not in (None, "RECTIFIED_LEFT_VIRTUAL_CAMERA"):
        warning(
            f"session.json 声明的坐标系为 {coordinate_system!r}，"
            "并非预期的 RECTIFIED_LEFT_VIRTUAL_CAMERA"
        )
    depth_unit = depth_camera.get("depth_unit")
    if depth_unit not in (None, "millimeter"):
        raise ValueError(
            f"session.json 声明的深度单位为 {depth_unit!r}，脚本只接受 millimeter"
        )
    source_name = depth_camera.get("intrinsics_source", "session.json")
    return Intrinsics(
        matrix=matrix,
        width=image_width,
        height=image_height,
        source=f"session.json effective_intrinsics ({source_name})",
    )


def find_cam_b(calibration: dict, calibration_path: Path) -> dict:
    camera_data = calibration.get("cameraData")
    if not isinstance(camera_data, list):
        raise ValueError(f"calibration_active.json 缺少 cameraData：{calibration_path}")

    for item in camera_data:
        if not isinstance(item, list) or len(item) != 2:
            continue
        socket, data = item
        if str(socket) == str(CAM_B_SOCKET) and isinstance(data, dict):
            return data
    raise ValueError(
        f"calibration_active.json 中找不到 CAM_B（socket {CAM_B_SOCKET}）标定"
    )


def load_calibration_intrinsics(
    session_dir: Path,
    image_width: int,
    image_height: int,
) -> Intrinsics:
    calibration_path = session_dir / "calibration_active.json"
    calibration = load_json(calibration_path)
    camera = find_cam_b(calibration, calibration_path)
    label = "calibration_active.json 的 CAM_B intrinsicMatrix"
    matrix = validate_matrix(camera.get("intrinsicMatrix"), label)
    source_width = positive_dimension(camera.get("width"), f"{label} 的 width")
    source_height = positive_dimension(camera.get("height"), f"{label} 的 height")
    matrix = scale_intrinsics(
        matrix,
        source_width,
        source_height,
        image_width,
        image_height,
        label,
    )
    warning(
        "session.json 没有 geometry.depth_camera.effective_intrinsics；"
        "正在回退到 calibration_active.json 的 CAM_B 原始内参。"
        "旧数据通常可以转换，但校正后虚拟相机的有效内参可能略有差异。"
    )
    return Intrinsics(
        matrix=matrix,
        width=image_width,
        height=image_height,
        source="calibration_active.json CAM_B fallback",
    )


def load_intrinsics(session_dir: Path, width: int, height: int) -> Intrinsics:
    effective = load_effective_intrinsics(session_dir, width, height)
    if effective is not None:
        return effective
    return load_calibration_intrinsics(session_dir, width, height)


def read_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise OSError(f"无法读取深度图：{path}")
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise TypeError(
            f"深度图必须是单通道 uint16 PNG：{path}，"
            f"实际 shape={depth.shape}, dtype={depth.dtype}"
        )
    return depth


def read_grayscale(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"无法读取左校正图：{path}")
    if image.dtype != np.uint8 or image.shape != expected_shape:
        raise ValueError(
            f"左校正图必须与深度图同尺寸且为 uint8：{path}，"
            f"实际 shape={image.shape}, dtype={image.dtype}，"
            f"期望 shape={expected_shape}"
        )
    return image


def list_depth_frames(session_dir: Path) -> list[tuple[int, Path]]:
    depth_dir = session_dir / "depth_mm"
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"缺少深度图目录：{depth_dir}")

    frames: list[tuple[int, Path]] = []
    seen: set[int] = set()
    for path in depth_dir.glob("*.png"):
        try:
            index = int(path.stem)
        except ValueError:
            warning(f"忽略非数字帧文件名：{path.name}")
            continue
        if index in seen:
            raise ValueError(f"存在重复帧编号 {index}：{depth_dir}")
        seen.add(index)
        frames.append((index, path))
    frames.sort(key=lambda item: item[0])
    if not frames:
        raise FileNotFoundError(f"没有找到深度 PNG：{depth_dir}")
    return frames


def select_frames(
    frames: list[tuple[int, Path]],
    frame_index: int,
    convert_all: bool,
    step: int,
) -> list[tuple[int, Path]]:
    if convert_all:
        return frames[::step]
    by_index = dict(frames)
    if frame_index not in by_index:
        raise FileNotFoundError(
            f"找不到帧 {frame_index}；可用帧编号范围为 {frames[0][0]}..{frames[-1][0]}"
        )
    return [(frame_index, by_index[frame_index])]


def depth_to_vertices(
    depth: np.ndarray,
    intrinsics: Intrinsics,
    stride: int,
    min_depth_mm: int,
    max_depth_mm: int,
    grayscale: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    sampled_depth = depth[::stride, ::stride]
    valid = (
        (sampled_depth != INVALID_DEPTH)
        & (sampled_depth != SATURATED_DEPTH)
        & (sampled_depth >= min_depth_mm)
        & (sampled_depth <= max_depth_mm)
    )
    sampled_rows, sampled_cols = np.nonzero(valid)
    z = sampled_depth[sampled_rows, sampled_cols].astype(np.float32) / 1000.0
    u = (sampled_cols * stride).astype(np.float32)
    v = (sampled_rows * stride).astype(np.float32)
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    points = np.column_stack((x, y, z)).astype(np.float32, copy=False)

    colors: np.ndarray | None = None
    if grayscale is not None:
        sampled_gray = grayscale[::stride, ::stride]
        colors = sampled_gray[sampled_rows, sampled_cols]
    return points, colors, int(sampled_depth.size)


def write_binary_ply(
    path: Path,
    points: np.ndarray,
    grayscale: np.ndarray | None,
) -> None:
    properties = [
        "property float x",
        "property float y",
        "property float z",
    ]
    dtype_fields: list[tuple[str, str]] = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
    ]
    if grayscale is not None:
        properties.extend(
            ("property uchar red", "property uchar green", "property uchar blue")
        )
        dtype_fields.extend((("red", "u1"), ("green", "u1"), ("blue", "u1")))

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment generated by depth_to_pointcloud.py",
        "comment units meter",
        "comment coordinate_system rectified_left: x_right y_down z_forward",
        f"element vertex {points.shape[0]}",
        *properties,
        "end_header",
        "",
    ]
    vertices = np.empty(points.shape[0], dtype=np.dtype(dtype_fields, align=False))
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    if grayscale is not None:
        vertices["red"] = grayscale
        vertices["green"] = grayscale
        vertices["blue"] = grayscale

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as ply_file:
        ply_file.write("\n".join(header_lines).encode("ascii"))
        vertices.tofile(ply_file)


def convert_frame(
    session_dir: Path,
    depth_path: Path,
    output_path: Path,
    intrinsics: Intrinsics,
    stride: int,
    min_depth_mm: int,
    max_depth_mm: int,
    color: bool,
) -> FrameResult:
    started = time.perf_counter()
    depth = read_depth(depth_path)
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(
            f"帧尺寸不一致：{depth_path} 是 {depth.shape[1]}x{depth.shape[0]}，"
            f"首帧/内参是 {intrinsics.width}x{intrinsics.height}"
        )

    grayscale: np.ndarray | None = None
    if color:
        grayscale = read_grayscale(
            session_dir / "left_rectified" / depth_path.name,
            depth.shape,
        )
    points, point_grayscale, sampled_pixels = depth_to_vertices(
        depth,
        intrinsics,
        stride,
        min_depth_mm,
        max_depth_mm,
        grayscale,
    )
    write_binary_ply(output_path, points, point_grayscale)
    return FrameResult(
        depth_path=depth_path,
        output_path=output_path,
        points=int(points.shape[0]),
        sampled_pixels=sampled_pixels,
        colorized=point_grayscale is not None,
        elapsed_seconds=time.perf_counter() - started,
    )


def main() -> int:
    args = parse_args()
    session_dir = args.session.resolve()
    if not session_dir.is_dir():
        raise FileNotFoundError(f"会话目录不存在：{session_dir}")

    frames = list_depth_frames(session_dir)
    selected = select_frames(frames, args.frame, args.all, args.step)
    first_depth = read_depth(selected[0][1])
    image_height, image_width = first_depth.shape
    intrinsics = load_intrinsics(session_dir, image_width, image_height)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else session_dir / "pointcloud_ply"
    )
    output_paths = [output_dir / f"{index:08d}.ply" for index, _ in selected]
    collisions = [path for path in output_paths if path.exists()]
    if collisions and not args.overwrite:
        preview = ", ".join(str(path) for path in collisions[:3])
        suffix = " ..." if len(collisions) > 3 else ""
        raise FileExistsError(
            f"已有 {len(collisions)} 个输出文件：{preview}{suffix}；"
            "如需覆盖请加 --overwrite"
        )

    print(
        f"[内参] {intrinsics.source} | image={image_width}x{image_height} | "
        f"fx={intrinsics.fx:.6f}, fy={intrinsics.fy:.6f}, "
        f"cx={intrinsics.cx:.6f}, cy={intrinsics.cy:.6f}",
        flush=True,
    )
    print(
        f"[选择] {len(selected)} 帧 | stride={args.stride} | "
        f"depth={args.min_depth_mm}..{args.max_depth_mm} mm | "
        f"color={'gray RGB' if args.color else 'none'}",
        flush=True,
    )

    results: list[FrameResult] = []
    for ((_, depth_path), output_path) in zip(selected, output_paths):
        result = convert_frame(
            session_dir=session_dir,
            depth_path=depth_path,
            output_path=output_path,
            intrinsics=intrinsics,
            stride=args.stride,
            min_depth_mm=args.min_depth_mm,
            max_depth_mm=args.max_depth_mm,
            color=args.color,
        )
        results.append(result)
        print(
            f"[完成] {depth_path.name} -> {output_path.name} | "
            f"points={result.points}/{result.sampled_pixels} | "
            f"color={'yes' if result.colorized else 'no'} | "
            f"{result.elapsed_seconds:.3f}s",
            flush=True,
        )

    total_points = sum(result.points for result in results)
    total_bytes = sum(result.output_path.stat().st_size for result in results)
    print(
        f"[汇总] frames={len(results)}, points={total_points}, "
        f"size={total_bytes / (1024 * 1024):.2f} MiB, output={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"[错误] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
