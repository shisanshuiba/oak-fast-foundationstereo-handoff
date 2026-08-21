#!/usr/bin/env python3
"""Collect synchronized stereo data while rendering a metric point cloud.

This script reuses the tested PoE connection and capture helpers from
``collect_stereo_dataset.py``.  The saved dataset layout stays compatible with
the original collector, while ``session.json`` additionally stores the
effective intrinsic matrix carried by the actual rectified-left depth frame.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import collect_stereo_dataset as base


cv2 = base.cv2
dai = base.dai
np = base.np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "同步保存左右校正图和毫米深度图，并实时渲染左视角三维点云。"
        )
    )
    parser.add_argument(
        "--device",
        default=base.DEFAULT_DEVICE,
        help="相机 IP、名称或 MXID（默认：%(default)s）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recordings/stereo_depth_pointcloud"),
        help="采集根目录（默认：%(default)s）",
    )
    parser.add_argument("--fps", type=int, default=10, help="采集帧率（默认：%(default)s）")
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="保存帧组数；0 表示持续采集（默认：%(default)s）",
    )
    parser.add_argument("--retries", type=int, default=3, help="PoE 连接次数")
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=10.0,
        help="连接失败后的等待秒数（默认：%(default)s）",
    )
    parser.add_argument(
        "--sync-threshold-ms",
        type=float,
        default=10.0,
        help="设备时间戳最大同步误差，毫秒（默认：%(default)s）",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        choices=range(10),
        default=1,
        metavar="0..9",
        help="PNG 压缩级别（默认：%(default)s）",
    )
    parser.add_argument("--save-npy", action="store_true", help="同时保存深度 NPY")
    parser.add_argument("--no-extended", action="store_true", help="关闭扩展视差")
    parser.add_argument("--no-preview", action="store_true", help="关闭二维采集预览")
    parser.add_argument(
        "--measure-roi",
        type=int,
        default=11,
        help="二维预览测距邻域边长，必须为正奇数（默认：%(default)s）",
    )
    parser.add_argument(
        "--no-pointcloud",
        action="store_true",
        help="关闭 Open3D 点云窗口；用于无界面采集或测试",
    )
    parser.add_argument(
        "--pointcloud-stride",
        type=int,
        default=2,
        help="点云像素采样步长；2 表示横纵各取一半（默认：%(default)s）",
    )
    parser.add_argument(
        "--pointcloud-fps",
        type=float,
        default=5.0,
        help="点云窗口刷新帧率；采集仍按 --fps 保存（默认：%(default)s）",
    )
    parser.add_argument(
        "--pointcloud-min-mm",
        type=int,
        default=200,
        help="点云显示最小深度，毫米（默认：%(default)s）",
    )
    parser.add_argument(
        "--pointcloud-max-mm",
        type=int,
        default=10000,
        help="点云显示最大深度，毫米（默认：%(default)s）",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.5,
        help="Open3D 点大小（默认：%(default)s）",
    )
    parser.add_argument(
        "--pointcloud-color",
        choices=("gray", "depth"),
        default="gray",
        help="点云着色：左目灰度或按深度伪彩色（默认：%(default)s）",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps 必须大于 0")
    if args.frames < 0:
        parser.error("--frames 不能小于 0")
    if args.retries <= 0:
        parser.error("--retries 必须大于 0")
    if args.retry_delay < 0:
        parser.error("--retry-delay 不能小于 0")
    if args.sync_threshold_ms <= 0:
        parser.error("--sync-threshold-ms 必须大于 0")
    if args.measure_roi <= 0 or args.measure_roi % 2 == 0:
        parser.error("--measure-roi 必须为正奇数")
    if args.pointcloud_stride <= 0:
        parser.error("--pointcloud-stride 必须大于 0")
    if args.pointcloud_fps <= 0:
        parser.error("--pointcloud-fps 必须大于 0")
    if args.pointcloud_min_mm < 0:
        parser.error("--pointcloud-min-mm 不能小于 0")
    if args.pointcloud_max_mm <= args.pointcloud_min_mm:
        parser.error("--pointcloud-max-mm 必须大于 --pointcloud-min-mm")
    if args.point_size <= 0:
        parser.error("--point-size 必须大于 0")
    return args


def effective_intrinsics(
    depth_message: Any,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return the 3x3 intrinsic matrix of the depth frame as actually produced."""
    if expected_shape is not None:
        expected_height, expected_width = expected_shape
        frame_width, frame_height = depth_message.getTransformation().getSize()
        if (int(frame_height), int(frame_width)) != expected_shape:
            raise ValueError(
                "深度帧变换尺寸不一致："
                f"transformation={(frame_width, frame_height)}, "
                f"frame={(expected_width, expected_height)}"
            )
    matrix = np.asarray(
        depth_message.getTransformation().getIntrinsicMatrix(),
        dtype=np.float32,
    )
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"深度帧有效内参格式异常：shape={matrix.shape}")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError(f"深度帧有效焦距异常：fx={matrix[0, 0]}, fy={matrix[1, 1]}")
    if abs(float(matrix[2, 2])) < 1e-8:
        raise ValueError("深度帧有效内参 K[2,2] 为 0")
    matrix = matrix / matrix[2, 2]
    return matrix


def depth_to_points(
    depth_mm: np.ndarray,
    intrinsic: np.ndarray,
    stride: int,
    min_depth_mm: int,
    max_depth_mm: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project depth into the rectified-left virtual camera frame.

    Returns ``points_m``, the validity mask on the sampled grid, and sampled
    depth in millimeters.  Point order follows row-major valid pixels.
    """
    if depth_mm.ndim != 2 or depth_mm.dtype != np.uint16:
        raise TypeError("depth_mm 必须是二维 uint16 数组")
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic 必须是 3x3 矩阵")

    sampled = depth_mm[::stride, ::stride]
    rows = np.arange(0, depth_mm.shape[0], stride, dtype=np.float32)
    cols = np.arange(0, depth_mm.shape[1], stride, dtype=np.float32)
    u, v = np.meshgrid(cols, rows)

    valid = (
        (sampled > 0)
        & (sampled < np.iinfo(np.uint16).max)
        & (sampled >= min_depth_mm)
        & (sampled <= max_depth_mm)
    )
    z = sampled.astype(np.float32) / 1000.0
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.column_stack((x[valid], y[valid], z[valid])).astype(
        np.float32,
        copy=False,
    )
    return points, valid, sampled


class PointCloudRenderer:
    """Small Open3D visualizer for left-aligned stereo depth."""

    def __init__(
        self,
        stride: int,
        min_depth_mm: int,
        max_depth_mm: int,
        point_size: float,
        color_mode: str,
        refresh_fps: float,
    ) -> None:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError(
                "实时点云需要 Open3D。请在当前 venv 执行："
                "python -m pip install open3d"
            ) from exc

        self.o3d = o3d
        self.stride = stride
        self.min_depth_mm = min_depth_mm
        self.max_depth_mm = max_depth_mm
        self.color_mode = color_mode
        self.refresh_interval = 1.0 / refresh_fps
        self.last_render_time = 0.0
        self.running = True
        self.visualizer = o3d.visualization.VisualizerWithKeyCallback()
        if not self.visualizer.create_window(
            window_name="DepthAI live point cloud",
            width=1280,
            height=800,
        ):
            raise RuntimeError("无法创建 Open3D 窗口")
        render_option = self.visualizer.get_render_option()
        render_option.point_size = point_size
        render_option.background_color = np.asarray((0.02, 0.02, 0.02))
        self.cloud = o3d.geometry.PointCloud()
        self.geometry_added = False
        self.visualizer.register_key_callback(ord("Q"), self._request_stop)

    def _request_stop(self, _visualizer: Any) -> bool:
        self.running = False
        return False

    def poll(self) -> bool:
        if not self.running:
            return False
        window_running = bool(self.visualizer.poll_events())
        self.running = self.running and window_running
        if self.running:
            self.visualizer.update_renderer()
        return self.running

    def should_render(self, now: float) -> bool:
        return now - self.last_render_time >= self.refresh_interval

    def _colors(
        self,
        left: np.ndarray,
        valid: np.ndarray,
        sampled_depth: np.ndarray,
    ) -> np.ndarray:
        if self.color_mode == "gray":
            gray = left[:: self.stride, :: self.stride][valid].astype(np.float64)
            return np.repeat((gray / 255.0)[:, None], 3, axis=1)

        normalized = np.clip(
            (sampled_depth.astype(np.float32) - self.min_depth_mm)
            * 255.0
            / (self.max_depth_mm - self.min_depth_mm),
            0,
            255,
        ).astype(np.uint8)
        bgr = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        return bgr[..., ::-1][valid].astype(np.float64) / 255.0

    def update(
        self,
        depth: np.ndarray,
        left: np.ndarray,
        intrinsic: np.ndarray,
    ) -> tuple[bool, int]:
        points, valid, sampled_depth = depth_to_points(
            depth,
            intrinsic,
            self.stride,
            self.min_depth_mm,
            self.max_depth_mm,
        )
        self.last_render_time = time.monotonic()
        if points.size == 0:
            return self.poll(), 0
        colors = self._colors(left, valid, sampled_depth)
        self.cloud.points = self.o3d.utility.Vector3dVector(points.astype(np.float64))
        self.cloud.colors = self.o3d.utility.Vector3dVector(colors)

        if not self.geometry_added:
            self.visualizer.add_geometry(self.cloud)
            control = self.visualizer.get_view_control()
            control.set_front((0.0, 0.0, -1.0))
            control.set_up((0.0, -1.0, 0.0))
            control.set_lookat((0.0, 0.0, 1.0))
            control.set_zoom(0.35)
            self.geometry_added = True
        else:
            self.visualizer.update_geometry(self.cloud)

        return self.poll(), int(points.shape[0])

    def close(self) -> None:
        self.visualizer.destroy_window()


def main() -> int:
    args = parse_args()
    os.environ["DEPTHAI_DEVICE_NAME_LIST"] = args.device

    renderer: PointCloudRenderer | None = None
    if not args.no_pointcloud:
        try:
            import open3d  # noqa: F401
        except ImportError:
            print(
                "[错误] 实时点云需要 Open3D。请在当前 venv 执行："
                "python -m pip install open3d",
                file=sys.stderr,
                flush=True,
            )
            return 1

    device: dai.Device | None = None
    pipeline: dai.Pipeline | None = None
    pipeline_started = False
    return_code = 0
    frames_saved = 0
    session_path: Path | None = None
    session_data: dict[str, Any] = {}
    inspector: base.DepthInspector | None = None
    intrinsic: np.ndarray | None = None
    nonempty_pointcloud_reported = False

    try:
        device = base.connect_device(args.device, args.retries, args.retry_delay)
        pipeline, output_queue = base.build_pipeline(
            device,
            args.fps,
            args.sync_threshold_ms,
            not args.no_extended,
        )
        if not args.no_pointcloud:
            renderer = PointCloudRenderer(
                args.pointcloud_stride,
                args.pointcloud_min_mm,
                args.pointcloud_max_mm,
                args.point_size,
                args.pointcloud_color,
                args.pointcloud_fps,
            )

        session_name = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        session_path = args.output.resolve() / session_name
        left_dir = session_path / "left_rectified"
        right_dir = session_path / "right_rectified"
        depth_dir = session_path / "depth_mm"
        for directory in (left_dir, right_dir, depth_dir):
            directory.mkdir(parents=True, exist_ok=False)
        if args.save_npy:
            (session_path / "depth_npy").mkdir()

        base.export_calibration(device, session_path)
        device_info = device.getDeviceInfo()
        session_data = {
            "schema_version": 2,
            "started_at": datetime.now().astimezone().isoformat(),
            "completed": False,
            "device": {
                "name": str(device_info.name),
                "mxid": str(device_info.deviceId),
                "state": str(device_info.state),
                "protocol": str(device_info.protocol),
                "platform": str(device_info.platform),
            },
            "capture": {
                "fps": args.fps,
                "requested_frames": args.frames,
                "sync_threshold_ms": args.sync_threshold_ms,
                "extended_disparity": not args.no_extended,
                "left_right_check": True,
                "depth_alignment": "RECTIFIED_LEFT",
                "depth_unit": "millimeter",
                "invalid_depth_value": 0,
                "saturated_depth_value": 65535,
                "png_compression": args.png_compression,
                "measurement_roi_size": args.measure_roi,
            },
            "pointcloud_preview": {
                "enabled": not args.no_pointcloud,
                "stride": args.pointcloud_stride,
                "refresh_fps": args.pointcloud_fps,
                "min_depth_mm": args.pointcloud_min_mm,
                "max_depth_mm": args.pointcloud_max_mm,
                "color": args.pointcloud_color,
            },
            "frames_saved": 0,
        }
        session_json_path = session_path / "session.json"
        base.write_session_json(session_json_path, session_data)

        metadata_path = session_path / "metadata.csv"
        fieldnames = [
            "index",
            "filename",
            "left_sequence",
            "right_sequence",
            "depth_sequence",
            "left_device_timestamp_ns",
            "right_device_timestamp_ns",
            "depth_device_timestamp_ns",
            "sync_interval_ns",
            "valid_depth_pixels",
            "saturated_depth_pixels",
            "total_pixels",
            "min_depth_mm",
            "median_depth_mm",
            "max_depth_mm",
        ]

        if not args.no_preview:
            inspector = base.DepthInspector(args.measure_roi)
            cv2.namedWindow(base.PREVIEW_WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(base.PREVIEW_WINDOW, inspector.on_mouse)

        pipeline.start()
        pipeline_started = True
        print(f"[采集] 输出目录：{session_path}", flush=True)
        print(
            "[点云] 实时点云使用深度帧有效内参；关闭 Open3D 窗口或按 Q 停止。",
            flush=True,
        )

        with metadata_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            last_wait_notice = time.monotonic()
            while pipeline.isRunning():
                group = output_queue.get(timedelta(milliseconds=100))
                if group is None:
                    if renderer is not None and not renderer.poll():
                        break
                    now = time.monotonic()
                    if now - last_wait_notice >= 10.0:
                        print("[等待] 10 秒未收到同步帧，继续等待 ...", flush=True)
                        last_wait_notice = now
                    continue
                last_wait_notice = time.monotonic()

                left_msg = group["left"]
                right_msg = group["right"]
                depth_msg = group["depth"]
                left = left_msg.getCvFrame()
                right = right_msg.getCvFrame()
                depth = depth_msg.getFrame()

                if left.dtype != np.uint8 or right.dtype != np.uint8:
                    raise TypeError("左/右校正图不是预期的 uint8")
                if depth.dtype != np.uint16:
                    raise TypeError(f"深度图不是预期的 uint16，而是 {depth.dtype}")
                if left.shape != depth.shape or right.shape != depth.shape:
                    raise ValueError(
                        f"图像尺寸不一致：left={left.shape}, right={right.shape}, "
                        f"depth={depth.shape}"
                    )

                if intrinsic is None:
                    intrinsic = effective_intrinsics(depth_msg, depth.shape)
                    session_data["geometry"] = {
                        "depth_camera": {
                            "effective_intrinsics": intrinsic.tolist(),
                            "width": int(depth.shape[1]),
                            "height": int(depth.shape[0]),
                            "reference_stream": "left_rectified",
                            "camera_socket": "CAM_B",
                            "image_geometry": "rectified",
                            "projection_model": "pinhole",
                            "coordinate_system": "RECTIFIED_LEFT_VIRTUAL_CAMERA",
                            "axis_convention": "RDF_RIGHT_DOWN_FORWARD",
                            "depth_unit": "millimeter",
                            "depth_scale_to_meter": 0.001,
                            "intrinsics_source": "ImgTransformation.getIntrinsicMatrix",
                        }
                    }
                    base.write_session_json(session_json_path, session_data)
                    print(
                        "[点云] 已保存有效内参："
                        f"fx={intrinsic[0, 0]:.3f}, fy={intrinsic[1, 1]:.3f}, "
                        f"cx={intrinsic[0, 2]:.3f}, cy={intrinsic[1, 2]:.3f}",
                        flush=True,
                    )
                else:
                    current_intrinsic = effective_intrinsics(depth_msg, depth.shape)
                    if not np.allclose(current_intrinsic, intrinsic, rtol=0.0, atol=1e-4):
                        raise RuntimeError("采集中深度帧有效内参发生变化，已停止以避免错误点云")

                filename = f"{frames_saved:08d}.png"
                left_path = left_dir / filename
                right_path = right_dir / filename
                depth_path = depth_dir / filename
                base.write_png(left_path, left, args.png_compression)
                base.write_png(right_path, right, args.png_compression)
                base.write_png(depth_path, depth, args.png_compression)
                if args.save_npy:
                    np.save(session_path / "depth_npy" / f"{frames_saved:08d}.npy", depth)

                if frames_saved == 0:
                    depth_readback = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                    if (
                        depth_readback is None
                        or depth_readback.dtype != np.uint16
                        or not np.array_equal(depth_readback, depth)
                    ):
                        raise RuntimeError("首张深度 PNG 无损回读校验失败")

                valid_mask = (depth > 0) & (depth < np.iinfo(np.uint16).max)
                valid = depth[valid_mask]
                saturated_pixels = int(
                    np.count_nonzero(depth == np.iinfo(np.uint16).max)
                )
                writer.writerow(
                    {
                        "index": frames_saved,
                        "filename": filename,
                        "left_sequence": left_msg.getSequenceNum(),
                        "right_sequence": right_msg.getSequenceNum(),
                        "depth_sequence": depth_msg.getSequenceNum(),
                        "left_device_timestamp_ns": base.timestamp_ns(left_msg),
                        "right_device_timestamp_ns": base.timestamp_ns(right_msg),
                        "depth_device_timestamp_ns": base.timestamp_ns(depth_msg),
                        "sync_interval_ns": group.getIntervalNs(),
                        "valid_depth_pixels": int(valid.size),
                        "saturated_depth_pixels": saturated_pixels,
                        "total_pixels": int(depth.size),
                        "min_depth_mm": int(valid.min()) if valid.size else "",
                        "median_depth_mm": int(np.median(valid)) if valid.size else "",
                        "max_depth_mm": int(valid.max()) if valid.size else "",
                    }
                )
                frames_saved += 1
                if frames_saved == 1 or frames_saved % 10 == 0:
                    csv_file.flush()
                    print(f"[采集] 已保存 {frames_saved} 组", flush=True)

                stop_requested = False
                if renderer is not None:
                    now = time.monotonic()
                    if renderer.should_render(now):
                        running, point_count = renderer.update(depth, left, intrinsic)
                        if point_count > 0 and not nonempty_pointcloud_reported:
                            print(
                                f"[点云] 首个非空点云：{point_count} 个有效点",
                                flush=True,
                            )
                            nonempty_pointcloud_reported = True
                        elif frames_saved == 1 or frames_saved % 30 == 0:
                            print(f"[点云] 当前渲染 {point_count} 个有效点", flush=True)
                    else:
                        running = renderer.poll()
                    stop_requested = not running

                if not args.no_preview:
                    assert inspector is not None
                    cv2.imshow(
                        base.PREVIEW_WINDOW,
                        base.make_preview(
                            left,
                            right,
                            depth,
                            frames_saved - 1,
                            inspector,
                        ),
                    )
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q")):
                        stop_requested = True
                    if key in (ord("u"), ord("U")):
                        inspector.locked = False

                if stop_requested or (args.frames and frames_saved >= args.frames):
                    break

    except KeyboardInterrupt:
        print("\n[采集] 收到 Ctrl+C，正在安全结束 ...", flush=True)
    except Exception as exc:
        return_code = 1
        print(f"[错误] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        if pipeline_started and pipeline is not None:
            try:
                pipeline.stop()
                print("[关闭] Pipeline 已正常停止", flush=True)
            except Exception as exc:
                return_code = 1
                print(f"[关闭] Pipeline 停止失败：{exc}", file=sys.stderr, flush=True)
        elif device is not None:
            try:
                device.close()
            except Exception as exc:
                print(f"[关闭] Device 关闭失败：{exc}", file=sys.stderr, flush=True)

        if renderer is not None:
            renderer.close()
        if not args.no_preview:
            cv2.destroyAllWindows()

        if session_path is not None and session_data:
            session_data["completed"] = return_code == 0
            session_data["ended_at"] = datetime.now().astimezone().isoformat()
            session_data["frames_saved"] = frames_saved
            try:
                base.write_session_json(session_path / "session.json", session_data)
            except OSError as exc:
                return_code = 1
                print(f"[错误] 无法更新 session.json：{exc}", file=sys.stderr, flush=True)

    if session_path is not None:
        print(f"[完成] 共保存 {frames_saved} 组：{session_path}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
