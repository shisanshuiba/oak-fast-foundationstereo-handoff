#!/usr/bin/env python3
"""Interactively collect independent synchronized stereo/depth segments.

The camera and pipeline stay running while capture is started and stopped.
Each recording segment gets its own self-contained directory. Depth PNG files
remain uint16 millimeter data; value 0 means invalid depth.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import TextIO

try:
    import msvcrt
except ImportError:  # pragma: no cover - this collector targets Windows
    msvcrt = None

import cv2
import numpy as np

# Reuse the tested connection, calibration, PNG, mouse-inspection, and timeout
# helpers without changing the original collector.
import collect_stereo_dataset as base


dai = base.dai
PREVIEW_WINDOW = "Stereo dataset interactive: left | right | depth"
DEFAULT_SAVE_FPS = 4.0
DEFAULT_WARMUP_FRAMES = 20
QUEUE_TIMEOUT = timedelta(milliseconds=250)

METADATA_FIELDS = [
    "index",
    "filename",
    "left_sequence",
    "right_sequence",
    "depth_sequence",
    "left_device_timestamp_ns",
    "right_device_timestamp_ns",
    "depth_device_timestamp_ns",
    "sync_interval_ns",
    "left_exposure_time_us",
    "right_exposure_time_us",
    "left_sensitivity_iso",
    "right_sensitivity_iso",
    "valid_depth_pixels",
    "saturated_depth_pixels",
    "total_pixels",
    "min_depth_mm",
    "median_depth_mm",
    "max_depth_mm",
]


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "相机和 Pipeline 常驻；R 开始、S 停止、空格切换，"
            "每次开始/停止生成一个独立采集目录。"
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
        default=Path("recordings/stereo_depth"),
        help="所有分段的根目录（默认：%(default)s）",
    )
    parser.add_argument(
        "--camera-fps",
        "--fps",
        dest="camera_fps",
        type=int,
        default=10,
        help="相机及预览帧率；--fps 是兼容别名（默认：%(default)s）",
    )
    parser.add_argument(
        "--save-fps",
        type=float,
        default=DEFAULT_SAVE_FPS,
        help="实际落盘的同步帧组频率，可设 3 或 4（默认：%(default)s）",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=DEFAULT_WARMUP_FRAMES,
        help="启动后只预览不保存的曝光预热帧数（默认：%(default)s）",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="每个分段最多保存的帧组数；0 表示不限（默认：%(default)s）",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="PoE 连接尝试次数（默认：%(default)s）",
    )
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
        help="PNG 压缩级别；越低写入越快（默认：%(default)s）",
    )
    parser.add_argument(
        "--measure-roi",
        type=int,
        default=11,
        help="实时深度测量的正方形邻域边长，必须为正奇数（默认：%(default)s）",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="除 16 位 PNG 外，再保存一份深度 .npy",
    )
    parser.add_argument(
        "--no-extended",
        action="store_true",
        help="关闭扩展视差；中远距离采集可使用此项",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="不显示窗口；在 PowerShell 中仍可用 R/S/空格/Q 控制",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="预热结束后自动开始第一个分段",
    )
    args = parser.parse_args()

    if args.camera_fps <= 0:
        parser.error("--camera-fps/--fps 必须大于 0")
    if args.save_fps <= 0:
        parser.error("--save-fps 必须大于 0")
    if args.save_fps > args.camera_fps:
        parser.error("--save-fps 不能大于 --camera-fps")
    if args.warmup_frames < 0:
        parser.error("--warmup-frames 不能小于 0")
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
    if args.no_preview and msvcrt is None:
        parser.error("--no-preview 的键盘控制只支持 Windows 控制台")
    return args


def build_pipeline(
    device: dai.Device,
    camera_fps: int,
    sync_threshold_ms: float,
    extended_disparity: bool,
) -> tuple[dai.Pipeline, dai.MessageQueue]:
    """Build a synchronized pipeline with a bounded non-blocking host queue."""
    pipeline = dai.Pipeline(device)

    left_camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right_camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    left_output = left_camera.requestFullResolutionOutput(fps=camera_fps)
    right_output = right_camera.requestFullResolutionOutput(fps=camera_fps)

    stereo = pipeline.create(dai.node.StereoDepth)
    left_output.link(stereo.left)
    right_output.link(stereo.right)
    stereo.setRectification(True)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(extended_disparity)
    stereo.setDepthAlign(
        dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT
    )

    sync = pipeline.create(dai.node.Sync)
    sync.setRunOnHost(True)
    sync.setTimestampSource(dai.node.Sync.TimestampSource.DEVICE)
    sync.setSyncThreshold(timedelta(milliseconds=sync_threshold_ms))
    sync.setSyncAttempts(-1)
    stereo.rectifiedLeft.link(sync.inputs["left"])
    stereo.rectifiedRight.link(sync.inputs["right"])
    stereo.depth.link(sync.inputs["depth"])

    queue = sync.out.createOutputQueue(maxSize=4, blocking=False)
    return pipeline, queue


class SaveRateLimiter:
    """Select frames using device timestamps without sleeping or catch-up bursts."""

    def __init__(self, save_fps: float) -> None:
        self.period_ns = max(1, round(1_000_000_000 / save_fps))
        self.next_timestamp_ns: int | None = None
        self.last_timestamp_ns: int | None = None

    def due(self, timestamp_ns: int) -> bool:
        if (
            self.last_timestamp_ns is not None
            and timestamp_ns < self.last_timestamp_ns
        ):
            self.next_timestamp_ns = None
        self.last_timestamp_ns = timestamp_ns

        if self.next_timestamp_ns is None:
            self.next_timestamp_ns = timestamp_ns + self.period_ns
            return True
        if timestamp_ns < self.next_timestamp_ns:
            return False

        missed_periods = (timestamp_ns - self.next_timestamp_ns) // self.period_ns
        self.next_timestamp_ns += (missed_periods + 1) * self.period_ns
        return True


def create_unique_segment_dir(root: Path, segment_number: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    base_name = f"{stamp}_segment{segment_number:03d}"
    for collision in range(1000):
        suffix = "" if collision == 0 else f"_{collision:03d}"
        candidate = root / f"{base_name}{suffix}"
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"无法创建唯一分段目录：{root / base_name}")


def exposure_time_us(message: dai.ImgFrame) -> int | str:
    try:
        return int(round(message.getExposureTime().total_seconds() * 1_000_000))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return ""


def sensitivity_iso(message: dai.ImgFrame) -> int | str:
    try:
        return int(message.getSensitivity())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return ""


def validate_frames(
    left: np.ndarray,
    right: np.ndarray,
    depth: np.ndarray,
) -> None:
    if left.dtype != np.uint8 or right.dtype != np.uint8:
        raise TypeError("左/右校正图不是预期的 uint8")
    if depth.dtype != np.uint16:
        raise TypeError(f"深度图不是预期的 uint16，而是 {depth.dtype}")
    if left.shape != depth.shape or right.shape != depth.shape:
        raise ValueError(
            f"图像尺寸不一致：left={left.shape}, right={right.shape}, "
            f"depth={depth.shape}"
        )


class SegmentRecorder:
    """Own all files and accounting for one start/stop recording segment."""

    def __init__(
        self,
        args: argparse.Namespace,
        device: dai.Device,
        segment_number: int,
    ) -> None:
        self.args = args
        self.segment_number = segment_number
        self.path = create_unique_segment_dir(args.output.resolve(), segment_number)
        self.left_dir = self.path / "left_rectified"
        self.right_dir = self.path / "right_rectified"
        self.depth_dir = self.path / "depth_mm"
        for directory in (self.left_dir, self.right_dir, self.depth_dir):
            directory.mkdir()
        self.depth_npy_dir = self.path / "depth_npy"
        if args.save_npy:
            self.depth_npy_dir.mkdir()

        self.frames_saved = 0
        self.groups_seen = 0
        self.groups_rate_skipped = 0
        self.first_device_timestamp_ns: int | None = None
        self.last_device_timestamp_ns: int | None = None
        self.started_monotonic = time.monotonic()
        self.closed = False
        self.geometry_attempted = False
        self.rate_limiter = SaveRateLimiter(args.save_fps)

        device_info = device.getDeviceInfo()
        self.session_data = {
            "schema_version": 2,
            "capture_mode": "interactive_segment",
            "segment_number": segment_number,
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
                "camera_fps": args.camera_fps,
                "save_fps": args.save_fps,
                "requested_frames_per_segment": args.frames,
                "startup_warmup_frames": args.warmup_frames,
                "sync_threshold_ms": args.sync_threshold_ms,
                "extended_disparity": not args.no_extended,
                "left_right_check": True,
                "depth_alignment": "RECTIFIED_LEFT",
                "depth_unit": "millimeter",
                "invalid_depth_value": 0,
                "saturated_depth_value": base.INVALID_DEPTH_MAX,
                "png_compression": args.png_compression,
                "measurement_roi_size": args.measure_roi,
            },
            "frames_saved": 0,
            "groups_seen_while_recording": 0,
            "groups_skipped_by_rate_limit": 0,
        }
        self.session_json_path = self.path / "session.json"
        base.write_session_json(self.session_json_path, self.session_data)

        self.csv_file: TextIO | None = None
        try:
            base.export_calibration(device, self.path)
            self.csv_file = (self.path / "metadata.csv").open(
                "w",
                newline="",
                encoding="utf-8-sig",
            )
            self.writer = csv.DictWriter(self.csv_file, fieldnames=METADATA_FIELDS)
            self.writer.writeheader()
            self.csv_file.flush()
        except Exception as exc:
            self.session_data.update(
                {
                    "completed": False,
                    "ended_at": datetime.now().astimezone().isoformat(),
                    "stop_reason": "start_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            base.write_session_json(self.session_json_path, self.session_data)
            if self.csv_file is not None:
                self.csv_file.close()
            self.closed = True
            raise

    def _capture_effective_intrinsics(
        self,
        depth_message: dai.ImgFrame,
        depth: np.ndarray,
    ) -> None:
        if self.geometry_attempted:
            return
        self.geometry_attempted = True
        try:
            transform = depth_message.getTransformation()
            matrix = np.asarray(
                transform.getIntrinsicMatrix(),
                dtype=np.float64,
            ).reshape(3, 3)
            if (
                not np.all(np.isfinite(matrix))
                or matrix[0, 0] <= 0
                or matrix[1, 1] <= 0
                or matrix[2, 2] == 0
            ):
                raise ValueError("有效内参矩阵包含非法数值")
            matrix = matrix / matrix[2, 2]
            self.session_data["geometry"] = {
                "depth_camera": {
                    "width": int(depth.shape[1]),
                    "height": int(depth.shape[0]),
                    "effective_intrinsics": matrix.tolist(),
                    "alignment": "RECTIFIED_LEFT",
                    "coordinate_system": "RDF: +X right, +Y down, +Z forward",
                    "depth_unit": "millimeter",
                }
            }
            base.write_session_json(self.session_json_path, self.session_data)
            print("[几何] 已记录当前深度帧的有效内参", flush=True)
        except Exception as exc:
            print(f"[几何] 有效内参记录失败，标定 JSON 仍已保存：{exc}", flush=True)

    def save_group(
        self,
        group: dai.MessageGroup,
        left: np.ndarray,
        right: np.ndarray,
        depth: np.ndarray,
    ) -> bool:
        if self.closed:
            raise RuntimeError("当前分段已关闭，不能继续写入")

        self.groups_seen += 1
        left_msg = group["left"]
        right_msg = group["right"]
        depth_msg = group["depth"]
        reference_timestamp_ns = base.timestamp_ns(left_msg)
        if not self.rate_limiter.due(reference_timestamp_ns):
            self.groups_rate_skipped += 1
            return False

        filename = f"{self.frames_saved:08d}.png"
        left_path = self.left_dir / filename
        right_path = self.right_dir / filename
        depth_path = self.depth_dir / filename
        base.write_png(left_path, left, self.args.png_compression)
        base.write_png(right_path, right, self.args.png_compression)
        base.write_png(depth_path, depth, self.args.png_compression)
        if self.args.save_npy:
            np.save(self.depth_npy_dir / f"{self.frames_saved:08d}.npy", depth)

        if self.frames_saved == 0:
            depth_readback = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if (
                depth_readback is None
                or depth_readback.dtype != np.uint16
                or not np.array_equal(depth_readback, depth)
            ):
                raise RuntimeError("首张深度 PNG 无损回读校验失败")
            print(
                f"[校验] 深度 PNG 无损：shape={depth.shape}, dtype={depth.dtype}",
                flush=True,
            )
            self._capture_effective_intrinsics(depth_msg, depth)

        valid_mask = (depth > 0) & (depth < base.INVALID_DEPTH_MAX)
        valid = depth[valid_mask]
        saturated_pixels = int(np.count_nonzero(depth == base.INVALID_DEPTH_MAX))
        assert self.csv_file is not None
        self.writer.writerow(
            {
                "index": self.frames_saved,
                "filename": filename,
                "left_sequence": left_msg.getSequenceNum(),
                "right_sequence": right_msg.getSequenceNum(),
                "depth_sequence": depth_msg.getSequenceNum(),
                "left_device_timestamp_ns": base.timestamp_ns(left_msg),
                "right_device_timestamp_ns": base.timestamp_ns(right_msg),
                "depth_device_timestamp_ns": base.timestamp_ns(depth_msg),
                "sync_interval_ns": group.getIntervalNs(),
                "left_exposure_time_us": exposure_time_us(left_msg),
                "right_exposure_time_us": exposure_time_us(right_msg),
                "left_sensitivity_iso": sensitivity_iso(left_msg),
                "right_sensitivity_iso": sensitivity_iso(right_msg),
                "valid_depth_pixels": int(valid.size),
                "saturated_depth_pixels": saturated_pixels,
                "total_pixels": int(depth.size),
                "min_depth_mm": int(valid.min()) if valid.size else "",
                "median_depth_mm": int(np.median(valid)) if valid.size else "",
                "max_depth_mm": int(valid.max()) if valid.size else "",
            }
        )

        if self.first_device_timestamp_ns is None:
            self.first_device_timestamp_ns = reference_timestamp_ns
        self.last_device_timestamp_ns = reference_timestamp_ns
        self.frames_saved += 1

        if self.frames_saved == 1 or self.frames_saved % 10 == 0:
            self.csv_file.flush()
            self.session_data["frames_saved"] = self.frames_saved
            self.session_data["groups_seen_while_recording"] = self.groups_seen
            self.session_data["groups_skipped_by_rate_limit"] = (
                self.groups_rate_skipped
            )
            base.write_session_json(self.session_json_path, self.session_data)
            print(
                f"[采集 #{self.segment_number:03d}] 已保存 {self.frames_saved} 组",
                flush=True,
            )
        return True

    def finalize(
        self,
        completed: bool,
        reason: str,
        error: str | None = None,
    ) -> None:
        if self.closed:
            return
        self.closed = True

        close_error: Exception | None = None
        if self.csv_file is not None:
            try:
                self.csv_file.flush()
            except Exception as exc:
                close_error = exc
            try:
                self.csv_file.close()
            except Exception as exc:
                close_error = close_error or exc

        duration_seconds = max(0.0, time.monotonic() - self.started_monotonic)
        self.session_data.update(
            {
                "completed": completed and close_error is None,
                "ended_at": datetime.now().astimezone().isoformat(),
                "stop_reason": reason,
                "frames_saved": self.frames_saved,
                "groups_seen_while_recording": self.groups_seen,
                "groups_skipped_by_rate_limit": self.groups_rate_skipped,
                "duration_seconds": round(duration_seconds, 6),
                "actual_save_fps": round(
                    self.frames_saved / duration_seconds,
                    6,
                )
                if duration_seconds > 0
                else 0.0,
                "first_device_timestamp_ns": self.first_device_timestamp_ns,
                "last_device_timestamp_ns": self.last_device_timestamp_ns,
            }
        )
        if error is not None:
            self.session_data["error"] = error
        if close_error is not None:
            self.session_data["close_error"] = (
                f"{type(close_error).__name__}: {close_error}"
            )
        base.write_session_json(self.session_json_path, self.session_data)
        print(
            f"[分段 #{self.segment_number:03d}] 已停止："
            f"{self.frames_saved} 组，原因={reason}，目录={self.path}",
            flush=True,
        )


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.62,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def make_interactive_preview(
    left: np.ndarray,
    right: np.ndarray,
    depth: np.ndarray,
    inspector: base.DepthInspector,
    recorder: SegmentRecorder | None,
    warmup_seen: int,
    warmup_total: int,
    pending_start: bool,
    save_fps: float,
) -> np.ndarray:
    saved = recorder.frames_saved if recorder is not None else 0
    panel = base.make_preview(
        left,
        right,
        depth,
        saved - 1,
        inspector,
    )
    height, width = panel.shape[:2]
    cv2.rectangle(panel, (0, 0), (width - 1, 68), (0, 0, 0), -1)
    cv2.rectangle(panel, (0, height - 38), (width - 1, height - 1), (0, 0, 0), -1)

    if warmup_seen < warmup_total:
        state = f"WARMUP {warmup_seen}/{warmup_total}"
        if pending_start:
            state += " (START QUEUED)"
        state_color = (0, 215, 255)
    elif recorder is not None:
        state = (
            f"RECORDING  segment={recorder.segment_number:03d}  "
            f"saved={recorder.frames_saved}  target={save_fps:g} Hz"
        )
        state_color = (0, 0, 255)
    else:
        state = f"STANDBY  target={save_fps:g} Hz"
        state_color = (0, 200, 0)

    cv2.circle(panel, (17, 20), 8, state_color, -1, cv2.LINE_AA)
    draw_text(panel, state, (34, 27))

    assert inspector.x is not None and inspector.y is not None
    point_depth, median_depth, roi_valid, roi_total = base.measure_depth(
        depth,
        inspector.x,
        inspector.y,
        inspector.roi_size,
    )
    point_text = (
        f"{point_depth} mm/{point_depth / 1000.0:.3f} m"
        if point_depth is not None
        else "N/A"
    )
    median_text = (
        f"{median_depth} mm/{median_depth / 1000.0:.3f} m"
        if median_depth is not None
        else "N/A"
    )
    mode = "LOCKED" if inspector.locked else "LIVE"
    draw_text(
        panel,
        (
            f"[{mode}] (x,y)=({inspector.x},{inspector.y})  pixel={point_text}  "
            f"ROI{inspector.roi_size} median={median_text}  "
            f"valid={roi_valid}/{roi_total}"
        ),
        (15, 56),
        0.57,
    )
    draw_text(
        panel,
        "R: record | S: stop | SPACE: toggle | U: live measure | Q: quit",
        (15, height - 13),
        0.56,
    )
    return panel


def poll_key(no_preview: bool) -> int | None:
    if not no_preview:
        key = cv2.waitKey(1)
        return None if key < 0 else key & 0xFF

    assert msvcrt is not None
    if not msvcrt.kbhit():
        return None
    character = msvcrt.getwch()
    if character in ("\x00", "\xe0"):
        if msvcrt.kbhit():
            msvcrt.getwch()
        return None
    return ord(character)


def main() -> int:
    args = parse_args()
    os.environ["DEPTHAI_DEVICE_NAME_LIST"] = args.device

    device: dai.Device | None = None
    pipeline: dai.Pipeline | None = None
    pipeline_started = False
    recorder: SegmentRecorder | None = None
    inspector: base.DepthInspector | None = None
    return_code = 0
    normal_exit = False
    exit_reason = "pipeline_stopped"
    error_text: str | None = None
    segment_number = 0
    warmup_seen = 0
    pending_start = args.auto_start
    last_toggle_time = 0.0
    last_wait_notice = time.monotonic()
    last_frames: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def start_segment() -> SegmentRecorder:
        nonlocal segment_number
        segment_number += 1
        new_recorder = SegmentRecorder(args, device, segment_number)  # type: ignore[arg-type]
        print(
            f"[分段 #{segment_number:03d}] 开始采集：{new_recorder.path}",
            flush=True,
        )
        return new_recorder

    try:
        device = base.connect_device(args.device, args.retries, args.retry_delay)
        pipeline, output_queue = build_pipeline(
            device,
            args.camera_fps,
            args.sync_threshold_ms,
            not args.no_extended,
        )

        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=True)
        if not args.no_preview:
            inspector = base.DepthInspector(args.measure_roi)
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(PREVIEW_WINDOW, inspector.on_mouse)

        pipeline.start()
        pipeline_started = True
        print(f"[常驻] Pipeline 已启动，分段根目录：{args.output}", flush=True)
        print(
            f"[预热] 前 {args.warmup_frames} 组只预览、不保存；"
            f"相机={args.camera_fps} FPS，保存={args.save_fps:g} 组/秒。",
            flush=True,
        )
        print(
            "[控制] R=开始，S=停止，空格=开始/停止，Q=安全退出，U=解除测距锁定。",
            flush=True,
        )
        if args.no_preview:
            print("[控制] 请保持此 PowerShell 窗口处于焦点以接收按键。", flush=True)

        while pipeline.isRunning():
            group = output_queue.get(QUEUE_TIMEOUT)
            was_warmup_frame = False
            current_group: dai.MessageGroup | None = None
            current_frames: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

            if group is None:
                if time.monotonic() - last_wait_notice >= 10.0:
                    print("[等待] 10 秒未收到新的同步帧，继续保持连接 ...", flush=True)
                    last_wait_notice = time.monotonic()
            else:
                last_wait_notice = time.monotonic()
                current_group = group
                left = group["left"].getCvFrame()
                right = group["right"].getCvFrame()
                depth = group["depth"].getFrame()
                validate_frames(left, right, depth)
                current_frames = (left, right, depth)
                last_frames = current_frames

                if warmup_seen < args.warmup_frames:
                    was_warmup_frame = True
                    warmup_seen += 1
                    if (
                        warmup_seen == args.warmup_frames
                        or warmup_seen == 1
                        or warmup_seen % 5 == 0
                    ):
                        print(
                            f"[预热] {warmup_seen}/{args.warmup_frames} 组",
                            flush=True,
                        )

            if not args.no_preview and last_frames is not None:
                assert inspector is not None
                cv2.imshow(
                    PREVIEW_WINDOW,
                    make_interactive_preview(
                        *last_frames,
                        inspector,
                        recorder,
                        warmup_seen,
                        args.warmup_frames,
                        pending_start,
                        args.save_fps,
                    ),
                )

            key = poll_key(args.no_preview)
            window_closed = False
            if not args.no_preview and last_frames is not None:
                try:
                    window_closed = (
                        cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1
                    )
                except cv2.error:
                    window_closed = True

            if window_closed or key in (ord("q"), ord("Q")):
                normal_exit = True
                exit_reason = "window_closed" if window_closed else "user_quit"
                break

            if key in (ord("u"), ord("U")) and inspector is not None:
                inspector.locked = False

            toggle_requested = key == ord(" ")
            if toggle_requested:
                now = time.monotonic()
                if now - last_toggle_time < 0.3:
                    toggle_requested = False
                else:
                    last_toggle_time = now

            start_requested = key in (ord("r"), ord("R")) or (
                toggle_requested and recorder is None and not pending_start
            )
            stop_requested = key in (ord("s"), ord("S")) or (
                toggle_requested and (recorder is not None or pending_start)
            )

            if start_requested:
                if recorder is not None:
                    print("[控制] 当前已在采集，忽略重复开始命令", flush=True)
                elif warmup_seen < args.warmup_frames:
                    pending_start = True
                    print("[控制] 已预约开始；预热完成后自动采集", flush=True)
                else:
                    recorder = start_segment()
                    pending_start = False

            if stop_requested:
                if recorder is not None:
                    recorder.finalize(True, "user_stop")
                    recorder = None
                elif pending_start:
                    pending_start = False
                    print("[控制] 已取消预热后的自动开始", flush=True)
                else:
                    print("[控制] 当前处于待机，没有正在采集的分段", flush=True)

            if (
                pending_start
                and recorder is None
                and warmup_seen >= args.warmup_frames
            ):
                recorder = start_segment()
                pending_start = False

            if (
                recorder is not None
                and current_group is not None
                and current_frames is not None
                and not was_warmup_frame
            ):
                recorder.save_group(current_group, *current_frames)
                if args.frames and recorder.frames_saved >= args.frames:
                    recorder.finalize(True, "frame_limit")
                    recorder = None
                    print("[控制] 已达到本分段帧数上限，返回待机", flush=True)

    except KeyboardInterrupt:
        normal_exit = True
        exit_reason = "ctrl_c"
        print("\n[常驻] 收到 Ctrl+C，正在安全结束 ...", flush=True)
    except Exception as exc:
        return_code = 1
        exit_reason = "error"
        error_text = f"{type(exc).__name__}: {exc}"
        print(f"[错误] {error_text}", file=sys.stderr, flush=True)
    finally:
        if recorder is not None:
            try:
                recorder.finalize(normal_exit, exit_reason, error_text)
            except Exception as exc:
                return_code = 1
                print(f"[关闭] 分段收尾失败：{exc}", file=sys.stderr, flush=True)

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

        if not args.no_preview:
            cv2.destroyAllWindows()

    print(f"[完成] 本次程序共创建 {segment_number} 个采集分段", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
