#!/usr/bin/env python3
"""Collect synchronized rectified stereo images and uint16 depth frames.

The output depth PNG stores the original millimeter values without
visualization normalization. A value of 0 means invalid depth.
"""

from __future__ import annotations

import os

# These settings must be present before importing depthai. They target this
# RVC2 PoE camera and keep slow link-local network boots from timing out.
os.environ.setdefault("DEPTHAI_SEARCH_TIMEOUT", "60000")
os.environ.setdefault("DEPTHAI_CONNECT_TIMEOUT", "60000")
os.environ.setdefault("DEPTHAI_BOOTUP_TIMEOUT", "60000")
os.environ.setdefault("DEPTHAI_WATCHDOG_INITIAL_DELAY", "60000")
os.environ.setdefault("DEPTHAI_PROTOCOL", "tcpip")
os.environ.setdefault("DEPTHAI_PLATFORM", "rvc2")

# depthai 3.9.0 can crash while serializing an RVC2 crash dump during normal
# shutdown. This disables that collection path, not the device watchdog.
os.environ.setdefault("DEPTHAI_CRASHDUMP", "0")
os.environ.setdefault("DEPTHAI_CRASHDUMP_TIMEOUT", "0")

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import depthai as dai
import numpy as np


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_DEVICE = (
    os.environ.get("DEPTHAI_DEVICE_NAME_LIST", "169.254.1.222")
    .split(",")[0]
    .strip()
    or "169.254.1.222"
)
PREVIEW_WINDOW = "Stereo dataset: left | right | depth"
PREVIEW_HEIGHT = 360
INVALID_DEPTH_MAX = int(np.iinfo(np.uint16).max)


def measure_depth(
    depth: np.ndarray,
    x: int,
    y: int,
    roi_size: int,
) -> tuple[int | None, int | None, int, int]:
    """Return point depth and robust local median in millimeters."""
    height, width = depth.shape
    x = min(max(int(x), 0), width - 1)
    y = min(max(int(y), 0), height - 1)
    point_value = int(depth[y, x])
    point_depth = point_value if 0 < point_value < INVALID_DEPTH_MAX else None

    half = roi_size // 2
    x0 = max(0, x - half)
    x1 = min(width, x + half + 1)
    y0 = max(0, y - half)
    y1 = min(height, y + half + 1)
    roi = depth[y0:y1, x0:x1]
    valid = roi[(roi > 0) & (roi < INVALID_DEPTH_MAX)]
    median_depth = int(np.median(valid)) if valid.size else None
    return point_depth, median_depth, int(valid.size), int(roi.size)


class DepthInspector:
    """Map preview mouse positions to the left-aligned depth image."""

    def __init__(self, roi_size: int) -> None:
        self.roi_size = roi_size
        self.source_width = 0
        self.source_height = 0
        self.left_rect = (0, 0, 0, 0)
        self.depth_rect = (0, 0, 0, 0)
        self.x: int | None = None
        self.y: int | None = None
        self.locked = False

    def update_geometry(
        self,
        source_shape: tuple[int, int],
        left_shape: tuple[int, int],
        right_shape: tuple[int, int],
        depth_shape: tuple[int, int],
    ) -> None:
        self.source_height, self.source_width = source_shape
        left_height, left_width = left_shape
        right_height, right_width = right_shape
        depth_height, depth_width = depth_shape
        if left_height != right_height or left_height != depth_height:
            raise ValueError("预览面板高度不一致，无法映射鼠标坐标")
        self.left_rect = (0, 0, left_width, left_height)
        self.depth_rect = (
            left_width + right_width,
            0,
            depth_width,
            depth_height,
        )
        if self.x is None or self.y is None:
            self.x = self.source_width // 2
            self.y = self.source_height // 2
        else:
            self.x = min(max(self.x, 0), self.source_width - 1)
            self.y = min(max(self.y, 0), self.source_height - 1)

    def preview_to_source(self, preview_x: int, preview_y: int) -> tuple[int, int] | None:
        if (
            self.source_width <= 0
            or self.source_height <= 0
        ):
            return None

        selected_rect = next(
            (
                rect
                for rect in (self.left_rect, self.depth_rect)
                if rect[0] <= preview_x < rect[0] + rect[2]
                and rect[1] <= preview_y < rect[1] + rect[3]
            ),
            None,
        )
        if selected_rect is None:
            # Depth is aligned to the left image, not to the right pane.
            return None
        x0, y0, pane_width, pane_height = selected_rect
        local_x = preview_x - x0
        local_y = preview_y - y0

        source_x = min(
            self.source_width - 1,
            int(local_x * self.source_width / pane_width),
        )
        source_y = min(
            self.source_height - 1,
            int(local_y * self.source_height / pane_height),
        )
        return source_x, source_y

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_RBUTTONDOWN:
            self.locked = False
            mapped = self.preview_to_source(x, y)
            if mapped is not None:
                self.x, self.y = mapped
            return

        mapped = self.preview_to_source(x, y)
        if mapped is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x, self.y = mapped
            self.locked = True
        elif event == cv2.EVENT_MOUSEMOVE and not self.locked:
            self.x, self.y = mapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "采集同步的左/右校正灰度图和 uint16 毫米深度图；"
            "按 q 或 Ctrl+C 停止。"
        )
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="相机 IP、名称或 MXID（默认：%(default)s）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recordings/stereo_depth"),
        help="采集根目录（默认：%(default)s）",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="相机与采集帧率（默认：%(default)s）",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="保存帧组数；0 表示持续采集（默认：%(default)s）",
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
        help="PNG 压缩级别；数值越低写入越快（默认：%(default)s）",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="不显示实时预览；此时用 Ctrl+C 停止",
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
        help="关闭扩展视差；远距离场景可使用此项",
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
    return args


def matching_connected_device(identifier: str) -> tuple[dai.DeviceInfo | None, list]:
    """Return a freshly discovered descriptor, including BOOTED devices."""
    try:
        devices = list(dai.Device.getAllConnectedDevices())
    except RuntimeError as exc:
        print(f"[发现] 暂时无法枚举设备：{exc}", flush=True)
        return None, []

    match = next(
        (
            info
            for info in devices
            if str(info.name) == identifier
            or str(info.deviceId) == identifier
            or str(info.getDeviceId()) == identifier
        ),
        None,
    )
    return match, devices


def connect_device(identifier: str, retries: int, retry_delay: float) -> dai.Device:
    """Connect using a fresh DeviceInfo on every PoE retry."""
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        fresh_info, devices = matching_connected_device(identifier)
        if devices:
            found = "; ".join(str(info) for info in devices)
            print(f"[发现] {found}", flush=True)
        else:
            print(f"[发现] 当前未枚举到 {identifier}，改用定向连接", flush=True)

        device_info = fresh_info if fresh_info is not None else dai.DeviceInfo(identifier)
        print(f"[连接] 第 {attempt}/{retries} 次连接 {identifier} ...", flush=True)

        try:
            device = dai.Device(device_info)
            print(f"[连接] 成功：{device.getDeviceInfo()}", flush=True)
            return device
        except RuntimeError as exc:
            last_error = exc
            error_text = str(exc)
            print(f"[连接] 本次失败：{error_text}", flush=True)

            if "ALREADY_IN_USE" in error_text or "already in use" in error_text.lower():
                raise RuntimeError(
                    "相机正被其他程序占用。请关闭 OAK Viewer、其他 Python "
                    "程序或旧采集窗口后重试。"
                ) from exc

            if attempt < retries:
                print(
                    f"[连接] 等待 {retry_delay:g} 秒，让 PoE 固件和网络链路恢复 ...",
                    flush=True,
                )
                time.sleep(retry_delay)

    raise RuntimeError(
        f"连续 {retries} 次无法连接 {identifier}。请确认没有其他程序占用相机；"
        "若设备仍停留在 BOOTED 状态，请等待 watchdog 恢复，或只进行一次 PoE 断电重启。"
    ) from last_error


def build_pipeline(
    device: dai.Device,
    fps: int,
    sync_threshold_ms: float,
    extended_disparity: bool,
) -> tuple[dai.Pipeline, dai.MessageQueue]:
    pipeline = dai.Pipeline(device)

    left_camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right_camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    left_output = left_camera.requestFullResolutionOutput(fps=fps)
    right_output = right_camera.requestFullResolutionOutput(fps=fps)

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

    output_queue = sync.out.createOutputQueue(maxSize=8, blocking=True)
    return pipeline, output_queue


def timestamp_ns(message: dai.ImgFrame) -> int:
    return int(message.getTimestampDevice().total_seconds() * 1_000_000_000)


def make_preview(
    left: np.ndarray,
    right: np.ndarray,
    depth: np.ndarray,
    index: int,
    inspector: DepthInspector,
) -> np.ndarray:
    valid_mask = (depth > 0) & (depth < INVALID_DEPTH_MAX)
    depth_u8 = np.zeros(depth.shape, dtype=np.uint8)

    if np.any(valid_mask):
        valid = depth[valid_mask]
        low, high = np.percentile(valid, (2, 98))
        if high <= low:
            high = low + 1
        depth_u8[valid_mask] = np.clip(
            (depth[valid_mask].astype(np.float32) - low) * 255.0 / (high - low),
            0,
            255,
        ).astype(np.uint8)

    depth_color = cv2.applyColorMap(255 - depth_u8, cv2.COLORMAP_TURBO)
    depth_color[~valid_mask] = 0
    left_color = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    right_color = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

    def resize(image: np.ndarray, interpolation: int) -> np.ndarray:
        width = round(image.shape[1] * PREVIEW_HEIGHT / image.shape[0])
        return cv2.resize(
            image,
            (width, PREVIEW_HEIGHT),
            interpolation=interpolation,
        )

    left_preview = resize(left_color, cv2.INTER_AREA)
    right_preview = resize(right_color, cv2.INTER_AREA)
    depth_preview = resize(depth_color, cv2.INTER_NEAREST)
    inspector.update_geometry(
        depth.shape,
        left_preview.shape[:2],
        right_preview.shape[:2],
        depth_preview.shape[:2],
    )
    panel = np.hstack((left_preview, right_preview, depth_preview))

    assert inspector.x is not None and inspector.y is not None
    point_depth, median_depth, roi_valid, roi_total = measure_depth(
        depth,
        inspector.x,
        inspector.y,
        inspector.roi_size,
    )

    def source_to_preview(rect: tuple[int, int, int, int]) -> tuple[int, int]:
        x0, y0, pane_width, pane_height = rect
        marker_x = x0 + min(
            pane_width - 1,
            int(inspector.x * pane_width / depth.shape[1]),
        )
        marker_y = y0 + min(
            pane_height - 1,
            int(inspector.y * pane_height / depth.shape[0]),
        )
        return marker_x, marker_y

    for rect in (inspector.left_rect, inspector.depth_rect):
        cv2.drawMarker(
            panel,
            source_to_preview(rect),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=22,
            thickness=2,
            line_type=cv2.LINE_AA,
        )

    point_text = (
        f"{point_depth} mm ({point_depth / 1000.0:.3f} m)"
        if point_depth is not None
        else "N/A"
    )
    median_text = (
        f"{median_depth} mm ({median_depth / 1000.0:.3f} m)"
        if median_depth is not None
        else "N/A"
    )
    mode = "LOCKED" if inspector.locked else "LIVE"
    text_lines = (
        f"saved={index + 1}  LEFT (depth reference) | RIGHT | DEPTH (mm)",
        (
            f"[{mode}] (x,y)=({inspector.x},{inspector.y})  "
            f"pixel={point_text}  ROI{inspector.roi_size} median={median_text}  "
            f"valid={roi_valid}/{roi_total}"
        ),
        "Move over LEFT/DEPTH: measure | Left click: lock | Right click or U: live | Q: quit",
    )

    def draw_text(text: str, origin: tuple[int, int], scale: float = 0.65) -> None:
        cv2.putText(
            panel,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    draw_text(text_lines[0], (15, 27))
    draw_text(text_lines[1], (15, 55))
    draw_text(text_lines[2], (15, PREVIEW_HEIGHT - 14), 0.55)
    return panel


def write_png(path: Path, image: np.ndarray, compression: int) -> None:
    ok = cv2.imwrite(
        str(path),
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, compression],
    )
    if not ok:
        raise OSError(f"写入失败：{path}")


def export_calibration(device: dai.Device, session_dir: Path) -> None:
    active_path = session_dir / "calibration_active.json"
    if not device.getCalibration().eepromToJsonFile(active_path):
        raise RuntimeError(f"无法导出活动标定：{active_path}")

    try:
        factory_path = session_dir / "calibration_factory.json"
        if not device.readFactoryCalibration().eepromToJsonFile(factory_path):
            print("[标定] 未能导出工厂标定；活动标定已保存", flush=True)
    except RuntimeError as exc:
        print(f"[标定] 没有可导出的工厂标定：{exc}", flush=True)


def write_session_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    os.environ["DEPTHAI_DEVICE_NAME_LIST"] = args.device

    device: dai.Device | None = None
    pipeline: dai.Pipeline | None = None
    pipeline_started = False
    return_code = 0
    frames_saved = 0
    session_path: Path | None = None
    session_data: dict = {}
    inspector: DepthInspector | None = None

    try:
        device = connect_device(args.device, args.retries, args.retry_delay)
        pipeline, output_queue = build_pipeline(
            device,
            args.fps,
            args.sync_threshold_ms,
            not args.no_extended,
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

        export_calibration(device, session_path)
        device_info = device.getDeviceInfo()
        session_data = {
            "schema_version": 1,
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
            "frames_saved": 0,
        }
        session_json_path = session_path / "session.json"
        write_session_json(session_json_path, session_data)

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
            inspector = DepthInspector(args.measure_roi)
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(PREVIEW_WINDOW, inspector.on_mouse)

        pipeline.start()
        pipeline_started = True
        print(f"[采集] 输出目录：{session_path}", flush=True)
        print(
            "[采集] 保存 rectified left/right PNG + uint16 depth PNG；"
            "按 q 或 Ctrl+C 停止。",
            flush=True,
        )
        if not args.no_preview:
            print(
                "[测距] 鼠标在左图或深度图上移动可实时测量；"
                "左键锁定，右键或 U 恢复跟随。",
                flush=True,
            )

        with metadata_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            while pipeline.isRunning():
                group = output_queue.get(timedelta(seconds=10))
                if group is None:
                    print("[等待] 10 秒未收到同步帧，继续等待 ...", flush=True)
                    continue

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

                filename = f"{frames_saved:08d}.png"
                left_path = left_dir / filename
                right_path = right_dir / filename
                depth_path = depth_dir / filename
                write_png(left_path, left, args.png_compression)
                write_png(right_path, right, args.png_compression)
                write_png(depth_path, depth, args.png_compression)
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
                    print(
                        f"[校验] 深度 PNG 无损：shape={depth.shape}, dtype={depth.dtype}",
                        flush=True,
                    )

                valid_mask = (depth > 0) & (depth < np.iinfo(np.uint16).max)
                valid = depth[valid_mask]
                saturated_pixels = int(np.count_nonzero(depth == np.iinfo(np.uint16).max))
                writer.writerow(
                    {
                        "index": frames_saved,
                        "filename": filename,
                        "left_sequence": left_msg.getSequenceNum(),
                        "right_sequence": right_msg.getSequenceNum(),
                        "depth_sequence": depth_msg.getSequenceNum(),
                        "left_device_timestamp_ns": timestamp_ns(left_msg),
                        "right_device_timestamp_ns": timestamp_ns(right_msg),
                        "depth_device_timestamp_ns": timestamp_ns(depth_msg),
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

                if not args.no_preview:
                    assert inspector is not None
                    cv2.imshow(
                        PREVIEW_WINDOW,
                        make_preview(
                            left,
                            right,
                            depth,
                            frames_saved - 1,
                            inspector,
                        ),
                    )
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q")):
                        break
                    if key in (ord("u"), ord("U")):
                        inspector.locked = False

                if args.frames and frames_saved >= args.frames:
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

        if not args.no_preview:
            cv2.destroyAllWindows()

        if session_path is not None and session_data:
            session_data["completed"] = return_code == 0
            session_data["ended_at"] = datetime.now().astimezone().isoformat()
            session_data["frames_saved"] = frames_saved
            try:
                write_session_json(session_path / "session.json", session_data)
            except OSError as exc:
                return_code = 1
                print(f"[错误] 无法更新 session.json：{exc}", file=sys.stderr, flush=True)

    if session_path is not None:
        print(f"[完成] 共保存 {frames_saved} 组：{session_path}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
