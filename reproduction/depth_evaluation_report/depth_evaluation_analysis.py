"""Reproducible evaluation for the two previously generated OAK stereo results.

The OAK ``depth_mm`` images are treated as reference (pseudo-GT), not as an
independent absolute-depth ground truth.  All report metrics are computed at
the saved prediction resolution of 400x640.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent

DEPTH_MIN_M = 0.2
DEPTH_MAX_M = 2.0
ERROR_VIS_MAX_M = 0.2

CASES = (
    {
        "id": "test_170600_00000032",
        "label": "Test 170600 / 00000032",
        "frame": "00000032",
        "output": "output/oak_test",
        "source": "FastFoundationStereo_OAK_Dataset/test/20260819_170600",
    },
    {
        "id": "train_172516_00000034",
        "label": "Train 172516 / 00000034",
        "frame": "00000034",
        "output": "output/pseudocolor_smoke_00000034",
        "source": "FastFoundationStereo_OAK_Dataset/train/20260819_172516",
    },
)


def _project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def load_case(spec: dict) -> dict:
    output_dir = _project_path(spec["output"])
    source_dir = _project_path(spec["source"])
    filename = spec["frame"] + ".png"

    pred = np.load(output_dir / "depth_meter.npy").astype(np.float32)
    left = cv2.imread(str(output_dir / "left.png"), cv2.IMREAD_COLOR)
    gt_mm = cv2.imread(
        str(source_dir / "depth_mm" / filename), cv2.IMREAD_UNCHANGED
    )
    valid_raw = cv2.imread(
        str(source_dir / "valid_mask" / filename), cv2.IMREAD_UNCHANGED
    )
    source_left = cv2.imread(
        str(source_dir / "left" / filename), cv2.IMREAD_UNCHANGED
    )

    if pred.ndim != 2:
        raise ValueError(f"{spec['id']}: prediction must be 2D, got {pred.shape}")
    if any(value is None for value in (left, gt_mm, valid_raw, source_left)):
        raise FileNotFoundError(f"{spec['id']}: one or more required inputs are missing")

    height, width = pred.shape
    gt = cv2.resize(
        gt_mm, (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(np.float32) / 1000.0
    valid_mask = cv2.resize(
        valid_raw, (width, height), interpolation=cv2.INTER_NEAREST
    ) > 0

    resized_source = cv2.resize(
        source_left, (width, height), interpolation=cv2.INTER_AREA
    )
    if resized_source.ndim == 2:
        resized_source = np.repeat(resized_source[..., None], 3, axis=2)
    if not np.array_equal(left, resized_source):
        max_difference = int(
            np.abs(left.astype(np.int16) - resized_source.astype(np.int16)).max()
        )
        raise ValueError(
            f"{spec['id']}: saved input does not align with source; "
            f"max pixel difference={max_difference}"
        )

    reference_valid = (
        valid_mask & np.isfinite(gt) & (gt > 0) & (gt < 65.535)
    )
    prediction_valid = np.isfinite(pred) & (pred > 0)
    work_mask = (
        reference_valid & (gt >= DEPTH_MIN_M) & (gt <= DEPTH_MAX_M)
    )
    eroded_work_mask = cv2.erode(
        work_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ) > 0

    loaded = dict(spec)
    loaded.update(
        {
            "pred": pred,
            "left": left,
            "gt": gt,
            "reference_valid": reference_valid,
            "prediction_valid": prediction_valid,
            "work_mask": work_mask,
            "eroded_work_mask": eroded_work_mask,
        }
    )
    return loaded


def calculate_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    eligible: np.ndarray,
    *,
    invalid_as_zero: bool = True,
) -> dict:
    eligible = np.asarray(eligible, dtype=bool)
    prediction_valid = np.isfinite(pred) & (pred > 0)
    eligible_count = int(eligible.sum())
    if eligible_count == 0:
        raise ValueError("metric mask contains no eligible pixels")

    coverage = float((eligible & prediction_valid).sum() / eligible_count)
    scored = eligible if invalid_as_zero else eligible & prediction_valid
    pred_for_score = np.where(prediction_valid, pred, 0.0)
    gt_values = gt[scored].astype(np.float64)
    pred_values = pred_for_score[scored].astype(np.float64)

    signed_error = pred_values - gt_values
    absolute_error = np.abs(signed_error)
    relative_error = absolute_error / gt_values
    ratio = np.maximum(
        np.divide(
            pred_values,
            gt_values,
            out=np.zeros_like(pred_values),
            where=gt_values > 0,
        ),
        np.divide(
            gt_values,
            pred_values,
            out=np.full_like(gt_values, np.inf),
            where=pred_values > 0,
        ),
    )

    return {
        "eligible_pixels": eligible_count,
        "scored_pixels": int(scored.sum()),
        "coverage": coverage,
        "mae_m": float(absolute_error.mean()),
        "rmse_m": float(np.sqrt(np.mean(signed_error**2))),
        "median_ae_m": float(np.median(absolute_error)),
        "abs_rel": float(relative_error.mean()),
        "bias_m": float(signed_error.mean()),
        "delta1": float((ratio < 1.25).mean()),
        "delta2": float((ratio < 1.25**2).mean()),
        "delta3": float((ratio < 1.25**3).mean()),
        "over_10cm": float((absolute_error > 0.10).mean()),
        "over_10pct": float((relative_error > 0.10).mean()),
    }


def concatenate_cases(cases: list[dict], key: str) -> np.ndarray:
    return np.concatenate([case[key].reshape(-1) for case in cases])


def build_quantitative_results(cases: list[dict]) -> tuple[dict, dict]:
    per_frame = {}
    for case in cases:
        per_frame[case["id"]] = {
            "label": case["label"],
            "main": calculate_metrics(
                case["gt"], case["pred"], case["work_mask"]
            ),
            "main_intersection": calculate_metrics(
                case["gt"],
                case["pred"],
                case["work_mask"],
                invalid_as_zero=False,
            ),
            "full": calculate_metrics(
                case["gt"], case["pred"], case["reference_valid"]
            ),
            "eroded_main": calculate_metrics(
                case["gt"], case["pred"], case["eroded_work_mask"]
            ),
        }

    gt_all = concatenate_cases(cases, "gt")
    pred_all = concatenate_cases(cases, "pred")
    reference_all = concatenate_cases(cases, "reference_valid").astype(bool)
    work_all = concatenate_cases(cases, "work_mask").astype(bool)
    eroded_all = concatenate_cases(cases, "eroded_work_mask").astype(bool)

    aggregate = {
        "main": calculate_metrics(gt_all, pred_all, work_all),
        "main_intersection": calculate_metrics(
            gt_all, pred_all, work_all, invalid_as_zero=False
        ),
        "full": calculate_metrics(gt_all, pred_all, reference_all),
        "full_intersection": calculate_metrics(
            gt_all, pred_all, reference_all, invalid_as_zero=False
        ),
        "eroded_main": calculate_metrics(gt_all, pred_all, eroded_all),
    }

    band_definitions = (
        ("0.2-0.5 m", reference_all & (gt_all >= 0.2) & (gt_all < 0.5)),
        ("0.5-1.0 m", reference_all & (gt_all >= 0.5) & (gt_all < 1.0)),
        ("1.0-2.0 m", reference_all & (gt_all >= 1.0) & (gt_all <= 2.0)),
        (">2.0 m", reference_all & (gt_all > 2.0)),
    )
    depth_bands = {
        label: calculate_metrics(gt_all, pred_all, mask)
        for label, mask in band_definitions
    }

    pred_valid_all = np.isfinite(pred_all) & (pred_all > 0)
    pred_for_score = np.where(pred_valid_all, pred_all, 0.0)
    absolute_error_all = np.abs(pred_for_score - gt_all)
    full_error_sum = float(absolute_error_all[reference_all].sum())
    tail_mask = reference_all & (gt_all > 2.0)
    tail_error_share = float(
        absolute_error_all[tail_mask].sum() / full_error_sum
    )
    main_large_error = work_all & (absolute_error_all > 0.10)
    boundary_large_error_share = float(
        (main_large_error & ~eroded_all).sum() / main_large_error.sum()
    )

    diagnostics = {
        "tail_gt_over_2m_pixels": int(tail_mask.sum()),
        "tail_gt_over_2m_rate": float(tail_mask.sum() / reference_all.sum()),
        "tail_absolute_error_share": tail_error_share,
        "work_0p5_to_1m_share": float(
            depth_bands["0.5-1.0 m"]["eligible_pixels"]
            / aggregate["main"]["eligible_pixels"]
        ),
        "large_error_outside_eroded_share": boundary_large_error_share,
        "mae_reduction_after_erosion": float(
            1.0
            - aggregate["eroded_main"]["mae_m"] / aggregate["main"]["mae_m"]
        ),
    }

    results = {
        "configuration": {
            "prediction_resolution": [400, 640],
            "reference_resize": "cv2.INTER_NEAREST",
            "reference_unit": "depth_mm / 1000 = meters",
            "main_range_m": [DEPTH_MIN_M, DEPTH_MAX_M],
            "invalid_prediction_policy": "saved invalid depth remains 0 m and is scored as failure",
            "bias_definition": "mean(prediction - reference)",
            "reference_status": "OAK StereoDepth pseudo-GT from the same stereo pair",
        },
        "per_frame": per_frame,
        "aggregate": aggregate,
        "depth_bands": depth_bands,
        "diagnostics": diagnostics,
    }

    chart_data = build_chart_data(
        cases, per_frame, aggregate, depth_bands, gt_all, pred_all, work_all
    )
    return results, chart_data


def build_chart_data(
    cases: list[dict],
    per_frame: dict,
    aggregate: dict,
    depth_bands: dict,
    gt_all: np.ndarray,
    pred_all: np.ndarray,
    work_all: np.ndarray,
) -> dict:
    summary = aggregate["main"]
    summary_rows = [
        {
            "scope": "0.2-2.0 m aggregate",
            "coverage": summary["coverage"],
            "mae_cm": summary["mae_m"] * 100.0,
            "rmse_cm": summary["rmse_m"] * 100.0,
            "median_ae_mm": summary["median_ae_m"] * 1000.0,
            "abs_rel": summary["abs_rel"],
            "delta1": summary["delta1"],
            "over_10cm": summary["over_10cm"],
            "bias_cm": summary["bias_m"] * 100.0,
            "eligible_pixels": summary["eligible_pixels"],
        }
    ]

    frame_rows = []
    for case in cases:
        metrics = per_frame[case["id"]]["main"]
        frame_rows.append(
            {
                "frame": case["frame"],
                "session": case["label"],
                "eligible_pixels": metrics["eligible_pixels"],
                "coverage": metrics["coverage"],
                "mae_cm": metrics["mae_m"] * 100.0,
                "rmse_cm": metrics["rmse_m"] * 100.0,
                "median_ae_mm": metrics["median_ae_m"] * 1000.0,
                "abs_rel": metrics["abs_rel"],
                "delta1": metrics["delta1"],
                "over_10cm": metrics["over_10cm"],
                "bias_cm": metrics["bias_m"] * 100.0,
            }
        )

    band_rows = []
    for order, (label, metrics) in enumerate(depth_bands.items(), start=1):
        band_rows.append(
            {
                "order": order,
                "depth_band": label,
                "eligible_pixels": metrics["eligible_pixels"],
                "coverage": metrics["coverage"],
                "mae_cm": metrics["mae_m"] * 100.0,
                "rmse_cm": metrics["rmse_m"] * 100.0,
                "median_ae_mm": metrics["median_ae_m"] * 1000.0,
                "abs_rel": metrics["abs_rel"],
                "delta1": metrics["delta1"],
                "over_10cm": metrics["over_10cm"],
                "bias_cm": metrics["bias_m"] * 100.0,
            }
        )

    pred_valid = np.isfinite(pred_all) & (pred_all > 0)
    pred_for_score = np.where(pred_valid, pred_all, 0.0)
    error = np.abs(pred_for_score[work_all] - gt_all[work_all])
    bin_edges = np.array(
        [0.0, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500, np.inf]
    )
    bin_labels = (
        "0-5 mm",
        "5-10 mm",
        "10-20 mm",
        "20-50 mm",
        "5-10 cm",
        "10-20 cm",
        "20-50 cm",
        ">50 cm",
    )
    counts, _ = np.histogram(error, bins=bin_edges)
    error_bins = [
        {
            "order": index + 1,
            "bin_label": label,
            "count": int(count),
            "share": float(count / error.size),
        }
        for index, (label, count) in enumerate(zip(bin_labels, counts))
    ]

    return {
        "summary": summary_rows,
        "frames": frame_rows,
        "depth_bands": band_rows,
        "error_bins": error_bins,
    }


def depth_to_bgr(
    depth: np.ndarray,
    valid: np.ndarray,
    *,
    near: float = DEPTH_MIN_M,
    far: float = DEPTH_MAX_M,
    invalid_color: tuple[int, int, int] = (82, 82, 82),
) -> np.ndarray:
    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[valid] = np.clip(
        (far - depth[valid]) / (far - near), 0.0, 1.0
    )
    color_index = np.rint(normalized * 255.0).astype(np.uint8)
    output = cv2.applyColorMap(color_index, cv2.COLORMAP_TURBO)
    output[~valid] = invalid_color
    return output


def error_to_bgr(
    absolute_error: np.ndarray,
    valid: np.ndarray,
    *,
    maximum: float = ERROR_VIS_MAX_M,
    invalid_color: tuple[int, int, int] = (82, 82, 82),
) -> np.ndarray:
    normalized = np.zeros(absolute_error.shape, dtype=np.float32)
    normalized[valid] = np.clip(absolute_error[valid] / maximum, 0.0, 1.0)
    color_index = np.rint(normalized * 255.0).astype(np.uint8)
    output = cv2.applyColorMap(color_index, cv2.COLORMAP_INFERNO)
    output[~valid] = invalid_color
    return output


def _text(
    image: np.ndarray,
    label: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.72,
    color: tuple[int, int, int] = (240, 240, 240),
    thickness: int = 2,
) -> None:
    cv2.putText(
        image,
        label,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _titled_tile(image: np.ndarray, title: str, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    tile = np.full((height + 42, width, 3), 245, dtype=np.uint8)
    tile[42:] = resized
    _text(tile, title, (12, 29), color=(28, 28, 28), scale=0.68)
    return tile


def _legend_bar(
    width: int,
    *,
    kind: str,
    left_label: str,
    right_label: str,
) -> np.ndarray:
    indices = np.linspace(255, 0, width, dtype=np.uint8)[None, :]
    if kind == "depth":
        bar = cv2.applyColorMap(indices, cv2.COLORMAP_TURBO)
    elif kind == "error":
        indices = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        bar = cv2.applyColorMap(indices, cv2.COLORMAP_INFERNO)
    else:
        raise ValueError(kind)
    bar = cv2.resize(bar, (width, 22), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((58, width, 3), 245, dtype=np.uint8)
    canvas[:22] = bar
    _text(canvas, left_label, (2, 48), scale=0.48, color=(35, 35, 35), thickness=1)
    right_size = cv2.getTextSize(
        right_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
    )[0]
    _text(
        canvas,
        right_label,
        (width - right_size[0] - 2, 48),
        scale=0.48,
        color=(35, 35, 35),
        thickness=1,
    )
    return canvas


def case_visuals(case: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = case["left"].copy()
    reference = depth_to_bgr(case["gt"], case["work_mask"])
    prediction_mask = case["work_mask"] & case["prediction_valid"]
    prediction = depth_to_bgr(case["pred"], prediction_mask)
    error = np.abs(np.where(case["prediction_valid"], case["pred"], 0.0) - case["gt"])
    error_image = error_to_bgr(error, prediction_mask)
    return left, reference, prediction, error_image


def render_depth_comparison(cases: list[dict]) -> Path:
    tile_width, tile_height = 560, 350
    gap = 10
    label_width = 172
    top_height = 60
    row_height = tile_height + 42
    canvas_width = label_width + 4 * tile_width + 3 * gap
    canvas_height = top_height + 2 * row_height + gap + 82
    canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)
    _text(
        canvas,
        "Depth comparison: input | OAK reference | prediction | absolute error",
        (18, 39),
        scale=0.92,
        color=(22, 22, 22),
    )

    for row_index, case in enumerate(cases):
        left, reference, prediction, error_image = case_visuals(case)
        if case["frame"] == "00000034":
            cv2.rectangle(left, (180, 35), (465, 315), (0, 220, 255), 3)
            cv2.rectangle(left, (455, 45), (635, 305), (255, 120, 30), 3)
            _text(left, "A", (190, 60), color=(0, 220, 255), scale=0.8)
            _text(left, "B", (590, 70), color=(255, 120, 30), scale=0.8)

        images = (left, reference, prediction, error_image)
        titles = (
            "Left input",
            "OAK reference depth",
            "FFS predicted depth",
            "Absolute error (clipped)",
        )
        y0 = top_height + row_index * (row_height + gap)
        _text(
            canvas,
            case["frame"],
            (14, y0 + 176),
            scale=0.78,
            color=(30, 30, 30),
        )
        for column_index, (image, title) in enumerate(zip(images, titles)):
            tile = _titled_tile(image, title, tile_width, tile_height)
            x0 = label_width + column_index * (tile_width + gap)
            canvas[y0 : y0 + row_height, x0 : x0 + tile_width] = tile

    legend_y = top_height + 2 * row_height + gap + 10
    depth_legend = _legend_bar(
        700, kind="depth", left_label="near 0.2 m (red)", right_label="far 2.0 m (blue)"
    )
    error_legend = _legend_bar(
        700, kind="error", left_label="0 m", right_label=">=0.2 m"
    )
    canvas[legend_y : legend_y + 58, label_width : label_width + 700] = depth_legend
    error_x = canvas_width - 700
    canvas[legend_y : legend_y + 58, error_x : error_x + 700] = error_legend

    output_path = REPORT_DIR / "qualitative_depth_comparison.jpg"
    if not cv2.imwrite(
        str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 91]
    ):
        raise RuntimeError(f"failed to write {output_path}")
    return output_path


def render_detail_comparison(case: dict) -> Path:
    visuals = case_visuals(case)
    crops = (
        ("A - center plate", (180, 35, 465, 315)),
        ("B - right rack", (455, 45, 635, 305)),
    )
    titles = ("Left", "OAK reference", "Prediction", "Absolute error")
    tile_width, tile_height = 420, 320
    gap = 10
    label_width = 190
    top_height = 58
    row_height = tile_height + 42
    canvas_width = label_width + 4 * tile_width + 3 * gap
    canvas_height = top_height + 2 * row_height + gap
    canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)
    _text(
        canvas,
        "Detail crops from frame 00000034",
        (18, 38),
        scale=0.92,
        color=(22, 22, 22),
    )

    for row_index, (crop_label, (x0, y0, x1, y1)) in enumerate(crops):
        row_y = top_height + row_index * (row_height + gap)
        _text(
            canvas,
            crop_label,
            (12, row_y + 174),
            scale=0.64,
            color=(30, 30, 30),
        )
        for column_index, (image, title) in enumerate(zip(visuals, titles)):
            crop = image[y0:y1, x0:x1]
            tile = _titled_tile(crop, title, tile_width, tile_height)
            column_x = label_width + column_index * (tile_width + gap)
            canvas[
                row_y : row_y + row_height,
                column_x : column_x + tile_width,
            ] = tile

    output_path = REPORT_DIR / "qualitative_detail_comparison.jpg"
    if not cv2.imwrite(
        str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 91]
    ):
        raise RuntimeError(f"failed to write {output_path}")
    return output_path


def _camera_projection(
    points: np.ndarray,
    eye: np.ndarray,
    look_at: np.ndarray,
    up: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    forward = look_at - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)

    relative = points - eye
    view_x = relative @ right
    view_y = relative @ true_up
    view_z = relative @ forward
    finite = np.isfinite(points).all(axis=1) & np.isfinite(view_z) & (view_z > 0)

    x_low, x_high = np.percentile(view_x[finite], [0.2, 99.8])
    y_low, y_high = np.percentile(view_y[finite], [0.2, 99.8])
    padding = 24
    scale = min(
        (width - 2 * padding) / max(x_high - x_low, 1e-6),
        (height - 2 * padding) / max(y_high - y_low, 1e-6),
    )
    center_x = (x_low + x_high) / 2.0
    center_y = (y_low + y_high) / 2.0
    pixel_x = np.rint((view_x - center_x) * scale + width / 2).astype(np.int32)
    pixel_y = np.rint(-(view_y - center_y) * scale + height / 2).astype(np.int32)
    visible = (
        finite
        & (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
    )
    return pixel_x, pixel_y, view_z, visible


def _paint_projected_cloud(
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    view_z: np.ndarray,
    visible: np.ndarray,
    rgb_colors: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.full((height, width, 3), (38, 41, 46), dtype=np.uint8)
    valid_ids = np.flatnonzero(visible)
    order = valid_ids[np.argsort(view_z[valid_ids])[::-1]]
    bgr_colors = np.rint(np.clip(rgb_colors, 0.0, 1.0) * 255.0).astype(np.uint8)[:, ::-1]
    for point_id in order:
        x = int(pixel_x[point_id])
        y = int(pixel_y[point_id])
        canvas[max(0, y - 1) : min(height, y + 2), max(0, x - 1) : min(width, x + 2)] = bgr_colors[point_id]
    return canvas


def render_pointcloud_comparison(case: dict) -> Path:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for point-cloud comparison") from exc

    output_dir = _project_path(case["output"])
    texture_cloud = o3d.io.read_point_cloud(str(output_dir / "cloud.ply"))
    pseudo_cloud = o3d.io.read_point_cloud(
        str(output_dir / "cloud_pseudocolor.ply")
    )
    points = np.asarray(pseudo_cloud.points)
    texture_points = np.asarray(texture_cloud.points)
    texture_colors = np.asarray(texture_cloud.colors)
    pseudo_colors = np.asarray(pseudo_cloud.colors)
    if len(points) == 0 or len(points) != len(pseudo_colors):
        raise ValueError("pseudo-color point cloud is empty or missing colors")
    if len(points) != len(texture_points) or not np.allclose(
        points, texture_points, rtol=0.0, atol=1e-7
    ):
        raise ValueError("texture and pseudo-color point clouds are not aligned")

    panel_width, panel_height = 900, 560
    point_mask = (
        np.isfinite(case["pred"])
        & (case["pred"] >= 0.1)
        & (case["pred"] <= 20.0)
    )
    front_texture = case["left"].copy()
    front_texture[~point_mask] = (38, 41, 46)
    front_pseudo = depth_to_bgr(case["pred"], point_mask)
    front_texture = cv2.resize(
        front_texture, (panel_width, panel_height), interpolation=cv2.INTER_AREA
    )
    front_pseudo = cv2.resize(
        front_pseudo, (panel_width, panel_height), interpolation=cv2.INTER_NEAREST
    )

    eye = np.array([0.85, -0.55, -0.15], dtype=np.float64)
    look_at = np.array([0.055, -0.004, 0.667], dtype=np.float64)
    up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    pixel_x, pixel_y, view_z, visible = _camera_projection(
        points, eye, look_at, up, panel_width, panel_height
    )
    oblique_texture = _paint_projected_cloud(
        pixel_x,
        pixel_y,
        view_z,
        visible,
        texture_colors,
        panel_width,
        panel_height,
    )
    oblique_pseudo = _paint_projected_cloud(
        pixel_x,
        pixel_y,
        view_z,
        visible,
        pseudo_colors,
        panel_width,
        panel_height,
    )

    panels = (
        _titled_tile(front_texture, "Front view - grayscale texture", panel_width, panel_height),
        _titled_tile(front_pseudo, "Front view - depth pseudo-color", panel_width, panel_height),
        _titled_tile(oblique_texture, "Oblique view - grayscale texture", panel_width, panel_height),
        _titled_tile(oblique_pseudo, "Oblique view - depth pseudo-color", panel_width, panel_height),
    )
    gap = 12
    header = 62
    panel_full_height = panel_height + 42
    legend_height = 70
    canvas = np.full(
        (
            header + 2 * panel_full_height + gap + legend_height,
            2 * panel_width + gap,
            3,
        ),
        245,
        dtype=np.uint8,
    )
    _text(
        canvas,
        "Point-cloud color comparison - frame 00000034",
        (18, 40),
        scale=0.96,
        color=(22, 22, 22),
    )
    for index, panel in enumerate(panels):
        row, column = divmod(index, 2)
        x0 = column * (panel_width + gap)
        y0 = header + row * (panel_full_height + gap)
        canvas[y0 : y0 + panel_full_height, x0 : x0 + panel_width] = panel
    legend = _legend_bar(
        760,
        kind="depth",
        left_label="near 0.2 m (red)",
        right_label="far 2.0 m (blue)",
    )
    legend_x = (canvas.shape[1] - legend.shape[1]) // 2
    canvas[-legend_height + 6 : -legend_height + 64, legend_x : legend_x + 760] = legend

    output_path = REPORT_DIR / "pointcloud_color_comparison.jpg"
    if not cv2.imwrite(
        str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]
    ):
        raise RuntimeError(f"failed to write {output_path}")
    return output_path


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(results: dict, chart_data: dict) -> None:
    metrics_path = REPORT_DIR / "metrics_summary.json"
    metrics_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (REPORT_DIR / "chart_data.json").write_text(
        json.dumps(chart_data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_csv(REPORT_DIR / "per_frame_metrics.csv", chart_data["frames"])
    write_csv(REPORT_DIR / "depth_band_metrics.csv", chart_data["depth_bands"])
    write_csv(REPORT_DIR / "error_distribution.csv", chart_data["error_bins"])


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    loaded_cases = [load_case(spec) for spec in CASES]
    results, chart_data = build_quantitative_results(loaded_cases)
    write_outputs(results, chart_data)
    render_depth_comparison(loaded_cases)
    render_detail_comparison(loaded_cases[1])
    render_pointcloud_comparison(loaded_cases[1])

    main_metrics = results["aggregate"]["main"]
    print("Depth evaluation completed")
    print(f"  eligible pixels: {main_metrics['eligible_pixels']:,}")
    print(f"  coverage: {main_metrics['coverage'] * 100:.3f}%")
    print(f"  MAE: {main_metrics['mae_m'] * 100:.3f} cm")
    print(f"  RMSE: {main_metrics['rmse_m'] * 100:.3f} cm")
    print(f"  median AE: {main_metrics['median_ae_m'] * 1000:.3f} mm")
    print(f"  delta1: {main_metrics['delta1'] * 100:.3f}%")


if __name__ == "__main__":
    main()
