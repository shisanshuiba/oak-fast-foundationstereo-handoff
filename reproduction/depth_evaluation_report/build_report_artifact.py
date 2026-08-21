"""Build the canonical portable report artifact from reviewed evaluation outputs."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPORT_DIR = Path(__file__).resolve().parent
TITLE = "Fast-FoundationStereo OAK 深度一致性评估报告"
SOURCE_ID = "evaluation_source"


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def markdown_block(block_id: str, body: str, *, sourced: bool = False) -> dict:
    block = {"id": block_id, "type": "markdown", "body": body}
    if sourced:
        block["sourceId"] = SOURCE_ID
    return block


def html_figure(block_id: str, image_path: Path, alt: str, caption: str) -> dict:
    body = (
        '<figure style="margin:0">'
        f'<img src="{image_data_uri(image_path)}" alt="{alt}" '
        'style="display:block;width:100%;height:auto;border-radius:8px">'
        f'<figcaption style="margin-top:8px;color:#5b6472">{caption}</figcaption>'
        '</figure>'
    )
    return {"id": block_id, "type": "html", "body": body, "sourceId": SOURCE_ID}


def card(card_id: str, description: str, label: str, field: str, fmt: str, unit: str | None = None) -> dict:
    metric = {"label": label, "field": field, "format": fmt}
    if unit:
        metric["unit"] = unit
    return {
        "id": card_id,
        "description": description,
        "dataset": "summary",
        "sourceId": SOURCE_ID,
        "metrics": [metric],
    }


def column(field: str, label: str, col_type: str = "number", fmt: str | None = None, unit: str | None = None) -> dict:
    result = {"field": field, "label": label, "type": col_type}
    if fmt:
        result["format"] = fmt
    if unit:
        result["unit"] = unit
    return result


def build_artifact() -> dict:
    metrics = json.loads((REPORT_DIR / "metrics_summary.json").read_text(encoding="utf-8"))
    chart_data = json.loads((REPORT_DIR / "chart_data.json").read_text(encoding="utf-8"))

    aggregate = metrics["aggregate"]
    main = aggregate["main"]
    full = aggregate["full"]
    eroded = aggregate["eroded_main"]
    diagnostics = metrics["diagnostics"]

    scope_rows = [
        {
            "scope": "主口径：0.2-2.0 m",
            "eligible_pixels": main["eligible_pixels"],
            "coverage": main["coverage"],
            "mae_cm": main["mae_m"] * 100.0,
            "rmse_cm": main["rmse_m"] * 100.0,
            "median_ae_mm": main["median_ae_m"] * 1000.0,
            "abs_rel": main["abs_rel"],
            "delta1": main["delta1"],
            "bias_cm": main["bias_m"] * 100.0,
        },
        {
            "scope": "全有效参考范围",
            "eligible_pixels": full["eligible_pixels"],
            "coverage": full["coverage"],
            "mae_cm": full["mae_m"] * 100.0,
            "rmse_cm": full["rmse_m"] * 100.0,
            "median_ae_mm": full["median_ae_m"] * 1000.0,
            "abs_rel": full["abs_rel"],
            "delta1": full["delta1"],
            "bias_cm": full["bias_m"] * 100.0,
        },
        {
            "scope": "3x3 腐蚀后：0.2-2.0 m",
            "eligible_pixels": eroded["eligible_pixels"],
            "coverage": eroded["coverage"],
            "mae_cm": eroded["mae_m"] * 100.0,
            "rmse_cm": eroded["rmse_m"] * 100.0,
            "median_ae_mm": eroded["median_ae_m"] * 1000.0,
            "abs_rel": eroded["abs_rel"],
            "delta1": eroded["delta1"],
            "bias_cm": eroded["bias_m"] * 100.0,
        },
    ]

    generated_at = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    cards = [
        card("coverage_card", "主工作范围内产生有限正深度的比例", "覆盖率", "coverage", "percent"),
        card("mae_card", "0.2-2.0 m 内的平均绝对深度误差", "MAE", "mae_cm", "number", "cm"),
        card("rmse_card", "对长尾离群误差更敏感的均方根误差", "RMSE", "rmse_cm", "number", "cm"),
        card("median_card", "一半有效像素不超过该绝对误差", "中位绝对误差", "median_ae_mm", "number", "mm"),
        card("delta1_card", "max(pred/ref, ref/pred) < 1.25 的像素比例", "δ1", "delta1", "percent"),
    ]

    charts = [
        {
            "id": "error_distribution_chart",
            "title": "绝对深度误差分布",
            "subtitle": "两帧合并，0.2-2.0 m，N=312,000；无效预测按 0 m 计失败",
            "showDescription": True,
            "type": "histogram",
            "dataset": "error_bins",
            "sourceId": SOURCE_ID,
            "valueFormat": "number",
            "unit": "pixels",
            "xAxisTitle": "绝对误差区间",
            "yAxisTitle": "像素数",
            "encodings": {
                "x": {"field": "bin_label", "type": "ordinal", "label": "误差区间"},
                "y": {"field": "count", "type": "quantitative", "label": "像素数"},
            },
        },
        {
            "id": "frame_mae_chart",
            "title": "逐帧平均绝对误差",
            "subtitle": "统一 0.2-2.0 m 工作范围；00000034 的长尾误差更明显",
            "showDescription": True,
            "type": "bar",
            "dataset": "frames",
            "sourceId": SOURCE_ID,
            "valueFormat": "number",
            "unit": "cm",
            "xAxisTitle": "帧号",
            "yAxisTitle": "MAE (cm)",
            "encodings": {
                "x": {"field": "frame", "type": "nominal", "label": "帧号"},
                "y": {"field": "mae_cm", "type": "quantitative", "label": "MAE", "unit": "cm"},
            },
        },
        {
            "id": "depth_band_mae_chart",
            "title": "工作范围内分距离段平均绝对误差",
            "subtitle": "0.2-0.5 m 仅有 1,496 像素，主要位于小区域与边界，稳定性较低",
            "showDescription": True,
            "type": "bar",
            "dataset": "work_depth_bands",
            "sourceId": SOURCE_ID,
            "valueFormat": "number",
            "unit": "cm",
            "xAxisTitle": "OAK 参考深度范围",
            "yAxisTitle": "MAE (cm)",
            "encodings": {
                "x": {"field": "depth_band", "type": "ordinal", "label": "深度段"},
                "y": {"field": "mae_cm", "type": "quantitative", "label": "MAE", "unit": "cm"},
            },
        },
    ]

    common_metric_columns = [
        column("eligible_pixels", "评测像素", "number", "number"),
        column("coverage", "覆盖率", "number", "percent"),
        column("mae_cm", "MAE", "number", "number", "cm"),
        column("rmse_cm", "RMSE", "number", "number", "cm"),
        column("median_ae_mm", "MedAE", "number", "number", "mm"),
        column("abs_rel", "AbsRel", "number", "percent"),
        column("delta1", "δ1", "number", "percent"),
        column("bias_cm", "Bias", "number", "number", "cm"),
    ]
    tables = [
        {
            "id": "frame_metrics_table",
            "title": "逐帧主指标",
            "subtitle": "正 Bias 表示预测偏远；所有指标使用同一主掩码定义",
            "showDescription": True,
            "dataset": "frames",
            "sourceId": SOURCE_ID,
            "density": "spacious",
            "defaultSort": {"field": "frame", "direction": "asc"},
            "columns": [
                column("frame", "帧号", "text"),
                *common_metric_columns,
                column("over_10cm", ">10 cm", "number", "percent"),
            ],
        },
        {
            "id": "depth_band_table",
            "title": "分距离段指标",
            "subtitle": "超过 2 m 的行仅作数据质量诊断，不参与主结论",
            "showDescription": True,
            "dataset": "depth_bands",
            "sourceId": SOURCE_ID,
            "density": "spacious",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                column("order", "序号", "number", "number"),
                column("depth_band", "参考深度段", "text"),
                *common_metric_columns,
                column("over_10cm", ">10 cm", "number", "percent"),
            ],
        },
        {
            "id": "scope_comparison_table",
            "title": "主范围、全范围与边界鲁棒性对比",
            "subtitle": "全范围 RMSE 受不到 1% 的 >2 m 长尾显著放大",
            "showDescription": True,
            "dataset": "scope_comparison",
            "sourceId": SOURCE_ID,
            "density": "spacious",
            "defaultSort": {"field": "eligible_pixels", "direction": "desc"},
            "columns": [
                column("scope", "口径", "text"),
                *common_metric_columns,
            ],
        },
    ]

    technical_summary = f"""## 技术摘要

- **主工作范围内覆盖完整、典型误差较小。** 两帧合并、`0.2–2.0 m` 共 {main['eligible_pixels']:,} 个参考有效像素，覆盖率 {main['coverage'] * 100:.3f}%，MAE {main['mae_m'] * 100:.3f} cm，RMSE {main['rmse_m'] * 100:.3f} cm，中位绝对误差 {main['median_ae_m'] * 1000:.3f} mm，δ1 {main['delta1'] * 100:.3f}%。
- **误差呈明显长尾并集中在结构边界。** 中位误差仅约 4.93 mm，但 RMSE 达 14.11 cm；`3×3` 有效区腐蚀后 MAE 降至 {eroded['mae_m'] * 100:.3f} cm，说明孔板、线缆、遮挡轮廓和货架薄边主导大误差。
- **汇总结果主要代表 0.5–1.0 m。** 该距离段占主工作范围 {diagnostics['work_0p5_to_1m_share'] * 100:.2f}%，不能把当前两帧外推为整个 0.2–2.0 m 的均衡性能。
- **这是一致性评估，不是绝对精度认证。** OAK `depth_mm` 由同一对双目图像生成，且有效掩码内仍存在最高 64.788 m 的远距长尾；需要独立测距真值才能确认物理绝对精度。
"""

    blocks = [
        markdown_block("title", f"# {TITLE}"),
        markdown_block("technical_summary", technical_summary, sourced=True),
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": [item["id"] for item in cards]},
        markdown_block(
            "error_shape",
            """## 半数像素误差不超过 5 mm，但少数离群点拉高均值

误差分布显示，约 50.35% 的主工作范围像素落在 `0–5 mm`，66.06% 不超过 `10 mm`；与此同时，8.90% 超过 `10 cm`。因此只看 MAE 或 RMSE 会低估主体平面的稳定性，只看中位数又会掩盖边界失败，两类指标需要同时解释。""",
            sourced=True,
        ),
        {"id": "error_distribution", "type": "chart", "chartId": "error_distribution_chart"},
        markdown_block(
            "frame_stability",
            """## 00000034 的主体中位误差更低，但严重离群点更多

`00000034` 的中位绝对误差为 4.40 mm，优于 `00000032` 的 5.56 mm；但它的 MAE、RMSE 和 `>10 cm` 比例均更高。这个组合说明第二帧的大部分区域更贴近参考深度，但孔洞、货架条纹和遮挡边缘产生了更多严重离群误差。""",
            sourced=True,
        ),
        {"id": "frame_mae", "type": "chart", "chartId": "frame_mae_chart"},
        {"id": "frame_table", "type": "table", "tableId": "frame_metrics_table"},
        markdown_block(
            "distance_behavior",
            """## 偏差随距离反转，单一常数校准无法解决

`0.5–1.0 m` 占评测像素的主体，MAE 为 4.16 cm、Bias 为 `+3.27 cm`，即平均略偏远；`1.0–2.0 m` 的 Bias 则为 `−8.65 cm`，呈平均偏近。`0.2–0.5 m` 只有 1,496 个像素且多位于边界，小样本 MAE 不能当作稳定的近距性能估计。""",
            sourced=True,
        ),
        {"id": "depth_band_mae", "type": "chart", "chartId": "depth_band_mae_chart"},
        {"id": "depth_band_detail", "type": "table", "tableId": "depth_band_table"},
        markdown_block(
            "qualitative_overview",
            """## 预测更连续，误差主要沿深度突变轮廓出现

下图将两帧输入、OAK 参考深度、模型预测和绝对误差放在同一色标下。预测在中央大平面和孔洞内部更稠密连续；高误差主要贴着孔板边缘、斜线缆、底部前景轮廓和右侧重复货架分布。OAK 参考本身存在水平条纹、散斑及局部缺测，因此这些边界差异不能全部归因于模型。""",
            sourced=True,
        ),
        html_figure(
            "qualitative_depth_image",
            REPORT_DIR / "qualitative_depth_comparison.jpg",
            "两帧左图、OAK参考深度、Fast-FoundationStereo预测深度和绝对误差的并排对比",
            "两帧共享 0.2–2.0 m 深度色标；绝对误差在 0.2 m 截断；灰色区域不参与主指标。",
        ),
        markdown_block(
            "qualitative_details",
            """## 孔板和细货架揭示了边界平滑与参考噪声的叠加

局部 A 中，模型保留了孔洞和大平面结构，但孔边缘出现膨胀及跨边界平滑；局部 B 中，货架薄边是最显著的连续高误差区域，同时 OAK 参考呈明显条纹与破碎标签。这个现象与腐蚀掩码后的指标改善相互印证。""",
            sourced=True,
        ),
        html_figure(
            "qualitative_detail_image",
            REPORT_DIR / "qualitative_detail_comparison.jpg",
            "00000034中央孔板与右侧货架的局部深度和误差对比",
            "A 为中央孔板，B 为右侧重复货架；色标与总览保持一致。",
        ),
        markdown_block(
            "pointcloud_readability",
            """## 伪彩色点云显著提升了前后层次可读性

原始 OAK 左图为灰度，因此纹理点云只能靠亮度辨认结构；深度伪彩将近处映射为红色、远处映射为蓝色，在斜视图中能直接区分孔板、货架和前景物体的相对 Z 层次。`cloud_pseudocolor.ply` 含 241,768 个点及等量 RGB 颜色，颜色与保存的米制深度逐点一致。""",
            sourced=True,
        ),
        html_figure(
            "pointcloud_image",
            REPORT_DIR / "pointcloud_color_comparison.jpg",
            "00000034灰度纹理点云与深度伪彩点云的正视和斜视对比",
            "上排为正视投影，下排为同一斜视投影；左右点坐标完全相同，仅颜色编码不同。",
        ),
        markdown_block(
            "scope_definition",
            """## 评测回答的是与 OAK StereoDepth 的一致性

- 预测：已保存的 `depth_meter.npy`，`400×640`、单位米。
- 参考：OAK `depth_mm` 与 `valid_mask`，从 `800×1280` 最近邻缩放至预测尺寸，深度除以 1000 转为米。
- 主掩码：`valid_mask > 0`、参考深度有限且 `0.2≤D≤2.0 m`。
- 预测无效策略：保存的 `0 m` 保留在主口径中并计作失败；仅有效交集作为辅助敏感性检查。
- 指标：MAE、RMSE、MedAE、AbsRel、Bias、δ1/δ2/δ3、覆盖率、`>10 cm` 与 `>10%` 大误差比例。
""",
        ),
        markdown_block(
            "methodology",
            """## 两帧以统一尺度、单位和分母计算

两份输出的 `left.png` 均与对应源图按推理脚本缩放后逐像素完全一致，排除了错帧和空间错位。所有汇总指标按像素加权，而不是先算逐帧均值再平均；δ 指标使用 `max(pred/reference, reference/pred) < 1.25^k`，Bias 定义为 `mean(pred-reference)`。计算代码和已执行 notebook 随报告一并交付。""",
            sourced=True,
        ),
        markdown_block(
            "robustness",
            f"""## 边界腐蚀使 MAE 下降 {diagnostics['mae_reduction_after_erosion'] * 100:.1f}%，全范围则被远距长尾主导

对主工作区掩码做 `3×3` 腐蚀后仍保留 {eroded['eligible_pixels']:,} 个像素，MAE 从 {main['mae_m'] * 100:.2f} cm 降到 {eroded['mae_m'] * 100:.2f} cm，δ1 从 {main['delta1'] * 100:.2f}% 升到 {eroded['delta1'] * 100:.2f}%；被腐蚀区域包含约 {diagnostics['large_error_outside_eroded_share'] * 100:.2f}% 的全部 `>10 cm` 错误。另一方面，只有 {diagnostics['tail_gt_over_2m_rate'] * 100:.3f}% 的参考像素超过 2 m，却贡献约 {diagnostics['tail_absolute_error_share'] * 100:.2f}% 的全范围绝对误差，使全范围 RMSE 升到 {full['rmse_m']:.3f} m。""",
            sourced=True,
        ),
        {"id": "scope_table", "type": "table", "tableId": "scope_comparison_table"},
        markdown_block(
            "limitations",
            """## 当前证据不足以声明绝对物理精度或广泛泛化

1. 只有两帧，且 95.1% 的主工作像素位于 0.5–1.0 m，场景与距离覆盖有限。
2. OAK 参考由同一双目图像生成，不是独立 LiDAR、结构光或人工测距真值；共同的纹理与遮挡失败模式会影响比较。
3. `valid_mask` 内仍保留最高 64.788 m 的异常远值，说明参考数据质量并非完全受控。
4. 3×3 腐蚀是边界敏感性检查，不是新的主指标；它会移除约三分之一像素，不能替代完整掩码结果。
""",
        ),
        markdown_block(
            "next_steps",
            """## 下一步应引入独立真值并扩大距离覆盖

1. 用卷尺或激光测距仪布置 0.3、0.5、0.8、1.0、1.5、2.0 m 的平面靶，单独报告中心平面误差与边界误差。
2. 采集至少数十帧，按场景、纹理、遮挡强度和距离分层汇总，并给出置信区间或帧间分布。
3. 使用独立 RGB-D/LiDAR 参考验证绝对精度；OAK StereoDepth 继续作为一致性基线，而不是唯一真值。
4. 对孔板、线缆和货架薄边建立边界专用子集，比较原掩码、腐蚀掩码与显式边界带指标。
""",
        ),
        markdown_block(
            "further_questions",
            """## 仍需回答的问题

- 误差随真实距离是否连续变化，还是主要由特定纹理与遮挡触发？
- 固定 Bias 校正是否能改善 0.5–1.0 m，却恶化 1.0–2.0 m？
- 使用独立绝对真值后，OAK 条纹/飞点与模型边界平滑各自贡献多少误差？
""",
        ),
    ]

    source_manifest = {
        "id": SOURCE_ID,
        "label": "Fast-FoundationStereo OAK 两帧深度评估",
        "path": "output/depth_evaluation_report/depth_evaluation_analysis.py",
    }
    source_query = {
        "id": SOURCE_ID,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_json_auto('output/depth_evaluation_report/chart_data.json')",
            "description": "读取由审计脚本生成的有界指标与图表数据；原始尺度对齐和误差计算保存在 depth_evaluation_analysis.py。",
            "tables_used": [
                "output/oak_test/depth_meter.npy",
                "output/pseudocolor_smoke_00000034/depth_meter.npy",
                "FastFoundationStereo_OAK_Dataset/test/20260819_170600/depth_mm/00000032.png",
                "FastFoundationStereo_OAK_Dataset/test/20260819_170600/valid_mask/00000032.png",
                "FastFoundationStereo_OAK_Dataset/train/20260819_172516/depth_mm/00000034.png",
                "FastFoundationStereo_OAK_Dataset/train/20260819_172516/valid_mask/00000034.png",
            ],
            "filters": [
                "OAK valid_mask > 0",
                "主结果使用 0.2 <= reference_depth_m <= 2.0",
                "无效预测保留为 0 m 并计入主指标",
            ],
            "metric_definitions": [
                "MAE = mean(abs(prediction_m - reference_m))",
                "RMSE = sqrt(mean((prediction_m - reference_m)^2))",
                "AbsRel = mean(abs(prediction_m - reference_m) / reference_m)",
                "Bias = mean(prediction_m - reference_m)",
                "delta_k = mean(max(pred/reference, reference/pred) < 1.25^k)",
                "Coverage = finite positive predictions / eligible reference pixels",
            ],
            "executed_at": generated_at,
        },
    }

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [source_manifest],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "datasets": {
                "summary": chart_data["summary"],
                "frames": chart_data["frames"],
                "depth_bands": chart_data["depth_bands"],
                "work_depth_bands": chart_data["depth_bands"][:3],
                "error_bins": chart_data["error_bins"],
                "scope_comparison": scope_rows,
            },
        },
        "sources": [source_query],
    }


def main() -> None:
    artifact = build_artifact()
    output_path = REPORT_DIR / "artifact.json"
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Artifact written to {output_path}")
    print(f"Artifact size: {size_mb:.3f} MiB")


if __name__ == "__main__":
    main()
