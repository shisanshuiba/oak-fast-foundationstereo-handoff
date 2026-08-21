# 来源与打包范围

## 本地来源

- 采集脚本来自 DepthAI v3 `examples/python/StereoDepth` 工作目录。
- 论文代码来自 Fast-FoundationStereo `master` 下载快照。
- 复现输入来自自建 OAK 双目数据集的会话 `20260819_172516`。
- 本仓库只保留一组脱敏样例和轻量结果；完整数据集与完整运行输出由交接方另行保存。

Fast-FoundationStereo 本地目录是从 `master` 下载的快照，不含 `.git`，因此不能从本地元数据可靠恢复精确 commit SHA。上游地址：

https://github.com/NVlabs/Fast-FoundationStereo

## 纳入内容

- 当前采集/交互采集/实时点云/离线点云脚本及历史备份。
- 上游源码、文档、许可证、demo、assets，以及本地新增的 `scripts/run_demo_pseudocolor.py`。
- 会话 `20260819_172516` 的两组实际输入（帧 34、48）。
- `output/` 和 `output_docker/` 下的现有复现结果，排除缓存字节码。
- 环境版本、权重哈希、文件清单与 SHA256。

## 有意排除

- 完整数据集：979 文件，664,650,665 bytes（约 633.86 MiB）；未重复打包。
- 权重文件与冗余下载 ZIP：权重许可不明确，按官方链接恢复。
- 5.24 GiB Conda 环境目录：改用精确包版本清单恢复。
- `__pycache__/`、`*.pyc` 等可再生缓存。

## 数据集校验

- `dataset_manifest.json` SHA256：`f418d6d656af595a9a8a137a5d309c635ff45ebfb92fa45adc03010dbea4b09e`
- `checksums.sha256` SHA256：`05c8f158d999ffa032c9c7b52a57ed82dcaf3c2a2f9c2951f20fe3d98b391702`
- `README_CN.md` SHA256：`58aa04ae8ed4ffa57961b4516f883afc625d2419ce79997dcc3962f10652422b`

E 盘仓库内数据集副本与 C 盘主副本的文件数、总字节数及上述根文件哈希一致。
