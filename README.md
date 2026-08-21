# OAK PoE Stereo + Fast-FoundationStereo 备份

这是工作交接用的轻量 GitHub 备份，包含：

- `capture_tools/`：双目采集、交互分段采集、实时点云和离线点云脚本；
- `fast_foundationstereo/`：Fast-FoundationStereo 源码快照及本地新增的伪彩输出脚本；
- `reproduction/sample_input/`：一组经过校正的 OAK 左右输入、K、OAK 深度和有效掩码；
- `reproduction/sample_output/`：对应的论文模型输入副本与视差可视化；
- `reproduction/depth_evaluation_report/`：深度对比分析源码、数据和报告；
- `docs/`：环境、权重哈希、范围及交接说明。

## 运行采集

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
python -m venv venv
./venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install "depthai==3.9.0" "numpy<3" "opencv-python<5" "open3d==0.19.0"
python capture_tools/collect_stereo_dataset_interactive.py --device "169.254.1.222" --output "E:/Depth_Data" --camera-fps 10 --save-fps 4 --warmup-frames 20
```

`R` 开始新分段，`S` 停止当前分段，`Q` 安全退出。交互脚本依赖同目录的 `collect_stereo_dataset.py`。

## 运行论文样例

权重不在仓库中。下载及 SHA256 见 `docs/WEIGHTS_MANIFEST.md`，放到：

`fast_foundationstereo/weights/23-36-37/model_best_bp2_serialize.pth`

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
Set-Location "fast_foundationstereo"
python scripts/run_demo.py `
  --model_dir "weights/23-36-37/model_best_bp2_serialize.pth" `
  --left_file "../reproduction/sample_input/left/00000048.png" `
  --right_file "../reproduction/sample_input/right/00000048.png" `
  --intrinsic_file "../reproduction/sample_input/K.txt" `
  --out_dir "../reproduction/rerun_00000048" `
  --scale 0.5 --get_pc 1 --remove_invisible 0 --denoise_cloud 0 `
  --valid_iters 8 --max_disp 192 --zfar 100
```

## 说明

- 模型原生输出是以左图为参考的视差；深度和点云由内参、基线和视差派生。
- OAK `depth_mm` 是设备算法产生的伪真值，不是独立外部真值。
- 权重、完整数据集、PLY/NPY/PFM 和虚拟环境均被 `.gitignore` 排除，避免许可证、隐私和 GitHub 大文件问题。
- Fast-FoundationStereo 仅限非商业研究使用；详见 `fast_foundationstereo/LICENSE.txt`。

