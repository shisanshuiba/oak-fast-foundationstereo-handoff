# OAK PoE 双目深度分段采集器使用说明（Windows）

本文档说明如何在另一台 Windows 电脑上配置环境并运行
`collect_stereo_dataset_interactive.py`。该程序会让相机和 Pipeline 持续在线，
可反复开始、停止采集；每次采集生成一个独立目录，不需要每段都重新连接相机。

## 1. 功能概览

- 同步采集左校正图、右校正图和左视角对齐的深度图。
- 相机默认以 10 FPS 工作，实际默认保存 4 组/秒。
- 程序启动后默认丢弃前 20 组，只进行预览和自动曝光预热。
- 按 `R` 开始、按 `S` 停止，Pipeline 在停止后继续运行。
- 每次开始都会创建新的独立目录，帧号从 `00000000` 重新计数。
- 深度文件保存为原始 `uint16 PNG`，单位为毫米，不是伪彩色图片。
- 预览窗口支持鼠标测距、测量点锁定和 11×11 ROI 中值测距。

## 2. 需要发送的文件

请将以下三个文件放在同一个目录中：

```text
OAK_Stereo_Collector/
├── README_collect_stereo_dataset_interactive.md
├── collect_stereo_dataset_interactive.py
└── collect_stereo_dataset.py
```

其中：

- `collect_stereo_dataset_interactive.py` 是交互式分段采集主程序。
- `collect_stereo_dataset.py` 提供已经验证过的 PoE 连接、标定导出、PNG 写入和测距函数。
- 两个 Python 文件必须位于同一目录，否则会出现
  `ModuleNotFoundError: No module named 'collect_stereo_dataset'`。
- `collect_stereo_dataset_backup_*.py` 和 `depth_to_pointcloud.py` 不是运行采集器的必需文件。

## 3. 硬件和系统要求

推荐配置：

- Windows 10/11 64 位。
- 64 位 Python 3.12；Python 3.9 及以上通常也可使用。
- OAK RVC2 PoE 双目相机，左右相机对应 `CAM_B` 和 `CAM_C`。
- PoE 注入器或 PoE 交换机，以及能够传输数据的网线。
- 足够的磁盘空间；如果启用 `--save-npy`，磁盘占用会明显增加。

程序固定使用 TCP/IP、RVC2、`CAM_B/C`。不同型号、OAK4 或相机插座布局
不同的设备可能需要修改 Pipeline 配置。

## 4. 创建 Python 环境

以下步骤只需要在第一次部署时执行。

### 4.1 打开 PowerShell

每次新开 PowerShell 后，建议首先设置 UTF-8 输出：

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
```

进入收到的程序目录，例如：

```powershell
Set-Location "C:/OAK_Stereo_Collector"
```

### 4.2 确认 Python

```powershell
python --version
python -c "import struct; print('Python bits:', struct.calcsize('P') * 8)"
```

建议看到 Python 3.12.x 和 `Python bits: 64`。

### 4.3 创建并激活虚拟环境

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
```

成功后，PowerShell 提示符前通常会出现 `(venv)`。

如果 PowerShell 阻止激活脚本，可仅对当前窗口临时放行，然后重新激活：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
./venv/Scripts/Activate.ps1
```

### 4.4 安装依赖

本程序不需要完整下载 DepthAI 示例模型，只需安装以下 Python 包：

```powershell
python -m pip install --upgrade pip
python -m pip install "depthai==3.9.0" "numpy<3" "opencv-python<5"
```

验证导入：

```powershell
python -c "import depthai as dai, cv2, numpy as np; print('DepthAI:', dai.__version__); print('OpenCV:', cv2.__version__); print('NumPy:', np.__version__)"
```

如果输出 `DepthAI: 3.9.0` 且没有异常，Python 环境已经可用。

本程序已在 Python 3.12.7、DepthAI 3.9.0、NumPy 2.5.2 和
OpenCV 4.14.0.94 环境中验证。

> 在部分 DepthAI 3.9.0 Windows wheel 上，`python -m pip check` 可能显示
> `depthai 3.9.0 is not supported on this platform`。如果上面的实际导入成功，
> 该提示通常是 wheel 元数据检查问题，不代表扩展无法运行。

## 5. 网络连接

### 5.1 电脑通过 PoE 注入器直连相机

接线顺序：

```text
电脑有线网口 ──> PoE 注入器的 DATA IN / LAN
OAK PoE 相机 <── PoE 注入器的 POE OUT / DATA+POWER
PoE 注入器   <── 适配器供电
```

普通 PoE 注入器只负责供电和传输数据，并不提供 DHCP。相机处于默认动态网络
配置、网络中没有 DHCP 且没有写入其他静态地址时，RVC2 OAK PoE 通常回退到：

```text
169.254.1.222
```

电脑有线网卡需要位于同一个 `/16` 网段。Windows 通常会自动分配
`169.254.x.x` 地址；也可将电脑网卡手动设置为：

```text
IP 地址：169.254.1.10
子网掩码：255.255.0.0
默认网关：留空
DNS：留空
```

设置静态地址前，请确认选中的是连接相机的有线网卡，不要修改 Wi-Fi 网卡。

多台相机不能同时使用相同的 `169.254.1.222`，否则会产生 IP 冲突。多设备场景
应使用 DHCP 地址保留或为各相机配置不同地址。

### 5.2 相机接入 PoE 交换机或路由器

普通非三层 PoE 交换机也不一定提供 DHCP；只有网络中实际存在 DHCP 服务器时，
相机才会获得 DHCP 地址。该地址可能不是
`169.254.1.222`，也可能在重启后变化。

建议：

- 在路由器的 DHCP 客户端列表中查找相机地址。
- 在路由器中为相机配置 DHCP 地址保留，使地址保持稳定。
- 运行程序时通过 `--device "实际IP"` 指定当前地址。
- 电脑与相机应位于同一网络，且网络不能隔离客户端或屏蔽本地广播。

`--device` 也接受名称或 MXID，但这仍依赖设备发现。对于 DHCP 或跨网段场景，
明确传入当前可路由的相机 IP 最稳妥。

### 5.3 检查网卡和相机

先检查物理网卡：

```powershell
Get-NetAdapter |
  Format-Table Name,InterfaceDescription,Status,LinkSpeed -Auto
```

连接相机的有线网卡必须存在，并且理想状态为 `Up`。如果只看到 Wi-Fi、
VPN 或虚拟网卡，则应先恢复有线网卡，继续修改 Python 或 IP 没有作用。

查看 IPv4 地址：

```powershell
Get-NetIPConfiguration |
  Format-List InterfaceAlias,InterfaceDescription,IPv4Address,IPv4DefaultGateway
```

直连模式下测试默认地址：

```powershell
ping 169.254.1.222
```

最后用 DepthAI 枚举设备：

```powershell
python -c "import depthai as dai; print('Connected:', dai.Device.getAllConnectedDevices())"
```

正常情况下会显示相机 IP、MXID、TCP/IP 协议和设备状态。`Connected: []`
表示当前环境没有发现设备，应先排查网络，不要立即重装 Python。

## 6. 启动采集器

确保虚拟环境已激活，并进入两个 Python 文件所在目录：

```powershell
Set-Location "C:/OAK_Stereo_Collector"
./venv/Scripts/Activate.ps1
```

### 6.1 推荐命令：默认 IP、每秒保存 4 组

```powershell
python collect_stereo_dataset_interactive.py `
  --device "169.254.1.222" `
  --output "E:/Depth_Data" `
  --save-fps 4
```

### 6.2 每秒保存 3 组

```powershell
python collect_stereo_dataset_interactive.py `
  --device "169.254.1.222" `
  --output "E:/Depth_Data" `
  --save-fps 3
```

### 6.3 使用 DHCP 地址

例如相机获得了 `192.168.1.80`：

```powershell
python collect_stereo_dataset_interactive.py `
  --device "192.168.1.80" `
  --output "E:/Depth_Data" `
  --save-fps 4
```

连接 PoE 相机及上传固件可能需要一段时间。看到预览窗口和
`[常驻] Pipeline 已启动` 后，再开始操作。

## 7. 采集操作

程序启动后会先进行一次预热。默认前 20 组只显示、不保存，用于避开
启动阶段的自动曝光过渡。Pipeline 在后续待机和分段之间始终保持运行，
因此不会在每次开始时重新经历连接和启动曝光过程。

快捷键：

| 按键 | 功能 |
|---|---|
| `R` | 开始一个新分段；预热期间按下会预约在预热完成后开始 |
| `S` | 停止并完成当前分段，但不关闭相机和 Pipeline |
| `Space` | 在开始和停止之间切换 |
| `Q` | 完成当前分段并安全退出整个程序 |
| `U` | 解除测距点锁定，恢复鼠标跟随 |
| 鼠标左键 | 在左图或深度图上锁定测量点 |
| 鼠标右键 | 解除测量点锁定 |

注意：

- 预览窗口需要处于焦点时才能接收窗口快捷键。
- 测距只支持 `LEFT` 和 `DEPTH` 面板，因为深度默认对齐左校正视角；右图同坐标
  不是该深度像素的直接对应位置。
- 待机状态不会保存文件，但会持续取流、预览和维持自动曝光。
- 建议使用 `S` 停止当前分段，再使用 `R` 开始下一段。
- 优先使用 `Q` 或 `Ctrl+C` 安全退出，不要直接强制结束 Python 进程。

## 8. 输出目录

每次按 `R` 都会在 `--output` 指定的根目录下创建新目录，例如：

```text
E:/Depth_Data/
├── 20260820_141530_123_segment001/
│   ├── left_rectified/
│   │   ├── 00000000.png
│   │   └── 00000001.png
│   ├── right_rectified/
│   │   ├── 00000000.png
│   │   └── 00000001.png
│   ├── depth_mm/
│   │   ├── 00000000.png
│   │   └── 00000001.png
│   ├── metadata.csv
│   ├── session.json
│   ├── calibration_active.json
│   └── calibration_factory.json（设备提供时）
└── 20260820_141545_456_segment002/
    └── ...
```

如果使用 `--save-npy`，每个分段还会包含 `depth_npy/`。

数据含义：

- `left_rectified/`：左目校正灰度图，`uint8`。
- `right_rectified/`：右目校正灰度图，`uint8`。
- `depth_mm/`：对齐到左校正视角的 `uint16` 毫米深度。
- 深度值 `0`：无效或无法匹配的像素。
- 深度值 `65535`：饱和或保留值，不应作为正常距离使用。
- `metadata.csv`：三路帧号、设备时间戳、同步跨度、曝光时间、ISO、
  有效深度数量及深度统计。
- `session.json`：设备、参数、帧数、停止原因、有效内参和完成状态。
- `calibration_active.json`：当前活动标定；`calibration_factory.json` 仅在设备
  能够提供工厂标定时生成。

Windows 资源管理器直接显示 16 位深度 PNG 时通常会看起来很黑，且只有少数亮点。
这是因为文件保存的是毫米数值，不是显示用彩色图。预览窗口中的彩色深度只是
可视化，不会替换原始深度数据。

## 9. 常用参数

查看全部参数：

```powershell
python collect_stereo_dataset_interactive.py --help
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--device` | `169.254.1.222` | 相机 IP、名称或 MXID；网络部署优先使用实际 IP |
| `--output` | `recordings/stereo_depth` | 所有分段的根目录 |
| `--camera-fps` | `10` | 相机和预览帧率；`--fps` 是兼容别名 |
| `--save-fps` | `4` | 实际保存频率，必须不大于相机帧率 |
| `--warmup-frames` | `20` | 启动时丢弃的曝光预热帧数 |
| `--frames` | `0` | 每段最多保存多少组；0 表示不限 |
| `--measure-roi` | `11` | 实时测距的正方形 ROI 边长，必须为奇数 |
| `--save-npy` | 关闭 | 同时保存深度 NPY |
| `--no-extended` | 关闭 | 关闭扩展视差，适合更关注中远距离的场景 |
| `--auto-start` | 关闭 | 预热结束后自动开始第一段 |
| `--no-preview` | 关闭 | 不显示预览，使用 PowerShell 焦点接收快捷键 |

如果启动后仍有过曝，可以增加预热帧数：

```powershell
python collect_stereo_dataset_interactive.py `
  --device "169.254.1.222" `
  --output "E:/Depth_Data" `
  --save-fps 4 `
  --warmup-frames 30
```

限制每段保存 100 组，达到后自动停止并返回待机：

```powershell
python collect_stereo_dataset_interactive.py `
  --device "169.254.1.222" `
  --output "E:/Depth_Data" `
  --save-fps 4 `
  --frames 100
```

## 10. 常见问题

### 10.1 `No module named 'collect_stereo_dataset'`

确认两个 Python 文件在同一个目录，并从该目录启动主程序。

### 10.2 `No DepthAI device found` 或 `Cannot find any device`

依次检查：

1. 相机和 PoE 注入器是否供电，网口 Link 灯是否亮。
2. `Get-NetAdapter` 是否能看到真实有线网卡且状态为 `Up`。
3. 电脑和相机是否处于同一网段。
4. DHCP 网络中的相机实际 IP 是否与 `--device` 一致。
5. 是否有 OAK Viewer、旧版 DepthAI Demo 或其他 Python 程序正在占用相机。
6. 是否存在 VPN、虚拟网卡、客户端隔离或防火墙规则影响本地发现。

DepthAI PoE 设备发现通常使用 UDP 11491，XLink 数据连接使用 TCP 11490。
应针对实际网络和程序配置防火墙规则，不建议长期关闭整个 Windows 防火墙。

### 10.3 `X_LINK_DEVICE_NOT_FOUND`

- 关闭其他相机程序。
- 等待 PoE 相机的固件和网络链路恢复后再试。
- 确认当前 IP 可达。
- 必要时只进行一次 PoE 断电重启，不要连续快速断电或强杀程序。

### 10.4 Windows 中看不到 Realtek/以太网网卡

这属于主机网卡或驱动问题，不是 Python 环境问题。重新插拔 USB 网卡、换 USB
接口或重启电脑；板载网卡仍不出现时，应检查 BIOS/UEFI 中的 LAN 设置和电脑厂商
提供的网卡驱动。在真实有线网卡恢复前，相机连接重试不会成功。

### 10.5 深度图片看起来几乎全黑

`depth_mm/*.png` 保存的是 16 位毫米值。普通图片浏览器没有按照有效距离范围进行
归一化，所以缩略图很黑是正常现象。请使用程序预览、OpenCV 读取原值，或使用
点云转换程序查看三维结果。

### 10.6 按键没有反应

- 使用预览模式时，先单击 OpenCV 预览窗口，再按快捷键。
- 使用 `--no-preview` 时，保持运行程序的 PowerShell 窗口处于焦点。
- `R/S/Q/U` 不区分大小写。

## 11. 交付前检查清单

- [ ] 三个交付文件位于同一目录。
- [ ] Python 为 64 位，虚拟环境可以正常激活。
- [ ] `depthai==3.9.0`、NumPy 和 OpenCV 可以正常导入。
- [ ] 真实有线网卡存在且状态为 `Up`。
- [ ] 已确认使用直连地址还是 DHCP 地址。
- [ ] DepthAI 枚举命令能够看到目标相机。
- [ ] 输出盘符存在并有足够空间。
- [ ] 使用 `R` 和 `S` 完成至少两个短分段测试。
- [ ] 每段左右图、深度图数量与 `metadata.csv` 数据行数一致。
- [ ] 使用 `Q` 安全退出后，`session.json` 中 `completed` 为 `true`。

## 12. 官方参考

- DepthAI v3：https://docs.luxonis.com/software-v3/depthai/
- OAK PoE 部署：https://docs.luxonis.com/hardware/platform/deploy/poe-deployment-guide/
- StereoDepth：https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/stereo_depth
