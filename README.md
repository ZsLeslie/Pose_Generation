# 三相机位姿生成流程

使用 `extrinsics/rig_extrinsics.yaml` 中的刚体外参，将 `colmap/sparse` 中每一帧前相机位姿转换为左、中、右相机位姿。中相机记为 `mid`，与 YAML 一致。

## 主程序

推荐入口为项目根目录的 `main.py`，按以下顺序调用项目模块：

1. 通过 `transformZ.align_model()`，根据距离文件恢复整个 sparse 模型的米制尺度，并以参考前相机的光学坐标系定义新世界坐标系。
2. 从转换后的模型读取前相机位姿，调用 `generate_poses()` 计算左、中、右相机位姿，输出四相机的 JSON 和 CSV。

需要 Python 3.10+、`requirements.txt` 中的依赖，以及运行环境中的 COLMAP 命令行程序。导入项目模块不会执行 COLMAP；只有调用主流程或独立对齐程序时才检查该程序是否可用。

```powershell
python -m pip install -r requirements.txt
python main.py --input colmap/sparse --distance distance.xlsx --extrinsics extrinsics/rig_extrinsics.yaml --output outputs/pipeline
```

COLMAP 未加入 PATH 时，通过 `--colmap-bin "C:\path\to\colmap.exe"` 指定程序路径。可以通过 `--reference-image front_1785227493507.jpg` 指定参考帧；不指定时，选择按图像名自然排序的第一个已注册前相机。参考帧必须属于 `--front-name-pattern` 和 `--front-camera-id` 筛选后的前相机集合。

参考帧相机中心变为 `(0, 0, 0)`，相机到世界、世界到相机变换均为单位矩阵；新世界轴为该帧光学 X 向右、Y 向下、Z 向前。这里的 Z 轴是参考相机光轴，不代表重力方向。世界坐标变换为 `X_new = s * (R_ref @ X_old + t_ref)`，同时作用于模型位姿和三维点。

**尺度只恢复一次。** 第一步已经将模型变为米制，第二步固定使用 `meters_per_colmap_unit=1.0`，直接将以米为单位的机间外参与前相机位姿组合。原始尺度因子 `s` 单独保存在报告中，不会再次应用于三相机基线。

默认输出结构：

```text
outputs/pipeline/
  sparse_metric/             # 已恢复尺度并对齐的完整 COLMAP 模型
    transform_sim3.txt       # COLMAP 使用的八项 Sim3 参数
    T_new_from_old.txt        # 新旧世界坐标之间的 4x4 相似变换矩阵
    scale_report.json        # 尺度拟合及模型变换验证报告
  poses/
    poses.json               # 四相机位姿，含参考坐标系、单位、标定与对齐元数据
    poses.csv                # 每帧 front/left/mid/right 各一行，length_unit=m
  pipeline_report.json       # 完整流程的输入、输出、参考帧、尺度和统计信息
```

重复运行时，显式传入 `--overwrite` 才会替换已有主流程结果。输入 sparse 模型、距离文件和外参文件必须放在输出目录之外。完整模型在临时目录完成转换和验证后才发布到 `sparse_metric`。

尺度拟合延用原程序的累计相机路径弦长最小二乘方法；输入位姿的离群点、采样稀疏和距离数据误差都会影响拟合。流程保留原始帧，不自动删除、插值或平滑位姿，可检查 `scale_report.json` 中的误差统计。

COLMAP 的相似变换文件采用 `scale qw qx qy qz tx ty tz` 八项格式，见 [COLMAP Sim3 文件实现](https://github.com/colmap/colmap/blob/main/src/colmap/geometry/sim3.cc)。

## 代码结构

```text
main.py                  # 参数解析与主程序入口
src/
  pipeline.py            # 配置、返回结果与两阶段流程编排
  transformZ.py          # 尺度估计、参考相机对齐、COLMAP 调用与结果验证
  distance_io.py         # TXT 距离读取及 Excel 到 TXT 的转换
  colmap_io.py           # COLMAP BIN/TXT 位姿读取、前相机筛选和排序
  geometry.py            # 四元数、旋转矩阵、相机中心及位姿求逆
  generate_poses.py      # 机间外参读取、位姿计算与 JSON/CSV 导出
  visualize_poses.py     # 轨迹图与统计导出
```

Python 调用接口：

```python
from pathlib import Path
from src.pipeline import PipelineConfig, run_pipeline

result = run_pipeline(PipelineConfig(
    input_model=Path("colmap/sparse"),
    distance_file=Path("distance.xlsx"),
    extrinsics_file=Path("extrinsics/rig_extrinsics.yaml"),
    output_dir=Path("outputs/pipeline"),
    reference_image="front_1785227493507.jpg",
    colmap_bin="colmap",
))
print(result.poses_csv)
```

## 独立运行位姿生成

需要 Python 3.10 或以上版本：

```powershell
python -m pip install -r requirements.txt
python src/generate_poses.py
```

默认读取 `images.bin`，没有二进制文件时读取 `images.txt`；默认只选择文件名匹配 `front_*` 的已注册图像，并按图像名排序。输入、输出默认路径相对于项目目录，可从其他工作目录运行脚本。

可指定路径及筛选条件：

```powershell
python src/generate_poses.py --model colmap/sparse --extrinsics extrinsics/rig_extrinsics.yaml --output outputs --front-camera-id 1
```

如果前相机图像没有 `front_` 前缀，设置 `--front-name-pattern "*"`，并按需结合 `--front-camera-id` 筛选。`--model` 应指向直接包含 `images.bin` 或 `images.txt` 的目录，例如其他项目中的 `sparse/0`。

## 尺度与精度

直接对未定标模型独立运行 `generate_poses.py` 时，已知以米为单位的相机间外参不能单独确定绝对尺度。省略尺度参数会暂时假设 `1 COLMAP 单位 = 1 米`，在终端警告，并记录 `scale_status = assumed`。这些位置仅为暂定结果；旋转不受尺度参数影响。主程序 `main.py` 先完成定标，不会使用这个暂定模式。

得到尺度后，重新运行。例如，测量场景中两点的真实距离为 `2 m`，在 COLMAP 重建中距离为 `5`，则 `s = 2 / 5 = 0.4 m/COLMAP 单位`：

```powershell
python src/generate_poses.py --meters-per-colmap-unit 0.4
```

输出始终保留原 COLMAP 世界坐标系和单位，标定平移除以 `s` 后再参与计算，因此能够与原始点云对齐。若还需要米制世界坐标，将输出的平移、相机中心及对应点云坐标统一乘以 `s`。`scale_status = provided` 表示尺度由调用者提供，并不表示代码验证了定标准确性。

上述说明适用于直接输入未定标模型的独立模式。对已经米制对齐的模型，可运行 `python src/generate_poses.py --model outputs/pipeline/sparse_metric --metric-model`，此时输出单位为米，并且不再次缩放模型。

此外，当前 YAML 声明了 `extrinsics_approximate: 1`：前相机工作姿态由绕光心俯仰的假设得到，旋转轴和光心位移未经测量。结果会保留该精度说明。

## 坐标系与计算公式

采用 [COLMAP 官方位姿约定](https://colmap.github.io/format.html#images-txt)：四元数顺序为 `(qw, qx, qy, qz)`，位姿为世界到相机的变换；相机光学坐标轴为 X 向右、Y 向下、Z 向前。所有矩阵对列向量左乘。

令前相机的 COLMAP 位姿为 `H_front_from_world = [R_f, t_f]`，YAML 外参为 `H_camera_from_front = [R_cf, t_cf_m]`，尺度为 `s`：

```text
R_camera_from_world = R_cf @ R_f
t_camera_from_world = R_cf @ t_f + t_cf_m / s
H_world_from_camera = inverse(H_camera_from_world)
camera_center_world = -R_camera_from_world.T @ t_camera_from_world
```

当前 YAML 的 `H_left_from_front`、`H_mid_from_front`、`H_right_from_front` **已经包含**从 `front_capture` 到 `front_work` 的 90° 姿态转换，默认 `--front-frame work` 直接使用这三个矩阵，不能再重复加 90°。

若 COLMAP 使用的是标定时朝向的 `front_capture` 图像，应明确指定：

```powershell
python src/generate_poses.py --front-frame capture
```

此时相机间变换为 `H_camera_from_front_work @ H_front_work_from_front_capture`。该参数只选择 COLMAP 前相机位姿对应的光学坐标系，不会改动输入位姿或点云。

## 输出

- `outputs/poses.json`：每帧的源图像名称、源图像 ID、源相机 ID，以及 `front`、`left`、`mid`、`right` 的四元数、平移、相机中心、世界到相机和相机到世界的 4×4 矩阵；元数据记录尺度假设、前相机姿态选择及标定精度。
- `outputs/poses.csv`：每帧前、左、中、右各一行。`qw...qz` 和 `tx...tz` 表示世界到相机变换，`cx...cz` 表示世界坐标中的相机中心；同时记录尺度和姿态选择。

当前数据含 117 帧前相机位姿，因此计算 351 条左中右位姿，加上原始前相机位姿后 CSV 共 468 行数据。每组三相机位姿对应同一前相机曝光时刻，前提是刚性安装和图像同步；脚本不进行时间插值或侧相机图像匹配。输出是位姿文件，不包含侧相机内参或特征观测，不能作为完整的 COLMAP 稀疏模型直接导入。

## 四相机轨迹可视化

```powershell
python src/generate_poses.py
python src/visualize_poses.py
```

本机也可以直接使用现有环境：

```powershell
& 'D:\Software\miniconda\envs\lidar\python.exe' src/visualize_poses.py
```

可视化仅从 CSV 读取四相机位姿，不依赖 JSON、外参文件或 COLMAP 模型。旧版 CSV 只有左中右相机，需先运行新版 `generate_poses.py` 补齐前相机；若已有真实尺度，请保留生成时使用的 `--meters-per-colmap-unit` 参数。

生成以下文件：

- `outputs/camera_trajectories.html`：可离线打开的交互式三维图。拖动旋转、滚轮缩放、点击图例切换相机；帧滑块同步查看该帧四相机的位置和光轴 +Z 方向。空心圆表示起点，空心菱形表示终点。HTML 内嵌 Plotly，无需服务器或联网。
- `outputs/camera_trajectories.png`：三维轨迹和 XY、XZ、YZ 三个平面投影。圆点为起点，X 为终点，箭头为初始光轴朝向。三维坐标轴保持真实长度比例；二维投影采用独立坐标轴比例，以便观察较小方向上的偏移。
- `outputs/trajectory_summary.csv`：每个相机的帧数、相邻采样点累计路程、起终点直线距离及起终点坐标；长度单位为 COLMAP 单位。累计路程取决于采样密度及位姿噪声，不代表经过平滑或补帧的真实行驶里程。

轨迹使用相机中心 `cx, cy, cz`，不能用世界到相机的平移 `tx, ty, tz` 直接绘制。相机朝向由世界到相机旋转的转置计算。帧按图像名称中的数字自然排序，不按无时序含义的 COLMAP 图像 ID 排序。图中 X、Y、Z 为原始 COLMAP 世界轴，并未假设某个轴对应重力方向。

可指定输入、输出和光轴显示长度：

```powershell
python src/visualize_poses.py --csv outputs/poses.csv --output outputs --arrow-length 0.3 --dpi 200
```

可视化主流程的米制结果：

```powershell
python src/visualize_poses.py --csv outputs/pipeline/poses/poses.csv --output outputs/pipeline/plots
```

可视化程序读取 `length_unit` 标注坐标和路程单位；米制结果使用 `m`，旧版无该列的 CSV 仍按 COLMAP 单位显示。

尺度尚未定标时，图标题会明确标出位置为暂定值，不将坐标直接标为米。绘图与交互导出采用 [Matplotlib](https://matplotlib.org/stable/) 和 [Plotly 的自包含 HTML 导出](https://plotly.com/python/interactive-html-export/)。

## transformZ 距离文件输入

`src/transformZ.py` 的 `--distance` 参数接受 TXT 或 Excel（`.xlsx`、`.xls`、`.xlsm`、`.excel`）。Excel 默认读取第一个工作表，支持当前 `distance.xlsx` 的 `name / distance / time` 列：按列名提取 `name` 和 `distance`，忽略 `time`。也支持无表头的“距离、图像名”或“图像名、距离”两列数据。距离单位为米。

Excel 会先转换为同目录的同名 TXT，例如 `distance.xlsx` → `distance.txt`，每行格式为 `距离 图像名`，然后继续原有尺度计算。转换时校验缺失值、非有限数、负距离和重复图像名；公式单元格使用 Excel 保存的计算结果，需先在 Excel 中计算并保存。若已有 TXT 内容不同，会另存为 `distance.converted.txt` 等名称；内容相同则直接复用。

例如，在原有运行命令中改为 `--distance distance.xlsx` 即可，不需要手动转换。Excel 读取依赖 `openpyxl`（现代 Excel 格式）或 `xlrd`（旧版 XLS），已加入 `requirements.txt`。

## 验证

```powershell
python -m unittest discover -s tests -v
```
