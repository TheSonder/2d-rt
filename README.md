# 2D-RT

`2D-RT` 是一个面向 2D 场景传播边界提取的工程，当前主能力不是通用 ray tracing demo，而是：

- 从场景 JSON 加载建筑轮廓与发射机位置
- 构建 2D 几何索引与可见性判断
- 提取 `LoS / Reflection / Diffraction / RR / RD / DR` 等传播边界
- 导出结构化边界 JSON
- 生成用于人工检查的可视化 PNG
- 保留一个基于 PyTorch 自定义算子的 C/C++/CUDA `raytrace` 示例能力

项目采用 CMake 构建原生扩展，Python 层负责场景加载、边界提取、导出和可视化。

## 当前能力概览

### 1. 边界提取

Python 包 `rt2d` 当前已经实现以下能力：

- `extract_scene_boundaries(...)`
  - 按场景和 TX 提取传播边界
  - 支持 `max_interactions=0/1/2`
  - 默认输出 `LoS / Reflection / RR`
  - 传入 `include_diffraction=True` 后额外输出 `Diffraction / RD / DR`
- `build_geometry(...)`
  - 将场景预处理成可复用几何索引
- `compute_visible_subsegments(...)`
  - 计算边段在给定 TX 下的可见子段
- `export_boundaries_json(...)`
  - 导出统一 JSON 结构

当前边界输出中包含：

- `type`: `los` / `reflection` / `diffraction` / `mixed`
- `p0`, `p1`: 边界起止点
- `source`: 来源建筑、边或顶点
- `mechanism`: 传播机制描述
- `sequence`: 例如 `L`、`R`、`D`、`RR`、`RD`、`DR`
- `role`: 例如 `visibility`、`reflection_face`、`reflection_shadow`、`diffraction_edge`
- `scene_id`, `tx_id`

### 2. 可视化

仓库内置边界可视化脚本，支持三种模式：

- `pretty`
  - 常规 matplotlib 展示图，便于人工浏览
- `aligned`
  - 按图像网格坐标导出，可与 RadioMapSeer 风格 PNG 对齐
- `aligned-geometry`
  - 只导出建筑和边界线，不叠加底图

可视化默认按 `tx_id` 分文件输出，每个发射机一张图。

### 3. 原生 Torch 算子

项目仍然保留 `rt2d.raytrace(...)`：

- 原生实现位于 C / C++ / CUDA
- 通过 `torch.ops.load_library(...)` 动态加载
- 当前示例功能是计算射线与球体的交点距离

这部分现在更像原生扩展示例或底层实验能力，而不是仓库的主要业务接口。

## 仓库结构

```text
.
|-- CMakeLists.txt
|-- CHANGELOG.md
|-- data/
|   `-- environment/
|       |-- antenna/
|       `-- polygon/
|-- include/rt2d/
|   |-- c_scene.h
|   `-- raytrace.h
|-- python/
|   |-- examples/
|   |   |-- demo.py
|   |   |-- extract_boundaries.py
|   |   `-- visualize_boundaries.py
|   `-- rt2d/
|       |-- __init__.py
|       |-- boundary.py
|       |-- loader.py
|       `-- scene.py
|-- scripts/
|   `-- build.ps1
|-- src/
|   |-- c/c_scene.c
|   |-- cpp/ops.cpp
|   |-- cpp/raytrace_cpu.cpp
|   `-- cuda/raytrace_cuda.cu
`-- tests/
    `-- test_boundaries.py
```

## 场景数据格式

默认场景目录是：

```text
data/environment/
  antenna/<scene_id>.json
  polygon/<scene_id>.json
```

其中：

- `antenna/<scene_id>.json` 是发射机坐标数组
- `polygon/<scene_id>.json` 是建筑多边形数组

示例：

```json
{
  "antenna": [[12.0, 34.0], [56.0, 78.0]],
  "polygons": [
    [[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0], [0.0, 0.0]]
  ]
}
```

实际加载时两类数据分开存储，`rt2d.load_scene(scene_id)` 会组合成：

```python
{
    "scene_id": "0",
    "root_dir": "...",
    "antenna_path": "...",
    "polygon_path": "...",
    "antenna": [[...], [...]],
    "polygons": [[[...], ...]],
}
```

## 环境要求

### 提取与可视化

- Python 3.10+
- `matplotlib`
- `Pillow`

如果只使用纯 Python 的场景加载、边界提取和导出，不依赖 PyTorch。

### 构建原生扩展

- Python 3.10+
- PyTorch
- CMake 3.24+
- 支持 C++17 的编译器
- CUDA Toolkit 与 NVIDIA Driver

如果只需要 Python 边界提取能力，可以先不构建原生扩展。

## 构建原生库

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1 -Config Release
```

如果要指定 Python 解释器：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1 -Config Release -PythonExe "C:\path\to\python.exe"
```

构建脚本会执行：

1. 配置 CMake
2. 编译 `rt2d_torch`
3. 安装动态库到 `python/rt2d`

可选地启用 `FetchContent` 拉取依赖：

```powershell
cmake -S . -B build -DRT2D_USE_FETCH_DEPS=ON
```

当前可选依赖：

- `fmt`
- `glm`

## Python 使用

### 1. 加载场景

```python
import rt2d

scene = rt2d.load_scene("0")
print(scene["scene_id"])
print(len(scene["antenna"]))
print(len(scene["polygons"]))
```

### 2. 提取边界

```python
import rt2d

payload = rt2d.extract_scene_boundaries(
    "0",
    tx_ids=[0],
    max_interactions=2,
    include_diffraction=True,
)

print(payload["scene_id"])
print(payload["boundary_count"])
print(payload["boundaries"][0])
```

如果你已经拿到场景字典，也可以直接传入：

```python
scene = rt2d.load_scene("0")
payload = rt2d.extract_scene_boundaries(scene, tx_ids=[0], max_interactions=1)
```

### 3. 复用中间几何结果

```python
import rt2d

scene = rt2d.load_scene("0")
geom = rt2d.build_geometry(scene)
tx = geom.antennas[0]
edge = geom.edges[0]

visible_segments = rt2d.compute_visible_subsegments(tx, edge, geom)
print(visible_segments)
```

### 4. 导出边界 JSON

```python
import rt2d

payload = rt2d.extract_scene_boundaries(
    "0",
    tx_ids=[0],
    max_interactions=1,
    output_path="build/scene_0_boundaries.json",
)
```

## 命令行示例

先设置 Python 包搜索路径：

```powershell
$env:PYTHONPATH = "$PWD\python"
```

### 1. 提取边界 JSON

```powershell
python .\python\examples\extract_boundaries.py 0 --tx-id 0 --max-interactions 2 --output .\build\scene_0.json
```

启用绕射：

```powershell
python .\python\examples\extract_boundaries.py 0 --tx-id 0 --max-interactions 2 --with-diffraction
```

### 2. 可视化边界

常规展示图：

```powershell
python .\python\examples\visualize_boundaries.py 0 --tx-id 0 --mode pretty
```

与底图对齐：

```powershell
python .\python\examples\visualize_boundaries.py 0 --tx-id 0 --mode aligned --canvas-size 256 --radiomap-png .\path\to\radiomap.png
```

仅导出建筑和边界线：

```powershell
python .\python\examples\visualize_boundaries.py 0 --tx-id 0 --mode aligned-geometry --canvas-size 256
```

### 3. 运行原生 raytrace 示例

```powershell
python .\python\examples\demo.py
```

## `rt2d.raytrace(...)` 示例

```python
import torch
import rt2d

device = "cuda" if torch.cuda.is_available() else "cpu"
origins = torch.zeros((4096, 3), device=device, dtype=torch.float32)
directions = torch.randn((4096, 3), device=device, dtype=torch.float32)
directions = directions / directions.norm(dim=1, keepdim=True)
sphere_center = torch.tensor([0.0, 0.0, 3.0], device=device, dtype=torch.float32)

distances = rt2d.raytrace(origins, directions, sphere_center, sphere_radius=1.0)
```

注意：

- `raytrace(...)` 依赖已编译的 `rt2d_torch` 动态库
- `rt2d` 包本身支持惰性导入 `torch`
- 只有在真正调用 `raytrace(...)` 或 `load_library(...)` 时才需要 PyTorch

## 当前实现边界

截至当前代码状态：

- `max_interactions` 仅支持 `0 / 1 / 2`
- `max_interactions > 2` 会直接报错
- 二阶组合当前实现到 `RR / RD / DR`
- `DD` 和三阶及以上组合尚未实现
- 默认关闭绕射扩展，避免结果过于密集

## 测试

当前测试集中在边界提取逻辑，主要文件是：

- `tests/test_boundaries.py`

覆盖内容包括：

- 单矩形场景的 LoS / Reflection 输出
- 部分遮挡下的可见子段裁剪
- LoS / Reflection 边界在后续建筑处停止
- 反射管内的 shadow boundary 输出
- `max_interactions=2` 下的 `RR / RD / DR`
- 场景 `0 / 1` 的 JSON 导出结构

## 设计说明

这个仓库当前有两条能力线：

- 纯 Python 的 2D 场景边界提取主线
- C/C++/CUDA + Torch 扩展实验线

如果你的目标是做场景传播边界生成、导出和可视化，优先使用：

- `rt2d.load_scene(...)`
- `rt2d.extract_scene_boundaries(...)`
- `python/examples/extract_boundaries.py`
- `python/examples/visualize_boundaries.py`

如果你的目标是验证原生扩展编译链路或 Torch 自定义算子，再使用：

- `rt2d.raytrace(...)`
- `python/examples/demo.py`
