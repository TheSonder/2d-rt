# CHANGELOG

## 2026-03-27

### Changed
- 重写 `README.md`，按当前项目真实能力重新整理项目定位、目录结构、场景数据格式、边界提取 API、可视化脚本、原生扩展构建方式和使用示例。
- 明确区分仓库的两条能力线：Python 侧的 2D 传播边界提取主线，以及 C/C++/CUDA + Torch 自定义算子实验线。
- 补充 `extract_scene_boundaries(...)`、`build_geometry(...)`、`compute_visible_subsegments(...)`、`extract_boundaries.py`、`visualize_boundaries.py`、`demo.py` 的 README 使用说明。
- 补充 `pretty / aligned / aligned-geometry` 三种可视化模式说明，以及 `max_interactions`、`include_diffraction` 的当前行为说明。

### Breaking
- 无兼容性影响。

### Migration
- 无需修改现有调用代码。
- 如需了解当前推荐用法，请以新的 `README.md` 为准，不再参考旧的 “sphere raytrace starter project” 描述。

## 2026-03-22

### Changed
- 新增 `python/rt2d/boundary.py`，实现几何展开、轻量拓扑、Uniform Grid 空间索引、LoS/Reflection/Diffraction 边界提取、边界合并和场景级 JSON 导出。
- 新增 `rt2d.extract_scene_boundaries(...)`、`build_geometry(...)`、`compute_visible_subsegments(...)` 等 Python 接口，并在 `python/rt2d/__init__.py` 中导出。
- 新增 `python/examples/extract_boundaries.py`，支持按 `scene_id` 抽取并导出边界 JSON。
- 新增 `python/examples/visualize_boundaries.py`，支持将建筑、TX 与 `los/reflection/diffraction/mixed` 边界叠加导出为 PNG。
- 调整 `python/examples/*.py`，从仓库根目录直接运行时自动补齐 `python/` 到 `sys.path`。
- 新增 `tests/test_boundaries.py`，覆盖单矩形、半遮挡反射面、`scene 0/1` 结构化导出。
- 调整边界延长逻辑：`LoS` 与 `Reflection` 终点现在会停在首个环境碰撞，否则才落到场景外边界，不再直接穿过后续建筑。
- 调整边界起射过滤逻辑：若边界从顶点/边界点出发后立即进入源建筑内部，则直接丢弃，不再输出穿入建筑体的伪边界。
- 新增 `max_interactions` 配置，当前支持 `0/1/2` 阶交互展开。
- 新增 `sequence` 输出字段，使用 `L/R/D/RR/RD/DR` 标记边界来源序列。
- 新增二阶边界展开：先实现 `RR / RD / DR`，暂未实现 `DD` 与三阶以上组合。
- 将边界生成主流程重构为“射线管状态机”递推：每次交互都会形成新的 `TubeState`，后续目标边/点先经过当前射线管裁剪，再做可见性与镜像/绕射判定。
- 为非根 `TubeState` 新增“状态可见性边界”提取，反射射线管内被后续建筑遮挡时，现在会输出对应的同序列阴影边界。
- 将 `LoS` 也并入统一的 `TubeState` 可见性边界流程，根状态下的视距边界现在与 `R/D` 共享同一套生成逻辑。
- 调整反射态可见性判定：镜像源后的后续可见性会排除父反射建筑的全部边，避免父反射体错误挡住反射射线管内的目标。
- 调整反射态阴影轮廓提取：非根状态下优先使用可见顶点做轮廓极值，而不是仅用逐边子段端点，修正矩形遮挡物阴影顶点选错的问题。
- 新增 `role` 输出字段，当前可区分 `visibility / reflection_face / reflection_shadow / diffraction_edge`，并在可视化中将 `reflection_shadow` 单独高亮显示。
- 将 `Diffraction` 从点事件升级为线边界输出，并让可视化脚本按线段绘制。
- 新增测试场景 `reflection_shadow_demo`，场景布局对应“单 TX + 大反射建筑 + 反射管内小建筑遮挡”的示意图，便于人工核对一阶反射与阴影边界。
- 调整 `python/rt2d/__init__.py` 与 `python/rt2d/loader.py`，将 `torch` 改为惰性导入，使纯几何边界提取不再依赖导入时就存在 PyTorch。
- 调整 `python/examples/visualize_boundaries.py`，新增 `--mode aligned`、`--canvas-size` 与 `--radiomap-png`，支持按 `256x256` 图像坐标导出无边框 PNG，并与 RadioMapSeer 的 radio map 底图直接叠加对齐。
- 将 `aligned` 模式进一步改为基于像素栅格的 raster 渲染，使用像素边界坐标映射而不是 matplotlib 矢量填充，减少与 RadioMapSeer PNG 的半像素偏移。
- 修正 `aligned` 模式的像素落位：去掉统一的 `-0.5` 偏移，改为按像素中心对齐并对边界点做栅格范围裁剪，修复整体“偏左偏上”的问题。
- 修正 `aligned` 模式的比例尺：移除误加入的 `+5` 像素硬偏移，并禁止场外延长边界参与对齐范围缩放，避免底图与边界出现“比例尺不一致”。
- 调整 `python/examples/visualize_boundaries.py` 的出图策略：不再绘制任何 TX 标记，且按 `tx_id` 单独输出 PNG，一张图只保留对应发射机的边界结果。
- 新增 `visualize_boundaries.py --mode aligned-geometry`，输出与 RadioMapSeer 图像网格同坐标的“仅建筑物 + 边界线”单张 PNG，不叠加 radio map 底图。
- 调整边界提取默认机制：`extract_scene_boundaries(...)` 现在默认关闭绕射扩展，仅保留 `LoS / Reflection / RR`；如需 `D / RD / DR`，必须显式传入 `include_diffraction=True` 或在示例脚本里使用 `--with-diffraction`。
- 调整 [python/examples/extract_boundaries.py](/d:/TheSonder/0.5%20Code/Python/2d-rt/python/examples/extract_boundaries.py) 和 [python/examples/visualize_boundaries.py](/d:/TheSonder/0.5%20Code/Python/2d-rt/python/examples/visualize_boundaries.py)，新增 `--with-diffraction` 命令行开关，默认关闭绕射以降低高阶结果复杂度。
- 调整可视化样式：将紫色 `reflection_shadow` 线宽从 `2.0` 收窄到 `1.0`，减少对齐图中遮挡边界的视觉压迫感。

### Breaking
- `rt2d` 现在默认暴露边界提取相关新接口；`raytrace` 仍保留原调用方式，但 `torch` 不再在包导入阶段立即加载。
- `LoS`/`Reflection` 边界的 `p1` 语义从“扩展到场景包围盒”改为“扩展到首个环境碰撞或场景外边界”。
- `LoS`/`Reflection` 现在会过滤掉从源建筑边界点出发但立刻进入该建筑内部的无效边界。
- 边界 JSON 新增 `sequence` 字段；`Diffraction` 不再只输出点事件，而是输出带终点的线边界。
- 边界 JSON 新增 `role` 字段；下游如果直接按旧 key 集合校验，需要同步更新。
- `extract_scene_boundaries(..., max_interactions=...)` 目前对 `max_interactions > 2` 直接报错，不再静默退化。
- `python/examples/visualize_boundaries.py` 新增 `aligned` 导出模式；该模式会使用图像坐标系并翻转 `y` 轴，输出不再带标题、坐标轴、网格与边距。
- `aligned` 模式的内部渲染方式改为像素栅格绘制；若你之前依赖 matplotlib 的抗锯齿外观，输出观感会改变，但几何落位会更接近 RadioMapSeer。
- `aligned` 模式的像素坐标语义进一步调整为像素中心落位；如果你手头有基于旧 `-0.5` 偏移生成的对比图，需要重新导出。
- `aligned` 模式不再让场外延长边界改变对齐 extent；如果你之前依赖“整条延长边界都完整显示”，现在会改为优先保持和 RadioMapSeer 底图同尺度。
- `visualize_boundaries.py` 默认从“多 TX 合成一张图”改为“每个 TX 一张图”；如果你之前依赖单文件汇总图，需要改用多个输出文件或显式传入单个 `--tx-id`。
- `visualize_boundaries.py` 新增 `aligned-geometry` 模式；如果你之前依赖“不给底图路径时由 `aligned` 隐式生成几何图”，现在建议显式切到新模式。
- `extract_scene_boundaries(...)` 的默认机制集合发生变化：默认不再输出 `diffraction / RD / DR`；如果你之前默认依赖这些边界，需要显式打开绕射。
- 可视化里的紫色阴影边界显示会比之前更细；无接口兼容性影响。
- 无额外兼容性影响。

### Migration
- 如需提取传播边界，直接调用 `rt2d.extract_scene_boundaries(scene_id_or_scene, tx_ids=..., output_path=...)`。
- 如需复用中间几何结果，调用 `rt2d.build_geometry(scene)` 和 `rt2d.compute_visible_subsegments(tx, edge, geom)`。
- 如有下游逻辑依赖旧的超长边界线段，需要改为使用新的 `p1`，它现在表示几何上真实的终止碰撞点或出界点。
- 如有下游逻辑曾依赖这些“穿入建筑内部”的伪边界，需要按新输出重新消费。
- 如需对齐一阶/二阶交互，改为显式传入 `max_interactions=0/1/2`。
- 下游若之前把 `Diffraction` 当点事件处理，需要改为读取新的 `p0/p1/sequence` 线边界表示。
- 如需区分反射面本身的边界和反射射线管内的阴影边界，读取新增的 `role` 字段。
- 如需与 RadioMapSeer 的 `256x256` PNG 对齐，改为使用 `python/examples/visualize_boundaries.py --mode aligned --canvas-size 256`；如需叠到底图上，额外传入 `--radiomap-png <path>`。
- 如需验证像素级对齐，优先使用 `--mode aligned` 并直接传入实际的 RadioMapSeer PNG；该模式现在会按像素边界做坐标映射。
- 重新验证像素级对齐时，使用最新的 `aligned` 导出；当前版本已改为按像素中心映射，而不是旧的半像素左上偏移。
- 若要与 RadioMapSeer 底图严格同尺度，使用最新的 `aligned` 导出；该模式现在以场景网格为准，不再把场外延长边界算进缩放范围。
- 如需只导出单个发射机，继续传 `--tx-id <id>`；如不传，则脚本会按 payload 中的每个 `tx_id` 自动生成带 `_tx{id}` 后缀的多张 PNG。
- 如需导出和原图同尺寸、同坐标、但只包含建筑与边界线的图，使用 `python/examples/visualize_boundaries.py --mode aligned-geometry`。
- 如需恢复旧的包含绕射的结果，调用 `rt2d.extract_scene_boundaries(..., include_diffraction=True)`，或在示例脚本后追加 `--with-diffraction`。
- 如需导出更轻量的紫色阴影边界展示，直接使用当前版本的可视化脚本，无需额外参数。
- 既有 `rt2d.raytrace(...)` 调用无需修改；如果上层代码依赖“导入 `rt2d` 时必须立即失败于缺少 PyTorch”，需要改为在调用 `raytrace` 或 `load_library` 时处理该错误。

