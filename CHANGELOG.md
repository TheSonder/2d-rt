# CHANGELOG

## 2026-03-28

### Changed
- 为 `LoS` 主流程新增 `los_shadow` 边界输出，使根状态下也能导出视距遮挡边界。
- 调整可视化样式，新增 `los_shadow` 的渲染颜色与图例项。
- 新增 `python/examples/generate_radiomapseer_los_dataset.py`，支持从 `RadioMapSeer` 数据目录批量生成 `LoS + LoS-shadow` 标签图，并与 `png/buildings_complete/{map_id}.png` 叠加输出。
- 新增对应测试，校验 `max_interactions=0` 时可以输出 `los_shadow`。

### Breaking
- 无兼容性影响。

### Migration
- 如需生成与 `png/buildings_complete` 叠加后的视距与视距遮挡边界标签图，调用 `python/examples/generate_radiomapseer_los_dataset.py`。
- 如需在现有提取结果中区分视距遮挡边界，读取新增的 `role="los_shadow"`。

## 2026-03-27

### Changed
- 清理并重写 `CHANGELOG.md`，将原有乱码内容统一整理为中文 UTF-8 文本。
- 清理并重写 `README.md`，将项目说明同步到当前真实能力，并统一为中文 UTF-8 文本。
- 收缩边界提取主流程，移除绕射链路，只保留反射链路与反射管内的遮挡区域边界。
- 调整 `extract_scene_boundaries(...)`、示例脚本、测试和导出说明，使仓库对外语义统一为“只支持反射边界”。
- 为 `extract_scene_boundaries(...)` 新增 `tx_id` 范围校验，非法索引现在直接报 `ValueError`。
- 调整边界去重键，将 `role` 纳入判定，避免几何重合但语义不同的边界被错误合并。
- 将 `max_interactions` 的上限从 `2` 扩展到 `4`，允许主流程展开到四阶反射。
- 新增四阶反射相关测试和文档说明。

### Breaking
- `extract_scene_boundaries(...)` 不再支持绕射输出，也不再接受 `include_diffraction` 参数。
- `rt2d` 不再导出 `extract_diffraction_events(...)`。
- `python/examples/extract_boundaries.py` 和 `python/examples/visualize_boundaries.py` 不再提供 `--with-diffraction` 开关。
- 边界输出现在只保留 `los / reflection` 两类 `type`，以及 `L / R / RR / RRR / RRRR` 序列。

### Migration
- 如果旧代码传入了 `include_diffraction=True`，请直接删除该参数。
- 如果旧代码依赖 `D / RD / DR` 或 `diffraction_edge`，需要改为只消费 `reflection_face` 和 `reflection_shadow`。
- 如果旧脚本使用了 `--with-diffraction`，请直接删除该命令行参数。
- 如需三阶或四阶反射，直接把 `max_interactions` 提高到 `3` 或 `4`。

## 2026-03-22

### Changed
- 新增 `python/rt2d/boundary.py`，实现 2D 几何展开、轻量拓扑、Uniform Grid 空间索引、LoS / Reflection 边界提取、边界合并以及场景级 JSON 导出。
- 新增 `rt2d.extract_scene_boundaries(...)`、`build_geometry(...)`、`compute_visible_subsegments(...)` 等 Python 接口，并在 `python/rt2d/__init__.py` 中导出。
- 新增 `python/examples/extract_boundaries.py`，支持按 `scene_id` 提取并导出边界 JSON。
- 新增 `python/examples/visualize_boundaries.py`，支持将建筑与 `los / reflection` 边界渲染导出为 PNG。
- 调整 `python/examples/*.py`，从仓库根目录直接运行时自动补齐 `python/` 到 `sys.path`。
- 新增 `tests/test_boundaries.py`，覆盖单矩形、半遮挡反射、场景 `0 / 1` 结构化导出等测试。
- 调整边界延长逻辑：`LoS` 和 `Reflection` 的终点现在优先停在首个环境碰撞点，否则才落到场景外边界，不再直接穿过后续建筑。
- 调整边界起射过滤逻辑：如果边界从顶点或边界点出发后立刻进入源建筑内部，则直接丢弃，不再输出穿入建筑体的伪边界。
- 新增 `max_interactions` 配置，当前支持高阶反射展开。
- 新增 `sequence` 输出字段，使用 `L / R / RR / RRR / RRRR` 标记边界来源序列。
- 将边界生成主流程重构为基于“射线管状态机”的递推过程：每次交互形成新的 `TubeState`，后续目标边或点先经过当前射线管裁剪，再做可见性与镜像判定。
- 为非根 `TubeState` 新增“状态可见性边界”提取；当反射射线管内部被后续建筑遮挡时，现在会输出对应的同序列阴影边界。
- 将 `LoS` 也纳入统一的 `TubeState` 可见性边界流程，根状态下的视距边界与反射态共用同一套生成逻辑。
- 调整反射态可见性判定：镜像源下的后续可见性会排除父反射建筑的全部边，避免父反射体错误遮挡反射射线管内部目标。
- 调整反射态阴影轮廓提取：非根状态下优先使用可见顶点做轮廓极值，而不是仅用逐边子段端点，修正矩形遮挡物阴影顶点选择错误的问题。
- 新增 `role` 输出字段，当前可区分 `visibility / reflection_face / reflection_shadow / los_shadow`，并在可视化中将阴影边界单独高亮显示。
- 新增测试场景 `reflection_shadow_demo`，用于人工核对一阶反射与阴影边界。
- 调整 `python/rt2d/__init__.py` 和 `python/rt2d/loader.py`，将 `torch` 改为惰性导入，使纯几何边界提取不再依赖导入时就存在 PyTorch。
- 调整 `python/examples/visualize_boundaries.py`，新增 `--mode aligned`、`--canvas-size` 和 `--radiomap-png`，支持按 `256x256` 图像坐标导出无边框 PNG，并直接叠加到 RadioMapSeer 风格底图。
- 将 `aligned` 模式进一步改为基于像素栅格的 raster 渲染，使用像素边界坐标映射而不是 matplotlib 矢量填充，减少与 RadioMapSeer PNG 的半像素偏移。
- 修正 `aligned` 模式的像素落位：去掉统一的 `-0.5` 偏移，改为按像素中心对齐并对边界点做栅格范围裁剪，修复整体偏左偏上的问题。
- 修正 `aligned` 模式的比例尺度：移除误加入的 `+5` 像素硬偏移，并禁止场外延长边界参与对齐范围缩放，避免底图与边界出现比例尺不一致。
- 调整 `python/examples/visualize_boundaries.py` 的出图策略：不再绘制任何 TX 标记，且按 `tx_id` 单独输出 PNG，一张图只保留对应发射机的边界结果。
- 新增 `visualize_boundaries.py --mode aligned-geometry`，输出与 RadioMapSeer 图像网格同坐标的“仅建筑物 + 边界线”单张 PNG，不叠加 radio map 底图。
- 调整可视化样式：将紫色 `reflection_shadow` 线宽从 `2.0` 收窄到 `1.0`，减少对齐图中遮挡边界的视觉压迫感。

### Breaking
- `rt2d` 现在默认暴露边界提取相关新接口；`raytrace` 仍保留原调用方式，但 `torch` 不再在包导入阶段立即加载。
- `LoS / Reflection` 边界的 `p1` 语义从“扩展到场景包围盒”改为“扩展到首个环境碰撞或场景外边界”。
- `LoS / Reflection` 现在会过滤掉从源建筑边界点出发但立刻进入该建筑内部的无效边界。
- 边界 JSON 新增 `sequence` 和 `role` 字段；如果下游直接按旧 key 集合校验，需要同步更新。
- `python/examples/visualize_boundaries.py` 新增 `aligned` 导出模式；该模式使用图像坐标系并翻转 `y` 轴，输出不再带标题、坐标轴、网格和边距。
- `aligned` 模式的内部渲染方式改为像素栅格绘制；如果之前依赖 matplotlib 的矢量抗锯齿外观，视觉效果会变化，但几何落位更接近 RadioMapSeer。
- `aligned` 模式的像素坐标语义调整为像素中心落位；如果之前基于旧 `-0.5` 偏移生成对比图，需要重新导出。
- `aligned` 模式不再让场外延长边界改变对齐 extent；如果之前依赖“整条延长边界都完整显示”，现在会改为优先保持和 RadioMapSeer 底图同尺度。
- `visualize_boundaries.py` 默认从“多 TX 合成一张图”改为“每个 TX 一张图”；如果之前依赖单文件汇总图，需要改为读取多个输出文件或显式传入单个 `--tx-id`。
- `visualize_boundaries.py` 新增 `aligned-geometry` 模式；如果之前依赖“不提供底图路径时由 `aligned` 隐式输出几何图”，建议显式改用新模式。

### Migration
- 如需提取传播边界，直接调用 `rt2d.extract_scene_boundaries(scene_id_or_scene, tx_ids=..., output_path=...)`。
- 如需复用中间几何结果，调用 `rt2d.build_geometry(scene)` 和 `rt2d.compute_visible_subsegments(tx, edge, geom)`。
- 如果下游逻辑依赖旧的超长边界线段，需要改为使用新的 `p1` 语义；它现在表示真实的碰撞终止点或出界点。
- 如果下游逻辑曾依赖穿入建筑内部的伪边界，需要按新的过滤规则重新消费结果。
- 如需高阶反射，显式传入 `max_interactions=0 / 1 / 2 / 3 / 4`。
- 如需区分反射面本身的边界与反射射线管内的阴影边界，读取新增的 `role` 字段。
- 如需与 RadioMapSeer 的 `256x256` PNG 对齐，使用 `python/examples/visualize_boundaries.py --mode aligned --canvas-size 256`；如需叠到底图上，额外传入 `--radiomap-png <path>`。
- 如需验证像素级对齐，优先使用最新的 `aligned` 导出；当前版本已改为按像素中心映射，而不是旧的半像素左上偏移。
- 如需与 RadioMapSeer 底图严格同尺度，使用最新 `aligned` 导出；该模式现在以场景网格为准，不再把场外延长边界算入缩放范围。
- 如需只导出单个发射机，继续传 `--tx-id <id>`；如不传，脚本会按 payload 中的每个 `tx_id` 自动生成带 `_tx{id}` 后缀的多张 PNG。
- 如需导出与原图同尺寸、同坐标、但只包含建筑与边界线的图，使用 `python/examples/visualize_boundaries.py --mode aligned-geometry`。
- 既有 `rt2d.raytrace(...)` 调用无需修改；如果上层代码依赖“导入 `rt2d` 时必须立即因缺少 PyTorch 失败”，需要改为在调用 `raytrace` 或 `load_library` 时处理该错误。



