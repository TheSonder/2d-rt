# CHANGELOG

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
- 将 `Diffraction` 从点事件升级为线边界输出，并让可视化脚本按线段绘制。
- 调整 `python/rt2d/__init__.py` 与 `python/rt2d/loader.py`，将 `torch` 改为惰性导入，使纯几何边界提取不再依赖导入时就存在 PyTorch。

### Breaking
- `rt2d` 现在默认暴露边界提取相关新接口；`raytrace` 仍保留原调用方式，但 `torch` 不再在包导入阶段立即加载。
- `LoS`/`Reflection` 边界的 `p1` 语义从“扩展到场景包围盒”改为“扩展到首个环境碰撞或场景外边界”。
- `LoS`/`Reflection` 现在会过滤掉从源建筑边界点出发但立刻进入该建筑内部的无效边界。
- 边界 JSON 新增 `sequence` 字段；`Diffraction` 不再只输出点事件，而是输出带终点的线边界。
- `extract_scene_boundaries(..., max_interactions=...)` 目前对 `max_interactions > 2` 直接报错，不再静默退化。
- 无额外兼容性影响。

### Migration
- 如需提取传播边界，直接调用 `rt2d.extract_scene_boundaries(scene_id_or_scene, tx_ids=..., output_path=...)`。
- 如需复用中间几何结果，调用 `rt2d.build_geometry(scene)` 和 `rt2d.compute_visible_subsegments(tx, edge, geom)`。
- 如有下游逻辑依赖旧的超长边界线段，需要改为使用新的 `p1`，它现在表示几何上真实的终止碰撞点或出界点。
- 如有下游逻辑曾依赖这些“穿入建筑内部”的伪边界，需要按新输出重新消费。
- 如需对齐一阶/二阶交互，改为显式传入 `max_interactions=0/1/2`。
- 下游若之前把 `Diffraction` 当点事件处理，需要改为读取新的 `p0/p1/sequence` 线边界表示。
- 既有 `rt2d.raytrace(...)` 调用无需修改；如果上层代码依赖“导入 `rt2d` 时必须立即失败于缺少 PyTorch”，需要改为在调用 `raytrace` 或 `load_library` 时处理该错误。
