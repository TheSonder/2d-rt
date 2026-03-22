# CHANGELOG

## 2026-03-22

### Changed
- 新增 `python/rt2d/boundary.py`，实现几何展开、轻量拓扑、Uniform Grid 空间索引、LoS/Reflection/Diffraction 边界提取、边界合并和场景级 JSON 导出。
- 新增 `rt2d.extract_scene_boundaries(...)`、`build_geometry(...)`、`compute_visible_subsegments(...)` 等 Python 接口，并在 `python/rt2d/__init__.py` 中导出。
- 新增 `python/examples/extract_boundaries.py`，支持按 `scene_id` 抽取并导出边界 JSON。
- 新增 `tests/test_boundaries.py`，覆盖单矩形、半遮挡反射面、`scene 0/1` 结构化导出。
- 调整 `python/rt2d/__init__.py` 与 `python/rt2d/loader.py`，将 `torch` 改为惰性导入，使纯几何边界提取不再依赖导入时就存在 PyTorch。

### Breaking
- `rt2d` 现在默认暴露边界提取相关新接口；`raytrace` 仍保留原调用方式，但 `torch` 不再在包导入阶段立即加载。
- 无额外兼容性影响。

### Migration
- 如需提取传播边界，直接调用 `rt2d.extract_scene_boundaries(scene_id_or_scene, tx_ids=..., output_path=...)`。
- 如需复用中间几何结果，调用 `rt2d.build_geometry(scene)` 和 `rt2d.compute_visible_subsegments(tx, edge, geom)`。
- 既有 `rt2d.raytrace(...)` 调用无需修改；如果上层代码依赖“导入 `rt2d` 时必须立即失败于缺少 PyTorch”，需要改为在调用 `raytrace` 或 `load_library` 时处理该错误。
