# CHANGELOG

## 2026-03-28

### Changed
- 为 `LoS` 主流程新增 `los_shadow` 边界输出，使根状态下也能导出视距遮挡边界。
- 调整可视化样式，新增 `los_shadow` 的渲染颜色与图例项。
- 新增 `python/examples/generate_radiomapseer_los_dataset.py`，支持从 `RadioMapSeer` 数据目录批量生成 `LoS + LoS-shadow` 标签图，并与 `png/buildings_complete/{map_id}.png` 叠加输出。
- 新增对应测试，校验 `max_interactions=0` 时可以输出 `los_shadow`。`r`n- 调整 `python/examples/generate_radiomapseer_physics_boundary_dataset.py`，新增 `--max-interactions` 参数，并支持渲染 `reflection_face / reflection_shadow`，可直接生成一阶反射数据集图像。
- 新增 `rt2d.compute_rx_visibility(...)`，支持在 `xoy` 平面按 `1m` 栅格生成 `RX` 点，并按最小传播阶数输出 `LoS / 1阶 / 2阶 / 3阶 / 4阶` 可见性标签。
- 新增 `python/rt2d/coverage.py`，实现基于传播状态展开的 `RX` 可见性求解，当前支持 `Reflection`、几何 `Diffraction` 以及 `RR / RD / DR / DD` 等混合高阶组合。
- 新增 `python/examples/extract_rx_visibility.py`，支持按场景导出 `RX` 栅格可见性 JSON。
- 新增 `python/examples/visualize_rx_visibility.py`，支持将 `RX` 可见性结果按阶数着色渲染为 PNG，并提供 `pretty / aligned` 两种出图模式。
- 调整 `RX` 导出与可视化脚本的默认区域策略：对数字场景 `scene_id`，在未显式传入 `--bounds` 时默认对齐到 `RadioMapSeer` 的整张 `256x256` 区域，即 `0..255`。
- 为 `RX` 可视化新增 `layered-sequence` 渲染逻辑，并与原有 `minimal-order` 逻辑并存；新逻辑会按精确序列显示 `R / D / RR / RD / DR / DD ...`，并支持你指定的纯绕射区域覆写规则。
- 明确固化 `layered-sequence` 下的一阶同阶优先级：在去除 `LoS` 区域后，`1阶 R` 会覆盖 `1阶 D`。
- 为 `rt2d.compute_rx_visibility(...)` 新增可选的 `layered_sequence_grid` 辅助结果，可供新渲染逻辑直接消费。
- 新增实验性 `energy-pruned-sequence` 渲染逻辑：在保留原始几何命中序列的前提下，额外用简化距离/交互代价对高阶弱路径做裁剪，便于和 `RadioMapSeer IRT2` 一类能量图做对比试验。
- 新增 `tests/test_coverage.py`，覆盖空场景 LoS、一阶反射、一阶绕射、二阶反射和四阶上限支持。

### Breaking
- 无兼容性影响。

### Migration
- 如需生成与 `png/buildings_complete` 叠加后的视距与视距遮挡边界标签图，调用 `python/examples/generate_radiomapseer_los_dataset.py`。
- 如需在现有提取结果中区分视距遮挡边界，读取新增的 `role="los_shadow"`。`r`n- 如需生成一阶反射版本，调用 `python/examples/generate_radiomapseer_physics_boundary_dataset.py --max-interactions 1 --output-dir ...\DPM1`。
- 如需输出 `RX` 栅格的最小可达阶数，调用 `rt2d.compute_rx_visibility(scene_id_or_scene, tx_ids=..., max_interactions=..., grid_step=1.0, ...)`。
- 如需命令行导出 `RX` 可见性结果，使用 `python .\\python\\examples\\extract_rx_visibility.py <scene_id> --max-interactions 4 --output .\\build\\rx_visibility.json`。
- 如需将 `RX` 可见性结果直接出图，使用 `python .\\python\\examples\\visualize_rx_visibility.py <scene_id> --max-interactions 4 --output .\\build\\rx_visibility.png`；如需按像素图导出，可追加 `--mode aligned --scale 4`。
- 如需使用新的序列分层渲染逻辑，追加 `--render-logic layered-sequence`；如需同时把该分层结果写入 JSON，使用 `extract_rx_visibility.py --include-layered-sequence-grid`。
- 如需试验更接近能量图的裁剪效果，可使用 `visualize_rx_visibility.py --render-logic energy-pruned-sequence`；当前这一模式仍是经验型近似，不代表最终物理模型。
- 对数字 `scene_id`，`extract_rx_visibility.py` 和 `visualize_rx_visibility.py` 现在默认使用 `0..255` 的整图区域；如果你只想导出局部场景包围盒，显式传入 `--bounds min_x min_y max_x max_y`。
- `visibility_order_grid` 采用行优先且 `y` 从大到小的布局，标签约定为：`-2=建筑内/边界`, `-1=不可达`, `0=LoS`, `1..4=最小交互阶数`。

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




## 2026-03-29

### Changed
- 新增 `python/examples/compare_radiomapseer_feature_maps.py`，可直接读取 `RadioMapSeer` 的 `buildings_complete / antenna / gain(DPM, IRT2)`，生成基于当前 `RX visibility` 序列图的特征边界图，并与 `gain` 的高频纹理做对齐比较。
- 新脚本支持复用 `compute_rx_visibility(...)` 的 `layered_sequence_grid`、`sequence_hit_grids` 和当前 `energy_pruned` 渲染逻辑，统一比较 `L / R / D / RR / RD / DR / DD` 等不同组合。
- 新脚本会输出样本级对比 PNG、缓存 JSON、以及汇总 `summary.json / summary.csv`，便于继续调参和批量筛选更贴近 `RadioMapSeer` 纹理的序列组合。
- 调整 `compare_radiomapseer_feature_maps.py`，改为将 `DPM` 与 `IRT2` 分开评分：`DPM` 只比较 `LoS / 一阶 R/D`，`IRT2` 只比较二阶 `RR / DD / RD / DR`。
- 调整 `compare_radiomapseer_feature_maps.py` 的 `Best-boundary` 生成方式，使用更平滑的 contour 线替代原先的硬栅格边界，减轻低分辨率下的锯齿误差。
- 调整 `compare_radiomapseer_feature_maps.py` 的评分逻辑，加入覆盖中弱纹理的多阈值 `F1` 与 1 像素容忍匹配，并新增 `energy_capture` 指标。
- 继续增强 `compare_radiomapseer_feature_maps.py` 的 `IRT2` 纹理提取，改为结合多尺度纹理响应与相对 `DPM` 的残差响应，并将 `IRT2` 可视化参考区域的默认阈值放宽到更能覆盖浅纹理的水平。
- 将 `compare_radiomapseer_feature_maps.py` 的 `DPM` 参考提取重新独立出来，改为“边界梯度为主、局部细纹理为辅”的轻量响应，避免被 `IRT2 residual` 的浅纹理增强逻辑带偏。
- 回退 `compare_radiomapseer_feature_maps.py` 中会额外制造伪边界的 contour 式 `Best-boundary` 生成逻辑，恢复为更接近第一版的离散邻接边界掩码；同时移除仅用于显示的平滑放大渲染，避免 `Best-boundary` 比参考纹理多出明显不存在的边界。
- 继续收简 `compare_radiomapseer_feature_maps.py`：移除 `IRT2 residual` 参与纹理提取的逻辑，改为让 `DPM` 与 `IRT2` 共用同一套多尺度纹理提取器。
- 将 `Best-boundary` 的生成方式改为“先求标签边界线段起点/终点，再按原图像素网格直接栅格化”，减少因掩码膨胀带来的伪边界，并让锯齿更贴近原始 `RadioMapSeer` 图像坐标。

### Breaking
- 无兼容性影响。

### Migration
- 如需比较当前 RT 特征边界与 `RadioMapSeer gain` 的纹理对齐效果，运行 `python .\\python\\examples\\compare_radiomapseer_feature_maps.py --samples 0:0 0:1 0:2 --output-dir .\\build\\radiomapseer_feature_compare`。
- 如需复用已计算的 `RX visibility` 结果，保持相同 `--output-dir` 并去掉 `--force-recompute`，脚本会直接读取缓存 JSON 重新评分和出图。
- 新版本会分别输出 `summary_dpm.csv` 和 `summary_irt2.csv`；如需看 `Best-overlay`，建议结合新的平滑 `Best-boundary` 与多阈值评分一起判断，不再只看最强纹理区域。
- 新版 `IRT2` 面板额外包含 `IRT2-residual`，用于辅助判断哪些浅纹理是二阶新增结构，而不是直接沿用 `DPM` 的纹理。
- `DPM` 与 `IRT2` 现在使用不同的参考纹理提取器；如果你在看 `Best-overlay`，不要再假设两者的红色参考区域来自同一套提取逻辑。
- 如果你需要和第一版 `Best-boundary` 的视觉风格保持一致，现在优先参考当前版本导出的离散边界图，而不是此前中间版本的平滑 contour 图。
- 当前版本的 `DPM` 对比已改为优先使用 `rt2d.extract_scene_boundaries(...)` 提供的真实 `LoS / Reflection` 线段，再直接栅格化到 `256x256` 图像网格；不再从 `coverage` 标签网格反推 `DPM Best-boundary`。
- 当前版本已移除 `IRT2 residual` 参与纹理提取和出图，`DPM` 与 `IRT2` 重新统一为同一套多尺度纹理提取逻辑。
- 当前版本已不再使用 `IRT2-residual` 参与纹理提取或出图；如需对比 `DPM` 与 `IRT2`，请直接查看统一口径下的 `DPM-texture` 与 `IRT2-texture`。
- 当前对齐图中 `0..255` 的世界坐标直接对应 `256x256` 像素网格，因此 `1` 个世界单位约等于 `1px`；但为了贴近 `RadioMapSeer` 真值图中的视觉线宽，RT 线段当前按 `2px` 直接栅格化显示。
- 将 `python/examples/compare_radiomapseer_feature_maps.py` 重写为只比较 `partition` 路线，不再混入 `tube / 真实边界线` 方案；当前脚本的核心问题改为“`RT` 栅格分区与 `DPM / IRT2 gain` 真值的对应好坏”。
- 当前 `compare_radiomapseer_feature_maps.py` 中，`DPM` 候选包含 `minimal_order1 / layered_reflection / layered_diffraction / layered_full`，`IRT2` 候选包含 `minimal_order2 / layered_rr / layered_dd / layered_rd / layered_dr / layered_order2_* / energy_pruned_order2`。
- 当前对齐图中的第三栏已改为 `Best-partition`，显示的是分区结果本身；`Best-overlay` 和 `Energy-overlay` 只用于查看分区边界和 `gain` 真值纹理是否对齐。
- 新增 `python/examples/evaluate_partition_batch.py`，支持按 `map` 范围和固定 `tx` 组批量评估 `partition` 候选在 `DPM / IRT2` 上的稳定性。
- 新增 `docs/radiomapseer_partition_strategy.md`，整理当前 `RadioMapSeer partition` 对齐策略、覆盖原则、候选排列组合和评分口径，便于后续直接理解当前实验设定。
- 为 `rt2d.compute_rx_visibility(...)` 新增可选的 `torch` 加速后端，当前保持原有算法和状态展开逻辑不变，只将 `state -> rx` reachability 判定改为基于 PyTorch 的分块批量计算。
- 新增 `acceleration_backend / torch_device / torch_state_chunk_size / torch_point_chunk_size / torch_edge_chunk_size` 参数，用于控制 `compute_rx_visibility(...)` 的新后端。
- 新增 `rt2d.build_rx_visibility_runtime(...)` 与 `rt2d.compute_rx_visibility_runtime(...)`，用于在加载 scene 后缓存 geometry、`256x256` 栅格、outdoor mask、GPU 张量和按 `tx / order / enable_reflection / enable_diffraction` 维度缓存的状态展开结果。
- 新增 `rt2d.warm_rx_visibility_runtime(...)`，可显式预热指定 `tx / order` 的 `state expansion` 与 `TorchStateBatch` 缓存，减少首次推理时的 CPU 准备开销。
- 进一步强化 `scene-wise runtime cache`：当前不仅缓存 Python 侧 `state expansion`，还缓存对应的 `TorchStateBatch`，并将 `LoS` 的单状态 batch 单独缓存为 runtime 内的可复用对象。
- 调整 `python/examples/compare_radiomapseer_feature_maps.py` 和 `python/examples/evaluate_partition_batch.py`，使其支持直接透传新的加速后端参数。
- 调整 `python/examples/evaluate_partition_batch.py`，改为按 `scene-wise runtime` 路径批量评估：每张 map 只构建一次 runtime，然后在同一 runtime 上处理固定 tx 组。
- 新增 `tests/test_coverage.py` 中的 CPU / torch 后端等价性测试，使用 `torch_device='cpu'` 校验两种后端在小场景上的输出一致。
- 新增 `tests/test_coverage.py` 中的 runtime builder / runtime warm 一致性测试，校验新引入的运行时缓存接口不会改变结果。
- 在 `rt2d_env` 实测 `scene 0, tx 0, max_interactions=1, 256x256` 下，`cpu` 约 `69.7s`，`torch-cpu` 约 `25.4s`，`torch-cuda` 约 `16.7s`。
- 在 `rt2d_env` 的 runtime 复用模式下，`scene 0` 的 GPU 预处理约 `12.4s`，随后单个 `tx 0`、`max_interactions=1` 的推理约 `2.7s ~ 4.3s`，可将“场景预处理”和“单次推理”拆开计时。
- 在 `rt2d_env` 的 runtime + state cache 路径下，`scene 0, tx 0, max_interactions=1` 的单次推理稳定在约 `2.5s`，同一 runtime 上重复推理时不再重复构建 `state expansion / TorchStateBatch`。
- 清理 `python/rt2d/coverage.py` 中旧的 `compute_rx_visibility(...)` 残留死代码，当前入口已经统一走 runtime 路径。
- 在 `feat/path-family-los-r-preview` 分支上新增最小 `LoS + R` 的 `path_family` 预览实现，新增 `python/rt2d/path_family/` 包，并可输出 `rx -> RayHit` 以及基于 `LoS` 优先、`R` 次之的分区图。
- 新增 `python/examples/preview_path_family_los_r.py`，可直接将最小 `LoS + R` path-family 分区结果叠加到 `RadioMapSeer` 的建筑物 PNG 上做人工检查。
- 新增 `tests/test_coverage.py` 中的最小 `path_family LoS + R` smoke test，验证新入口至少能在小场景上返回合法的 `blocked / unreachable / L / R` 分区标签。
