# Path Family Grid-SBR Design

## 目标

在不破坏当前 `coverage.py` 主线的前提下，新开一条 `PathFamily / Grid-SBR` 路线，专门解决：

- 密集 `RX` 网格上的高效传播判定
- `TX -> ... -> RX` 完整作用点恢复
- 后续更自然的 GPU / tile / broad phase 优化

当前主线 `coverage.py` 的目标是：

- 判断某个 `RX` 是否被 `L / R / D / RR / RD / DR / DD` 命中

新路线的目标是：

- 先构建与 `RX` 无关的 `PathFamily`
- 再批量筛掉不可能命中的网格块/像素
- 最后只对命中的 `(family, rx)` 做精确作用点恢复

## 分支建议

- `feat/path-family-grid-sbr`

## 模块划分

建议新建一个 package：

- `python/rt2d/path_family/`

建议的模块拆分如下：

### 1. `types.py`

负责所有核心 dataclass / type alias：

- `PathFamily`
- `RayHit`
- `ReflectionInteractionRef`
- `DiffractionInteractionRef`
- `BeamRegion`
- `PathFamilyRuntime`

### 2. `runtime.py`

负责运行时缓存和 scene 级复用：

- `build_path_family_runtime(...)`
- `warm_path_family_runtime(...)`

缓存内容建议包括：

- `GeometryIndex`
- `edge / vertex / polygon` 索引
- `vertex spatial index`
- `rx tile layout`
- family expansion cache

### 3. `expand.py`

负责 family 扩张：

- `expand_root_families(...)`
- `expand_reflection_families(...)`
- `expand_diffraction_families(...)`
- `dedupe_families(...)`

这里的目标是：

- family expansion 只依赖几何和当前 family
- 不碰 `RX`

### 4. `broadphase.py`

负责从 family 找候选网格区域：

- `family_candidate_tiles(...)`
- `family_candidate_cells(...)`
- `beam_bbox_to_tiles(...)`

这层只做：

- beam / fan 几何命中
- tile / cell 粗筛

不做完整遮挡和精确作用点恢复。

### 5. `exact.py`

负责命中后的精确恢复：

- `family_reaches_rx(...)`
- `reconstruct_reflection_points(...)`
- `reconstruct_mixed_path_points(...)`
- `build_ray_hit(...)`

输入是：

- `family`
- `rx_point`

输出是：

- `RayHit`

### 6. `partition.py`

负责把 `family hits` 聚合成当前项目仍然需要的分区输出：

- `build_sequence_hit_grids_from_rays(...)`
- `build_minimal_order_grid_from_rays(...)`
- `build_partition_grid_from_rays(...)`

这层是新路线和当前 `coverage.py` 对齐的桥。

### 7. `api.py`

负责对外主入口：

- `compute_rx_rays_runtime(...)`
- `compute_rx_partition_runtime(...)`

其中：

- `compute_rx_rays_runtime(...)` 输出 `rx -> list[RayHit]`
- `compute_rx_partition_runtime(...)` 输出当前项目兼容的分区网格

## Dataclass 草图

### 1. `BeamRegion`

```python
@dataclass(frozen=True)
class BeamRegion:
    apex: tuple[float, float]
    left_dir: tuple[float, float] | None
    right_dir: tuple[float, float] | None
    bbox: tuple[float, float, float, float]
```

说明：

- 复用当前 `tube / fan` 的表达方式
- 不要一上来搞复杂多边形 beam
- MVP 先保持和现有 `state` 一致

### 2. `ReflectionInteractionRef`

```python
@dataclass(frozen=True)
class ReflectionInteractionRef:
    edge_id: int
    poly_id: int
    subsegment_t0: float
    subsegment_t1: float
    p0: tuple[float, float]
    p1: tuple[float, float]
```

说明：

- 反射 family 必须知道它依赖哪条边
- 以及在该边上有效的可见子段

### 3. `DiffractionInteractionRef`

```python
@dataclass(frozen=True)
class DiffractionInteractionRef:
    vertex_id: int
    poly_id: int
    point: tuple[float, float]
```

### 4. `PathFamily`

```python
@dataclass(frozen=True)
class PathFamily:
    family_id: int
    sequence: str
    order: int
    depth: int
    parent_family_id: int | None
    interaction_kind: str
    interaction_ref: ReflectionInteractionRef | DiffractionInteractionRef | None
    equiv_source: tuple[float, float]
    beam: BeamRegion
    exclude_edge_ids: tuple[int, ...]
```

说明：

- `equiv_source`：
  - 反射时是镜像源
  - 绕射时是绕射顶点
- `beam`：
  - 先只保留当前代码中已经验证过的 tube/fan 结构
- `parent_family_id`：
  - 后续恢复整条路径时要回溯

### 5. `RayHit`

```python
@dataclass(frozen=True)
class RayHit:
    family_id: int
    sequence: str
    rx_row: int
    rx_col: int
    rx_point: tuple[float, float]
    interaction_types: tuple[str, ...]
    interaction_points: tuple[tuple[float, float], ...]
    path_points: tuple[tuple[float, float], ...]
```

说明：

- `interaction_points`：
  - 只放真实的 `R / D` 点
- `path_points`：
  - 可以是完整 `TX -> ... -> RX`

## 第一阶段最小可落地范围

不要一开始就上 `RD / DR / DD / tile / pruning / GPU` 全家桶。

MVP 应该只做：

### 阶段 1A

- `LoS + R`
- 单 `scene`
- 单 `TX`
- CPU 实现
- family expansion 正确
- family -> candidate cell 正确
- reflection point reconstruction 正确

输出：

- `rx -> list[RayHit]`
- `sequence_hit_grids`
- `minimal_order_grid`

验证方式：

- 和当前 `coverage.py` 在 `enable_diffraction=False`、`max_interactions=1/2` 的结果对比
- 小场景下对 `reflection point` 做显式单测

### 阶段 1B

- 加入 `D`
- 但先只做纯 `D`
- 不碰 mixed `RD / DR`

### 阶段 1C

- 加入 `RR`
- 仍然不碰 mixed

### 阶段 1D

- 加入 `RD / DR / DD`
- 完成 mixed path reconstruction

## 实现顺序建议

### 第一步

复用现有几何：

- 继续使用 `build_geometry(...)`
- 继续使用 `compute_visible_subsegments(...)`
- 继续使用现有 front-face / critical vertex 逻辑

不要在第一版就重写全部底层几何。

### 第二步

先做 family expansion，不碰 RX：

- root -> `R`
- root -> `D`
- `R -> RR`
- `D -> DD`

先确认：

- family 数量
- family 去重
- beam 表达

### 第三步

做 broad phase：

- 先不必 tile 化到特别复杂
- MVP 可以先：
  - beam bbox -> candidate cells
  - 再 exact

### 第四步

做 exact reconstruction：

- `LoS`
- `R`
- `D`

最后才去搞：

- `RD / DR / DD / top-K / pruning`

## 和现有 `coverage.py` 共存的接入方式

核心原则：

- 不要直接重写 `coverage.py`
- 让它继续当 baseline 和 correctness oracle

建议做法：

### 1. 保留旧入口

- `compute_rx_visibility(...)`
- `compute_rx_visibility_runtime(...)`

它们继续服务当前分区主线。

### 2. 新入口独立命名

建议新增：

```python
compute_rx_rays_runtime(...)
compute_rx_partition_runtime_v2(...)
```

说明：

- `compute_rx_rays_runtime(...)`
  - 输出 `rx -> list[RayHit]`
- `compute_rx_partition_runtime_v2(...)`
  - 从 `RayHit` 聚合出分区图

### 3. compare 脚本支持 engine 选择

比如：

```text
--engine coverage
--engine path-family
```

这样可以在相同样本上直接 A/B。

### 4. 输出 schema 先兼容旧格式

至少先兼容：

- `visibility_order_grid`
- `sequence_hit_grids`
- `layered_sequence_grid`

这样现有分析脚本可以少改。

## 我建议的第一周范围

如果真的开分支实现，我建议第一周只做：

1. `types.py`
2. `runtime.py`
3. `expand.py`
   - 先只支持 `LoS + R`
4. `exact.py`
   - 先只恢复 reflection point
5. `api.py`
   - `compute_rx_rays_runtime(...)`
6. 一个最小 compare script
   - 同场景、同 TX 对比 `coverage.py` 与 `path_family`

暂时不做：

- mixed `RD / DR`
- aggressive pruning
- tile broad phase
- GPU

## 总结

当前建议不是替换 `coverage.py`，而是：

- 保留 `coverage.py` 当 baseline
- 新开 `path_family` package
- 先做 `LoS + R` MVP
- 跑通 `RayHit`
- 再逐步扩大到 `D / RR / RD / DR / DD`

这是风险最低、最容易逐步验证正确性的做法。
