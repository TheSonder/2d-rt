# RadioMapSeer Partition Strategy

## 目标

当前项目在 `RadioMapSeer` 上主要比较的是：

- 输入：建筑物 `polygon json`、发射机 `antenna json`
- 过程：基于 `rt2d.compute_rx_visibility(...)` 计算每个栅格 `RX` 点可被哪些传播机制命中
- 输出：`partition` 分区图
- 真值：`RadioMapSeer/gain/DPM` 与 `RadioMapSeer/gain/IRT2`

注意：

- 当前这份策略文件只描述 `partition` 路线
- 不再包含 `tube / 真实边界线` 的比较逻辑

## 真值理解

`RadioMapSeer/gain/*` 是真值图。

- 建筑物区域是纯黑
- 未覆盖区域也可能是纯黑
- 因此必须结合建筑物掩码一起解释 `gain`
- 当前比较时，建筑物区域会被单独剔除，不参与纹理评分

当前主要关注：

- `DPM`
- `IRT2`

## 覆盖原则

当前默认物理覆盖原则是累计覆盖，而不是纯单阶覆盖：

- `DPM`：`L ∪ R ∪ D`
- `IRT2`：`L ∪ R ∪ D ∪ RR ∪ RD ∪ DR ∪ DD`

这里分成两层：

1. Reachability
   判断某个点是否能被某种机制命中
2. Display Label
   当一个点同时被多种机制命中时，最终渲染成哪个标签

## 当前分区标签

当前 `partition` 渲染只使用以下抽象标签：

- `blocked`
- `unreachable`
- `L`
- `I1`
- `I2`

其中：

- `L`：LoS
- `I1`：一阶交互区
- `I2`：二阶交互区

## 当前比较模式

### 1. Minimal

按最小阶渲染：

- `DPM minimal_order1`
  - `L -> L`
  - `1阶 -> I1`
  - 其他 -> `unreachable`
- `IRT2 minimal_order2`
  - `L -> L`
  - `1阶 -> I1`
  - `2阶 -> I2`
  - 其他 -> `unreachable`

### 2. Layered

按精确机制序列筛选，再映射到抽象标签：

- `DPM layered_reflection`
  - `L, R`
- `DPM layered_diffraction`
  - `L, D`
- `DPM layered_full`
  - `L, R, D`

- `IRT2 layered_rr`
  - `RR`
- `IRT2 layered_dd`
  - `DD`
- `IRT2 layered_rd`
  - `RD`
- `IRT2 layered_dr`
  - `DR`
- `IRT2 layered_order2_no_rr`
  - `DD, RD, DR`
- `IRT2 layered_order2_no_dd`
  - `RR, RD, DR`
- `IRT2 layered_order2_full`
  - `RR, DD, RD, DR`

### 3. Energy

- `IRT2 energy_pruned_order2`
  - 先用 `sequence_hit_grids`
  - 再经过 `build_energy_pruned_sequence_render_grid(...)`
  - 最终再映射成 `I2`

## 当前真值纹理提取

当前 `DPM` 与 `IRT2` 使用统一的纹理提取器：

- 局部梯度
- 多尺度平滑差分
- 大尺度 front 响应
- 基于 TX 的径向响应

当前纹理提取的目的不是恢复真实物理能量，而是构造一个更适合和 `partition` 边界做对齐的真值纹理参考图。

## 当前评分

对每个候选分区图，先提取其分区边界，再和真值纹理图比较。

当前评分包含：

- 多阈值 `F1`
  - `60 / 70 / 80 / 90 / 95 / 97` 百分位
- `edge_lift`
- `energy_capture`

当前主排序指标：

1. `f1_mean`
2. `energy_capture`
3. `edge_lift`

## 当前已知趋势

基于当前阶段的小样本统计：

- `DPM` 往往更偏向 `layered_diffraction`
- `IRT2` 在 `layered_rr / layered_rd / layered_dr` 之间竞争
- `energy_pruned_order2` 目前通常不是最优

这只是当前趋势，不代表最终结论。

## 当前批量评估脚本

- 单样本比较：
  - `python/examples/compare_radiomapseer_feature_maps.py`
- 批量评估：
  - `python/examples/evaluate_partition_batch.py`

当前推荐先用批量评估查看：

- `DPM` 是否稳定由 `layered_diffraction` 获胜
- `IRT2` 是否稳定由 `layered_rr / layered_rd / layered_dr` 中某一个获胜
