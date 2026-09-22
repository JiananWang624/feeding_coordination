# Feeding Coordination：完整研究历史与技术交接

> 状态：截至 2026-09-22 的仓库审计。本文是项目级导航和研究记忆，不替代任何阶段的机器可读产物。文中的「冻结」指当前研究/复现契约，不等于物理系统已经验证。

## 0. 如何使用本交接文档

未来 Agent 或研究者应先读本文，再读所做阶段的 contract、`summary.json` / `manifest.json`，最后检查实现和测试。发生冲突时按「最新实际产物及 contract → 当前代码/测试 → 后期阶段文档 → README/ARCHITECTURE → 早期计划」判断。先核对工作树；不要把未提交展示产物当作冻结科学结果。不要因为早期文档使用了 `FROZEN` 一词，就把后来被实验推翻的假设当成现行结论。

本仓库根目录没有 `HANDOFF*.md`；`external/exact_sew/HANDOFF.md` 是**上游 Exact-SEW 工程交接**，不是本项目后期的人体协调研究结论。`README.md` 和 `ARCHITECTURE.md` 主要记录 Phase 0–3 边界；`docs/PHASE2_CONTRACT.md` 精确记录当时的 identity 虚拟工具实验，但不能覆盖 Robot R0 之后的 90° 工具语义。本文保留各历史版本，而不把项目改写成「从一开始就知道答案」。

未来任务开始前：

1. 确认 `git status` 和目标产物的存在性、hash、schema；保留用户改动。
2. 读本文件中相应阶段、该阶段 contract 与实际 `summary.json`。
3. 读当前实现及针对性测试；区分 OOF 评估模型和 full-data deployment 模型。
4. 除非出现可复现的新 blocker，不重开 Exact-SEW 数学、已停止的 PCRC/full-plan 主线或物理含义已被否定的 identity `P→U` 假设。

## 1. 项目一句话定义、记号与研究边界

本研究在固定的进食工具任务下，研究人类冗余协调（redundancy coordination）如何随工具运动演化，并通过冻结的 Exact-SEW 把工具轨迹（utensil/tool trajectory）`U(t)` 与 Stereo-SEW 冗余角 `ψ(t)` 映射为 Kinova Gen3 关节轨迹 `q(t)`。这里的「任务等价（task-equivalent）」指比较策略在**同一虚拟工具任务目标**下改变冗余选择；并不表示人与机器人肘部坐标应逐点重合。

项目**不是**重新发明 human-to-robot retargeting、SEW/IK 或以人体每个关节作为机器人拟合目标。后期增加上下文到工具轨迹生成：上下文 ProMP（Contextual ProMP）先预测 `U(t)`，StrongLocal 再预测冗余演化，Exact-SEW 实现 `q(t)`。ProMP、ridge 和 Exact-SEW 本身都不是本项目声称的新算法。

## 2. 为什么开始：最初背景与计划

采集层包含 feeding demonstrations、肩/肘/腕轨迹、上游派生的 hand、被跟踪的 fork、plate 位置和 mouth-related 派生点。单纯跟踪末端工具会留下 Gen3 七自由度手臂的冗余选择：相同工具任务下可能出现任意或突变的 whole-arm motion。作者此前已有 SEW/cSEW/Stereo-SEW 和 Gen3 Exact-SEW 工作，因此将其作为冻结机器人实现层，把论文问题转向**任务等价条件下的人类冗余协调**。

原计划的科学假设是：完整预先规划的工具轨迹可能含有「当前局部状态」以外的冗余信息，值得研究 PCRC / full-plan residual。其计划链是：

```text
human demonstrations
  → measured U(t), human ψ(t)
  → current/local → causal history → future summary → full plan/PCRC
  → frozen Exact-SEW → robot q(t)
```

当时这很合理：进食动作有朝 plate / recipient 的阶段性意图，未来终点与路径也许会提前影响人的肘部选择。之后 Phase 3、3.1 用完整 take 留出和 parent-bite 统计检验了它；该假设**没有得到可靠支持**，不是被忘记实施。原始 Phase 0 委托和初始提交 `4c83e54` 还明确把 tool generation、calibration、PCRC 留到后续，而不是在依赖冻结阶段开始。阶段次序由 Git 提交 `4c83e54 → 6f224f1 → 69bb78b → 299ea61 → acf20ec → dc0b1a2 → ecf45f4 → 48525a0 → 97efc8c → d78f2fc/f58feed → 4b70370 → 001886a → dd114c8 → cda26aa → c59b6b3` 重建；这些提交主要发生于 2026-09-15/16，不应把同一天内的时间顺序误说成长期采集时间线。

## 3. 表征、坐标和不能混淆的语义

### `U(t)`：被跟踪的叉子刚体，而非叉尖

最终 `U` 是 OptiTrack 名为 `fork` 的**tracked rigid-body origin and axes**。Motive 人体 landmark 原坐标 `L` 为 mm，项目 `B` 为 m：`p_B = 0.001 R_B←L p_L + t_B`。原始 recorder 的叉子位置本来就是 m，故其映射**没有** `0.001`：`p_B = R_B←L p_L + t_B`。姿态来自原始 xyzw quaternion，并按 `R_B←L R_L←U` 转换。它不是物理叉尖或食物接触点；fork-tip/food-point 标定仍缺失。

早期代码沿用 `Wrist_XYZ` 和所谓 hand/canonical orientation。数据审计显示 `Hand_XYZ` 实为 fork centre 沿局部负 Y 的 100 mm 上游派生点，并非独立解剖学 hand marker；raw wrist Euler 也是上游 fork quaternion 派生的 extrinsic xyz degrees。Phase 2 诊断进一步证明 raw fork frame 与旧 canonical hand orientation 相差固定约 90°。这属于**目标 frame 语义**，不应通过重写 Exact-SEW 或再施加内部 `R_robot_align` 来「修复」。Plate 仅有位置；`mouth_target_position` 是 transfer-end 派生点，G0/G1 称 `mouth_proxy`，既不是独立测得的嘴部 pose，也未经部署验证。

### `ψ(t)`：人的冗余协调，不是机器人关节或偏好分数

Phase 1/1.5 用测得的 shoulder/elbow/wrist 与冻结 Stereo-SEW 参考计算冗余角；几何有效性、近奇异、连续有效 run、wrapped/unwrapped 分别保存。肩部随时间运动被保留，没有把人体肩部人为固定。`ψ` 描述在给定工具任务下的肘部/手臂平面冗余状态；它不等于某个机器人关节角，不证明人类舒适、用户偏好或物理安全。当前学习目标是 `Δψ(t)=ψ(t)-ψ₀`，其中 `ψ₀` 作为重建偏置，而**不是** ridge 的线性输入特征。

### 机器人层：冻结的 `U, ψ → q`

Exact-SEW pinned commit 为 `52d74327d98a89b82012d311dde8d6dba066b843`，见 `DEPENDENCY_LOCK.json`。薄 adapter 输入 native Gen3 `base_link` 中的 m、SO(3)、rad 与 canonical aligned pinch orientation；一条轨迹/有效 run 一个新的 stateful solver，run 内保留内部 continuation。接口**没有外部 `q0` 参数**：早期概念图中的 `q0` 并非 Phase 0 adapter API。失败返回原状态和 `q=None`，不能在科学结果里用上一帧 q 冒充成功。依赖内部已经处理 `R_robot_align`，adapter 不再施加它。后续 RobotSmooth 的候选 ψ 搜索属于单独机器人策略，不是重写生产 Exact-SEW。验证见 `src/feeding_coordination/contracts/exact_sew.md` 和 `tests/test_exact_sew_adapter.py`。

## 4. 按时间顺序的开发与决策链

### Phase 0 — Exact-SEW dependency freeze

- **为什么做 / 问题：**先隔离已完成的 retargeting/IK 研究，问新项目能否稳定调用原实现，而不让新论文把它再当 novelty。
- **输入、方法、技术选择：**用 Git submodule 锁定上述 commit；薄 `ExactSewTrajectoryAdapter` 原样转发位置、旋转、ψ，建立依赖资产、API、状态、对齐、成功/失败 contract。每轨迹新 solver、轨迹内沿用状态，不加 fallback、平滑、clipping、retiming。
- **验证与结论：**adapter/direct parity、依赖身份、stateful lifecycle、失败传播与 no-fallback 合约测试；冻结的机器人实现层可复用。**冻结：**Exact-SEW revision/API、`R_robot_align` 所有权、Stereo-SEW/Gen3 几何，不重做 IK。**不在此阶段做：**人体数据、PCRC、工具生成或物理标定。
- **位置：**`DEPENDENCY_LOCK.json`、`src/feeding_coordination/adapters/exact_sew.py`、`src/feeding_coordination/contracts/exact_sew.md`、`tests/test_dependency_lock.py`、`tests/test_exact_sew_adapter.py`。

### Phase 1 — Human data / representation

- **为什么做 / 问题：**建立可审计的人体样本、坐标与缺失掩码，才能讨论人类 `ψ`；当时还需弄清工具位姿是否需要 hand-to-utensil calibration。
- **输入、方法、选择：**7 个完整 takes（0012/0013/0014/0015/0017/0019/0020），127 parent bites，分为 127 transfer 与 127 withdrawal，合计 254 segments、33,999 frames；记录 Motive frame/time、B-frame S/E/W、上游派生 hand/Euler、位置型 plate/mouth target；仅在同段内最多 3 帧且 0.03 s 的短洞插值，保留 raw/interpolated/long-missing 掩码。Stereo-SEW 参考从 frozen Gen3 native base 以 `Rx(+90°)` 映射至 B；`ψ` 在连续有效 run 内 unwrap，近奇异显式标记。
- **验证与数字：**32,234 完整 source rows、1,765 不完整；原始 wrist XYZ 缺 1,698 rows、elbow 缺 197；每段 52–332 帧，中位 126.5。数据审计指出只有 trial_0015 映射到 D1，其余六个 take 为 D0；recipient identity 未知，不能把 trial_0015 当成充分的跨示范者泛化证据。
- **发现、决定、冻结：**原本以为缺 H→U calibration 会使工具 pose 不可用；该判断在 1.5 被修正。Phase 1 冻结数据身份、S/E/W 与 `ψ` 计算、缺失数据及 B-frame 语义；不把 derived hand/mouth 改称独立测量。
- **位置：**`docs/DATASET_AUDIT.md`、`docs/PHASE1_CONTRACT.md`、`configs/phase1.json`、`src/feeding_coordination/phase1.py`、`outputs/phase1/manifest.json`。

### Phase 1.5 — Fork rigid-body recovery：第一次关键纠错

- **为什么做 / 问题：**Phase 1 的「无解剖学 hand→utensil 标定，所以工具 pose 无效」阻断后续；需要回查 recorder，而不是猜变换。
- **方法、验证：**发现 OptiTrack 直接跟踪名为 `fork` 的 rigid body；严格用 `motive_frame == natnet_frame_number` 做 exact join，不按最近时间硬配。原始 fork 位置是 m、xyzw quaternion 归一化后给姿态，并保留 tracking-valid、match/status、旋转及 derived-hand 一致性诊断。
- **数字：**33,999/33,999 行 exact matched，unmatched 0，tracking-invalid 18，valid tool pose 33,981/33,999（99.9471%）；orientation consistency 最大 `0.0001664 rad`（门限 0.001），derived hand discrepancy 最大 `1.718×10⁻⁵ m`（门限 0.0001）。254/254 段有 fork rigid-body 轨迹，但 **254/254 都没有 physical fork-tip calibration**。
- **决定与影响：**将 `U(t)` 正式定义为 tracked fork rigid-body frame；不需要再假装求一个 H→U；叉尖、真实嘴部和跨传感器同步精度仍未确认。随后 Phase 2/3 才能使用同一可信工具定义。
- **位置：**`src/feeding_coordination/phase15.py`、`docs/PHASE1_CONTRACT.md`、`outputs/phase1/manifest.json`、`tests/test_phase15.py`。

### Phase 2 — Exact-SEW oracle replay 与 35.8% 误导性结果

- **为什么做 / 问题：**在不学习策略的前提下，检验 measured `U,ψ` 能否经固定 Gen3 mounting 和冻结 solver 实现，建立 task-equivalent robot oracle。
- **方法、选择：**每 take 在最早有限 shoulder 处固定一次 robot base，`Rx(+90°)` 加 `[0,0.15,0.2] m` world offset；后续 shoulder 运动不 remount。将 B-frame fork pose 转成 base target，当时**虚拟** `P→U=I`；无效样本不插值、按 frame gap 分 run，各 run fresh stateful adapter；测得 `ψ` 给 oracle。
- **数字与验证：**33,999 rows 中 32,216 有效目标，288 valid runs；`SUCCESS_EXACT` 11,529/32,216＝35.7866%；`JOINT_LIMIT` 6,662、`NO_VALID_BRANCH` 14,025、`INPUT_NOT_SOLVED` 1,783。成功帧的 pose/ψ residual 很小，但这只证明被选目标求解精度。旧 trial_0012 replay 看似成功，却与新结果冲突，触发专项诊断。
- **当时结论及修正：**35.8% 是**identity 工具/当时 raw target** 下的数值回放，不是物理 feeding feasibility，更不能归咎为 Exact-SEW 必然失败；后续 R0 用正确语义显著改变结果。**保留为历史 baseline，不再当现行 `P→U`。**
- **位置：**`docs/PHASE2_CONTRACT.md`、`configs/phase2.json`、`src/feeding_coordination/phase2.py`、`outputs/phase2/summary.json`、`outputs/phase2/manifest.json`。

### Phase 2 diagnostic — A/B/C/D 隔离目标语义

- **为什么做：**不修改 solver，解释 trial_0012 旧 replay 与 Phase 2 的冲突。
- **四臂实验：**A＝旧 wrist + canonical orientation；B＝fork position + canonical orientation；C＝wrist position + raw fork orientation；D＝fork + raw fork orientation（即 Phase 2）。同一 4,327 common frames 上，A 成功 4,215（97.4116%），B 3,670（84.8163%），C 1,522（35.1745%），D 1,073（24.7978%）。A/D 在旧 single solver 与当前 per-run reset 生命周期之间成功率变化均为 0 percentage points；mounting 与 adapter parity 差为 0。
- **观察与归因：**canonical-to-raw fork orientation geodesic 均值 `1.5708003 rad`（约 90°）；旧 wrist-to-fork 距离均值 `0.1290 m`。分类是 **primary `ORIENTATION_TARGET_CHANGE`，secondary `POSITION_TARGET_CHANGE`**；排除 `MOUNTING_DIFFERENCE`、`SOLVER_LIFECYCLE_DIFFERENCE`、`ADAPTER_DIFFERENCE`。
- **路线影响：**不「修」Exact-SEW；先纠正虚拟工具坐标语义，再进行公平机器人策略比较。位置差仍提醒 physical P-to-U translation、fork-tip 尚未标定。此诊断限 trial_0012，不把 97.4% 外推至所有 takes。
- **位置：**`outputs/diagnostics/phase2_trial0012/summary.json`、`frame_results.csv`；Git `dc0b1a2`。

### Phase 3 — 原始 M1–M6 信息层级

- **为什么做 / 问题：**独立于 Phase 2 的机器人求解，检验局部、历史、未来摘要、完整工具 plan 是否给 `Δψ` 增加信息；这是原 PCRC 假设的直接前置检验。
- **输入、方法：**只用 Phase 1.5 的 measured U 与 measured ψ，绝不读 robot status 或 derived mouth。每段 first-valid-tool 局部 frame：`p_rel=R₀ᵀ(p−p₀)`，旋转向量 `log(R₀ᵀR)`；速度只从同一有效连续 run 的立即前驱获得。`ψ` run 由连续 `ψ_valid` 定义，目标 `ψ−ψ₀`；7 个完整 take 外层留一、内层在训练 take 选 alpha，parent bite 等权。
- **M1–M6 与数字（overall bite-balanced circular MAE / RMSE，rad）：**M1 pose `0.08585/0.11933`；M2 pose+velocity `0.08460/0.11740`；M3 当时 strong-local `0.10406/0.16524`；M4 history `0.10174/0.16319`；M5 future `0.10431/0.17423`；M6 full-plan `0.10004/0.13871`。原门限比较里 M4 相对**有缺陷的 M3** 似乎有改善（MAE 差 `−0.002323`，CI 全负），future 并无改善，M6 也无可靠胜出。
- **发现与影响：**M3 异常差，不能据此说「历史真的比正确 local 好」。为排除特征/ψ₀ 表征伪影，进入 Phase 3.1。M1–M6 保留为历史实验，不是最终生产模型。
- **位置：**`docs/PHASE3_CONTRACT.md`、`src/feeding_coordination/phase3.py`、`outputs/phase3/summary.json`、`fold_metrics.csv`、`bite_metrics.csv`。

### Phase 3.1 — StrongLocal sanity audit：PCRC 停止点

- **为什么做 / 问题：**拆开 M3 的局部特征，检验失败到底来自「局部信息不够」还是表征错误；重新对 history/future/full-plan 做公平比较。
- **方法：**L0＝M2；L1＝L0+phase；L2＝L1+raw `ψ₀`；L3＝L1+plate-relative/context mask；L4＝原 M3。嵌套选择在**七个 outer folds 均独立选 L3**，定义 `StrongLocal* = L3`。
- **关键证据：**L0/L1/L2/L3/L4 的 circular RMSE 分别 `0.11740/0.11734/0.17537/0.11389/0.16524 rad`。trial_0014 的 `ψ₀` 在 ±π 环形分支出现训练分布外值：约 19.87% 超训练范围，最大绝对标准分 `23.54`；加 raw `ψ₀` 令该 take MAE 增 `0.11439 rad`、RMSE 增 `0.23599 rad`。这不是「不能用初始冗余」，而是不能把其环形值作为普通线性 scalar feature 跨 branch 外推。
- **empirical kill criteria：**H* 相对 StrongLocal* MAE 差 `−0.001461 rad`、95% CI `[−0.003130,+0.000316]`；F* `+0.001258`、CI `[−0.000732,+0.003326]`；P* `+0.002868`、CI `[−0.001359,+0.007486]`。三者均不可靠改善，P* 点估计更差。**停止 PCRC / full-plan 主线；history/future 不再作为主方法。**这是否定当前数据与评估契约下的可靠增益，不是任何未来信息都不可能有价值的普遍定理。
- **位置：**`src/feeding_coordination/phase31.py`、`outputs/phase31/summary.json`、`local_ablation.csv`、`nested_selection.csv`。

### Phase 3.2 — 最终 human predictor freeze

- **为什么做 / 问题：**把被 3.1 支持的局部方法固定成可复现的 OOF 研究结果及一个单独的 full-data deployment artifact，并用简单 baseline 校准 claim。
- **模型：**19D `strong-local-l3-v1`：相对工具位置 3 + 相对姿态 rotvec 3 + 线速度 3 + 角速度 3 + normalized phase `s` 1 + transfer/withdrawal one-hot 2 + plate-relative position 3 + plate-valid mask 1。目标为 `Δψ=ψ_unwrapped−ψ₀`，重建 `ψ=ψ₀+predicted Δψ`；**`ψ₀` 不是 ML feature**。标准化加 bite 加权 ridge；OOF 仍每 take 留一。full-data deployment `alpha=1.0`，model replay error `0.0`，与 OOF 测试用途分开。
- **结果（bite-balanced circular MAE / RMSE，rad；127 bites、32,216 valid frames）：**hold `ψ₀` `0.130925/0.185194`；phase-only `0.114240/0.141682`；pose-only `0.085846/0.119330`；pose+velocity `0.084599/0.117400`；StrongLocal `0.086067/0.113894`。StrongLocal 比 hold 的 **RMSE 降 38.5%**，但 MAE 比 pose-only / pose+velocity 略高；不得写成它在所有指标与 B2/B3 上绝对最优。人的平均 `Δψ` range `0.27590 rad`，withdrawal 的冗余变化更大。
- **结论、冻结、放弃：**「`ψ` 有系统演化、局部 U 信息比只持有 `ψ₀` 有用」得到支持；「完整 plan 提供可靠新增信息」不支持且未重开。冻结特征顺序、目标、scaler/ridge/alpha、OOF split；放弃 raw `ψ₀` 线性特征及 PCRC 主线。
- **异质性警告：**take/session 间特征分布与错误并不相同（trial_0014 的环形分支是最强例子）；trial_0015 是唯一 D1 take，示范者效应与该 session 条件无法在当前七 take 中独立分离。不要把整体 bite-balanced 数字误说成每个 take 都有相同改进，也不要以逐帧显著性掩盖聚类结构。
- **位置：**`outputs/phase32/summary.json`、`outputs/phase32/final_model/model_contract.json`、`model.npz`、`src/feeding_coordination/phase32.py`。

### Robot R0 — task-equivalent 对照框架与虚拟工具语义修正

- **为什么做 / 问题：**Phase 2 的低成功率与 90° 诊断意味着不能直接讨论各冗余方法；先在**同一工具目标**下建立受控比较。
- **方法、选择：**仍用每 take 固定 provisional mounting，但虚拟 `P→U` 姿态采用 canonical 90° correction：`R_P = R_U @ R_input_align`；translation 仍 `[0,0,0] m`。这**不是物理标定**，也不等于再次使用内部 `R_robot_align`。在同一目标/同一起点比较 B0＝hold `ψ₀`、B1＝phase-only、B2＝pose-only、B3＝pose+velocity、StrongLocal、H*＝Phase 3.1 history OOF、HumanGT＝measured human ψ；后续 R1 才加入 RobotSmooth。HumanGT 是离线 oracle/reference，并非部署可用预测器。
- **验证与数字：**HumanGT frame success 从 Phase 2 identity 的 `35.7866%` 到当前 virtual orientation 的 `96.4055%`（描述性差 `+60.6189` 百分点）。B0 `96.0113%`、B2 `96.7625%`、B3 `96.7656%`、StrongLocal `96.7780%`。初始共同目标和 saved FK/task residual 校验，使冗余比较不被目标 U 差异污染。
- **决定与限制：**冻结**模拟研究的** 90° orientation convention 与基线集合；但零平移、mount、叉尖几何和真实可达性均 provisional。不要把 R0 数值称为实际机器人喂食可行率。
- **位置：**`src/feeding_coordination/robot_r0.py`、`configs/robot_r0.json`、`outputs/robot_r0/summary.json`、`manifest.json`。

### Robot R1 — 连续性诊断、RobotSmooth 与只读可视化

- **为什么做 / 问题：**即便每帧 `SUCCESS_EXACT`，`q(t)` 也可能突然跳到另一离散分支；需区分工具/ψ 跳变与 solver branch 行为，并建立只在连续区间定义的运动指标。
- **方法、选择：**相邻同一 source run、连续 Motive frame 且均成功时，最大 wrapped joint step `>0.5 rad` 记为当前帧 continuity violation；失败/数据洞/violation 处分割 clean intervals。near-π 事件定义为最大 step `≥0.95π`；71/71 near-π 事件伴随保存的 branch/search-branch 变化，支持离散 branch 变化是主要机制，而非把 U/ψ 本身误报为跳变。
- **RobotSmooth：**独立的 robot-centric causal heuristic，从相同已解初始状态出发，在上一 `ψ` 附近按固定 local offsets 尝试 Exact-SEW 候选；无解时用 32 点全局 ψ 网格恢复，按 wrapped q step、margin、ψ 变化等既定排序选一个。它保留任务 U，但可改变 ψ；不是人体学习模型，也不是物理平滑/retiming。
- **数字：**overall frame success StrongLocal/HumanGT/RobotSmooth 为 `96.7780% / 96.4055% / 95.3843%`；continuity violations `30 / 20 / 3`；RobotSmooth continuity-clean frame proportion `99.9902%`。它改善连续性但可能牺牲覆盖与 human consistency，不能冠以全局最好。只读 R1 viewer 读取 saved q、失败仅 viewer hold，展示 human 与 robot 几何、desired/actual U、连续性事件。
- **位置：**`src/feeding_coordination/robot_r1.py`、`robot_visualization.py`、`outputs/robot_r1/summary.json`、`strategy_metrics.csv`、`docs/ROBOT_VISUALIZATION.md`。

### Robot R2 — 机器人质量评估与进一步收缩 claim

- **为什么做 / 问题：**单看 ψ 误差与帧成功率，无法说明 whole-arm 形状、运动代价、关节余量；还需查碰撞/retiming 指标是否真的可定义。
- **方法与统计口径：**只读 R1 saved q/status/continuity；计算 wrapped q、frozen Gen3 SEW geometry 的 elbow 与 arm-plane、joint travel、clean-interval 速度/加速度/jerk、position limit margin、分别的 translational/rotational `pinch_site` Jacobian singular values。支持集 A＝source-valid，B＝两法共同 `SUCCESS_EXACT`，C＝在共同成功中按双方 violation 并集切 clean intervals；HumanGT 成对一致性采用 A/B/HumanGT 三者共同成功。聚合按 parent bite 平衡，bootstrap unit 是 parent bite，**帧不是独立统计单位**。
- **关键数字与结论：**StrongLocal−B2 在成对 bite-balanced 支持上的 ψ MAE `+0.001441 rad`、robot elbow `+0.000205 m`、travel `+0.194889 rad`；StrongLocal−B3 的 ψ MAE `+0.001825 rad`、elbow `+0.000304 m`、travel `−0.016809 rad`。这些混合或不利数值不支持「StrongLocal 全面可靠优于 B2/B3」。RobotSmooth−HumanGT 的 ψ MAE `+0.125568 rad`、elbow `+0.033320 m`，RMS velocity 差 `−0.05190 rad/s`、violation 差 `−0.06693`（bite-level mean）：这是人体一致性与机器人连续性/运动的 tradeoff。
- **不可报告的量：**`self_collision_audit.json` 明确为 `SELF_COLLISION_METRIC_UNAVAILABLE_MODEL_NOT_VALIDATED`：8 collision meshes、0 explicit pairs、0 excludes，零配置邻接 base/shoulder contact distance `−0.012045 m`。`retiming_readiness.json` 为 `RETIMING_LIMITS_NOT_YET_FROZEN`：XML 没有权威速度/加速度限值，不能把 q2/q4/q6 的位置限值冒充速度限值。R2 未做 collision、retiming、标定或安全论证。
- **位置：**`docs/ROBOT_R2.md`、`outputs/robot_r2/summary.json`、`feasibility_metrics.csv`、`human_consistency_metrics.csv`、`motion_metrics.csv`、`kinematic_metrics.csv`、`pairwise_bootstrap.csv`。

### Robot R3 — `ψ₀ ±0.25 rad` 初始状态稳健性

- **为什么做 / 问题：**最终学习的是相对初始状态的 `Δψ`；若 deployment 从不同初始冗余起步，策略排名是否稳定？不能默认 nominal `ψ₀` 结论直接外推。
- **方法、验证：**将各 frozen Δψ 演化锚到 `ψ₀−0.25`、nominal、`ψ₀+0.25 rad`；每 run/条件新 stateful Exact-SEW，RobotSmooth 从该条件已解第一帧继续。864 run-conditions 中 849 成功首帧；每条件 283/288＝98.2639% 首帧成功，5 个 `NO_VALID_BRANCH`；共同 first target/status/q parity，最大首帧 q 差 0。clean support、parent-bite bootstrap 延续 R2 语义。
- **结果与影响：**StrongLocal nominal/−0.25/+0.25 frame success 约 `94.9901%/95.3067%/94.4127%`，violations `56/21/19`；RobotSmooth 对应 success 均 `95.3843%`，violations `3/1/2`。RobotSmooth 连续性最低 violation、HumanGT/RobotSmooth tradeoff、B2/B3 与 StrongLocal 竞争性仍成立；但 velocity、feasibility、margin、Jacobian、q-similarity 与具体排序随 offset 改变，withdrawal 更 demanding。R3 并未解决部署初始 `ψ₀` 的选择。
- **位置：**`docs/ROBOT_R3.md`、`src/feeding_coordination/robot_r3.py`、`outputs/robot_r3/summary.json`、`strategy_metrics.csv`、`pairwise_bootstrap.csv`。

### G0 — Context → tool trajectory generation 的算法可行性

- **为什么在此时做：**先把人体冗余科学问题与机器人对照厘清，避免生成器误差掩盖 Phase 3 的信息层级结论；再补上完整系统的工具轨迹工程层。
- **输入与方法：**仅从 phase、初始 tracked fork pose、plate 位置、上游 derived mouth proxy 生成 101 点完整 SE(3) fork trajectory。轨迹在初始 fork frame 用相对平移 + `SO(3)` log 表示；transfer/withdrawal 分开。完整 take 留一，训练集内部统计 duration（取同 phase median）；不会用 held-out take 选 context scaler、ridge、retrieval 或 duration。
- **两个 baseline：**Retrieval 在 same-phase 训练集按标准化 6D plate/mouth-proxy context 最近邻取相对路径，transfer/withdrawal 对平移终点做 cubic smoothstep 适配，不发明终点姿态；Contextual ProMP 用 12 个 Gaussian/RBF bases × 6 个 relative SE(3) channels，以多输出 ridge 回归标准化 context→权重，仅输出 conditional mean，不采样 covariance。G0 不读 human ψ、q 或 robot outcome。
- **数字：**254 records × 2 方法＝508 次生成、数值生成成功率 100%；ProMP vs Retrieval held-out overall position RMS `0.067928 vs 0.086760 m`，orientation RMS `0.324355 vs 0.419162 rad`；按 parent bite bootstrap 的位置 RMS 差 `−0.018833 m`，95% CI `[−0.022388,−0.015386]`。支持当前数据/上下文下 ProMP 优于 retrieval，不证明真实 mouth sensing、fork-tip 位置或物理 feeding success。
- **决定：**以 Contextual ProMP 作为工程生成器进入 G1，保留 Retrieval 基线；不因规模小就引入 diffusion、Transformer、CVAE。G0 的 `mouth_proxy_is_independently_measured=false` 和 `deployment_context_validated=false` 绝不删。
- **位置：**`docs/GENERATOR_G0.md`、`configs/generator_g0.json`、`src/feeding_coordination/generator_g0.py`、`outputs/generator_g0/summary.json`、`representative_examples.json`。

### G1 — End-to-end generated pipeline

- **为什么做 / 问题：**G0 的工具 pose 精度优势能否沿 `U → Δψ → q` 传播？仅工具误差低不等于机器人完整轨迹成功。
- **输入、方法、控制：**246/254 G0 records 在其**初始 fork pose 的同一 Phase 1.5 行**有有效 human `ψ₀`，8 段 `NO_VALID_INITIAL_PSI` 不用后面的值偷换。七个 fold-specific StrongLocal 仅以另六 take 训练，其 measured-feature OOF replay parity 最大 `3.3823×10⁻¹³`（门限 `10⁻¹⁰`）；**不使用 full-data deployment 模型评估 held-out take**。generated `U` 预测 `Δψ` 并锚定首点，再经 frozen Exact-SEW 得 `q`；B0、StrongLocal、RobotSmooth 与 measured-U/StrongLocal reference 构成对照，G0 101 点 provisional time 不做物理 retiming。
- **主要结果，246 条 eligible records 的 record 平均 / 比率：**ProMP vs Retrieval：tool position RMS `0.068842 vs 0.087681 m`；generated-vs-measured-reference Δψ MAE `0.038297 vs 0.050763 rad`；StrongLocal frame success `97.907% vs 96.680%`；完整 pipeline success `224/246=91.057% vs 201/246=81.707%`；end-to-end wrapped q RMS deviation `0.2294 vs 0.3099 rad`。Parent-bite paired bootstrap 的 ProMP−Retrieval：tool RMS `−0.019071 m` CI `[−0.023026,−0.015133]`；ψ propagation `−0.012627 rad` CI `[−0.016699,−0.008832]`；complete trajectory success `+0.05118` CI `[+0.01969,+0.08268]`；q RMS `−0.08215 rad` CI `[−0.10824,−0.05963]`。注意 paired bite-balanced 差值**不必等于**两个单组百分率直接相减。
- **结论与负例：**工具生成优势确实传播至 ψ 与 robot pipeline；withdrawal 更困难（ProMP transfer/withdrawal ψ propagation MAE `0.01822/0.05711 rad`）。`trial_0012_bite_010_withdrawal` 两生成器都不是完整成功示范；`trial_0015_bite_019_transfer` 表明工具 U 数值有效仍可能下游机器人失败。generated-vs-measured-reference q/elbow 差是**完整链偏差**，混有 U、ψ、分支、joint-limit 与连续性影响，不是纯冗余预测误差。
- **未解决：**deployment initial redundancy selection、真实 mouth/context、物理叉尖、碰撞与 retiming；G1 是模拟算法可行性，不是安全或真实 feeding 成功。
- **位置：**`docs/GENERATOR_G1.md`、`configs/generator_g1.json`、`src/feeding_coordination/generator_g1.py`、`outputs/generator_g1/summary.json`、`pipeline_metrics.csv`、`robot_metrics.csv`、`representative_examples.json`。

### G1V — 只读演示与当前文件状态

- **为什么做 / 问题：**让导师能够看到 measured/generated U、human/reference/generated ψ 与两个或三个 stored-q 手臂的区别，以及 failure/continuity，而不是只看 CSV。
- **实现与验证：**`src/feeding_coordination/generator_g1_visual_comparison.py` 读取 G1 saved q，FK 仅用于 realized-U 展示/与保存结果的 parity；交互 viewer 与视频共享 replay frame，不调用 IK、不重训或改写 G1。展示 ψ 曲线/移动光标、tool error、wrapped q/elbow/arm-plane 差、plate、可选且明确标「derived mouth proxy」；failure 只 viewer hold，不能伪装成功。当前未提交工作树的说明/代码又加入 measured U + measured HumanGT q、measured-reference StrongLocal q、generated StrongLocal q 的三路 split 与同 take playlist；HumanGT 展示是从保存的 R1 q 在 G1 101 网格上**仅为显示**匹配，不是新的 G1 科学结果。
- **现状与冲突：**提交 `c59b6b3` 的 G1V 版本生成过五个比较 MP4 与截图，`outputs/generator_g1_visualization/manifest.json` 仍列出其路径、抽检 FK error 0；但截至本审计，工作树把那五个 MP4/六张截图标记为删除，当前目录只有新建的三路/playlist 视频与一张 smoke 截图。**不能声称旧五视频在当前磁盘可直接打开**，也不能为了写文档而恢复、重编码或删改用户展示工作。G1V 算法/展示工具已实现；完整会议产物清单须在提交/交付展示时重新核对。G1V 不改变前述科学结论。
- **位置：**`docs/GENERATOR_G1_VISUAL_COMPARISON.md`、`scripts/visualize_generator_g1_comparison.py`、`outputs/generator_g1_visualization/manifest.json`、`presets.json`、`videos/`。

## 5. 六次重大路线转折：原假设 → 验证 → 观察 → 决策

| 转折 | 原假设或早期做法 | 验证与实际观察 | 最终决策 |
| --- | --- | --- | --- |
| 1. Retargeting → coordination | 让机器人全手臂像人，或继续改 SEW | 旧项目已给出冻结 Exact-SEW；任务等价比较需要单独建模冗余 | SEW/IK 为依赖，研究问题改成人的 `ψ` 演化 |
| 2. Wrist/hand → fork `U` | 无 H→U calibration 就无工具轨迹 | Phase 1.5 查到直接 tracked fork；33,981 valid pose | `U` 定义为 fork rigid body；仍不冒称 fork tip |
| 3. identity `P→U` → canonical 90° | Phase 2 35.8% 似乎是机器人不可行 | A/B/C/D 找到约 90° orientation 主因，R0 HumanGT 96.4% | 修正**虚拟**目标 frame；物理平移仍待标定 |
| 4. full-plan/PCRC → local | 未来完整计划可显著增加 `ψ` 信息 | Phase 3.1 去除 raw `ψ₀` OOD 后，H*/F*/P* 对 L3 均无可靠增益 | 停止 PCRC，冻结 19D StrongLocal |
| 5. 「StrongLocal 最好」→ 受限 claim | 预测 ψ 更好应使机器人全面更好 | R2/R3 显示 B2/B3 竞争、初始 `ψ₀` 改变排序、HumanGT/RobotSmooth tradeoff | 分开报告 human consistency、feasibility、motion、robustness |
| 6. 固定 U 科学 → 生成 U 工程链 | 仅 measured U 不构成完整模拟流程 | G0/G1 LOTO：ProMP 相对 Retrieval 在 tool/ψ/q/complete success 有优势 | 加 Contextual ProMP 工程生成器；不把 ProMP 说成 novelty |

## 6. 已拒绝/停止的路线及原因

| 路线 | 当前处理 | 证据或边界 |
| --- | --- | --- |
| PCRC / full-plan residual 作主方法 | 停止 | Phase 3.1 P* 对 StrongLocal* 无可靠改善，点估计更差 |
| history、future summary 作主方法 | 停止 | 同一审计的 bootstrap CI 不支持可靠 gain |
| raw `ψ₀` 线性输入 | 拒绝 | trial_0014 ±π branch OOD；保留 `ψ₀` 作重建 offset |
| 新 SEW/IK、q-space optimizer | 排除 | Phase 0 pinned dependency 和论文问题边界 |
| diffusion/Transformer/CVAE | 当前无依据 | G0 小样本 baseline 已可对比；没有证据证明复杂度必要，非普遍禁止未来研究 |
| identity `P→U` 当物理真实 | 无效 | Phase 2 diagnostic 与 R0 虚拟修正；叉尖/平移仍无标定 |
| 自碰撞/安全结论 | 不可报告 | R2 模型 capability 未验证，邻接 mesh 零位接触 |
| 真实喂食成功、舒适、偏好 | 尚未建立 | 无可靠 mouth pose、recipient/真实执行、安全及人参与评价 |

## 7. 当前 source-of-truth pipeline 与策略角色

```text
固定工具科学实验：
Phase 1.5 measured fork U + measured ψ₀
    → {B0/B1/B2/B3, StrongLocal OOF, H*, HumanGT, RobotSmooth}
    → frozen Exact-SEW → stored Gen3 q
    → R1 continuity → R2 quality → R3 ψ₀-offset robustness

生成工具模拟链：
phase + plate + derived mouth proxy + initial tracked fork pose
    → G0 Contextual ProMP [Retrieval baseline]
    → generated U on 101 samples
    → G1 held-out-fold StrongLocal Δψ, anchored to measured initial ψ₀
       [B0/RobotSmooth controls]
    → frozen Exact-SEW + provisional virtual tool/base
    → Gen3 q → MuJoCo/read-only G1V presentation
```

Contextual ProMP＝engineering generator；StrongLocal＝冻结 human coordination predictor；RobotSmooth＝robot-centric causal engineering control；HumanGT＝离线 oracle/reference；Exact-SEW＝冻结 robot realization。G1 中 MeasuredUReference/StrongLocal 用 measured U 和**同一 fold model**，但**不用 human measured ψ 随时间的真值**。G1V 的 HumanGT 额外视觉轨迹来自 R1 stored q，不改变 G1 定量 reference 定义。

## 8. 关键数字速查：每行写明口径

| 阶段 / 指标 | 实际数值 | 口径与来源 |
| --- | ---: | --- |
| Phase 1 数据 | 7 takes；127 bites；254 segments；33,999 frames | `outputs/phase1/manifest.json` |
| Phase 1.5 fork pose | 33,981/33,999 valid（99.9471%） | tracked rigid-body，不是 fork tip；同上 |
| Phase 1.5 human ψ | 32,234/33,999 valid（94.8087%） | Stereo-SEW；同上 |
| Phase 2 identity oracle | 11,529/32,216 exact（35.7866%） | 历史虚拟 `P→U=I`；`outputs/phase2/summary.json` |
| Phase 3.2 hold vs StrongLocal | circular RMSE `0.185194 → 0.113894 rad`（−38.5%） | OOF、bite-balanced；`outputs/phase32/summary.json` |
| R0 HumanGT corrected virtual tool | 96.4055% frame success | **不是物理标定**；`outputs/robot_r0/summary.json` |
| R1 StrongLocal / HumanGT / RobotSmooth | violation `30 / 20 / 3` | `>0.5 rad`；`outputs/robot_r1/summary.json` |
| R2 StrongLocal−B2 | ψ MAE `+0.001441 rad`；elbow `+0.000205 m` | parent-bite balanced common support；`outputs/robot_r2/summary.json` |
| R3 三个初始 offset | 每条件首帧 `283/288=98.2639%` | `ψ₀−0.25/0/+0.25`；`outputs/robot_r3/summary.json` |
| G0 ProMP vs Retrieval | position RMS `0.067928 / 0.086760 m`；orientation `0.324355 / 0.419162 rad` | 254 held-out records；`outputs/generator_g0/summary.json` |
| G1 eligible | 246/254；8 个初始 ψ 无效排除 | `outputs/generator_g1/summary.json` |
| G1 StrongLocal full pipeline | ProMP `224/246=91.057%`；Retrieval `201/246=81.707%` | record rate，**不是** paired bite-bootstrap 差；`pipeline_metrics.csv` |
| G1 tool → ψ → q | ProMP/ Retrieval：tool RMS `0.068842/0.087681 m`；ψ propagation MAE `0.038297/0.050763 rad`；q RMS `0.2294/0.3099 rad` | eligible record 平均；同上 |

`summary.json`、CSV 的分组和支持集优先于此四舍五入速查表。尤其不要把 32,216 frames 当 32,216 个独立统计单位，也不要把 G0 的 254-record 误差与 G1 的 246-record 误差直接相减。完整轨迹成功率、逐帧成功率、数值生成成功率是三个不同概念。

## 9. 冻结项与 provisional 项要分别管理

**科学/复现契约冻结：**pinned Exact-SEW commit/API 与内部 `R_robot_align`；人体 Stereo-SEW 定义和 reference；tracked fork rigid-body `U` 定义；Phase 1.5 exact frame join 与状态掩码；19D `strong-local-l3-v1` 顺序/`Δψ` 目标/OOF 训练；PCRC 的当前 kill decision；RobotSmooth causal 策略定义；`0.5 rad` continuity threshold；模拟比较的 90° virtual `P→U` orientation；完整 take 外层留一、训练内调参、parent-bite bootstrap unit。不要因工程展示变化而静默改变这些定义。

**只是在模拟里暂定：**zero virtual `P→U` translation、每 take 的 Gen3 fixed mounting 和 `[0,0.15,0.2] m` offset、G0 duration、G1 initial human `ψ₀` 的已知来源、可视化中的 mouth proxy。这里的「冻结」只意味着**当前实验可复现**，不是物理校准完成。

**未解决或待测：**真实 mouth pose/感知、recipient identity 与泛化、叉尖/food-point 与 held-fork 物理变换、真实 Gen3 base/OptiTrack 外参、plate/recipient workspace、碰撞模型、权威速度/加速度限值、retiming、部署初始冗余选择、real Gen3、人体/环境 clearance、安全、舒适/偏好、真实 feeding outcome；跨传感器同步精度亦未建立。

## 10. Paper story 的演变及目前可辩护说法

**最初想讲：**完整 planned utensil trajectory 含有当前局部状态之外的人体冗余信息，因此用 PCRC/full-plan 预测 `ψ`。**数据实际显示：**原 M3 的差表现主要含 raw `ψ₀` 环形 OOD 伪影；改成 L3 后 history/future/full-plan 无可靠增益，P* 甚至点估计更差。StrongLocal 明显优于 hold `ψ₀`，但对 pose-only/pose+velocity 的优势取决于 RMSE/MAE，机器人层面也没有整体可靠压过 B2/B3。

**目前可辩护的叙事（仍需论文阶段严格审稿/复核）：**清晰的 fork-task 与人类 `ψ` 表征、完整 take 外推的透明负结果、同一虚拟工具目标下人体一致性与机器人连续性的 tradeoff、由 G0/G1 证实的模拟 generated-U 误差向 ψ/q/完整链的传播。价值来自表征、受控实验、系统链与负结果；**不应**把 ProMP、ridge、Stereo-SEW、Exact-SEW 本体、真实 feeding、安全或跨受试者普适性夸成新发现。

## 11. 当前停点、下一条主线与未来 Agent 禁止事项

**Current stopping point：**G1V 展示工具已实现，G0/G1、R0–R3 与 Phase 3.2 的科学产物冻结；但本审计时 G1V 旧 manifest 与工作区实际 MP4/截图不一致，展示交付前需另行核实。下一条**建议**主线是 *Physical Calibration Accuracy Validation*，不是复活 PCRC：

```text
P0  Gen3 base / OptiTrack / tracked-fork 外参标定
 → P1  tracked fork frame → physical fork-tip / food point 标定
 → P2  held-out calibration accuracy validation
 → P3  真实 workspace、plate 与 recipient placement
 → P4  无人体接触的 real Gen3 执行验证
 → P5  mannequin / 几何代理 feeding evaluation
 → Final  论文主张、统计支持与可复现产物冻结
```

这是一条未来工作**建议**，没有仓库实验产物可证明这些步骤已经完成。每一步在明确安全、权限和设备条件前都不能从当前模拟结果自动推出。

未来 Agent **不要**：重写 Exact-SEW / `R_robot_align`；把 PCRC 当当前方法；把 `mouth_proxy` 写成 measured mouth；把 virtual `P→U` 写成 physical calibration；用 full-data model 报既有 held-out 结果；隐藏 solver failure；把 robot elbow XYZ 与 human elbow XYZ 逐点相等当必要目标；把 trial_0015 当通用跨示范者证据；把零位模型接触当已验证 collision metric；声称 safety、comfort、preference 或 real feeding success。

## 12. 关键文件索引与冲突处理记录

| 需要回答的问题 | 先看哪里 |
| --- | --- |
| 依赖 commit、API、状态、对齐 | `DEPENDENCY_LOCK.json`；`src/feeding_coordination/contracts/exact_sew.md`；`src/feeding_coordination/adapters/exact_sew.py` |
| 数据身份、fork/hand/mouth 语义 | `docs/DATASET_AUDIT.md`；`docs/PHASE1_CONTRACT.md`；`outputs/phase1/manifest.json` |
| identity 工具的历史结果/诊断 | `docs/PHASE2_CONTRACT.md`；`outputs/phase2/summary.json`；`outputs/diagnostics/phase2_trial0012/summary.json` |
| M1–M6 与最终为何弃 PCRC | `docs/PHASE3_CONTRACT.md`；`outputs/phase3/summary.json`；`outputs/phase31/summary.json` |
| 最终人类模型和部署参数 | `outputs/phase32/summary.json`；`outputs/phase32/final_model/model_contract.json`；`outputs/phase32/final_model/model.npz` |
| 当前模拟虚拟工具和 R0/R1 | `outputs/robot_r0/summary.json`；`outputs/robot_r1/summary.json`；`src/feeding_coordination/robot_r1.py` |
| R2 质量与不可用能力 | `docs/ROBOT_R2.md`；`outputs/robot_r2/summary.json`；`outputs/robot_r2/self_collision_audit.json`；`outputs/robot_r2/retiming_readiness.json` |
| R3 初始状态稳健性 | `docs/ROBOT_R3.md`；`outputs/robot_r3/summary.json` |
| G0/G1 生成、统计、代表样本 | `docs/GENERATOR_G0.md`；`docs/GENERATOR_G1.md`；`outputs/generator_g0/summary.json`；`outputs/generator_g1/summary.json`；`outputs/generator_g1/pipeline_metrics.csv`；`outputs/generator_g1/representative_examples.json` |
| G1V 当前展示入口/产物 | `docs/GENERATOR_G1_VISUAL_COMPARISON.md`；`scripts/visualize_generator_g1_comparison.py`；`outputs/generator_g1_visualization/manifest.json` 与实际 `videos/` |

**已发现并裁决的内部描述冲突：**

1. `ARCHITECTURE.md`/Phase 2 contract 的 `P→U=I` 与 Robot R0 之后的 90° orientation 不冲突于时间线，但作为「当前系统描述」已过时；以 R0 summary/manifest 与当前代码为准。
2. 早期 Phase 3 的 M3/history 结论被 Phase 3.1 特征审计修正；以 3.1/3.2 summary 和最终 model contract 为准。
3. 根目录 README 未覆盖 R0–G1V，不代表这些阶段不存在；以提交、各阶段产物与当前代码为准。
4. G1V manifest 列出历史生成的五视频/六截图，但当前未提交工作树标记它们删除，另有新三路/playlist 文件；manifest **不是当前磁盘文件存在性的证明**。本文不对用户工作树做恢复或清理。
