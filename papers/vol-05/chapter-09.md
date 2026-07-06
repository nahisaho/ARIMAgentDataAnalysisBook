# 第9章　多目的 BO を Skill 化する — qEHVI と Pareto front

> [!IMPORTANT]
> **本章の到達目標**
>
> 1. 多目的 BO の骨格（Pareto front、hypervolume、qEHVI）を Skill 契約の言葉で説明できる
> 2. **エージェントが勝手に目的関数をスカラー化しない契約** を書ける
> 3. ARIM 材料開発の (硬度, 密度) トレードオフ例を Skill 契約に落とせる
> 4. `objectives_spec` フィールド（多目的用に拡張）と qEHVI の `reference_point` を宣言できる
>
> **本章で扱わないこと**
>
> - 単目的 acquisition → **第7章**
> - 制約付き BO（cEI / cEHVI）→ **第10章**
> - 階層 GP を multi-objective に組み合わせるフル展開 → **第11章 + 第14章**
> - Batch × Multi-objective の統合 → **第12章 + 第14章 capstone**

---

## 9.1 多目的 BO とは何か

材料開発では 1 個の目的関数だけ最適化することは稀です。**硬度と密度**、**収率と選択性**、**性能とコスト**——複数目的を同時に最適化する必要があります。

多目的 BO の骨格：

- **目的関数**：$\mathbf{f}(x) = (f_1(x), f_2(x), \ldots, f_M(x))$、$M \geq 2$
- **Pareto front**：**どの目的関数もそれ以上改善できない点の集合**——1 目的を改善するには他の目的を悪化させる必要がある
- **Pareto dominance**：$\mathbf{y}_1$ が $\mathbf{y}_2$ を dominate するとは、**全目的で $\mathbf{y}_1 \geq \mathbf{y}_2$、かつ少なくとも 1 目的で strict >** を満たすこと（最大化の場合）

多目的 BO の目標は **Pareto front を効率的にサンプリングする**こと——単一の best を探すのではなく、**トレードオフ曲線全体を明らかにする**。

---

## 9.2 エージェントが勝手にスカラー化しない契約

これが本章の中心的 Agentic 問題です。

### 9.2.1 問題

エージェントが多目的問題を受け取ったとき、**簡単のため加重平均でスカラー化する**傾向があります：

$$
f_{\text{scalar}}(x) = w_1 f_1(x) + w_2 f_2(x) + \ldots + w_M f_M(x)
$$

これは：

- **重みの選択が Human の暗黙的な優先順位を上書き**する
- **凸な Pareto front しか探索できない**（凹部分は捨てられる）
- **Pareto front 全体の形が見えなくなる**

**Skill 契約でこれを禁止する必要があります**（第3章 §3.5 の 5 逸脱に**多目的スカラー化の勝手変更**として第15章で追加）。

### 9.2.2 契約要件（概要）

Skill 契約に以下を宣言（詳細スキーマは §9.4 で定義）：

- `objectives_spec.mode = "multi_objective"`
- `objectives_spec.scalarization.allowed`：既定 `false`（スカラー化禁止）
- `objectives_spec.scalarization.method`：`null | linear | tchebycheff | augmented_tchebycheff`
- `objectives_spec.reference.reference_point`：Human 承認付きで宣言

**`scalarization.allowed: false` が既定**——エージェントが勝手にスカラー化すれば Skill fatal。

### 9.2.3 スカラー化を許可する場面

- **明確な優先順位が Human から与えられ、事前承認された**ケース（固定重み `linear`）
- **凸 Pareto front で十分と分かっている** 単純ケース
- **qParEGO 戦略が Human 承認済み**（per-iteration に重みをランダムサンプル、§9.4.3 参照）

いずれも `human_approved_strategy_id` を伴い、Skill 自律での変更は fatal（§9.4.4）。

---

## 9.3 qEHVI — expected hypervolume improvement

### 9.3.1 Hypervolume の定義

$M$-目的関数値のセット $Y = \{\mathbf{y}_i\}$ に対して、**参照点** $\mathbf{r}$（Human が事前宣言）を使って（全目的を最大化に正規化した後）：

$$
\text{HV}(Y; \mathbf{r}) = \text{Volume}\!\left(\bigcup_{\mathbf{y} \in \text{Pareto}(Y),\ \mathbf{y} \succeq \mathbf{r}} [\mathbf{r}, \mathbf{y}]\right)
$$

—— Pareto front と参照点で囲まれた $M$-次元領域の体積。**Pareto front の "広さ" を測る**単一指標。

> [!NOTE]
> **参照点との関係**：$\mathbf{r}$ を全成分で dominate しない点（すなわちある成分で $y_j < r_j$）は hypervolume 寄与ゼロ。したがって参照点を過度に厳しく設定すると初期観測が全て寄与ゼロになり、qEHVI が縮退（勾配ゼロ）します。参照点は "実用最低ライン" とし、初期観測の少なくとも数点はこれを超えるように選ぶこと。

> [!IMPORTANT]
> **スケール依存性**：Hypervolume は各目的の単位に敏感（GPa と g/cm³ を混ぜると数値スケールの支配関係が生じる）。Skill 契約では以下のいずれかを明示：
> (a) 生単位で運用し、Human が「単位選択も含めて hypervolume 意味を承認」と宣言、
> (b) `objectives_spec.objectives[*].target_range` を使って affine 正規化した空間で hypervolume を計算（`hypervolume_scale: normalized_by_target_range`）。

### 9.3.2 qEHVI

qEHVI (batch-EHVI) は、**次に $q$ 個の候補を選んだとき、hypervolume がどれだけ改善するか**の期待値：

$$
\alpha_{\text{qEHVI}}(\mathbf{X}_q) = \mathbb{E}\bigl[\text{HV}(Y \cup \mathbf{f}(\mathbf{X}_q); \mathbf{r}) - \text{HV}(Y; \mathbf{r})\bigr]
$$

- **強み**：Pareto front 全体を意識、batch 対応、BoTorch 実装が成熟
- **弱み**：計算コスト大（$M$ が大きいと partitioning が指数的）、参照点の選び方に依存
- **推奨**：材料 BO 多目的で $M \leq 4$、$q \leq 8$

### 9.3.3 参照点の選び方

参照点 $\mathbf{r}$ は Pareto front を囲む「悪い方の境界」。**Skill 契約で Human が明示的に宣言**する必要があります：

```yaml
qehvi_config:
  reference_point: [7.0, -6.0]      # 全 objective を maximize に正規化した後（density は minimize → -density）
  reference_point_rationale: |
    hardness 7.0 GPa 未満 / density 6.0 g/cm3 超は「実用不可」と Human が判断
  partitioning: "non_dominated"     # BoTorch FastNondominatedPartitioning (qEHVI で使用)
  hypervolume_scale: "raw_units"    # enum: raw_units | normalized_by_target_range
```

> [!WARNING]
> **参照点の変更は run 中は fatal**。Human 承認付きの緩め (loosening) / 厳しく (tightening) はいずれも **新 run** として扱う。特に「観測を見た後で緩める」のは第3章 §3.5 逸脱 5（stop_condition 緩め）と同種の bias 源のため、policy で禁止するか比較不能とマークすること。

### 9.3.4 BoTorch 実装 — qEHVI と qNEHVI

多目的 EHVI には 2 種類あります。**physical 実験は観測ノイズがあるので通常 qNEHVI（`qLogNoisyExpectedHypervolumeImprovement`）を推奨**します。

**(A) qLogEHVI — 観測が実質ノイズレス、posterior mean で Pareto front を評価する場合**：

```python
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.models import ModelListGP

model = ModelListGP(*[gp_j for gp_j in per_objective_gps])
ref_point = torch.tensor([7.0, -6.0])           # maximize-normalized

# Y_obs: (N, M) — maximize-normalized
partitioning = FastNondominatedPartitioning(
    ref_point=ref_point,
    Y=Y_obs_maximize_normalized,
)

acq = qLogExpectedHypervolumeImprovement(
    model=model,
    ref_point=ref_point,
    partitioning=partitioning,
)
```

**(B) qLogNEHVI — ノイズあり観測、後述の noise-aware 定式（材料実験の既定推奨）**：

```python
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)

acq = qLogNoisyExpectedHypervolumeImprovement(
    model=model,
    ref_point=ref_point,
    X_baseline=X_obs,     # 観測入力（N, d）— partitioning は内部で MC 積分
    prune_baseline=True,
)
```

> [!NOTE]
> **direction 正規化**：BoTorch 内部は全目的を最大化と仮定。`objectives_spec.direction: minimize` の目的は、outcome transform で $y_{\text{model}} = -y_{\text{raw}}$ に変換する（第7章 §7.1 の符号規約と同じ運用）。参照点も maximize-normalized 空間で宣言する。

> [!IMPORTANT]
> **qEHVI vs qNEHVI の選択**：
> - **qLogEHVI**：観測ノイズがほぼ無い / 決定論的シミュレーション / posterior mean で Pareto を評価。`FastNondominatedPartitioning` を明示的に構築。
> - **qLogNEHVI**：観測ノイズあり（物理実験）。`X_baseline` を渡すのみで、Pareto partitioning は MC 積分内で扱われる。**材料 BO の既定推奨**。

---

## 9.4 Skill 契約 — `objectives_spec` と `acquisition_spec` の役割分担

第5章 §5.2 で導入した目的関数構造を **`objectives_spec` として拡張**します（`kernel_spec` と同様、既存フィールドの内部拡張であり新規トップレベル追加ではない）。**acquisition の選択は Ch7 の `acquisition_spec` に集約**し、`objectives_spec` は目的関数の宣言に専念します。

**役割分担**：

| フィールド | 責務 | 例 |
|---|---|---|
| `objectives_spec` | 目的関数の宣言、direction、target_range、scalarization ポリシー、surrogate 紐付け、reference point | 何を最適化するか |
| `acquisition_spec` | どの acquisition を使うか、hyperparameters（reference_point の参照） | どう最適化するか |

### 9.4.1 objectives_spec

```yaml
objectives_spec:
  version: "v0.2"
  mode: "multi_objective"            # enum: single_objective | multi_objective
  objectives:
    - name: "hardness"
      unit: "GPa"
      direction: "maximize"
      target_range: [8.0, 15.0]
      surrogate_ref: "gp_hardness"   # 目的関数ごとに surrogate を紐付け
    - name: "density"
      unit: "g/cm^3"
      direction: "minimize"
      target_range: [2.5, 5.0]
      surrogate_ref: "gp_density"
  scalarization:
    allowed: false                   # 既定は禁止（後述 §9.4.3 例外あり）
  reference:
    reference_point: [7.0, -6.0]     # direction を全て maximize に正規化した後
    reference_point_rationale: "hardness 7.0 未満 / density 6.0 超は実用不可"
    hypervolume_scale: "raw_units"   # enum: raw_units | normalized_by_target_range
```

### 9.4.2 acquisition_spec（Ch7 の 2 層構造を踏襲）

```yaml
acquisition_spec:
  name: "qLogNEHVI"                  # canonical enum: qLogEHVI | qLogNEHVI | qParEGO | qNParEGO
  implementation:
    library: "botorch"
    class: "qLogNoisyExpectedHypervolumeImprovement"
    required_runtime_inputs:
      - "model"
      - "ref_point"                  # objectives_spec.reference.reference_point から参照
      - "X_baseline"
      - "prune_baseline"
  hyperparameters:
    prune_baseline: true
    ref_point_source: "objectives_spec.reference.reference_point"  # 明示参照
```

> [!IMPORTANT]
> `acquisition_spec.name` は Ch7 §7.4 の契約通り **immutable within run**。多目的でも同じ——iteration 途中で qLogNEHVI → qParEGO への切替は `acquisition_switch_events` に append し、Human 承認必須（Ch7 §7.4）。

### 9.4.3 スカラー化を許可する場合 (qParEGO / qNParEGO)

`scalarization.allowed: true` の場合、**weight sampling 戦略まで契約に含める**必要があります（Skill が per-iteration に勝手な重みを引かないため）：

```yaml
scalarization:
  allowed: true
  method: "augmented_tchebycheff"    # enum: linear | tchebycheff | augmented_tchebycheff
  strategy: "qParEGO_random_per_iteration"
  weight_sampling:
    distribution: "simplex_uniform"  # 単体上一様
    seed: 20261120
    deterministic_rule: "hash(seed || iteration_index)"
  human_approved_strategy_id: "approval:HITL-YYYYMMDD-XXXX"
```

**運用ルール**：
- 重みは `deterministic_rule` から再現可能に生成、iteration ごとの実効重みは Group C の provenance（`effective_scalarization_weights`）に記録
- `human_approved_strategy_id` は「qParEGO 戦略そのもの」に対する承認 ID。個別 iteration 承認は不要（戦略承認で covered）
- `weight_sampling.seed` の変更は fatal（第5章 `sequential_seed_provenance` と同種契約）

### 9.4.4 スカラー化変更イベントのスキーマ

固定重み方式（`method: linear`）で Human が重みを切り替える場合、または戦略自体を切り替える場合：

```yaml
weights_change_events:
  - event_id: "evt-mo-20261120-001"
    effective_from_iteration: 5
    from_method: "linear"
    to_method: "linear"
    from_weights: [0.5, 0.5]
    to_weights: [0.7, 0.3]
    approval_id: "approval:HITL-20261120-A17"
    approval_scope: "weights_only"      # enum: weights_only | method_change | strategy_change
    changed_at: "2026-11-20T15:00:00Z"
    reason: "顧客要求変更により硬度重視に転換"
    previous_event_hash: "sha256:abc123..."
```

append-only、Skill 自律での変更は fatal。`previous_event_hash` で改ざん検知。

**目的関数間の相関**：本章では **各目的関数に独立 GP**（`ModelListGP`）を割り当てる。相関を surrogate で扱う場合は **第11章の multi-task GP** を使う。

---

## 9.5 (硬度, 密度) トレードオフ例

ARIM 材料開発で頻出：**軽量高硬度材料**の開発。硬度 $\uparrow$、密度 $\downarrow$ が両立しない。

- $x$：合成温度、圧力、時間、雰囲気、facility_id（第5章 §5.2 の例と同じ）
- $\mathbf{y} = (\text{hardness (GPa)}, \text{density (g/cm}^3))$
- Direction：$(\text{maximize}, \text{minimize})$
- 参照点：$(7.0\ \text{GPa},\ 6.0\ \text{g/cm}^3)$ — maximize-normalized で $(7.0, -6.0)$

観測が蓄積すると Pareto front が明らかになり、Human が「硬度 10 GPa 以上で最も軽い」候補を選ぶ。**エージェントが勝手にスカラー化して「硬度 12 GPa、密度 5.5 g/cm3」だけを提案するのは禁止**。

---

## 9.6 多目的 acquisition の判断表

```
[Q1] 目的数 M は？
  ├─ 2      → qEHVI（推奨）
  ├─ 3-4    → qEHVI（計算コスト増、参照点厳選）
  └─ 5 以上 → qParEGO / qNParEGO（スカラー化の系統的サンプリング、下記）

[Q2] Batch サイズ q は？
  ├─ 1      → qEHVI で q=1（BoTorch は q≥1 統一）
  └─ 2-8    → qEHVI（第12章 batch acquisition と併用）

[Q3] 制約あり？
  → 第10章 cEHVI へ

[Q4] スカラー化を Human が明示的に許可？
  ├─ YES → qParEGO / linear scalarization
  └─ NO  → qEHVI（既定）
```

### qParEGO / qNParEGO

- **やること**：ランダムに重み $\mathbf{w}$ をサンプリングして Chebyshev / augmented Chebyshev スカラー化、それに qEI / qNEI を適用
- **強み**：$M \geq 5$ で qEHVI が計算不能な場合、Pareto front の代表点をサンプリング
- **弱み**：スカラー化を含むので、`scalarization.allowed: true` かつ Human 承認済み設定でのみ使用
- **BoTorch**：`qLogNoisyExpectedImprovement` + `GenericMCObjective`（Chebyshev） の組み合わせ

> [!IMPORTANT]
> qParEGO の重みは **iteration ごとにランダム再サンプリング**される（Pareto front を broad にカバーする戦術）。これは §9.2 の「勝手にスカラー化しない」契約と一見矛盾するように見えますが、**§9.4.3 のスキーマ**（`strategy: qParEGO_random_per_iteration` + `human_approved_strategy_id` + `weight_sampling.seed`）で「Human が qParEGO 全体戦略を承認済み」を宣言していれば、内部の per-iteration 重みサンプリングは Skill 挙動として合法。実効重みは Group C provenance `effective_scalarization_weights` に記録し、`weight_sampling.seed + iteration_index` から再現可能にする。

---

## 9.7 Skill 契約チェックリスト

- [ ] `objectives_spec.mode` が `multi_objective` に設定されている
- [ ] 各 objective に `direction` / `unit` / `target_range` が明示されている
- [ ] `scalarization.allowed` が明示的（既定 false）
- [ ] `objectives_spec.reference.reference_point` が Human 承認付きで宣言されている
- [ ] `reference_point_rationale` が書かれている
- [ ] `hypervolume_scale` が明示（`raw_units` または `normalized_by_target_range`）
- [ ] 各目的関数に個別 surrogate（`surrogate_ref`）が紐付いている
- [ ] `acquisition_spec.name` が canonical enum（qLogEHVI / qLogNEHVI / qParEGO / qNParEGO）
- [ ] `acquisition_spec.hyperparameters.ref_point_source` が `objectives_spec.reference.reference_point` を参照
- [ ] Skill 実行中に scalarization / reference_point を暗黙変更する経路が塞がれている
- [ ] `weights_change_events`（許可時）が append-only、`previous_event_hash` で改ざん検知、`approval_id` 付き
- [ ] qParEGO 使用時、`weight_sampling.seed` および `effective_scalarization_weights` が provenance に記録される

---

## 9.8 章末演習

**問 1**：3 目的（硬度、密度、コスト）のケース。qEHVI で参照点をどう決めますか？ Human 承認のプロセスを書けますか？

**問 2**：エージェントが「iteration 5 で weights を [0.6, 0.4] に変えたい」と提案。契約上、Skill が返すべき応答は？

**問 3**：qParEGO を使う場合、`scalarization.allowed` はどう設定しますか？ Human 承認は何回必要ですか？

**問 4**：Pareto front の凹部分を探索するにはどの acquisition を選びますか？（qEHVI vs qParEGO の比較）

**問 5**：多目的で「hardness を absolutely 優先」の場合、多目的 BO と単目的 + constraint（第10章）のどちらを選ぶべきか、判断根拠を書けますか？

---

## 9.9 参考資料

### 本書内

- 第3章 §3.5：5 逸脱パターン（第15章で「スカラー化の勝手変更」を追加）
- 第5章 §5.2：objective 関連フィールドの初出
- 第7章：単目的 acquisition の判断表（本章と対応）、符号規約
- 第10章（planned）：cEHVI / constrained multi-objective
- 第11章（planned）：hierarchical / multi-task GP × multi-objective
- 第12章（planned）：Batch × multi-objective
- 第14章（planned）：capstone（batch × multi-obj × constrained 統合）
- 第15章（planned）：Pareto 疑似収束、スカラー化の勝手変更

### 外部参考

- Daulton et al., "Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization", NeurIPS 2020 — qEHVI 原論文
- Daulton et al., "Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement", NeurIPS 2021 — qNEHVI
- Knowles, J., "ParEGO: A Hybrid Algorithm With On-Line Landscape Approximation for Expensive Multiobjective Optimization Problems", IEEE TEC 2006 — ParEGO
- BoTorch multi-objective tutorials

> [!NOTE]
> 次章（第10章）では、制約付き BO と safe BO を扱います。「温度上限を超える提案は絶対に返さない」契約と、cEI / cEHVI / SafeOpt の Skill 化。
