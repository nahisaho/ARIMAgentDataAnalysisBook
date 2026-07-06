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

### 9.2.2 契約要件

```yaml
objectives_spec:
  version: "v0.2"
  mode: "multi_objective"          # enum: single_objective | multi_objective
  objectives:
    - name: "hardness"
      unit: "GPa"
      direction: "maximize"
      target_range: [8.0, 15.0]    # Human が事前に宣言する意味のある範囲
    - name: "density"
      unit: "g/cm^3"
      direction: "minimize"
      target_range: [2.5, 5.0]
  scalarization:
    allowed: false                 # 既定は禁止
    method: null                   # enum: null | linear | tchebycheff | augmented_tchebycheff
    weights: null                  # allowed=true のとき Human が明示的に設定
    weights_change_policy: "immutable_within_run"  # enum: immutable_within_run | human_approved_switch
```

**`scalarization.allowed: false` が既定**——エージェントが勝手にスカラー化すれば Skill fatal。

### 9.2.3 スカラー化を許可する場面

- **明確な優先順位が Human から与えられ、事前承認された**ケース
- **凸 Pareto front で十分と分かっている** 単純ケース
- **Human 承認付きの weights_change_event**（下記）

Human が意図的に切り替える場合：

```yaml
weights_change_events:
  - iteration: 5
    from_weights: null
    to_weights: [0.7, 0.3]         # hardness を優先
    approver_id: "human:staff_id_XXXX"
    changed_at: "2026-11-20T15:00:00Z"
    reason: "顧客要求変更により硬度重視に転換"
```

append-only、Skill 自律での変更は fatal。

---

## 9.3 qEHVI — expected hypervolume improvement

### 9.3.1 Hypervolume の定義

$M$-目的関数値のセット $Y = \{\mathbf{y}_i\}$ に対して、**参照点** $\mathbf{r}$（Human が事前宣言）を使って（全目的を最大化に正規化した後）：

$$
\text{HV}(Y; \mathbf{r}) = \text{Volume}\!\left(\bigcup_{\mathbf{y} \in \text{Pareto}(Y)} [\mathbf{r}, \mathbf{y}]\right)
$$

—— Pareto front と参照点で囲まれた $M$-次元領域の体積。**Pareto front の "広さ" を測る**単一指標。

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
  partitioning: "non_dominated"     # BoTorch NondominatedPartitioning
```

> [!WARNING]
> 参照点を後から緩めるのは **第3章 §3.5 逸脱 5（stop_condition 緩め）と同種の逸脱**——Skill fatal。参照点を厳しくする（Pareto front の要求を上げる）のも context 変更なので新 run 扱い。

### 9.3.4 BoTorch 実装

```python
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.models import ModelListGP

# model_list: 各目的関数に SingleTaskGP を fit した ModelList
model = ModelListGP(*[gp_j for gp_j in per_objective_gps])

# reference point (M,) — 全 objective を maximize に正規化した空間で
ref_point = torch.tensor([7.0, -6.0])

# Y_obs: (N, M) — 同じく maximize-normalized
partitioning = FastNondominatedPartitioning(
    ref_point=ref_point,
    Y=Y_obs_maximize_normalized,
)

acq = qLogNoisyExpectedHypervolumeImprovement(
    model=model,
    ref_point=ref_point,
    X_baseline=X_obs,
    prune_baseline=True,
)
```

> [!NOTE]
> **direction 正規化**：BoTorch 内部は全目的を最大化と仮定。`objectives_spec.direction: minimize` の目的は、outcome transform で $y_{\text{model}} = -y_{\text{raw}}$ に変換する（第7章 §7.1 の符号規約と同じ運用）。参照点も maximize-normalized 空間で宣言する。

---

## 9.4 objectives_spec の Skill 契約

第5章 §5.2 で導入した目的関数構造を **`objectives_spec` として拡張**（`kernel_spec` と同様、既存フィールドの内部拡張であり新規トップレベル追加ではない）：

```yaml
objectives_spec:
  version: "v0.2"
  mode: "multi_objective"
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
    allowed: false
  qehvi_config:
    reference_point: [7.0, -6.0]     # direction を全て maximize に正規化した後
    reference_point_rationale: "..."
    partitioning: "non_dominated"
```

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
> qParEGO の重みは **iteration ごとにランダム再サンプリング**される（Pareto front を broad にカバーする戦術）。これは §9.2 の「勝手にスカラー化しない」契約と一見矛盾するように見えますが、**「Human が qParEGO 全体戦略を承認済み」**であれば、内部の per-iteration 重みサンプリングは Skill 挙動として合法。ただし `qparego_weight_seed`（重みサンプリングの seed）を provenance に記録し、再現可能にする。

---

## 9.7 Skill 契約チェックリスト

- [ ] `objectives_spec.mode` が `multi_objective` に設定されている
- [ ] 各 objective に `direction` / `unit` / `target_range` が明示されている
- [ ] `scalarization.allowed` が明示的（既定 false）
- [ ] `qehvi_config.reference_point` が Human 承認付きで宣言されている
- [ ] `reference_point_rationale` が書かれている
- [ ] 各目的関数に個別 surrogate（`surrogate_ref`）が紐付いている
- [ ] Skill 実行中に scalarization / reference_point を暗黙変更する経路が塞がれている
- [ ] `weights_change_events`（許可時）が append-only、Human 承認付き
- [ ] qParEGO 使用時、`qparego_weight_seed` が provenance に記録される

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
