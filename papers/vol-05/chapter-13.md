# 第13章　Active Learning と逐次実験計画 — BO 以外の逐次戦略

> [!IMPORTANT]
> **本章の到達目標**
>
> 1. Active learning (AL) の骨格（uncertainty sampling / query by committee / expected model change）を Skill 契約で説明できる
> 2. **BO vs Active Learning の判断分岐**（"最適化" と "学習" の目的分離）を書ける
> 3. Emukit / modAL の Skill 化テンプレートを持てる
> 4. **エージェントが目的を "最適化" と誤解しない契約**（`objective_mode` enum、`stop_condition_for_learning`）を設計できる
> 5. AL と BO を組み合わせる場合の切替 gate（`objective_mode_switch_events`）を書ける
>
> **本章で扱わないこと**
>
> - BO の骨格（surrogate / acquisition / stop condition） → **第6〜10章**
> - Batch BO → **第12章**（batch AL でも似た多様性契約を再利用）
> - Advanced Capstone → **第14章**（BO 側で AL を活用）
> - 失敗パターン全般 → **第15章**

---

## 13.1 なぜ Active Learning か — BO と何が違うのか

BO は **「目的関数を最適化」** する逐次戦略。Active Learning は **「モデルを効率よく学習」** する逐次戦略。同じ「次の実験点を選ぶ」でも、**目的が違えば選ぶべき点も違います**。

- **BO の目的**：$\arg\min_x f(x)$（あるいは $\arg\max$）を見つける。**exploit と explore のバランス**
- **AL の目的**：モデル $\hat{f}$ の誤差 $\|\hat{f} - f\|$ を減らす。**explore 主体**、exploit なし
- **共通点**：GP や BNN などの surrogate を使い、acquisition-like な選択規則で次の点を選ぶ
- **決定的違い**：**AL は「良い点を選ばない」**。むしろ「モデルが自信のない点」＝性能の悪そうな点を積極的に選ぶ

材料 R&D の例：

- **BO**：「最高硬度の合金組成を見つけたい」→ 有望領域を集中探索
- **AL**：「合金組成 → 硬度の予測モデルを作りたい」→ 探索空間全体をカバー、後で別の目的にも使える
- **混在**：「まず全体のマップを作り、その後で最適点を絞る」→ AL 期間 → BO 期間の切替

> [!WARNING]
> **エージェントの typical 失敗**：ユーザが「モデルを作りたい」と言っているのに、エージェントが「じゃあ最高値を探しますね」と BO を回してしまう。これは目的の取り違えで、**AL で得られるべき均等サンプルが得られない**。Skill 契約で `objective_mode` を明示する必要がある。

---

## 13.2 Active Learning の acquisition 骨格

### 13.2.1 Uncertainty Sampling — GP variance / BNN entropy

回帰の場合、predictive variance が大きい点を選ぶ：

$$
x^* = \arg\max_x \sigma^2(x)
$$

分類の場合、entropy が大きい点を選ぶ：

$$
x^* = \arg\max_x H[p(y|x)] = -\sum_c p(y=c|x)\log p(y=c|x)
$$

**契約**（BO の acquisition_spec を AL 用に転用）：

```yaml
active_learning_acquisition_spec:
  version: "v0.1"
  name: "uncertainty_sampling"                # enum: uncertainty_sampling | query_by_committee | expected_model_change | bald | variance_reduction | max_entropy
  target: "predictive_variance"               # enum: predictive_variance | class_entropy | posterior_entropy | mutual_information
  task_type: "regression"                     # enum: regression | classification | ranking
  batch_size: 4                                # AL でも batch 提案あり
  diversity_policy_ref: "vol-05:ch12:batch_diversity_policy_v0_2"  # 多様性は Ch12 と共有
```

### 13.2.2 Query by Committee (QBC)

複数モデル（committee）の予測が **食い違う点** を選ぶ：

$$
x^* = \arg\max_x \mathrm{Disagreement}\bigl(\{\hat{f}_k(x)\}_{k=1}^K\bigr)
$$

- 回帰：committee 予測の分散
- 分類：Vote entropy / KL divergence

**Skill 契約**：committee を明示：

```yaml
committee_spec:
  version: "v0.1"
  members:
    - {model_id: "gp_matern52_seed1", weight: 1.0}
    - {model_id: "gp_matern32_seed2", weight: 1.0}
    - {model_id: "random_forest_seed3", weight: 1.0}
  disagreement_metric: "variance"             # enum: variance | vote_entropy | kl_divergence | jensen_shannon
  refresh_policy: "every_iteration"           # enum: every_iteration | fixed_after_init | on_drift
```

### 13.2.3 Expected Model Change (EMC) / BALD

**EMC**：候補点 $x$ を追加したら、モデルパラメタがどれだけ変わるかを期待値評価

**BALD (Bayesian Active Learning by Disagreement)**：モデルパラメタ $\theta$ と予測 $y$ の相互情報量：

$$
\mathrm{BALD}(x) = H[y|x, \mathcal{D}] - \mathbb{E}_{\theta|\mathcal{D}}\bigl[H[y|x,\theta]\bigr]
$$

BNN や deep ensembles で扱いやすい。Ch8 の DKL / BNN と組合わせる。

---

## 13.3 BO vs Active Learning の判断分岐

**契約**：新規プロジェクト開始時、Skill は Human に「目的モード」を必ず宣言させる：

```yaml
objective_mode:
  version: "v0.1"
  mode: "optimization"                        # enum: optimization | learning | hybrid_learn_then_optimize | hybrid_multi_objective_and_learn
  declared_by: "human:project_lead_XXXX"
  declared_at: "2026-12-01T09:00:00Z"
  approval_id: "approval:HITL-20261201-OM01"
  rationale: "最高硬度の合金組成を探索するため BO を選択"
  # hybrid の場合、切替 gate を明示
  switch_gate:                                # optional、hybrid_* の場合のみ
    from: "learning"
    to: "optimization"
    condition: "surrogate_generalization_error < 0.15 AND n_observations >= 30"
    approval_required: true
  budget_split:                               # optional、hybrid_* の場合のみ
    learning_budget: 30                       # 実験数
    optimization_budget: 20
    total_budget: 50
```

### 13.3.1 判断表

| 状況 | 推奨モード | 理由 |
|---|---|---|
| 「最高値を見つけたい」 | `optimization` | BO 単体、exploit + explore |
| 「関係性を理解したい／予測モデルを作りたい」 | `learning` | AL 単体、全域探索 |
| 「まず全体を把握してから絞りたい」 | `hybrid_learn_then_optimize` | AL 期 → BO 期、budget split |
| 「予測モデルと最適点の両方が欲しい」 | `hybrid_multi_objective_and_learn` | Pareto 的：予測精度と最適値を同時目標に |
| 「装置差を理解しつつ最高値を探す」 | `hybrid_learn_then_optimize` + MTGP (Ch11) | 学習段で装置差を捉え、BO 段で MTGP surrogate として利用 |

### 13.3.2 目的モード誤認の防止契約

> [!WARNING]
> **Skill が objective_mode を自律的に変更するのは fatal**（第3章 §3.5 逸脱と同種）。切替は必ず Human 承認付きの `objective_mode_switch_events` として記録。

```yaml
objective_mode_switch_events:                 # append-only、hash chain（Ch12 §12.3 と同じ event schema）
  - event_id: "evt-mode-switch-20261215-001"
    event_hash: "sha256:..."
    previous_event_hash: "sha256:0000..."
    switched_at: "2026-12-15T14:00:00Z"
    from_mode: "learning"
    to_mode: "optimization"
    approved_by: "human:project_lead_XXXX"
    approval_id: "approval:HITL-20261215-OM02"
    switch_gate_evaluation:
      surrogate_generalization_error: 0.12    # threshold 0.15 未満
      n_observations: 32                       # threshold 30 以上
      gate_condition_met: true
    run_id: "run-2026-1215-01"
    skill_execution_id: "skill-exec-ABC"
```

---

## 13.4 Emukit / modAL の Skill 化

### 13.4.1 Emukit — 実験計画寄りの AL / BO 統合ライブラリ

Emukit は Amazon 製の decision-making under uncertainty library。BO / AL / sensitivity analysis を統一 API で提供。

```python
import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
from emukit.model_wrappers import GPyModelWrapper
import GPy

space = ParameterSpace([
    ContinuousParameter("x_temperature", 200.0, 800.0),
    ContinuousParameter("x_pressure", 0.1, 5.0),
])

# 初期データ
X_init = np.array([[300, 1.0], [500, 2.5], [700, 4.0]])
Y_init = np.array([[expensive_measurement(x)] for x in X_init])

# GP surrogate
gpy_model = GPy.models.GPRegression(X_init, Y_init)
emukit_model = GPyModelWrapper(gpy_model)

# Active Learning loop（uncertainty sampling）
acquisition = ModelVariance(emukit_model)
al_loop = ExperimentalDesignLoop(space, emukit_model, acquisition)
al_loop.run_loop(user_function=expensive_measurement, stopping_condition=10)
```

### 13.4.2 modAL — scikit-learn ベースの AL

modAL は scikit-learn model なら何でも AL に載せられる汎用ライブラリ。分類タスクに強い。

```python
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.ensemble import RandomForestClassifier

learner = ActiveLearner(
    estimator=RandomForestClassifier(n_estimators=100),
    query_strategy=uncertainty_sampling,
    X_training=X_init,
    y_training=y_init,
)

# 1 iteration: query → label → teach
query_idx, query_x = learner.query(X_pool)
y_new = human_label(query_x)                  # ラベル取得（Human）
learner.teach(query_x, y_new)
```

### 13.4.3 Skill 契約テンプレート — Active Learning Skill

```yaml
active_learning_skill:
  version: "v0.1"
  skill_name: "active_learning_regression_v1"
  objective_mode: "learning"                  # Ch13 §13.3、この Skill は learning 専用
  surrogate:
    model_family: "single_task_gp"            # Ch5 canonical enum
    kernel_spec: {name: "matern52", nu: 2.5}  # Ch6
  acquisition:
    name: "uncertainty_sampling"              # §13.2 canonical enum
    target: "predictive_variance"
  batch_size: 4
  diversity_policy_ref: "vol-05:ch12:batch_diversity_policy_v0_2"
  stop_condition_for_learning:                # BO の stop_condition とは別、学習用
    type: "generalization_error_threshold"    # enum: generalization_error_threshold | max_iterations | pool_exhausted | budget_exhausted
    threshold: 0.15                            # 予測誤差 15% 未満で停止
    validation_method: "leave_one_out_cv"     # enum: leave_one_out_cv | k_fold | held_out_test
    approval_to_stop: "requires_human_approval"
  experiment_launch_authorization_ref: "vol-05:ch05:experiment_launch_authorization_v0_1"
```

---

## 13.5 Agentic 特有の失敗パターン契約

- **目的の取り違え**：Human が「モデルを作りたい」と言ったのに、Skill が BO を実行 → `objective_mode` の宣言を必須化、宣言なしでの Skill 実行は fatal
- **勝手なモード切替**：AL 中に「良さそうな領域があった」と Skill が BO に切り替える → §13.3.2 の switch_events で防護
- **stop_condition_for_learning の勝手な緩和**：generalization error threshold を Skill が自律的に上げる → Ch10 と同じく fatal
- **committee の勝手な入れ替え**：QBC で committee メンバーを Skill が変更 → `committee_spec.refresh_policy` に従うことを機械検証、逸脱は fatal
- **pool の勝手な絞込み**：candidate pool（AL でよく使う）を Skill が事前フィルタしてしまうと bias 発生 → pool の変更は `pool_modification_events` に記録、Human 承認必須

```yaml
pool_modification_events:                     # append-only、hash chain
  - event_id: "evt-pool-mod-20261220-001"
    event_hash: "sha256:..."
    previous_event_hash: "sha256:0000..."
    modified_at: "2026-12-20T10:00:00Z"
    modification_type: "filter"               # enum: filter | expand | resample | replace
    approved_by: "human:project_lead_XXXX"
    approval_id: "approval:HITL-20261220-PM01"
    rationale: "実験不可能な組成領域を除外"
    pool_size_before: 10000
    pool_size_after: 8500
```

---

## 13.6 Skill 契約チェックリスト

- [ ] `objective_mode.mode` が Human 宣言済み、`approval_id` あり
- [ ] `objective_mode` の変更は `objective_mode_switch_events` で追跡、Skill 自律変更禁止
- [ ] `active_learning_acquisition_spec.name` が canonical enum
- [ ] `task_type` が明示（regression / classification / ranking）
- [ ] QBC 使用時、`committee_spec.members` と `disagreement_metric` が明示
- [ ] `batch_size >= 2` の場合、`diversity_policy_ref` で Ch12 の post-hoc 検証を利用
- [ ] `stop_condition_for_learning` が明示、Skill 自律緩和禁止
- [ ] `experiment_launch_authorization_ref` で Ch5 との整合
- [ ] Pool 変更は `pool_modification_events` で Human 承認付き
- [ ] Hybrid モードの場合、`switch_gate.condition` と `budget_split` が明示
- [ ] BO Skill (Ch7-12) と切り分けて別 Skill として登録（同一 Skill で兼務しない）

---

## 13.7 章末演習

**問 1**：Human は「合金組成の予測モデルを作りたい」と言っている。Skill が受けるべき正しい行動は？

**問 2**：AL 実行中に「有望領域が見つかった」と Skill が判断した。何をすべきか。

**問 3**：Query by Committee で committee のうち 1 つが GP fit に失敗した。契約上の応答は？

**問 4**：hybrid_learn_then_optimize モードで、learning phase の budget 30 のうち 25 実施して generalization error が 0.10 になった。次に何をすべきか？

**問 5**：Skill が candidate pool から「危険そうな領域」を勝手にフィルタしていた。監査で何をチェックすべきか？

---

## 13.8 参考資料

### 本書内

- 第3章 §3.5：Skill 逸脱パターン（objective_mode の自律変更禁止）
- 第5章 §5.2：`experiment_launch_authorization`, `surrogate_model_family` の初出
- 第6章：GP surrogate（AL でも同じ surrogate を使う）
- 第8章：BNN / DKL（BALD の基盤）
- 第10章：event_hash / previous_event_hash schema
- 第11章：MTGP（hybrid モードで装置差を捉える surrogate）
- 第12章：batch_diversity_policy（AL でも再利用）
- 第14章（planned）：Capstone（BO 側で AL を活用するケース）
- 第15章（planned）：objective_mode 取り違え、committee 勝手な変更の失敗パターン

### 外部参考

- Settles, "Active Learning Literature Survey", Univ. Wisconsin TR, 2010 — AL の baseline surveys
- Houlsby et al., "Bayesian Active Learning for Classification and Preference Learning", arXiv:1112.5745, 2011 — BALD 原論文
- MacKay, "Information-based objective functions for active data selection", Neural Computation 1992 — expected model change 系
- Emukit documentation (https://emukit.github.io/) — Amazon の AL/BO 統合 lib
- modAL documentation (https://modal-python.readthedocs.io/) — scikit-learn ベース AL

> [!NOTE]
> 次章（第14章）では、これまで積み上げた第6〜13章の Skill 契約を用いて、**因果 DAG による search space 絞込 → 単一装置 GP → BO 3 iteration**（14a：基本）と、**階層 GP × multi-objective × constrained × batch × 10 iteration**（14b：発展）を統合した Advanced Capstone を実装します。
