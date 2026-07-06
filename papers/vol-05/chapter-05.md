# 第5章 BO × Agentic Skill の設計原則

> **本章の到達目標**
> - **BO Skill の provenance 契約**（`iteration_index` / `search_space_bounds` / `surrogate_model_family` / `kernel_spec` / `acquisition_spec` / `pending_experiments` / `stop_condition` 等 15 フィールド）を、なぜそれぞれ pin するのかと合わせて説明できる
> - **`experiment_launch_authorization`** の完全設計（vol-04 L3 との parent 関係、bypass 禁止、両ゲート pass 必須）を書ける
> - **surrogate 再学習ポリシー** 3 種（every iteration / N iteration / threshold-triggered）を選び、Skill 契約に pin できる
> - **`stop_condition`** を budget / convergence / regret threshold の 3 スコープで operational 化する具体式を書ける
> - **`hallucinated_recommendation_detection`** の Skill 宣言レベル（operational 定義は第7章に委譲）を Skill 契約に組み込める
> - **Provenance field 使用マップ**——どのフィールドがどの章で導入・operational 化・監査されるかを一覧化できる
> - vol-01〜04 の Skill 契約テンプレートとの diff（BO × 逐次特有の追加フィールド）を明確にできる
>
> **本章で扱わないこと**
> - GP kernel と事前分布の具体設計（第6章）
> - Acquisition function ごとの数式と外挿検知 operational 定義（第7章）
> - Multi-task GP / 階層 GP の Skill 実装（第11章）
> - Batch BO の pending / batch_size 実装（第12章）
> - `budget_remaining` と `stop_condition` の operational コード（第14章 capstone）
> - MCP server 側の Skill 実装（付録 B）

---

## 5.1 なぜ BO Skill には vol-04 より多くの provenance が必要か

vol-04 の DoE Skill は「観測前に計画を確定する」——`randomization_seed`、`factorial_design_spec`、`blocking_factor`、`refutation_gate_v0_3` の 4 フィールドを pin すれば、実験全体が再現可能でした。

vol-05 の BO Skill は **「観測後に次を決める」**——iteration ごとに以下が変わります：

- 使う **surrogate model**（GP、RF、BNN、Deep Kernel）
- **kernel** と hyperparameter（推定されたもの）
- **acquisition function** の値
- **pending experiments** の状態
- **budget remaining** の減少
- 前 iteration の観測結果に基づく **候補提案**

このため、**「1 iteration を再現する」のに必要な情報が vol-04 の 4-5 倍**——**Core 15 フィールド + Gate/Policy 3 フィールド + 拡張フィールド 3**、合計 21 前後を pin する契約が必要になります。以下では 4 グループ（A: 起動時 pin / B: 状態 / C: 出力 / D: Gate/Policy）に分けて詳述します。

---

## 5.2 BO Skill provenance の 18 コアフィールド + 拡張

### グループ A：Skill 起動時に pin する契約フィールド（10 個）

これらは **Skill 起動時に決定して以降 immutable**——実行中の変更は fatal です。

| フィールド | 型 | 意味 | 章 |
|---|---|---|:-:|
| `iteration_index` | int | 現 iteration の番号（0-indexed）| 5 |
| `search_space_bounds` | dict | 連続変数の (low, high) + 離散変数の候補集合 + 単位 | 5 |
| `bo_library_stack` | enum | `botorch_direct` / `ax_botorch` / `skopt_ask_tell` / `hebo` / `emukit` / `modal_al` / `custom` | 4, 5 |
| `bo_library_stack_metadata` | dict | `library_version` / `model_backend` / `acquisition_impl` / `generation_strategy_uri`（Ax のみ）| 4, 5 |
| `surrogate_model_family` | enum | `single_task_gp` / `multi_task_gp` / `random_forest` / `bnn` / `deep_kernel` | 5 |
| `kernel_spec` | dict | kernel 型（Matérn 5/2 / RBF / Hamming / product / hierarchical）+ length_scale prior | 6 |
| `acquisition_spec` | dict | acquisition 名（qLogEI / qUCB / qKG / qMES / qEHVI / cEI / SafeOpt 等）+ hyperparameter | 7 |
| `surrogate_treatment_of_facility_variance` | enum | `categorical_input` / `hierarchical_shrinkage` / `blocked_out` / `not_applicable` | 3, 5 |
| `retraining_policy` | enum | `every_iteration` / `every_n_iterations` / `threshold_triggered` | 5 |
| `sequential_seed_provenance` | dict | 逐次乱数 seed の由来（初期 seed、iteration ごとの派生規則）| 5, 15 |

**pin される理由**：これらが iteration 間で勝手に変わると、**BO の性能保証（regret bound）が破れる**、**provenance から再現できなくなる**、または **第3章 §3.5 の逸脱**が発生します。

### グループ B：iteration ごとに更新される状態フィールド（5 個）

これらは iteration ごとに **append-only または monotonic** に変化。**キャンセルや上書きは fatal**。

| フィールド | 型 | 意味 | 更新規則 | 章 |
|---|---|---|---|:-:|
| `observed_data` | list | (X, Y) の観測データ全履歴 | append-only | 5 |
| `pending_experiments` | list | 承認済みだが未観測の候補 | append-only、キャンセルは Human 承認必須 | 12 |
| `budget_remaining` | dict | 期間予算・費用予算の残量 | monotonic 減少、iteration ごとに更新 | 5, 14 |
| `hallucination_events` | list | 外挿検知で flag が立った候補の履歴 | append-only | 7 |
| `batch_size` | int | Batch BO の場合の 1 iteration での提案数（非 Batch は 1） | 起動時 pin、iteration 内不変 | 12 |

### グループ C：Skill 出力に含める推論結果フィールド（3 個）

これらは **1 iteration の出力**——次 iteration の入力として次の Skill 起動時に渡されます。

| フィールド | 型 | 意味 | 章 |
|---|---|---|:-:|
| `next_candidate` | dict/list | 提案される (X_next, acquisition_value)、Batch の場合は list、各要素に `estimated_cost` と `estimated_calendar_hours` | 5, 7, 12 |
| `hallucination_flag` | dict | `{flag: bool, reasons: [str], scores: dict}` | 7 |
| `surrogate_diagnostics` | dict | fit quality（length_scale, noise, LOOCV RMSE, convergence status）| 6 |

### グループ D：Gate / Policy フィールド（Human 承認と停止判定、3 個）

これらは Skill の動作を規定するゲート・ポリシーで、`experiment_launch_authorization` は vol-04 L3 を parent として参照します。

| フィールド | 型 | 意味 | 章 |
|---|---|---|:-:|
| `experiment_launch_authorization` | dict | Human 承認記録（vol-04 L3 の `parent_authorization_id` を含む）| 5, §5.3 |
| `stop_condition` | dict | budget / convergence / regret threshold の 3 スコープ、`OR` 論理で結合 | 5, §5.5, 14 |
| `hallucinated_recommendation_detection` | dict | 宣言レベル（`declared: true` 必須、operational は Ch7 参照）| 5, §5.6, 7 |

### フィールド数のまとめ

| グループ | 個数 | 内訳 |
|---|:-:|---|
| A: 起動時 pin | 10 | 契約 immutable |
| B: 状態 | 5 | append-only / monotonic |
| C: 出力 | 3 | 次 iteration への渡し |
| D: Gate / Policy | 3 | 承認・停止・外挿検知 |
| **合計** | **21** | Core 18（A: 10、B の 4 主要、C: 3、D: 3、`batch_size` を除いた場合）+ 拡張 3（`bo_library_stack_metadata`、`sequential_seed_provenance`、`batch_size`）|

> [!NOTE]
> **「15 前後」から「Core 18 + 拡張 3」への拡張**：v0.1 では 15 フィールドで書きましたが、rubber-duck review で `sequential_seed_provenance`、`batch_size`、`bo_library_stack_metadata` の重要性が指摘され、21 フィールドの正式仕様としました。§5.8 使用マップも 21 フィールドで再構成しています。

### provenance の例（YAML）

```yaml
# BO Skill provenance v0.2（rubber-duck review 反映）
bo_skill_provenance:
  version: "v0.2"

  # --- Group A: 起動時 pin (immutable) ---
  iteration_index: 3
  search_space_bounds:
    temperature: { low: 300, high: 700, unit: "celsius" }
    pressure:    { low: 0.1, high: 10.0, unit: "MPa" }
    time:        { low: 1.0, high: 24.0, unit: "hours" }
    atmosphere:  { candidates: ["Ar", "N2", "O2", "vacuum"] }
    facility_id: { candidates: ["dev_A", "dev_B", "dev_C"] }
  bo_library_stack: "botorch_direct"
  bo_library_stack_metadata:
    library_version: "botorch==0.11.0, gpytorch==1.11"
    model_backend:   "gpytorch.ExactGP"
    acquisition_impl: "botorch.acquisition.logei.qLogExpectedImprovement"
    generation_strategy_uri: null  # Ax 経由時のみ設定
  surrogate_model_family: "single_task_gp"
  kernel_spec:
    continuous: { type: "matern52", ard: true, length_scale_prior: "gamma(3, 6)" }
    categorical: { type: "hamming" }
    composite:   "product"
  acquisition_spec:
    name: "qLogExpectedImprovement"
    hyperparameters: { num_restarts: 10, raw_samples: 128 }
  surrogate_treatment_of_facility_variance: "categorical_input"
  retraining_policy: "every_iteration"
  sequential_seed_provenance:
    initial_seed: 20261115
    derivation_rule: "seed_k = initial_seed + iteration_index"
    current_seed: 20261118  # iteration 3

  # --- Group B: 状態 (append-only / monotonic) ---
  observed_data:
    - { x: [450, 2.0, 4, "Ar", "dev_A"], y: 0.72, iter: 0 }
    - { x: [520, 5.5, 8, "N2", "dev_B"], y: 0.65, iter: 1 }
    - { x: [400, 1.0, 12, "O2", "dev_A"], y: 0.81, iter: 2 }
  pending_experiments: []
  budget_remaining:
    calendar_hours: 720
    money_jpy: 850000
  hallucination_events: []
  batch_size: 1

  # --- Group C: 出力 ---
  next_candidate:
    x: [480, 3.5, 6, "Ar", "dev_C"]
    acquisition_value: 0.043
    estimated_cost: { calendar_hours: 30, money_jpy: 45000 }
  hallucination_flag:
    flag: false
    reasons: []
    scores: { variance: 0.08, mahalanobis: 1.2, length_scale_ratio: 0.6 }
  surrogate_diagnostics:
    length_scale: [125, 2.1, 4.8]  # per ARD dim
    noise_variance: 0.012
    loocv_rmse: 0.045
    convergence_status: "ok"

  # --- Group D: Gate / Policy ---
  experiment_launch_authorization: null  # see §5.3、実験実行前に埋める
  stop_condition:
    logic: "OR"
    conditions:
      - { type: "budget", calendar_hours_min: 0, money_jpy_min: 0 }
      - { type: "convergence", lookback_iterations: 5, relative_change_threshold: 0.01 }
      - { type: "regret_threshold", target_value: 0.95, objective: "crystallinity" }
  hallucinated_recommendation_detection:
    declared: true
    detection_criteria_ref: "vol-05:ch07:hrd_operational_v0_1"
    action_on_flag: "return_needs_human_review"
    logged_events_field: "hallucination_events"
```

> [!TIP]
> **Skill が実行中に Group A のフィールドを変更しようとしたら、Skill は fatal で停止**します。Human が新規 Skill 実行として `iteration_index` を進めるしかない——これが第3章 §3.5 逸脱 2 / 5（acquisition drift、stop_condition 緩め）を防ぐ第一の仕組み。逸脱 1（Human 承認省略）は §5.3、逸脱 3（外挿を自信あり報告）は §5.6 と Ch7、逸脱 4（pending キャンセル）は Group B の append-only 契約と Ch12 で防がれます。5 逸脱 → 防止機構の対応表は §5.8 の直前に示します。

---

## 5.3 `experiment_launch_authorization` の完全設計

BO Skill の中で最も重要な設計は、**物理実験実行の承認ゲート**です。第1章 §1.4 で導入した「vol-04 L3 との parent 関係」をここで完全に展開します。

### 承認ゲートの構造

```
BO Skill 起動 (iteration_index = k)
    │
    ▼
[1] surrogate 再学習 + acquisition 最適化
    │
    ▼
[2] next_candidate 生成 + hallucination_flag チェック
    │
    ▼
[3] hallucination_flag.flag == true?
    │
    ├─ YES → [3a] Human review へ ─── needs_human_review 状態で終了
    │
    └─ NO  → [4] experiment_launch_authorization 要求
                │
                ▼
             [5] vol-04 L3 (L3_intervention_execution_authorization) が pass?
                │
                ├─ NO  → fatal（bypass 不可）
                │
                └─ YES → [6] experiment_launch_authorization pass
                             │
                             ▼
                          [7] 実験実行（装置予約→合成→測定）
                             │
                             ▼
                          [8] observed_data に append + iteration_index++
```

### 契約要件

**要件 1：vol-04 L3 を parent としてリンク**

```yaml
experiment_launch_authorization:
  version: "v0.2"
  authorization_id: "elauth_20261115_093015_iter3"
  status: "approved"  # enum: pending | approved | denied
  requested_at: "2026-11-15T09:30:00Z"
  approved_at: "2026-11-15T10:12:47Z"
  approver_id: "human:staff_id_XXXX"
  parent_authorization_id: "vol-04:L3_intervention_execution_authorization:auth_id_YYYY"
  parent_l3_status: "approved"  # L3 side pass 確認済み
  approval_scope:
    iteration_index: 3
    authorized_candidate_ids: ["cand_iter3_00"]
    candidate_hash: "sha256:9f2a...c7b3"  # candidate_x + iteration + bounds のハッシュ
    batch_size: 1
    max_experiments_authorized: 1
  device_reservation:
    facility_id: "dev_C"
    reservation_id: "resv_dev_C_20261116_090000"
    slot_start: "2026-11-16T09:00:00Z"
    slot_end:   "2026-11-16T17:00:00Z"
  budget_precheck:
    estimated_cost:      { calendar_hours: 30, money_jpy: 45000 }
    budget_remaining:    { calendar_hours: 720, money_jpy: 850000 }
    post_launch_remaining: { calendar_hours: 690, money_jpy: 805000 }
    check_pass: true  # post_launch_remaining の全 metric が >= 0
  human_notes: |
    温度・圧力・時間・雰囲気は operating envelope 内。
    hallucination_flag なし。装置予約 dev_C は 2026-11-16 09:00-17:00 で確保済み。
```

**要件 2：両ゲート pass が必須**

- vol-04 L3 のみ pass、`experiment_launch_authorization` なし → **fatal**（vol-04 承認だけで実験実行できない）
- `experiment_launch_authorization` のみ pass、vol-04 L3 なし → **fatal**（causal 妥当性の担保なしに実験実行できない）
- **両方 pass** → 実験実行 OK

**要件 3：bypass 禁止**

- Skill レベルで「承認済みとして扱う」flag を立てる操作は fatal
- `parent_authorization_id` が `null` の状態で実験実行に進む Skill は fatal
- Human 承認が長時間得られない iteration を「後で承認する」で通す運用は禁止

**要件 4：承認スコープの限定**

- 1 回の承認で許可される実験は **`max_experiments_authorized` 個まで**（通常 1、Batch BO の場合は batch_size）
- 承認済み candidate 以外の X で実験実行するのは fatal
- iteration_index が承認された値と異なる場合も fatal

> [!IMPORTANT]
> **物理実験を伴わない Skill（simulation-based BO、benchmark 関数最適化）では `experiment_launch_authorization` は不要**。ただし `parent_authorization_id` の代わりに `simulation_provenance: {simulator_name, version, deterministic_seed}` を pin してください——**実験と simulation を Skill レベルで混同する** ことも第3章 §3.5 逸脱の一種です。

### vol-04 L3 との差分（なぜ二重ゲートが必要か）

| ゲート | 責務 | 何を守るか |
|---|---|---|
| **vol-04 L3** | 介入実行の因果的妥当性（confounder / positivity / DAG 一貫性） | **causal validity**——観測データから真の因果効果を推定できる条件 |
| **vol-05 `experiment_launch_authorization`** | BO 特有の運用条件（hallucination なし、pending 状態一貫、budget 内、safe BO の場合は制約 pass） | **operational validity**——BO ループとして機能する条件 |

**二重ゲートの必然性**：causal validity が pass しても hallucination の可能性はあり、operational validity のみでは causal 妥当性が守られない——両者は独立で、両方の pass が必要です。

---

## 5.4 Surrogate 再学習ポリシー 3 種

`retraining_policy` は Group A の pin フィールドです。3 種類から選択：

### ポリシー 1：`every_iteration`

- **やり方**：毎 iteration で GP を最初から fit（`fit_gpytorch_mll`）
- **向いている場面**：観測データが少ない（< 30 点）、計算コストが低い
- **リスク**：hyperparameter が iteration ごとに大きく変動して acquisition が不安定
- **推奨**：本書のデフォルト、第14章 capstone Phase 1 で採用

### ポリシー 2：`every_n_iterations`

- **やり方**：**新観測点は毎 iteration で GP に取り込む**（posterior update）が、**hyperparameter θ の再最適化は N iteration ごとに限定**。それ以外の iteration では前回最適化された θ をそのまま再利用
- **向いている場面**：観測が中程度（30-100 点）、計算コストが中程度
- **リスク**：観測データが N iteration 分溜まる間、hyperparameter が古いため surrogate の calibration が劣化する可能性
- **設定例**：`N = 5`、fully Bayesian の場合は `N = 3-5`

### ポリシー 3：`threshold_triggered`

- **やり方**：現行 model で計算可能な指標を trigger にする——**新観測の standardized residual、negative log predictive density (NLPD)、予測区間外れ率、直近 K iteration の holdout error 増加**——閾値を超えたら θ を再最適化
- **向いている場面**：観測が多い（100+ 点）、計算コストが高い（fully Bayesian、階層 GP）
- **リスク**：閾値設計を誤ると再学習が発生せず、hallucination の原因になる
- **推奨**：第11章の階層 GP、第12章の Batch BO で採用検討
- **注意**：hyperparameter 相対変化を trigger にすると **再fit しないと測れないため循環**——current model で計算可能な指標に限定する

### 判断表

| ポリシー | 観測数 | 計算コスト | 実装難易度 | 使い所 |
|---|---|---|:-:|---|
| `every_iteration` | < 30 | 低 | ○ | 本書デフォルト、少データ現場 |
| `every_n_iterations` | 30-100 | 中 | ○ | 中規模、fully Bayesian |
| `threshold_triggered` | 100+ | 高 | △ | 大規模、階層 GP、Batch BO |

**pin される情報**：ポリシー名 + パラメータ（N の値、閾値）。実行中の変更は fatal。

---

## 5.5 `stop_condition` の 3 スコープ

vol-05 の `stop_condition` は **budget / convergence / regret threshold** の 3 スコープのみ（第1章 §1.5 で確立）。以下、各スコープの operational 定義を示します（フル実装は第14章 §14.4）。

### スコープ 1：Budget

- **やり方**：`budget_remaining.calendar_hours <= 0 OR budget_remaining.money_jpy <= 0` で停止
- **前提**：第3章 §3.4 のサイクル時間見積り $N_{\text{BO\_iter}}$ が Skill 起動時に確定している
- **launch 前チェック**：**次候補の `estimated_cost`（Group C）と `budget_remaining`（Group B）を比較**——`budget_remaining - estimated_cost < 0` の場合は `experiment_launch_authorization.budget_precheck.check_pass = false` となり、実験実行不可（§5.3 の budget_precheck ブロック）。「残り 1 時間で 8 時間の実験を launch する」を防ぐ
- **fatal 条件**：Skill が実行中に `budget_remaining` を **増やす**（緩める）ことは fatal——第3章 §3.5 逸脱 5

### スコープ 2：Convergence

- **やり方**：最近 M iteration の best-so-far の相対変化が閾値以下、または hyperparameter が安定
- **operational 定義例**：$\Delta_{\text{best}} = |y_{\text{best},k} - y_{\text{best},k-M}| / |y_{\text{best},k-M}| < \epsilon$（例：$M = 5$, $\epsilon = 0.01$）
- **注意**：多目的の場合、hypervolume の変化率で定義。第9章で拡張
- **fatal 条件**：Skill が M や ε を実行中に変更するのは fatal

### スコープ 3：Regret Threshold

- **やり方**：最良候補の目的関数値が事前に設定した閾値 $y_{\text{target}}$ を上回った時点で停止
- **operational 定義**：$y_{\text{best},k} \geq y_{\text{target}}$（最大化問題）
- **向いている場面**：明確な性能目標がある（例：結晶化度 0.95 以上、密度 5.0 g/cm³ 以上）
- **注意**：Regret bound の理論値ではなく、**実用的な "十分良い" 閾値** を Human が事前設定

### 複数スコープの組み合わせ

```yaml
stop_condition:
  version: "v0.1"
  logic: "OR"  # いずれかが満たされたら停止
  conditions:
    - type: "budget"
      calendar_hours_min: 0
      money_jpy_min: 0
    - type: "convergence"
      lookback_iterations: 5
      relative_change_threshold: 0.01
    - type: "regret_threshold"
      target_value: 0.95
      objective: "crystallinity"
```

> [!WARNING]
> **禁止されている `stop_condition` タイプ**：`MC_variance_saturation`（MCMC 分散飽和）は本書のスコープ外——理論的には可能ですが、operational 化が難しく、Human が閾値を直感的に理解できません。Skill 契約に登場したら **outline 違反として fatal**。

---

## 5.6 `hallucinated_recommendation_detection` の宣言レベル

本節では **Skill 契約における宣言レベルの記述** のみを扱います。**operational 定義（3 判定の合成基準）は第7章**——本章では「宣言としてどう Skill 契約に組み込むか」だけ。

### 宣言レベルの契約

```yaml
hallucinated_recommendation_detection:
  version: "v0.1"
  declared: true  # 本 Skill は外挿検知を実施する
  detection_criteria_ref: "vol-05:ch07:hrd_operational_v0_1"
  action_on_flag: "return_needs_human_review"  # fatal ではない
  logged_events_field: "hallucination_events"  # Group B の一部
```

- `declared: true` を必須——`declared: false` の BO Skill は物理実験用途では fatal
- `detection_criteria_ref` は第7章の operational 定義への参照（実装が第7章で更新されても、参照が Skill 契約に残る）
- `action_on_flag: "return_needs_human_review"` は必須——`ignore` や `retry_with_different_acquisition` は fatal

### hallucination_events の記録

```yaml
hallucination_events:
  - iteration: 5
    detected_at: "2026-11-20T14:22:15Z"
    x_flagged: [720, 8.5, 20, "N2", "dev_A"]
    reasons: ["mahalanobis_exceeded", "length_scale_ratio_exceeded"]
    scores:
      variance: 0.18
      mahalanobis: 4.2
      length_scale_ratio: 1.3
    human_decision: "reject"
    resolved_at: "2026-11-20T15:45:00Z"
```

- **append-only**——過去の event を削除・修正するのは fatal
- Human 判断（reject / accept_anyway / modify_bounds）を必ず記録
- `accept_anyway` は Human が **理由記述付き**で承認する場合のみ許容

### hallucination_flag 後の状態遷移

`hallucination_flag.flag = true` になった候補は、以下の 3 状態のいずれかに遷移します。**いずれの場合も `experiment_launch_authorization` と vol-04 L3 の両方が必須**——`accept_anyway` でも bypass 不可：

| 遷移 | 意味 | 次の行動 | 追加要件 |
|---|---|---|---|
| `reject` | 候補を破棄 | 次 iteration の acquisition を再最適化（constraints・bounds は変更しない） | 通常フロー |
| `modify_bounds` | 探索範囲を Human が調整 | Group A の `bounds` を Human 承認付きで更新、`iteration_index` を進めて Skill 再起動 | Group A 変更 → 新 Skill run 扱い（第3章 §3.5 逸脱 2 のガード再適用） |
| `accept_anyway` | 外挿と分かった上で実行 | `hallucination_events[].human_decision = "accept_anyway"` + `reason` 記録 | **`experiment_launch_authorization`（bypass 不可）＋ vol-04 L3 の両方が必須**、`human_notes` に外挿を認識した理由を明記 |

---

## 5.7 vol-01〜04 Skill 契約テンプレートとの diff

vol-04 の DoE Skill 契約テンプレート（vol-04 第5章）と、vol-05 の BO Skill 契約の差分：

### 追加されるフィールド（vol-05 で新設）

| フィールド | vol-04 | vol-05 |
|---|:-:|:-:|
| `iteration_index` | ✗ | ✓ |
| `bo_library_stack` | ✗ | ✓ |
| `surrogate_model_family` | ✗ | ✓ |
| `kernel_spec` | ✗ | ✓ |
| `acquisition_spec` | ✗ | ✓ |
| `retraining_policy` | ✗ | ✓ |
| `surrogate_treatment_of_facility_variance` | ✗ | ✓ |
| `sequential_seed_provenance` | ✗ | ✓（vol-04 の `randomization_seed` を逐次化して拡張）|
| `bo_library_stack_metadata` | ✗ | ✓ |
| `observed_data` | ✗ | ✓（vol-04 の観測ログを iteration-tagged monotonic append に）|
| `pending_experiments` | ✗ | ✓ |
| `budget_remaining` | ✗ | ✓ |
| `hallucination_events` | ✗ | ✓ |
| `batch_size` | ✗ | ✓（Ch12 で operational）|
| `next_candidate` | ✗ | ✓ |
| `surrogate_diagnostics` | ✗ | ✓ |
| `hallucination_flag` | ✗ | ✓ |
| `experiment_launch_authorization` | ✗ | ✓ |
| `stop_condition` | ✗ | ✓ |
| `hallucinated_recommendation_detection` | ✗ | ✓ |

### 継続するフィールド（vol-04 から継承）

| フィールド | vol-04 | vol-05 |
|---|:-:|:-:|
| `search_space_bounds`（vol-04 は `factorial_design_spec` に近い）| ✓ | ✓ |
| `L1_dag_authorization` | ✓ | ✓ |
| `L2_variable_selection_authorization` | ✓ | ✓ |
| `L3_intervention_execution_authorization`（vol-05 では `experiment_launch_authorization` の parent）| ✓ | ✓（parent） |
| `L4_facility_standard_promotion` | ✓ | ✓ |
| `refutation_gate_v0_3` | ✓ | ✓（BO 前提の因果 DAG 検証で使用） |
| `audit_manifest_v1` | ✓ | ✓（vol-05 では 19 checks を継承しつつ BO 固有 check を追加検討、詳細は第15章）|

### 削除されるフィールド（vol-04 で使ったが vol-05 では不要）

- `blocking_factor`（vol-04 §10.4）—— BO では blocking は `surrogate_treatment_of_facility_variance` に統合、独立フィールドは不要
- `taguchi_method_spec`（vol-04 §11.3）—— vol-05 は逐次実験のため一括タグチは対象外
- `randomization_seed`（vol-04）—— vol-05 では `sequential_seed_provenance` に置き換え（seed derivation ルール付き）

---

## 5.8 第3章 §3.5 5 逸脱と本章の防止機構の対応表

第3章 §3.5 で列挙した 5 逸脱が、本章のどのフィールド・契約要件で防がれるかの対応表：

| # | 逸脱シナリオ | 主要な防止フィールド / 契約要件 | 検出ポイント |
|:-:|---|---|---|
| 1 | Human 承認省略 | `experiment_launch_authorization`（§5.3、bypass 禁止、parent L3 必須）| Skill 実行時 fatal、audit 監査 |
| 2 | Acquisition 途中変更 | Group A の Skill 実行中 immutable 契約（§5.2）、`acquisition_spec` は起動時固定 | Skill fatal、iteration 途中の Group A 変更 = 新 run 扱い |
| 3 | 外挿を自信あり報告 | `hallucinated_recommendation_detection`（§5.6、宣言）+ Ch7 operational 3 判定、`hallucination_events` append-only | flag 発火時 `return_needs_human_review` |
| 4 | Pending 実験のキャンセル | `pending_experiments` append-only、Group B monotonic 契約（§5.2）| Skill fatal |
| 5 | Stop condition 緩め | `stop_condition` を Group A（起動時固定）、`budget_remaining` の増加 fatal、§5.5 の禁止 `MC_variance_saturation` | Skill fatal、audit 監査 |

> [!IMPORTANT]
> 「防ぐ」は「Skill 実行中の暗黙変更を fatal で止める」+「Human 判断としての変更は新 iteration/新 run として扱う」の二層。前者は Skill 内契約、後者は §5.3 の Human 承認ゲートで実現します。

---

## 5.9 Provenance field 使用マップ（章横断）

以下、21 フィールドが各章でどう扱われるかの一覧。**新規導入 (📝) / operational 化 (⚙️) / 監査 (🔍) / 拡張 (📈)**。

| フィールド | Ch5 | Ch6 | Ch7 | Ch8 | Ch9 | Ch10 | Ch11 | Ch12 | Ch13 | Ch14 | Ch15 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `iteration_index` | 📝 | | | | | | | | | ⚙️ | 🔍 |
| `search_space_bounds` | 📝 | | | | | | | | | ⚙️ | 🔍 |
| `constraints_declared` | 📝 | | | | | ⚙️ | | | | ⚙️ | 🔍 |
| `bo_library_stack` | 📝 | | | | | | | | | ⚙️ | 🔍 |
| `bo_library_stack_metadata` | 📝 | | | | | | | | | ⚙️ | 🔍 |
| `surrogate_model_family` | 📝 | 📈 | | 📈 | | | 📈 | | 📈 | ⚙️ | 🔍 |
| `kernel_spec` | 📝 | ⚙️ | | | | | 📈 | | | ⚙️ | 🔍 |
| `acquisition_spec` | 📝 | | ⚙️ | 📈 | 📈 | 📈 | | 📈 | 📈 | ⚙️ | 🔍 |
| `surrogate_treatment_of_facility_variance` | 📝 | 📈 | | | | | 📈 | | | ⚙️ | 🔍 |
| `retraining_policy` | 📝 | | | | | | 📈 | 📈 | | ⚙️ | 🔍 |
| `sequential_seed_provenance` | 📝 | | | | | | | 📈 | | ⚙️ | 🔍 |
| `observed_data` | 📝 | | | | | | | ⚙️ | | ⚙️ | 🔍 |
| `pending_experiments` | 📝 | | | | | | | ⚙️ | | ⚙️ | 🔍 |
| `budget_remaining` | 📝 | | | | | | | | | ⚙️ | 🔍 |
| `batch_size` | 📝 | | | | | | | ⚙️ | | ⚙️ | 🔍 |
| `hallucination_events` | 📝 | | ⚙️ | 📈 | | | | | | ⚙️ | 🔍 |
| `next_candidate` | 📝 | | ⚙️ | | 📈 | 📈 | | 📈 | 📈 | ⚙️ | 🔍 |
| `hallucination_flag` | 📝 | | ⚙️ | 📈 | | | | | | ⚙️ | 🔍 |
| `surrogate_diagnostics` | 📝 | ⚙️ | | 📈 | | | 📈 | | | ⚙️ | 🔍 |
| `experiment_launch_authorization` | 📝 | | | | | 📈 | | 📈 | | ⚙️ | 🔍 |
| `stop_condition` | 📝 | | | | 📈 | | | | | ⚙️ | 🔍 |
| `hallucinated_recommendation_detection` | 📝 | | ⚙️ | 📈 | | | | | | ⚙️ | 🔍 |

> [!TIP]
> このマップは **Skill 開発時のチェックリスト** としても使えます。第14章 capstone を実装する際、すべてのフィールドが operational（⚙️）に到達しているかを確認する。

---

## 5.10 章末演習 — 自分のケースで BO Skill 契約を書く

**問 1**：自分のケースで、Group A の主要フィールドを埋めてください（第2-4章の演習を継承）。

**問 2**：`retraining_policy` を 3 種から 1 つ選び、選定根拠を書けますか？

**問 3**：`stop_condition` を budget / convergence / regret threshold のどれで定義するか、閾値を含めて書けますか？

**問 4**：物理実験を伴う場合、vol-04 L3 の承認 ID をどのように取得しますか？ 承認プロセスと `experiment_launch_authorization` の順序を書き出してください。

**問 5**：`hallucinated_recommendation_detection.action_on_flag` を `return_needs_human_review` 以外に設定したい場面がありますか？——ない場合、なぜないかを 1 段落で説明してください。

---

## 5.11 次章に進む前のチェックリスト

- [ ] **§5.2 の 21 フィールド**をグループ分けして説明できる
- [ ] **§5.3 の `experiment_launch_authorization`** と vol-04 L3 の parent 関係を図で書ける
- [ ] **§5.4 の retraining policy 3 種**から自分のケースに合うものを選んだ
- [ ] **§5.5 の stop_condition** を自分のケースで 1 スコープ書ける
- [ ] **§5.6 の hallucination detection** を declared: true で契約に組み込む理由が言える
- [ ] **§5.7 の vol-04 との diff** を 3 フィールド以上挙げられる
- [ ] **§5.8 の使用マップ**で、自分のユースケースに関わる章がどこかを見当がついた

「うろ覚え」があれば、該当節に短く戻ってから第6章「GP surrogate の Skill 化」へ進んでください。Ch6 では `kernel_spec` の operational 化を扱います。

---

## 参考資料

### vol-05 の該当章

> [!NOTE]
> vol-05 の第6章以降および付録は執筆中（planned）。以下のリンクは章確定後に有効化されます。

- [第1章 vol-01〜04 の最小復習](./chapter-01.md)（§1.4 authorization gates、§1.7 provenance フィールド）
- [第2章 予測 → 因果 → 逐次 のラダー](./chapter-02.md)
- [第3章 ARIM × BO × Agentic 特有の課題](./chapter-03.md)（§3.3 装置差、§3.4 サイクル時間、§3.5 5 逸脱）
- [第4章 BO / active learning ライブラリ地図](./chapter-04.md)（§4.2 BoTorch/Ax gap、`bo_library_stack` enum）
- 第6章 GP surrogate の Skill 化（planned、`kernel_spec` operational 化、既定値表）
- 第7章 Acquisition function と外挿検知（planned、`acquisition_spec` operational 化、`hallucinated_recommendation_detection` の 3 判定）
- 第10章 制約付き BO / safe BO（planned、`experiment_launch_authorization` の safe BO 拡張）
- 第11章 マルチ装置 BO（planned、`surrogate_treatment_of_facility_variance: hierarchical_shrinkage`）
- 第12章 Batch BO（planned、`pending_experiments` operational 化、`batch_size` 拡張）
- 第14章 総合ハンズオン（planned、全 15 フィールドの operational 化）
- 第15章 失敗パターンと監査（planned、22 checks への拡張、`audit_manifest_v1`）
- 付録 A BO × Agentic Skill テンプレート集（planned、全フィールド YAML 雛形）

### vol-04 の該当章（Skill 契約テンプレートの前身）

- [第5章 vol-04 Skill 契約テンプレート](../vol-04/chapter-05.md)（DoE Skill の provenance）
- [第9章 refutation gate v0.3](../vol-04/chapter-09.md)（10 tests、vol-05 でも継続使用）
- [第14章 総合ハンズオン](../vol-04/chapter-14.md)（`audit_manifest_v1` の 19 checks、vol-05 で 22 に拡張）

### 外部参考

- Frazier, P. I. "A Tutorial on Bayesian Optimization" (arXiv:1807.02811, 2018) — regret bound と stop_condition の理論
- Balandat, M. et al. "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization" (NeurIPS 2020) — provenance の観点から surrogate/acquisition 実装
- Ax documentation on Generation Strategy — <https://ax.dev/tutorials/generation_strategy.html>（Ax を使う場合の GS 明示 pin）
