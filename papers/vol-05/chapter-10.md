# 第10章　制約付き BO と safe BO — 安全条件を Skill 契約に組み込む

> [!IMPORTANT]
> **本章の到達目標**
>
> 1. 制約付き BO（cEI / cEHVI）と safe BO（SafeOpt 系）の違いを Skill 契約の言葉で説明できる
> 2. **「温度上限を超える提案は Skill が絶対に返さない」契約** を書ける
> 3. `constraints_declared` フィールド（第5章）を operational 化できる
> 4. Skill の失敗モードとしての **制約破り** を検出・記録する仕組みを設計できる
>
> **本章で扱わないこと**
>
> - 単目的 acquisition の骨格 → **第7章**
> - 多目的 acquisition → **第9章**（cEHVI は本章で扱う）
> - 制約 × Batch → **第12章 + 第14章 capstone**
> - Constraint violation の失敗事例収集 → **第15章**

---

## 10.1 制約付き BO と safe BO の違い

**両者は目的が違う**——混同すると Skill 契約の要件を取り違えます。

| 観点 | 制約付き BO (cEI/cEHVI) | Safe BO (SafeOpt 系) |
|---|---|---|
| **前提** | 制約を評価するには実験が必要（**制約は事後にしか分からない**） | 制約は事前に評価可能、または保証付きで推定できる |
| **違反の扱い** | 探索中に制約違反が発生**しうる**（`P(feasible) < 1` の点を提案しうる） | 探索中に**絶対に制約違反しない** |
| **既知の初期点** | 不要 | **feasible な初期点が必須**（seed set） |
| **Skill 契約** | `feasibility_probability_threshold` で violation を許容範囲に絞る | **hard constraint、違反は fatal** |
| **代表 acquisition** | Expected Feasible Improvement (EFI) / cEI / cEHVI | SafeOpt / StageOpt |
| **用途** | 反応収率などの制約が実験しないと分からない場合 | 装置温度上限など、**絶対に破れない物理・安全制約** |

**Skill 契約の選択規則**：

- **「破っても実験の失敗で済む」制約** → **cEI/cEHVI**（feasibility を surrogate で学習）
- **「破ると装置損傷 / 人身事故 / 法令違反」制約** → **Safe BO + Skill 側の hard filter**

材料実験では**両方が混在**します。反応収率下限 (cEI) と装置温度上限 (safe BO + hard filter) を同時に扱う。

---

## 10.2 「温度上限を超える提案は Skill が絶対に返さない」契約

これが本章の中心的 Agentic 契約です。

### 10.2.1 3 層防御

```
[Layer 1] search_space_bounds の物理境界としての宣言
    └─ x_temperature ∈ [200, 800] は search space の hard bound として BO 内部でも守られる

[Layer 2] constraints_declared の hard/soft 区別
    └─ hard_constraints は Skill の post-filter で強制的に除外
    └─ soft_constraints は cEI/cEHVI で確率的に扱う

[Layer 3] experiment_launch_authorization の Human 承認直前 check
    └─ 提案候補が hard_constraints に違反していれば Skill fatal
    └─ Human が誤って承認しないよう、Skill 側で提示前にブロック
```

### 10.2.2 hard constraint と soft constraint の宣言

第5章 §5.2 の `constraints_declared` を operational 化：

```yaml
constraints_declared:
  version: "v0.1"
  hard_constraints:                     # Skill が絶対に破れない
    - name: "temperature_upper_bound"
      variable: "x_temperature"
      operator: "<="
      threshold: 800.0
      unit: "degC"
      rationale: "装置耐熱上限（メーカー仕様）"
      enforcement: "pre_authorization_filter"   # 提案候補は Skill が返す前にフィルタ
      violation_action: "fatal"                 # enum: fatal | warn | record_only
    - name: "pressure_upper_bound"
      variable: "x_pressure"
      operator: "<="
      threshold: 5.0
      unit: "MPa"
      rationale: "耐圧設計上限"
      enforcement: "pre_authorization_filter"
      violation_action: "fatal"
  soft_constraints:                     # 確率的に扱う、実験で評価が必要
    - name: "yield_lower_bound"
      output_variable: "yield_pct"
      operator: ">="
      threshold: 60.0
      unit: "percent"
      rationale: "採算ライン"
      enforcement: "cEI_feasibility_model"
      feasibility_probability_threshold: 0.5   # P(feasible | x) >= 0.5 の候補のみ許可
      surrogate_ref: "gp_yield"                # feasibility 用 surrogate
      violation_action: "record_only"          # 事後に判明、run 継続
```

### 10.2.3 pre-authorization filter の実装

Skill 実行フロー：

```
1. acquisition_spec に基づき候補 x_cand を最適化 (BoTorch)
     ↓
2. pre_authorization_filter: 全 hard_constraints を評価
     - 違反があれば Skill fatal（violation_event を append）
     - 違反がなければ次へ
     ↓
3. HRD（第7/8章）: 外挿検知
     ↓
4. experiment_launch_authorization: Human に提示
     ↓
5. Human 承認 → 実験実行
```

> [!IMPORTANT]
> **Layer 1（bounds）だけでは不十分**。BoTorch 最適化は bounds を守るが、Skill の後段処理（例：エージェントが候補を書き換えるループ）で違反が生じうる。**Layer 2 の pre-authorization filter は必ず独立に実装**すること（defense in depth）。

### 10.2.4 constraint violation event の記録

```yaml
constraint_violation_events:
  - event_id: "evt-cv-20261121-001"
    iteration: 7
    detected_at: "2026-11-21T10:23:45Z"
    detection_stage: "pre_authorization_filter"   # enum: pre_authorization_filter | post_experiment | audit
    constraint_name: "temperature_upper_bound"
    constraint_type: "hard"                       # enum: hard | soft
    candidate_x:
      x_temperature: 825.0
      x_pressure: 3.2
    threshold: 800.0
    actual: 825.0
    violation_action: "fatal"
    skill_response: "run_aborted"
    root_cause_hypothesis: "acquisition optimizer が bounds を無視、要調査"
    previous_event_hash: "sha256:def456..."
```

append-only、Skill が事後に隠蔽・修正するのは fatal。

---

## 10.3 cEI と cEHVI — feasibility を surrogate で学習

### 10.3.1 cEI (Constrained Expected Improvement)

**基本アイデア**：改善量 (EI) × 実行可能確率 (Feasibility)：

$$
\alpha_{\text{cEI}}(x) = \text{EI}(x) \cdot P(\text{feasible} \mid x)
$$

feasibility は **別の GP** で $g(x) \leq 0$ の確率を推定：

$$
P(\text{feasible} \mid x) = \prod_{k} P(g_k(x) \leq 0 \mid x)
$$

### 10.3.2 BoTorch 実装 — qLogNEI + constraint

BoTorch では `qLogNoisyExpectedImprovement` に `constraints` を渡す：

```python
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.models import ModelListGP

# obj_model: 目的関数の GP
# con_model: 制約 g(x) <= 0 の GP（複数可）
model = ModelListGP(obj_model, con_model_1, con_model_2)

# BoTorch 慣習: constraint は g(x) <= 0 の形（feasible が負側）
def constraint_callable(Z):
    # Z: (..., M+K) — model outputs (M objectives + K constraints)
    # 各 constraint output を返す
    return [Z[..., 1], Z[..., 2]]

acq = qLogNoisyExpectedImprovement(
    model=model,
    X_baseline=X_obs,
    constraints=constraint_callable,
    prune_baseline=True,
)
```

> [!NOTE]
> **符号規約**：BoTorch の constraint は $g(x) \leq 0$ で **feasible が負側**。`yield_lower_bound: yield >= 60` は $g(x) = 60 - \text{yield}(x) \leq 0$ に変換する。`constraints_declared.operator` から自動変換する adapter を Skill 内に持つと良い。

### 10.3.3 cEHVI — 制約付き多目的

多目的 (第9章) + 制約：

```python
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)

acq = qLogNoisyExpectedHypervolumeImprovement(
    model=model,           # ModelListGP with M objectives + K constraints
    ref_point=ref_point,
    X_baseline=X_obs,
    constraints=constraint_callable,
    prune_baseline=True,
)
```

---

## 10.4 Safe BO — SafeOpt / StageOpt

### 10.4.1 SafeOpt の骨格

**保証**：Lipschitz 定数 $L$ が既知の場合、feasible な seed point から、**探索中一度も制約を破らずに** safe expansion set を広げていく。

**要件**：

- **feasible seed set** $S_0$：事前に安全と分かっている点集合（**必須**）
- **Lipschitz 定数** $L$：制約関数の変化率上限（既知 or bounded posterior）
- **信頼区間パラメータ** $\beta_t$：探索の慎重さ

**Skill 契約**：

```yaml
safe_bo_config:
  version: "v0.1"
  enabled: true
  feasible_seed_set:
    - {x_temperature: 300.0, x_pressure: 1.0, verified_at: "2026-11-01T09:00:00Z"}
    - {x_temperature: 450.0, x_pressure: 2.0, verified_at: "2026-11-05T14:30:00Z"}
  lipschitz_bound:
    source: "human_declared"          # enum: human_declared | posterior_derived
    value: 0.15                       # 制約関数の per-unit 変化率上限
    rationale: "予備実験と物理モデルから推定、Human 承認済み"
  beta_t_schedule:
    type: "constant"                  # enum: constant | log_growth
    value: 3.0                        # ~99.7% CI
  expansion_policy: "conservative"    # enum: conservative | balanced
```

### 10.4.2 SafeOpt vs cEI の選択

```
[Q1] 制約を事前評価できる（実験不要）か？
  ├─ YES → search_space_bounds に組み込み（BO 内部で自動的に守られる）
  └─ NO  → Q2

[Q2] 制約違反が装置損傷 / 安全性リスクを伴うか？
  ├─ YES → Safe BO（SafeOpt / StageOpt）+ hard_constraints filter
  └─ NO  → Q3

[Q3] feasible seed point が事前に確保できるか？
  ├─ YES → Safe BO も選択肢
  └─ NO  → cEI/cEHVI（feasibility を surrogate で学習）
```

> [!WARNING]
> **SafeOpt を「安全」と誤解しない**：SafeOpt が保証するのは Lipschitz と β_t の仮定が正しい場合のみ。Lipschitz を過小評価すれば違反が起きる。**物理・法令由来の絶対制約は必ず Layer 2 の hard_constraints filter で二重防護**する。

### 10.4.3 実装ライブラリ

- **BoTorch 単体には SafeOpt はない**。研究コード `SafeOpt` パッケージ（Sui et al.）を使うか、Ax / BoTorch 上に自前実装する
- 材料 BO の実務では、多くの場合 **「hard_constraints filter + cEI」** で十分。SafeOpt を選ぶのは装置リスクが特に大きい場合

---

## 10.5 制約破りの失敗パターン（Ch15 プレビュー）

第15章で完全展開しますが、本章で扱うべき典型パターン：

| # | 失敗パターン | 検知手段 | 予防契約 |
|---|---|---|---|
| 1 | Skill が Layer 1 bounds を守れず違反候補を提案 | pre_authorization_filter | Layer 2 独立実装 |
| 2 | Human が誤って違反候補を承認 | pre_authorization_filter | Skill 側で提示前にブロック |
| 3 | cEI の `feasibility_probability_threshold` を Skill が勝手に緩める | provenance の閾値 immutable 契約 | 変更は fatal / 新 run |
| 4 | Safe BO の Lipschitz を過小に誤設定 | 事後の違反 → root cause | Human 承認 + 予備実験で bounded |
| 5 | soft_constraints を実際は hard で扱うべきだった | 事後の事故報告 | 分類を Human 承認プロセスに |
| 6 | 制約 surrogate の外挿誤用（HRD 不適用） | HRD (Ch7/8) を制約 surrogate にも適用 | 目的関数と同じ HRD 契約 |
| 7 | `constraint_violation_events` の事後改ざん | append-only + `previous_event_hash` | Skill fatal on hash mismatch |

---

## 10.6 Skill 契約チェックリスト

- [ ] `constraints_declared` に hard / soft の分類が明示されている
- [ ] 各 hard_constraint の `enforcement` が `pre_authorization_filter`
- [ ] 各 hard_constraint の `violation_action` が `fatal`
- [ ] soft_constraint に `feasibility_probability_threshold` と `surrogate_ref` が指定されている
- [ ] Layer 1 (bounds) と Layer 2 (filter) が独立に実装されている
- [ ] `constraint_violation_events` が append-only、`previous_event_hash` 付き
- [ ] cEI 使用時、feasibility surrogate にも HRD (Ch7/8) が適用されている
- [ ] Safe BO 使用時、`feasible_seed_set` と `lipschitz_bound.rationale` が Human 承認済み
- [ ] `constraints_declared` の閾値変更は fatal または新 run（immutable within run）
- [ ] acquisition_spec.name が canonical (qLogNEI + constraints / qLogNEHVI + constraints)

---

## 10.7 章末演習

**問 1**：「温度上限 800°C、圧力上限 5 MPa、収率下限 60%」の 3 制約。それぞれ hard / soft のどちらに分類しますか？ 根拠を述べよ。

**問 2**：Skill が iteration 6 で `feasibility_probability_threshold` を 0.5 → 0.3 に緩めたい旨を提案。契約上の正しい応答は？

**問 3**：Safe BO の Lipschitz を Human が 0.15 と宣言した。iteration 4 で実測値が 0.22 になった。Skill はどう振る舞うべきか。

**問 4**：hard_constraints filter を Skill が誤って skip した場合、監査で検出できる仕組みを設計せよ。

**問 5**：制約 surrogate が外挿領域を "feasible" と誤判定するリスク。第7/8章 HRD をどう組み合わせて防ぐか。

---

## 10.8 参考資料

### 本書内

- 第3章 §3.5：5 逸脱パターン（第15章で「制約破り」を追加）
- 第5章 §5.2：`constraints_declared` の初出、`experiment_launch_authorization`
- 第7章 / 第8章：HRD の 3 判定と v0.2 拡張（制約 surrogate にも適用）
- 第9章：多目的の枠組み（cEHVI で継承）
- 第12章（planned）：制約 × Batch
- 第14章（planned）：capstone 統合ケース
- 第15章（planned）：制約破りの失敗パターン完全展開

### 外部参考

- Gardner et al., "Bayesian Optimization with Inequality Constraints", ICML 2014 — cEI 原論文
- Gelbart et al., "Bayesian Optimization with Unknown Constraints", UAI 2014
- Sui et al., "Safe Exploration for Optimization with Gaussian Processes", ICML 2015 — SafeOpt 原論文
- Sui et al., "Stagewise Safe Bayesian Optimization with Gaussian Processes", ICML 2018 — StageOpt
- BoTorch constrained BO tutorials

> [!NOTE]
> 次章（第11章）では、マルチ装置・マルチタスク BO と階層 GP を扱います。ARIM 施設内の複数装置で並行探索する Skill の設計。
