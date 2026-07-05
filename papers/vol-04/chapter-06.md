# 第6章 介入効果の推定を Skill 化する（前半：Propensity / IPW / DR）

> **本章の到達目標**
> - **backdoor 基準を満たす adjustment set** が確定した観測データに対し、**Propensity Score Matching / IPW / Doubly Robust (DR)** で ATE / ATT を推定できる
> - **ML-based estimator**（DR-Learner、DML: Double/Debiased ML）が「なぜ ML を混ぜても identifiability を壊さないか」を Skill 契約レベルで理解する
> - **装置差を confounder として調整する ARIM 実例**——propensity score でどこまで装置バイアスを除去できるか——を Skill として実装できる
> - **Skill としての re-fit ポリシー**——いつ propensity model を再学習してよいか / いつ Human 承認が必要か——を書き下せる
> - **positivity 違反**の検出（extreme weight / propensity clipping）を Skill 契約に組み込める
>
> **本章で扱わないこと**
> - DAG 記法・identification 戦略の選定（第5章）
> - 準実験手法：DiD / IV / Synthetic Control（第7章）
> - CATE / ITE（第8章）
> - refutation・感度分析（第9章）
> - DoE（第10-12章）

---

## 6.1 なぜ ATE / ATT の Skill 化から始めるのか

第5章で DAG と adjustment set を pin しました。次のステップは **「その adjustment set で観測データから ATE / ATT を推定する Skill」** を書くことです。**backdoor 基準を満たす $Z$ が確定しているという前提**が本章の出発点であり、これが破綻していれば以下の推定はすべて意味を持ちません。

**vol-04 の Skill 契約の観点から見ると**、第4章 §4.9 のテンプレートで宣言する `identification_strategy: backdoor` が確定している場面で、`estimator_family` を **Propensity Matching / IPW / DR / DR-Learner / DML** のいずれかに pin する行為がここに相当します。

### 本章の 5 部構成

| 部 | 節 | 主題 |
|---|---|---|
| 第1部 | 6.2 | **Propensity Score の骨格**と **positivity assessment**（第0章 4 識別仮定の運用） |
| 第2部 | 6.3, 6.4 | **Matching** と **IPW**——古典的 estimator の Skill 化 |
| 第3部 | 6.5 | **Doubly Robust (DR)** と **ML-based**（DR-Learner、DML）——ML を混ぜても identifiability を壊さない設計 |
| 第4部 | 6.6 | **装置差を confounder として調整する ARIM ケーススタディ** |
| 第5部 | 6.7 | **Skill としての re-fit ポリシー**と `variable_selection_authorization` の運用 |

---

## 6.2 Propensity Score と positivity assessment

### 6.2.1 Propensity Score の定義

**Propensity Score**：共変量 $X$（＝ adjustment set $Z$）を与えたときの処置確率
$$e(x) = P(T = 1 \mid X = x)$$

**Rosenbaum & Rubin (1983) の定理**：
- **balancing property**：$X \perp T \mid e(X)$——propensity score で条件付ければ、$X$ の分布は処置群と対照群で揃う
- **strong ignorability**（＝ conditional exchangeability + positivity）が成立するなら、$e(X)$ で条件付けるだけで ATE が identifiable

**帰結**：**高次元の $X$ を 1 次元の $e(X)$ に圧縮できる**——これが propensity 系推定の魅力。

### 6.2.2 positivity assumption（第0章の 4 識別仮定の再確認）

**positivity（overlap）**：すべての $x$ で $0 < e(x) < 1$

- **violation の意味**：ある共変量値では**必ず処置される / 必ず対照になる**——**反実仮想が定義できない領域**が存在
- **empirical positivity**：観測データ範囲での $\hat{e}(x)$ が $[\epsilon, 1-\epsilon]$（典型的に $\epsilon = 0.05$）に収まっているか

> [!IMPORTANT]
> 第0章で導入した**4 識別仮定のなかで `checked` を経験的に主張できるのは positivity のみ**——SUTVA / consistency / exchangeability は理論的仮定であり、propensity 図で "見える" のは positivity だけです。**Skill 契約 (`identification_validity`) では `positivity: {status: checked, evidence: propensity_overlap_plot_uri}` を必須**にします（第4章 §4.9 テンプレート）。

### 6.2.3 propensity model の Skill 契約項目

```yaml
propensity_model:
  algorithm: logistic | gbm | random_forest | calibrated_classifier
  library: sklearn | lightgbm | xgboost
  library_version: <string>
  random_seed: <int>
  hyperparameter_tuning:
    method: cv | none
    cv_folds: 5
    metric: brier | log_loss | roc_auc
  calibration:
    method: isotonic | platt | none
    calibration_evidence_uri: <string>       # calibration plot artifact
  features:
    adjustment_set: [device, calibration_state, batch_id, ...]   # 第5章で pin
    non_adjustment_features_prohibited: true                     # DAG 外の変数追加は fatal
  positivity_assessment:
    epsilon: 0.05
    overlap_plot_uri: <string>
    trimming_policy:
      method: none | clip | drop
      clip_bounds: [0.05, 0.95]
      dropped_fraction_max: 0.10               # 超過なら fail-close
```

---

## 6.3 Propensity Score Matching — Skill 化

### 6.3.1 骨格

**1:1 matching**：処置群の各サンプル $i$ に対して、propensity score が最も近い対照群サンプル $j$ を対応付け、$Y_i - Y_j$ の平均で ATT を推定。

- **利点**：直感的、外れ値に頑健、outcome model の misspecification に鈍感
- **欠点**：データ効率が悪い（対照群の多くを捨てる）、matching 距離の選択が主観的

### 6.3.2 Skill 化のポイント

```yaml
estimator: propensity_matching
matching_config:
  ratio: "1:1" | "1:k"
  caliper: 0.2                       # propensity score 差の許容上限
  distance_metric: mahalanobis | logit | absolute
  replacement: true | false
  unmatched_policy: drop | flag
  matching_diagnostics:
    smd_threshold: 0.1               # standardized mean difference
    smd_report_uri: <string>          # matching 前後の共変量バランス
prohibited_actions:
  - increase_caliper_silently        # fatal
  - use_features_outside_adjustment_set   # fatal
  - accept_smd_over_threshold_without_review  # warning + human review
```

### 6.3.3 ARIM 例

- 処置 $T$ = プロトコル A 採用、対照 = プロトコル B、outcome $Y$ = 純度
- adjustment set：装置、較正状態、バッチ ID、環境温湿度（第5章で backdoor 判定済）
- **`ratio: "1:2"`**（対照が多い）、**`caliper: 0.1`**（近接 matching）、**SMD 全共変量で <0.1** を承認基準

---

## 6.4 IPW（Inverse Propensity Weighting）— Skill 化

### 6.4.1 骨格

**Horvitz-Thompson 型推定量**：
$$\widehat{\text{ATE}}_{\text{IPW}} = \frac{1}{n} \sum_i \left[ \frac{T_i Y_i}{\hat{e}(X_i)} - \frac{(1-T_i) Y_i}{1 - \hat{e}(X_i)} \right]$$

**stabilized IPW**（分散低減、実務ではこちらが標準）：
$$\widehat{\text{ATE}}_{\text{sIPW}} = \frac{\sum_i T_i Y_i / \hat{e}(X_i)}{\sum_i T_i / \hat{e}(X_i)} - \frac{\sum_i (1-T_i) Y_i / (1-\hat{e}(X_i))}{\sum_i (1-T_i) / (1-\hat{e}(X_i))}$$

### 6.4.2 IPW の危険：extreme weights

$\hat{e}(x)$ が 0 または 1 に近いサンプルで**重みが爆発**——**分散が発散**し、点推定は不安定に。

**対処**：
- **weight clipping**：$\hat{e}$ を $[\epsilon, 1-\epsilon]$ にクリップ（例：$\epsilon=0.05$）
- **weight truncation**：extreme weight のサンプルを除外
- **overlap trimming**：propensity 分布の非重複領域を事前除外

> [!WARNING]
> **weight clipping / trimming は "positivity 違反への対症療法"** です。**臨む処置は identifiable でない領域を "定義域から外す"** ことに等しく、**推定される estimand は "trimmed population" に限定**されます。**Skill 契約では `estimand_type: ate_trimmed` を明示**し、非トリム集団への外挿を禁止（第4章 Table 4.4 item 12b、`counterfactual_scope_gate` の fail）。

### 6.4.3 Skill 化のポイント

```yaml
estimator: ipw
ipw_config:
  variant: horvitz_thompson | stabilized
  clip_bounds: [0.05, 0.95]
  weight_diagnostics:
    max_weight_report_uri: <string>
    weight_ess_min: 30                # effective sample size 下限
  extreme_weight_policy:
    action: clip | truncate | fail_close
    action_threshold: 0.10             # 10% 超のサンプルが extreme なら fail_close
prohibited_actions:
  - silently_switch_variant                    # HT ↔ stabilized 切替は Human 承認 (fatal)
  - clip_without_recording_clipped_fraction   # fatal
  - use_effective_sample_size_below_min       # fatal
estimand_type: ate | ate_trimmed              # trimming したら trimmed 明示
```

---

## 6.5 Doubly Robust (DR) と ML-based estimator

### 6.5.1 DR estimator の骨格

**Doubly Robust** は **outcome model $\hat{\mu}(X, T)$** と **propensity model $\hat{e}(X)$** の**両方**を使い、**どちらか一方が正しければ ATE が consistent** な性質を持つ推定量：

$$\widehat{\text{ATE}}_{\text{DR}} = \frac{1}{n} \sum_i \left[ \hat{\mu}(X_i, 1) - \hat{\mu}(X_i, 0) + \frac{T_i (Y_i - \hat{\mu}(X_i, 1))}{\hat{e}(X_i)} - \frac{(1-T_i)(Y_i - \hat{\mu}(X_i, 0))}{1 - \hat{e}(X_i)} \right]$$

**双方正しければ**：$\sqrt{n}$-consistent + 標準的な CI

### 6.5.2 なぜ ML を混ぜても identifiability を壊さないか（DML の骨格）

**Double/Debiased ML (DML; Chernozhukov et al., 2018)** の中核：

1. **cross-fitting**：データを $K$-fold に分け、$\hat{\mu}$ と $\hat{e}$ を "他 fold" で学習——**overfitting による偏り**を除去
2. **orthogonal moment**：DR スコアは $\hat{\mu}, \hat{e}$ の推定誤差に対して 1 次で orthogonal——**ML の遅い収束率**（$n^{-1/4}$ 程度）でも ATE は $n^{-1/2}$ で consistent

**帰結**：ML（GBM、NN、causal forest 等）を nuisance model として使いつつ、ATE の識別性と CI の妥当性を保てる——**identifiability は DAG と backdoor 基準から来ており、estimator の ML 化は "nuisance の柔軟化" にすぎない**。

> [!IMPORTANT]
> **DML が identifiability を壊さない前提**：
> 1. adjustment set $Z$ が第5章で **backdoor を満たすと承認済み**（`variable_selection_authorization` 経由）
> 2. cross-fitting が**正しく実装**されている（同じ fold で学習と評価をしない）
> 3. positivity が守られている
>
> これらは **Skill 契約 (`identification_validity`, `cross_fitting_provenance`) で pin**——エージェントが「ML でよく当てはまるから adjustment set を追加した」等は**silent modification として fatal**（第4章 Table 4.4 item 3）。

### 6.5.3 DR-Learner との違い

| 用途 | estimator | 推定対象 | 主要ライブラリ |
|---|---|---|---|
| **ATE**（集団平均） | DR / DML | scalar effect | DoWhy, EconML (`LinearDML`), sklearn |
| **CATE**（$X$ で変化する効果） | **DR-Learner**（Ch8） | function of $X$ | EconML (`DRLearner`) |

**本章は ATE 中心**——DR-Learner の詳細は Ch8。ここでは「同じ DR スコアを、平均するか、$X$ に対する回帰にするか」が両者の違いと押さえておきます。

### 6.5.4 Skill 化のポイント

```yaml
estimator: dr | dml
dr_dml_config:
  outcome_model:
    algorithm: linear | gbm | rf | neural
    library: sklearn | lightgbm | xgboost | pytorch
    library_version: <string>
  propensity_model:
    (§6.2.3 と同じ)
  cross_fitting:
    n_folds: 5
    random_seed: <int>
    fold_assignment_uri: <string>      # fold ID の artifact
    library: econml | sklearn
  orthogonality_check:
    method: theoretical | empirical
    residual_correlation_bound: 0.05   # empirical の場合
prohibited_actions:
  - skip_cross_fitting                          # fatal（overfitting → biased ATE）
  - train_and_evaluate_on_same_fold             # fatal
  - use_untuned_ml_without_hyperparameter_provenance  # warning
identification_validity:
  strategy: backdoor
  adjustment_set: [...]                # 第5章 dag_of_record で承認済み
  positivity:
    status: checked
    evidence_uri: <string>
```

### 6.5.5 estimator 選択の判断表

| データの性質 | 推奨 estimator | 理由 |
|---|---|---|
| $Z$ が低次元、線形性が仮定できる | **IPW** or 回帰 | シンプル、解釈しやすい |
| $Z$ が高次元、outcome の非線形性が疑われる | **DR / DML** | ML nuisance でも $\sqrt{n}$-consistent |
| positivity が疑わしい | **Matching**（overlap 明示）or trimmed IPW | extreme weight 回避 |
| バランス評価を強調したい | **Propensity Matching** | SMD で共変量バランスを可視化 |
| 個別効果（CATE）が欲しい | **DR-Learner**（Ch8） | scalar ではなく関数を推定 |

---

## 6.6 ケーススタディ — 装置差を confounder として調整する

### 6.6.1 設定

**シナリオ**：ARIM 施設に装置 A / B / C の 3 台があり、それぞれ較正プロトコル・保守履歴が異なる。あるプロトコル $T$ の効果を評価したいが、**装置選択自体がユーザーの経験・研究方針で内生的**——装置が confounder。

**DAG**（第5章 §5.2.1 の confounder パターン）：
- $C = \text{装置 ID}$、$T = \text{プロトコル採否}$、$Y = \text{生成物特性}$
- $C \to T$、$C \to Y$、$T \to Y$
- adjustment set：$\{C, \text{較正状態}, \text{保守履歴期間}\}$（第5章 dag_of_record で承認済み）

### 6.6.2 実装フロー（DoWhy + EconML）

```python
# 疑似コード（Skill 内部の estimator）
from dowhy import CausalModel
from econml.dml import LinearDML

# 1. DAG は dag_of_record から読み込み（Skill 内で書き換え禁止）
model = CausalModel(
    data=df,
    treatment="protocol",
    outcome="purity",
    graph=dag_of_record_uri,     # 第5章の承認済み DAG
)

# 2. Identification（backdoor 承認済み adjustment set が確定）
identified_estimand = model.identify_effect(proceed_when_unidentifiable=False)

# 3. Positivity assessment（Skill 契約の identification_validity.positivity.checked）
propensity_plot(df, treatment="protocol", covariates=adjustment_set)

# 4. DML 推定（cross-fitting=5）
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dml.LinearDML",
    method_params={"init_params": {"cv": 5, "random_state": SEED}},
)
```

### 6.6.3 監査ポイント

- **装置 A / B / C の overlap プロット**を artifact 化——装置 C だけプロトコル採用サンプルが極端に少なければ**positivity violation** の可能性
- **`positivity_by_stratum`** を Skill 出力に含める——全体では positivity 満たしても、装置別で violated なら "装置横断の ATE" は identifiable ではない
- **`identification_report_uri`** に adjustment set の根拠（第5章 §5.6.2 の approval evidence）を明記

### 6.6.4 装置差以外の confounder が入ってきたら

エージェントが観測データを走査中に**新しい confounder 候補**（例：オペレータ ID）を発見した場合の Skill 契約：

- **単独で adjustment set に追加してはならない**（fatal、Table 4.4 item 3）
- **`dag_amendment_proposal`** artifact を生成し、**`dag_authorization` ゲートを再度通す**（第5章 §5.6.2）
- 承認されるまで**現行 dag_of_record での推定を継続**（silent 切替は監査違反）

---

## 6.7 Skill としての re-fit ポリシー

**re-fit** = propensity model / outcome model を新しいデータで再学習すること。これを**エージェント自律で行ってよい範囲**を Skill 契約で定めます。

### 6.7.1 4 段階の re-fit 権限

| Level | 状況 | 権限 | 発火するゲート |
|---|---|---|---|
| L1（自律） | 同一 adjustment set・同一データ分布・追加観測 | エージェント自律 | なし |
| L2（自律+ログ） | 同一 adjustment set、新サンプル追加、分布ドリフト検出時の再学習 | エージェント自律だが `refit_log_uri` 出力必須 | なし |
| L3（Human 通知） | ハイパーパラメータ再チューニング、cross-fitting fold 再分割 | エージェント実行 + Human 通知 | なし（監査で検知） |
| L4（Human 承認） | **adjustment set の変更、propensity 変数の追加/削除、algorithm family 変更** | Human 承認必須 | **`variable_selection_authorization`** |

### 6.7.2 分布ドリフト検出（L2）

**エージェントが自律で再学習してよい "drift" の定義**：

```yaml
distribution_drift_policy:
  drift_metric: psi | ks | js_divergence   # Population Stability Index 等
  drift_threshold: 0.2                     # 越えたら refit
  drift_scope:
    features: [device, calibration_state]   # adjustment_set の subset のみ監視
    outcome_drift_monitored: true
  refit_action:
    method: warm_restart | full_retrain
    refit_provenance_uri: <string>          # 前後モデルの hash + drift report
```

### 6.7.3 禁止事項

```yaml
prohibited_actions:
  # Level 判定違反
  - refit_at_l4_without_human_approval          # fatal
  - add_features_to_propensity_model_at_l1_l3   # fatal（adjustment set は §5.6 で pin）
  - drop_features_from_adjustment_set_silently  # fatal
  # ハイパーパラメータの silent 変更
  - change_algorithm_family_at_l1_l2            # fatal
  - change_random_seed_silently                 # fatal（reproducibility 破壊）
  # ドリフト対処
  - refit_without_drift_report                  # warning + human review
  - refit_when_drift_exceeds_threshold_without_dag_review  # fatal（分布が大きく変わったなら DAG 再検討が必要）
```

### 6.7.4 re-fit と `variable_selection_authorization` の関係

**エージェントが propensity model の feature 追加を提案する** = **adjustment set の変更** = **第5章 §5.6 の DAG レベルの意思決定**。

したがって：
- **`variable_selection_authorization` ゲート**（第4章 §4.6）を経由
- **DAG が変わるなら `dag_authorization`** も経由（第5章 §5.6）
- **どちらも承認されるまで、現行モデルの結論のみを出力**——silent な feature 追加は**因果 provenance の L2（identification assumptions）レイヤの破壊**（第4章 §4.4）

> [!IMPORTANT]
> **re-fit は "同じ Skill が同じ契約の下で動き続ける" ことを保証する仕組み**——契約自体（adjustment set / algorithm family / seed policy）を変えたら、**それは新しい Skill バージョン**であり、承認と provenance を再度取り直す必要があります。

---

## まとめ — 本章のチェックリスト

- [ ] **§6.2 Propensity Score の balancing property と positivity**——第0章 4 識別仮定の "checked" が propensity で成立する理由を説明できる
- [ ] **§6.3-6.4 Matching と IPW の Skill 契約**——caliper / clip bounds / SMD threshold / extreme weight policy を書き下せる
- [ ] **§6.5 DR / DML** の cross-fitting と orthogonal moment の意味——「ML を混ぜても identifiability を壊さない」条件を 3 つ列挙できる
- [ ] **§6.5.5 estimator 選択の判断表**を、自分のデータ特性に当てはめて選択できる
- [ ] **§6.6 装置差 ARIM ケース**——overlap plot + `positivity_by_stratum` の運用を Skill に組み込める
- [ ] **§6.7 4 段階 re-fit ポリシー**（L1-L4）とゲートの対応を暗記した
- [ ] **§6.7.3 禁止事項**——silent な feature 追加 / algorithm 変更 / seed 変更が fatal である理由を説明できる

---

## 章末演習

### 演習 6.1：propensity model の Skill 契約を書く

演習 5.1 で描いた自分の DAG から backdoor adjustment set を 1 つ選び、§6.2.3 のテンプレートを完全に埋めた **propensity model Skill 契約** を作成してください。

必須項目：
- [ ] algorithm、library、library_version、random_seed
- [ ] adjustment_set（第5章 approval 済のもの）
- [ ] positivity_assessment.overlap_plot_uri（合成データで実際にプロット）
- [ ] trimming_policy.dropped_fraction_max

### 演習 6.2：IPW vs DR / DML の選択

以下のシナリオそれぞれで、**どの estimator を使うか**を根拠付きで答えてください：

1. adjustment set が 3 変数（すべて low-cardinality categorical）、$n = 500$、outcome は線形にモデル化できそう
2. adjustment set が 20 変数（高次元）、$n = 5000$、outcome は非線形（分光ピーク由来の特徴量を含む）
3. 装置 C のプロトコル採用サンプルが 3% しかない
4. individual-level 効果（CATE）が欲しい

（解答は Ch7-8 で照合）

### 演習 6.3：re-fit ポリシーのシナリオ判定

自分の Skill 運用で以下が発生したとき、**L1-L4 のどの権限か**、**どのゲートを発火させるか**を答えてください：

1. 新しい月次データが 300 件追加された（分布ドリフトなし）
2. PSI が 0.15 → 0.25 に上昇（ドリフト閾値超過）
3. 装置 D が新設され、adjustment set に「装置 ID」を加えたい
4. propensity model を logistic → GBM に変更したい
5. random_seed を 42 → 2026 に変更したい

### 演習 6.4：装置差調整の限界を見る

演習用合成データ（付録 C）で、装置 C のプロトコル採用比率を 5% / 10% / 30% と変えたときの ATE 推定：

- [ ] propensity overlap plot の変化を可視化
- [ ] IPW（clipping なし / あり）と DML の点推定・SE を比較
- [ ] **`positivity_by_stratum` の判定**：装置 C 単独では positivity 違反、全体では OK な場面を再現
- [ ] Skill 契約上の estimand_type（`ate` vs `ate_trimmed`）の使い分けを説明

---

## 参考資料

### 内部 cross-reference

- 第0章：4 識別仮定（positivity / SUTVA / consistency / exchangeability）、`checked` は positivity のみ
- 第2章：装置差・オペレータ差の confounder 化（Pattern 2 との対比）
- 第3章：DoWhy / EconML / linearmodels の位置づけ
- **第4章**：`identification_validity`、`variable_selection_authorization`、Table 4.4 禁止事項、`counterfactual_scope_gate`
- **第5章**：backdoor 基準、DAG proposal/approval skill 分離、`dag_of_record`
- 第7章：DiD / IV / Synthetic Control（本章と対をなす準実験手法）
- 第8章：DR-Learner による CATE（本章の DR を関数化）
- 第9章：refutation / 感度分析（本章の推定値を "検算" する）
- **第13章 (13a Phase 1-2)**：本章と Ch5, Ch8 を統合して観測データ → DAG 仮説 → CATE 推定を実装
- 付録 A：ATE / ATT 推定 Skill テンプレート集
- 付録 D：propensity score / IPW / DR / DML / positivity の用語

### 外部参考文献

- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55.
- Chernozhukov, V., Chetverikov, D., Demirer, M., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal*, 21(1), C1-C68.
- Kennedy, E. H. (2016). Semiparametric theory and empirical processes in causal inference. In *Statistical Causal Inferences and Their Applications in Public Health Research*.
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
- DoWhy documentation: https://www.pywhy.org/dowhy/
- EconML documentation: https://econml.azurewebsites.net/

---

**本章の位置づけ**：backdoor 基準が使える場面での ATE / ATT 推定 Skill が揃いました。次章（Ch7）では、**backdoor が使えない場面**——未観測交絡・時間軸の前後・閾値ルール——に対する準実験手法（DiD / IV / Synthetic Control）を Skill 化していきます。
