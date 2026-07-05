# 第4章 因果 × Agentic Skill の設計原則

> **本章の到達目標**
> - vol-01 第7章「Skill 設計 6 要素」と **vol-02 第4章（統計/ML 版拡張）・vol-03 第4章（深層 × Agentic 版拡張）** を、**因果推論 × 実験計画 × Agentic** に拡張した設計原則を理解する
> - Skill 仕様書に **4 つの問い**——「何を identification 戦略とみなすか」「どの DAG で再現するか」「反実仮想の外挿範囲はどこまで許すか」「**エージェントに何を許すか**」——を書き下せる
> - **介入承認ゲートの 3 層分解**（`dag_authorization` / `variable_selection_authorization` / `intervention_execution_authorization`）を Skill 契約に組み込める
> - **`counterfactual_scope_gate`** の契約項目としての定義（vol-04 全体の source of truth）と、Ch8-9 での operational 判定・Ch11 での応用への橋渡しを理解する
> - **E-value による感度分析の provenance** を Skill 契約に含める
> - **因果 × Agentic Skill 仕様書テンプレート**と **3 層承認契約**を、本章の成果物として持ち帰る
>
> **本章で扱わないこと**
> - DAG 記法・SCM の形式的定義（第5章）
> - 具体的な estimator の実装（第6-8章）
> - refutation ツールの実装（第9章）
> - DoE の設計行列生成（第10-11章）
> - `sequential_experiment_stop_condition`（vol-05 の逐次実験計画へ移譲）

---

## 4.1 なぜ「設計」から始めるのか（因果 × Agentic 版の動機）

vol-03 第4章の設計原則では、**「何を成功とみなすか」＋「どの環境で再現するか」＋「エージェントに何を許すか」** の 3 問を Skill 仕様書に書き下しました。**因果推論 × 実験計画 × Agentic では、これに 1 問が加わり、既存 3 問も再解釈**されます。

**vol-04 の 4 つの問い**：

| # | 問い | 中心となる論点 | 対応節 |
|---|---|---|---|
| 1 | **何を identification 戦略とみなすか** | backdoor / frontdoor / IV / DiD / RDD / Synthetic Control のいずれかを Skill に pin、silent 切替を禁止 | §4.2, §4.3 |
| 2 | **どの DAG で再現するか** | `causal_graph_uri` + `causal_graph_sha256` の pin、confounder / mediator / collider の宣言 | §4.4 |
| 3 | **反実仮想の外挿範囲はどこまで許すか** | `counterfactual_scope_gate` の契約定義（Mahalanobis 距離 + CATE 予測分散） | §4.5 |
| 4 | **エージェントに何を許すか** | 3 層承認ゲート（DAG 変更 / 変数選択 / 介入実行）＋ E-value provenance | §4.6, §4.7 |

**vol-02 / vol-03 との対応**：

- vol-02 第4章の「成功条件 3 点セット」を、因果 × DoE 用に **identification validity / refutation pass / positivity check / external validity / DoE efficiency / randomization integrity** に拡張（§4.3）
- vol-03 第4章の「Agentic 学習権限 3 段階」を、**因果的判断の 3 層権限**——観測データからの効果推定は自律 / DAG 構造変更と変数選択は準自律 / 実際の介入実行は必ず Human 承認——に再解釈（§4.6）

---

## 4.2 6 要素の因果 × Agentic 拡張

vol-01 第7章の Skill 設計 6 要素（**① 目的 / ② 入力条件 / ③ 出力形式 / ④ 成功条件 / ⑤ 禁止事項 / ⑥ 再現性条件**）を、因果 × Agentic 用に拡張します。

### Table 4.1：6 要素の因果 × Agentic 拡張

| 要素 | vol-01 標準 | vol-02 拡張 | vol-03 拡張 | **vol-04 拡張（新規）** |
|---|---|---|---|---|
| ① 目的 | 何を計算するか | 統計/ML の推定量 | 深層モデルの推論 | **estimand（total / direct / indirect effect）** と causal question Type（第1章）を明示 |
| ② 入力条件 | データ範囲 | 前処理契約 | 深層学習向け augmentation 契約 | **`causal_graph_uri` / `confounders_declared` / `mediators_declared` / `colliders_declared` の宣言** |
| ③ 出力形式 | 数値・図表 | 統計モデルの推定結果 | モデル重みと予測 | **`identification_strategy` / `estimate + 信頼区間` / `refutation_results` / `sensitivity_analysis`** |
| ④ 成功条件 | 数値の正しさ | 統計的有意性・CV score | 汎化性能 | **§4.3 に詳述**（identification validity / refutation pass / positivity / external validity / DoE efficiency / randomization integrity） |
| ⑤ 禁止事項 | 過学習・データリーク | 事後の仮説改変 | 未承認 fine-tune | **§4.8 に詳述**（silent identification switch / unauthorized DAG modification / adjustment of collider / refutation skip / unauthorized intervention execution） |
| ⑥ 再現性条件 | seed 固定 | environment.lock | container SHA + GPU 型番 | **§4.4 に詳述**（`causal_graph_sha256` / `identification_strategy` / `randomization_seed` / `estimand_type` の pin） |

**新規契約フィールド**（§4.9 テンプレート参照）：

- `identification_strategy`：backdoor / frontdoor / IV / DiD / RDD / Synthetic Control のいずれか
- `causal_graph_uri` / `causal_graph_sha256`：DAG artifact の URI と SHA256
- `confounders_declared` / `mediators_declared` / `colliders_declared`：変数役割の宣言リスト
- `estimand_type`：total_effect / direct_effect / indirect_effect
- `refutation_tests_required`：DoWhy の正式 API 名リスト（`placebo_treatment_refuter` 等、第3章 §3.7）
- `positivity_check`：assessed 段階の記録
- `sutva_declared` / `consistency_declared` / `exchangeability_declared`：4 識別仮定の declared / assessed（第0章 §0.6）
- `sensitivity_analysis`：E-value / Rosenbaum bounds の値と閾値
- `counterfactual_scope_gate`：§4.5 で定義
- `authorization_gates`：3 層承認（§4.6）
- `experimental_design_provenance`：DoE 章向け（第10-12章で使用）
  - `randomization_seed`：擬似乱数 seed
  - `blocking_factors`：blocking の因子リスト
  - `design_type`：full_factorial / fractional_factorial / central_composite / box_behnken / latin_hypercube / plackett_burman / orthogonal_array

---

## 4.3 成功条件の 6 点セット — 因果 × DoE 拡張

vol-02 第4章の「成功条件 3 点セット（統計的有意 / CV score / holdout）」を、因果推論 × DoE 用に **6 点セット**に拡張します。

### Table 4.2：因果 × DoE 成功条件 6 点セット

| # | 成功条件 | 定義 | Skill 契約フィールド | 該当章 |
|---|---|---|---|---|
| 1 | **identification validity** | 与えられた DAG と変数役割で、目標 estimand が **identifiable** である | `identification_strategy` + `causal_graph_sha256` + DoWhy identifiability check | 第5-7章 |
| 2 | **refutation pass** | DoWhy の refutation 群（placebo, random common cause, data subset 等）を **全て pass** する | `refutation_tests_required` + `refutation_results` | 第9章 |
| 3 | **positivity check** | Treatment の割当が全 covariate strata で正の確率を持つ（連続 treatment では generalized propensity の common support） | `positivity_check` | 第2章 §2.4, 第6章 |
| 4 | **external validity** | 反実仮想が学習分布の外挿にならない | `counterfactual_scope_gate` | §4.5, 第8-9章 |
| 5 | **DoE efficiency**（DoE 章のみ） | 情報量あたりのコストが閾値内 | `doe_efficiency_metric` + `information_gain_threshold` | 第10-12章 |
| 6 | **randomization integrity**（DoE 章のみ） | randomization seed が pin されており、blocking が破綻していない | `randomization_seed` + `blocking_factors` | 第10章 |

**「全条件 AND」で成功**：因果推論 Skill は 1-4、DoE Skill は 5-6 も追加。1 つでも fail なら Skill は数値を返さず、Human に stop condition を返す（第2章 §2.4 IMPORTANT の「無理な問いには No」）。

---

## 4.4 どの DAG で再現するか — 因果 provenance の 3 レイヤ

vol-03 第4章の「GPU/Agentic provenance の 3 レイヤ」を、因果推論用に再構成します。

### Table 4.3：因果 provenance の 3 レイヤ

| レイヤ | 内容 | pin 対象 | 検証方法 |
|---|---|---|---|
| **L1: 識別レイヤ** | DAG 構造、変数役割、identification 戦略 | `causal_graph_uri`, `causal_graph_sha256`, `identification_strategy`, `confounders_declared`, `mediators_declared`, `colliders_declared`, `estimand_type` | DoWhy identifiability check、DAG SHA256 照合 |
| **L2: 識別仮定レイヤ** | positivity / SUTVA / consistency / exchangeability の declared / assessed 状態（第0章 §0.6） | `positivity_check`（**checked**）、`sutva_declared` / `consistency_declared` / `exchangeability_declared`（**assessed**） | 各仮定に対する数値・グラフ・ドメイン知識に基づく評価レポート |
| **L3: 実行レイヤ** | ライブラリバージョン、seed、乱数状態、コンテナ | `library_stack`（第3章 §3.7）、`environment.lock`（vol-02 資産）、`random_seed`、`container_sha256`（vol-03 資産） | pip freeze、docker inspect、seed reproducibility test |

**3 レイヤの独立性**：

- L1 が変わる → **新 Skill バージョン**（`dag_authorization` 経由、§4.6）
- L2 が変わる → **assessed の更新のみで OK**（vol-04 では `checked` は positivity のみ、他は declared のまま新エビデンスで assessed 更新）
- L3 が変わる → **新 Skill バージョン + 再現性テスト**（vol-02/03 と同様）

### DAG artifact の保存形式

`causal_graph_uri` は以下いずれかを指す：

- **DOT 形式**（GraphViz、第2章の演習データで採用）：人間可読、networkx 経由で DoWhy に入る
- **networkx pickle**：Python 完結、CI 内での再構築が高速
- **JSON**（{nodes, edges} 形式）：言語非依存、外部ツール（Neo4j 等）と連携する場合

**必ず SHA256 を併記**（`causal_graph_sha256`）——DAG artifact のバイト単位の変化を検知します。

---

## 4.5 反実仮想の外挿範囲 — `counterfactual_scope_gate` の契約定義

**本節は `counterfactual_scope_gate` の source of truth**です。Ch8-9 で operational 判定を実装、Ch11 で SMT 応答曲面に適用します。

### 4.5.1 なぜ scope gate が必要か

反実仮想（"もし装置を A ではなく B にしていたら Y は何になっていたか"）は、**学習分布の外側**の値を要求する場面が多発します。予測 Skill なら「外挿は精度が落ちる」で済みますが、因果 Skill では **「外挿は識別を無効化しうる」**——`consistency` 仮定が破れる可能性があるからです（第0章 §0.6、第2章 §2.4）。

### 4.5.2 契約フィールドの完全定義

```yaml
counterfactual_scope_gate:
  # --- 距離ベース判定 ---
  distance_metric: mahalanobis_distance   # or embedding_distance (画像/スペクトル)
  distance_threshold: 3.0                  # 反実仮想点と学習分布重心の距離上限
  neighbor_density_metric: k_nn_density    # k=10 近傍密度
  neighbor_density_threshold: 0.1          # 近傍密度の下限

  # --- 不確かさベース判定（CATE 予測分散） ---
  cate_variance_metric: prediction_variance  # EconML/BNN の予測分散
  cate_variance_threshold: 0.25              # 予測分散の上限

  # --- 判定ロジック ---
  gate_logic: any_fail_triggers_review     # 距離 or 密度 or 分散 のいずれか fail で review

  # --- fallback ---
  fallback: human_review                    # ゲート不通過時の挙動
  fallback_message_template: |
    Counterfactual point x' = {value} is out of scope.
    - distance = {distance} (threshold {distance_threshold})
    - neighbor density = {density} (threshold {neighbor_density_threshold})
    - CATE variance = {variance} (threshold {cate_variance_threshold})
```

### 4.5.3 データ型別の distance_metric 選択

| データ型 | 推奨 distance_metric | 補足 |
|---|---|---|
| 表形式（連続共変量） | mahalanobis_distance | 共分散を考慮した距離 |
| 表形式（カテゴリ共変量） | mixed_gower_distance | カテゴリ + 連続の混合 |
| スペクトル・時系列 | embedding_distance（vol-03 第7章の深層特徴） | 潜在空間での距離 |
| 画像・回折 | embedding_distance | 同上 |
| マルチモーダル | concatenated_embedding_distance | 各モーダルの embedding を結合 |

**閾値の初期値の決め方**：学習データの内部で **95 パーセンタイル**を計算し、それを threshold の初期値とします（第8章で校正手順）。

### 4.5.4 SMT 応答曲面への適用（Ch11 予告）

SMT の Kriging surrogate は外挿に弱いため（第3章 §3.3）、応答曲面の予測点が **`counterfactual_scope_gate` を通過**しない場合は Skill は最適条件を返しません。Ch11 で具体実装。

---

## 4.6 3 層承認ゲートの完全仕様（新設節）

**vol-04 の中核設計**です。vol-03 第4章の「Agentic 学習権限 3 段階」を、因果的判断向けに **役割分離**します。

### 4.6.1 3 層承認ゲートの位置づけ

| ゲート | 承認対象 | 承認者（推奨） | 反応時間目安 |
|---|---|---|---|
| **`dag_authorization`** | DAG 構造の変更、identification 戦略の切替 | Research lead（PI or 上級研究員） | 数時間〜1 日 |
| **`variable_selection_authorization`** | confounder / mediator / collider の判定変更、DoE の design parameter 変更（DiD window、IV 候補、RDD cutoff、blocking factor 等） | Research lead | 数時間 |
| **`intervention_execution_authorization`** | **実際の実験装置を動かす行為** | PI + Facility manager | 実験前の事前承認 |

**なぜ 3 層か**：

- **DAG は identification の根本**——変更すれば別の因果的問いになる（第2章 §2.5 パターン 2, 4）
- **変数役割は identification の妥当性**——confounder を collider に間違えれば bias 混入（第2章 §2.5 パターン 1, 3）
- **介入実行は現実世界への影響**——シミュレーションと違い revocable ではない

### 4.6.2 ゲートの発火条件（Skill 契約）

```yaml
authorization_gates:
  dag_authorization:
    required_for:
      - causal_graph_uri_change        # DAG artifact の変更
      - causal_graph_sha256_mismatch   # SHA 不一致
      - identification_strategy_change # backdoor→IV 等
    approver: research_lead
    provenance_field: dag_approved_by
    approval_evidence:
      - approver_signature
      - approval_timestamp
      - approval_scope                 # どの変更を承認したか

  variable_selection_authorization:
    required_for:
      - confounders_declared_change
      - mediators_declared_change
      - colliders_declared_change
      - design_parameter_change        # DiD window / IV / RDD / donor pool / blocking factor
    approver: research_lead
    provenance_field: variables_approved_by
    approval_evidence:
      - approver_signature
      - approval_timestamp
      - approval_scope

  intervention_execution_authorization:
    required_for:
      - physical_experiment_execution
    approver: pi_and_facility_manager
    provenance_field: intervention_approved_by
    approval_evidence:
      - pi_signature
      - facility_manager_signature
      - approval_timestamp
      - approved_experiment_id
      - safety_review_id               # 施設安全審査（該当あれば）
```

### 4.6.3 エージェントの自律範囲（第3章 §3.5 の再確認）

| 行為 | 権限 | 発火するゲート |
|---|---|---|
| 承認済み DAG での ATE 推定 | 自律 | なし |
| 承認済み confounder set での CATE 推定 | 自律 | なし |
| refutation の実行 | 自律 | なし |
| 感度分析（E-value 計算） | 自律 | なし |
| 反実仮想計算（scope 内） | 自律 | `counterfactual_scope_gate` 自動判定 |
| 反実仮想計算（scope 外） | Human review | `counterfactual_scope_gate` fallback |
| DAG の変更 | Human 承認 | `dag_authorization` |
| identification 戦略の切替 | Human 承認 | `dag_authorization` |
| confounder / mediator / collider の変更 | Human 承認 | `variable_selection_authorization` |
| DoE の design parameter 変更 | Human 承認 | `variable_selection_authorization` |
| **実際の実験実行** | **Human のみ** | **`intervention_execution_authorization`** |

### 4.6.4 ゲート違反時の挙動

Skill は以下を必須で実装します：

1. **fail-close**：ゲート違反を検知したら **数値を返さない**
2. **provenance 記録**：違反内容・時刻・呼び出し文脈を artifact に記録
3. **Human 通知**：`fallback_message_template` に沿ったメッセージ生成
4. **監査ログ**：全ゲート判定を tamper-evident なログに append（vol-03 第4章の Agentic provenance と統合）

---

## 4.7 E-value による感度分析の provenance

**識別仮定の一つ（exchangeability）は原理的に検証不可能**——unmeasured confounder の存在は否定できません。vol-04 では **E-value を Skill 契約に必須項目**として組み込みます。

### 4.7.1 E-value とは（要点のみ、詳細は第9章）

**E-value（VanderWeele & Ding, 2017）**：観測された効果を打ち消すのに必要な unmeasured confounder の強度。値が大きいほど、「別の説明が要る強い効果」と解釈できます。

- E-value = 1 → unmeasured confounder が RR = 1.0 でも効果を打ち消せる（robust ではない）
- E-value = 2 → unmeasured confounder が treatment・outcome の両方に対して RR = 2 の強度で関連する必要がある
- E-value = 5 → 打ち消しにはかなり強い confounder が要る

### 4.7.2 Skill 契約への組み込み

```yaml
sensitivity_analysis:
  method: e_value                          # or rosenbaum_bounds
  compute_for:
    - point_estimate                       # 点推定の E-value
    - lower_confidence_bound               # 信頼区間下限の E-value
  threshold:
    minimum_e_value: 1.5                   # この閾値未満ならフラグ
  provenance_field: sensitivity_report_uri  # 感度分析レポート artifact
  fail_action: flag_result_as_low_robustness
```

**E-value と `refutation_tests_required` の関係**：

- `refutation_tests_required` は **DoWhy の refutation 群**（第9章）——**別の側面の頑健性**（placebo、random common cause 等）
- `sensitivity_analysis`（E-value）は **unmeasured confounder に対する頑健性**
- **両者は独立**——両方 pass しないと Skill は結果を "high confidence" として返しません

---

## 4.8 因果 × Agentic Skill の禁止事項（severity 付き）

vol-03 第4章の禁止事項リストに、因果推論 × DoE 特有の項目を追加します。

### Table 4.4：禁止事項

| # | 禁止事項 | Severity | 検知フィールド | 対応する Ch2 パターン |
|---|---|---|---|---|
| 1 | Silent identification strategy switch（Skill 内部で backdoor→IV 等） | **fatal** | `identification_strategy` の差分 | パターン 4 |
| 2 | Unauthorized DAG modification | **fatal** | `causal_graph_sha256` の差分 | パターン 2 |
| 3 | Adjustment of collider without authorization | **fatal** | `confounders_declared` と DAG の整合 | パターン 1 |
| 4 | Mediator を confounder に格上げ | **fatal** | `estimand_type` と DAG の整合 | パターン 3 |
| 5 | Refutation skip | fatal | `refutation_tests_required` vs `refutation_results` | — |
| 6 | Positivity 違反を無視して estimation | fatal | `positivity_check` の状態 | — |
| 7 | `counterfactual_scope_gate` の閾値を Skill 内部で変更 | fatal | 契約 vs 実行時パラメータの差分 | — |
| 8 | E-value 計算のスキップ | warning | `sensitivity_analysis` の存在 | — |
| 9 | Unauthorized intervention execution（承認なしで実験装置を動かす） | **fatal** | `intervention_approved_by` の欠如 | — |
| 10 | Randomization seed の上書き（DoE 章） | fatal | `randomization_seed` の差分 | — |
| 11 | DoE の blocking 破綻 | warning | `blocking_factors` と実行データの整合 | — |
| 12 | SMT 応答曲面の外挿（scope gate 逸脱） | warning | `counterfactual_scope_gate` 判定 | — |

**Severity の意味**：

- **fatal**：Skill 停止、Human 介入必須、監査対象
- **warning**：Skill は数値を返すが `low_confidence` フラグ付き、Human review 推奨
- **flag**：数値を返すが provenance に記録して次回改善（vol-04 では未使用）

---

## 4.9 因果 × Agentic Skill 仕様書テンプレート（**成果物 1**）

以下は本章の**主要成果物**——因果 × Agentic Skill の最小契約テンプレートです。第3章 §3.7 の先取り紹介を **完全版に拡張**しました。

```yaml
skill:
  name: <skill_name>
  version: <semver>
  purpose: |
    <estimand と causal question Type を明示>

  # === ① 目的 ===
  estimand_type: total_effect        # or direct_effect / indirect_effect
  causal_question_type: type_2       # 第1章の 5 Type のいずれか

  # === ② 入力条件（L1: 識別レイヤ） ===
  identification_strategy: backdoor  # or frontdoor / iv / did / rdd / synthetic_control
  causal_graph_uri: "artifact://dags/<name>.dot"
  causal_graph_sha256: "<sha256>"
  confounders_declared: [<var1>, <var2>, ...]
  mediators_declared: [<var>, ...]
  colliders_declared: [<var>, ...]

  # === 識別仮定（L2） ===
  positivity_check:
    status: checked                  # vol-04 で唯一 checked
    method: propensity_common_support
  sutva_declared:
    status: assessed
    evidence_uri: "artifact://sutva_notes/<name>.md"
  consistency_declared:
    status: assessed
    evidence_uri: "artifact://consistency_notes/<name>.md"
  exchangeability_declared:
    status: assessed
    evidence_uri: "artifact://exchangeability_notes/<name>.md"

  # === ③ 出力形式 ===
  outputs:
    estimate: point_and_ci
    refutation_results: dict
    sensitivity_analysis: dict

  # === ④ 成功条件（§4.3 の 6 点セット） ===
  success_criteria:
    identification_validity: required
    refutation_pass: required
    positivity_check: required
    external_validity: required        # counterfactual_scope_gate
    doe_efficiency: not_applicable     # DoE Skill のみ
    randomization_integrity: not_applicable

  refutation_tests_required:
    - placebo_treatment_refuter
    - random_common_cause
    - data_subset_refuter

  sensitivity_analysis:
    method: e_value
    compute_for: [point_estimate, lower_confidence_bound]
    threshold:
      minimum_e_value: 1.5
    fail_action: flag_result_as_low_robustness

  counterfactual_scope_gate:
    distance_metric: mahalanobis_distance
    distance_threshold: 3.0
    neighbor_density_metric: k_nn_density
    neighbor_density_threshold: 0.1
    cate_variance_metric: prediction_variance
    cate_variance_threshold: 0.25
    gate_logic: any_fail_triggers_review
    fallback: human_review

  # === ⑤ 禁止事項（§4.8） ===
  prohibited_actions:
    - silent_identification_switch: fatal
    - unauthorized_dag_modification: fatal
    - collider_adjustment_unauthorized: fatal
    - mediator_to_confounder_upgrade: fatal
    - refutation_skip: fatal
    - positivity_violation_ignored: fatal
    - scope_gate_threshold_override: fatal
    - unauthorized_intervention_execution: fatal
    - randomization_seed_override: fatal      # DoE Skill

  # === ⑥ 再現性条件（L3: 実行レイヤ） ===
  library_stack:                                # 第3章 §3.7
    identification: dowhy==<ver>
    estimation: econml==<ver>
    refutation: dowhy==<ver>
  environment_lock_uri: "artifact://envs/<name>.lock"
  random_seed: 42
  container_sha256: "<sha256>"

  # === ⑦ 承認ゲート（§4.6） ===
  authorization_gates:
    dag_authorization:
      required_for: [causal_graph_uri_change, causal_graph_sha256_mismatch, identification_strategy_change]
      approver: research_lead
      provenance_field: dag_approved_by
    variable_selection_authorization:
      required_for: [confounders_declared_change, mediators_declared_change, colliders_declared_change, design_parameter_change]
      approver: research_lead
      provenance_field: variables_approved_by
    intervention_execution_authorization:
      required_for: [physical_experiment_execution]
      approver: pi_and_facility_manager
      provenance_field: intervention_approved_by

  # === ⑧ DoE 拡張（DoE Skill のみ、第10-12章） ===
  experimental_design_provenance:
    design_type: full_factorial       # or fractional_factorial / central_composite / ...
    randomization_seed: 42
    blocking_factors: [<factor>, ...]
    information_gain_metric: d_optimality
    information_gain_threshold: 0.8
```

**テンプレートの読み方**：

- **① - ⑥ は vol-01/02/03 と同構造**——読者は既存 Skill 契約を段階的に因果 × DoE 拡張できます
- **⑦ 承認ゲート**は vol-04 で新設——実装は付録 B（MCP 実装パターン）で完成
- **⑧ DoE 拡張**は第10-12章の Skill でのみ埋める——因果推論のみの Skill では `not_applicable`

**成果物 2（3 層承認契約）**：⑦ の `authorization_gates` セクションを、Skill から独立した契約書として運用する場合の雛形は付録 A（Skill テンプレート集）に収録。

---

## 章末チェックリスト

- [ ] **vol-04 の 4 つの問い**（identification / DAG 再現 / 反実仮想 scope / エージェント権限）を列挙できる
- [ ] **§4.2 Table 4.1 の 6 要素拡張**を、自分の Skill に照らして各セル埋められる
- [ ] **§4.3 Table 4.2 の 6 点セット**——因果推論 Skill は 1-4、DoE Skill は 1-6——を暗記した
- [ ] **§4.4 Table 4.3 の因果 provenance 3 レイヤ**（L1 識別 / L2 識別仮定 / L3 実行）の区別ができる
- [ ] **§4.5 `counterfactual_scope_gate` の 3 判定**（距離 / 密度 / CATE 予測分散）と `gate_logic: any_fail_triggers_review` を理解した
- [ ] **§4.6 3 層承認ゲート**——`dag_authorization` / `variable_selection_authorization` / `intervention_execution_authorization`——それぞれの `required_for` と承認者を言える
- [ ] **§4.7 E-value と refutation の独立性**（両方 pass で high confidence）を理解した
- [ ] **§4.8 Table 4.4 禁止事項 12 項目**を、Severity（fatal / warning）付きで整理できる
- [ ] **§4.9 の Skill 仕様書テンプレート**——① 目的から ⑧ DoE 拡張まで——を自分のテーマで埋められる

---

## 章末演習

### 演習 4.1：自分の Skill 仕様書の起草

第2章 演習 2.1 の DAG と、第3章 演習 3.1 のライブラリ選定を出発点に、**§4.9 のテンプレートを完全に埋めた Skill 仕様書**を作成してください。

必須項目：
- [ ] ① `estimand_type` と `causal_question_type`
- [ ] ② `identification_strategy` + `causal_graph_uri` + `confounders_declared`
- [ ] ② 識別仮定 4 つの status（`checked` は positivity のみ）
- [ ] ④ `success_criteria` の 6 項目（DoE 該当なしは `not_applicable`）
- [ ] ④ `refutation_tests_required`（少なくとも 2 つ）
- [ ] ④ `sensitivity_analysis.threshold.minimum_e_value`（デフォルト 1.5 か、自研究室基準）
- [ ] ④ `counterfactual_scope_gate` の 3 閾値（データ型に応じて distance_metric 選択）
- [ ] ⑤ `prohibited_actions` の Severity 付き列挙
- [ ] ⑥ `library_stack`（第3章 §3.7）
- [ ] ⑦ 3 層承認の承認者名（自研究室の役割で置換）

### 演習 4.2：ゲート違反シナリオの検知

以下のシナリオそれぞれについて、**どの禁止事項に該当し、どの承認ゲートで防ぐか**を答えてください：

1. エージェントが Skill 実行中に causal_graph_uri を上書きした
2. エージェントが後段の工程で得た selection collider を confounder 集合に追加した
3. エージェントが Skill 内部で backdoor → IV に切り替えた
4. エージェントが `counterfactual_scope_gate.distance_threshold` を 3.0 → 5.0 に上書きした
5. エージェントが Human 承認なしで実験装置に SET コマンドを送った

（解答は第14章の失敗パターン集で照合）

### 演習 4.3：3 レイヤ provenance の再現テスト

自分の Skill について、以下の再現テストを設計してください：

- [ ] **L1 再現**：`causal_graph_sha256` を保存 → 別の環境で SHA を再計算 → 一致確認
- [ ] **L2 再現**：identification 仮定 4 つの assessment レポート（`evidence_uri`）を artifact として保存 → 別レビュアーが独立に評価
- [ ] **L3 再現**：`random_seed` と `library_stack` を pin → CI 上で ATE 点推定を再実行 → 数値の完全一致（許容誤差 < 1e-10）

---

## 参考資料

### 内部 cross-reference

- 第0章：4 識別仮定（positivity / SUTVA / consistency / exchangeability）の declared / assessed / checked、fatal / warning / flag の 3 段階チェック
- 第1章：予測 → 介入 → 反実仮想のラダー、5 causal question Type
- 第2章：DAG の Agentic 特有課題、4 typical agent failure pattern（§2.5）、少データの 4 困難と対処道しるべ（§2.4）
- 第3章：library map、Table 3.3（ライブラリ × 権限マップ）、§3.7 の Skill 契約先取り
- **第5章（次章）**：DAG 記法、SCM、backdoor / frontdoor 基準
- 第6-7章：identification 戦略ごとの estimator 実装
- 第8章：EconML CATE、g-formula、`counterfactual_scope_gate` の operational 判定
- 第9章：DoWhy refutation、E-value、`counterfactual_scope_gate` の判定閾値校正
- 第10-11章：pyDOE2、SMT、`experimental_design_provenance` の運用
- 第12章：PyMC + pyDOE2、Bayesian DoE（one-shot）
- 第14章：因果 × Agentic 失敗パターン、演習 4.2 の解答
- **vol-01 第7章**：Skill 設計 6 要素の原型
- **vol-02 第4章**：統計/ML 版設計原則
- **vol-03 第4章**：深層 × Agentic 設計原則、Agentic 学習権限 3 段階
- 付録 A：Skill テンプレート集（3 層承認契約の独立雛形を含む）
- 付録 B：MCP による 3 層承認フロー実装
- 付録 D：因果推論用語集

### 外部 URL

- VanderWeele, T. J., & Ding, P. (2017). "Sensitivity Analysis in Observational Research: Introducing the E-Value." *Annals of Internal Medicine*, 167(4), 268-274. <https://www.acpjournals.org/doi/10.7326/M16-2607>
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- Sharma, A., & Kiciman, E. (2020). "DoWhy: An End-to-End Library for Causal Inference." *arXiv:2011.04216*. <https://arxiv.org/abs/2011.04216>
- Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.). Springer.（Rosenbaum bounds の原典）
