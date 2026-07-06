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
| 1 | **何を identification 戦略とみなすか** | backdoor / frontdoor / IV / DiD / RDD / Synthetic Control / **randomized_experiment**（第10-12章 DoE）のいずれかを Skill に pin、silent 切替を禁止 | §4.2, §4.3 |
| 2 | **どの DAG で再現するか** | `dag_of_record_uri` + `dag_of_record_sha256` の pin、confounder / mediator / collider の宣言 | §4.4 |
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
| ② 入力条件 | データ範囲 | 前処理契約 | 深層学習向け augmentation 契約 | **`dag_of_record_uri` / `confounders_declared` / `mediators_declared` / `colliders_declared` の宣言** |
| ③ 出力形式 | 数値・図表 | 統計モデルの推定結果 | モデル重みと予測 | **`identification_strategy` / `estimate + 信頼区間` / `test_results` (declared_required_tests に対応) / `sensitivity_analysis`** |
| ④ 成功条件 | 数値の正しさ | 統計的有意性・CV score | 汎化性能 | **§4.3 に詳述**（identification validity / refutation pass / positivity / external validity / DoE efficiency / randomization integrity） |
| ⑤ 禁止事項 | 過学習・データリーク | 事後の仮説改変 | 未承認 fine-tune | **§4.8 に詳述**（silent identification switch / unauthorized DAG modification / adjustment of collider / refutation skip / unauthorized intervention execution） |
| ⑥ 再現性条件 | seed 固定 | environment.lock | container SHA + GPU 型番 | **§4.4 に詳述**（`dag_of_record_sha256` / `identification_strategy` / `randomization_seed` / `estimand_type` の pin） |

**新規契約フィールド**（§4.9 テンプレート参照）：

- `identification_strategy`：backdoor / frontdoor / IV / DiD / RDD / Synthetic Control / **randomized_experiment**（第10-12章 DoE）のいずれか
- `dag_of_record_uri` / `dag_of_record_sha256`：**承認済み DAG artifact** の URI と SHA256。第5章 `dag_approval_skill` が発行する canonical 名（旧名 `causal_graph_uri` / `causal_graph_sha256` は本章の初期 draft でのみ使用、Ch5-9 と本章最終版は `dag_of_record_*` に統一）
- `confounders_declared` / `mediators_declared` / `colliders_declared`：変数役割の宣言リスト
- `estimand_type`：**population target を指す canonical enum**（`ate | att | late | 2sls_linear | itt_single_unit | cate | ite_labeled_prediction | ate_trimmed | att_trimmed | ate_clipped_weight | att_clipped_weight | ate_via_gformula` 等、Ch6-8 で定義）。**mediation 分解軸は別フィールド `mediation_role`（total / direct / indirect / not_applicable）で分離**（両者は直交する意味軸——population target と mediator を通る経路の役割）
- `mediation_role`：mediator を通る経路の役割。`total | direct | indirect | not_applicable`（mediation 分解を行わない場合は `not_applicable`）
- `declared_required_tests`：refutation の 10 概念名 enum（`e_value | rosenbaum_bounds | placebo | random_common_cause | unobserved_common_cause_strength | data_subset_validation | scope_gate_reverification | ite_prediction_coverage_refutation | prior_predictive_check | prior_data_alignment`、第9章 §9.7、canonical enum_version `ch09_v0_3`）。旧名 `refutation_tests_required` は DoWhy Python API 名（`placebo_treatment_refuter` 等）に紐付いていたが、Ch9 で概念名ベースの enum に統一
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
| 1 | **identification validity** | 与えられた DAG と変数役割で、目標 estimand が **identifiable** である | `identification_strategy` + `dag_of_record_sha256` + **戦略別 identifiability check**（backdoor/frontdoor は DoWhy、IV/DiD/RDD/SC は CausalPy/linearmodels 側の review artifact、pgmpy 直接推論は自作レポート）+ `identification_report_uri` | 第5-7章 |
| 2 | **refutation pass** | 第9章 `refutation_gate` の 10 test セット（e_value / rosenbaum_bounds / placebo / random_common_cause / unobserved_common_cause_strength / data_subset_validation / scope_gate_reverification / ite_prediction_coverage_refutation / prior_predictive_check / prior_data_alignment、canonical enum_version `ch09_v0_3`）を、estimator 別 applicability manifest に基づいて **required 全て pass** する | `declared_required_tests` + `test_results` + `applicability_manifest_uri` | 第9章 |
| 3 | **positivity check** | Treatment の割当が全 covariate strata で正の確率を持つ（連続 treatment では generalized propensity の common support） | `positivity_check` | 第2章 §2.4, 第6章 |
| 4 | **external validity** | 反実仮想が学習分布の外挿にならない | `counterfactual_scope_gate` | §4.5, 第8-9章 |
| 5 | **DoE efficiency**（DoE 章のみ） | 情報量あたりのコストが閾値内 | `information_gain_metric` + `information_gain_threshold`（§4.9 ⑫） | 第10-12章 |
| 6 | **randomization integrity**（DoE 章のみ） | randomization seed が pin されており、blocking が破綻していない | `randomization_seed` + `blocking_factors` | 第10章 |

**「全条件 AND」で成功**：因果推論 Skill は 1-4、DoE Skill は 5-6 も追加。1 つでも fail なら Skill は数値を返さず、Human に stop condition を返す（第2章 §2.4 IMPORTANT の「無理な問いには No」）。

---

## 4.4 どの DAG で再現するか — 因果 provenance の 3 レイヤ

vol-03 第4章の「GPU/Agentic provenance の 3 レイヤ」を、因果推論用に再構成します。

### Table 4.3：因果 provenance の 3 レイヤ

| レイヤ | 内容 | pin 対象 | 検証方法 |
|---|---|---|---|
| **L1: 識別レイヤ** | DAG 構造、変数役割、identification 戦略 | `dag_of_record_uri`, `dag_of_record_sha256`, `identification_strategy`, `confounders_declared`, `mediators_declared`, `colliders_declared`, `estimand_type`, `mediation_role`, `adjustment_set_approval_uri` | DoWhy identifiability check、DAG SHA256 照合、adjustment set 承認証跡照合 |
| **L2: 識別仮定レイヤ** | positivity / SUTVA / consistency / exchangeability の declared / assessed 状態（第0章 §0.6） | `positivity_by_stratum`（**checked**、§4.4.1 canonical shape）、`sutva_declared` / `consistency_declared` / `exchangeability_declared`（**assessed**） | 各仮定に対する数値・グラフ・ドメイン知識に基づく評価レポート |
| **L3: 実行レイヤ** | ライブラリバージョン、seed、乱数状態、コンテナ | `library_stack`（第3章 §3.7）、`environment.lock`（vol-02 資産）、`random_seed`、`container_sha256`（vol-03 資産） | pip freeze、docker inspect、seed reproducibility test |

**3 レイヤの独立性**：

- L1 が変わる → **新 Skill バージョン**（`dag_authorization` 経由、§4.6）
- L2 が変わる → **assessed の更新のみで OK**（vol-04 では `checked` は positivity のみ、他は declared のまま新エビデンスで assessed 更新）
- L3 が変わる → **新 Skill バージョン + 再現性テスト**（vol-02/03 と同様）

### DAG artifact の保存形式

`dag_of_record_uri` は以下いずれかを指す：

- **DOT 形式**（GraphViz、第2章の演習データで採用）：人間可読、networkx 経由で DoWhy に入る
- **networkx pickle**：Python 完結、CI 内での再構築が高速
- **JSON**（{nodes, edges} 形式）：言語非依存、外部ツール（Neo4j 等）と連携する場合

**必ず SHA256 を併記**（`dag_of_record_sha256`）——DAG artifact のバイト単位の変化を検知します。

### 4.4.1 `positivity_by_stratum` の canonical shape（B-3 統一定義）

Ch6/7/8 の各 estimator 契約が参照する `positivity_by_stratum` は **list-of-objects** を canonical shape として本節で確定します。過去の draft には nested object 形式（`positivity.by_stratum: {key: ..., min_treated: ...}`）と flat list 形式が混在していましたが、**list-of-objects に統一**——複数の stratification 軸（装置別 × 組成別など）を単一契約で扱えるためです。

```yaml
positivity_by_stratum:
  status: checked | not_applicable         # not_applicable は estimand_type ∈ {itt_single_unit} のみ許容
  positivity_not_applicable_rationale: <string>  # not_applicable の場合のみ必須
  strata:                                  # status: checked の場合の必須リスト
    - key: <string>                        # 例: device_id, composition_bin, batch_id
      min_treated: <int>                   # 各層で最小限必要な処置群 n
      min_control: <int>                   # 対照群 n の下限
      min_effective_sample_size: <int>     # optional（cate / ite_labeled_prediction では required）
      propensity_epsilon: <float>          # optional（cate / ite_labeled_prediction では required、propensity trim 下限）
      violation_policy: drop_stratum | fail_close
      stratum_report_uri: <string>         # 各層の overlap / effective_n 詳細レポート
  aggregate_report_uri: <string>           # 全 strata を通した集計レポート
```

**estimator 別の required/optional**：

| estimator ファミリ | `min_effective_sample_size` | `propensity_epsilon` | 備考 |
|---|---|---|---|
| PSM / IPW / DR / DML (Ch6) | optional | optional（IPW/DR/DML では推奨） | matching / weighting の失敗 mode |
| DiD / IV (Ch7) | optional | optional | 時系列・楽器の validity と別軸で必須 |
| Synthetic Control (Ch7) | — | — | `status: not_applicable` + rationale で置換 |
| CATE / g-formula (Ch8) | **required** | **required** | 局所推定の分散爆発を防ぐため |
| ite_labeled_prediction (Ch8) | **required** | **required** | 個体レベル予測の positivity 保証 |

**Ch6 の nested `positivity.by_stratum` は本節 canonical shape の下位互換ラッパー**——読み替え規則を Ch6 §6.2 冒頭 note に明記（後述 Ch6 修正で追加）。

---

## 4.5 反実仮想の外挿範囲 — `counterfactual_scope_gate` の契約定義

**本節は `counterfactual_scope_gate` の source of truth**です。Ch8 で operational 判定を実装（cluster-conditional Mahalanobis、閾値 calibration protocol）、Ch9 で独立再検証、Ch11 で SMT 応答曲面に適用します。**本節の契約は 4 判定・3 状態（pass / conditional_pass / fail）**——Ch8 の operational 実装と 1:1 対応します（過去の draft では 3 判定・2 状態を記載していましたが、`support_envelope_check` と `conditional_pass` 状態の追加は本節で確定）。

### 4.5.1 なぜ scope gate が必要か

反実仮想（"もし装置を A ではなく B にしていたら Y は何になっていたか"）は、**学習分布の外側**の値を要求する場面が多発します。予測 Skill なら「外挿は精度が落ちる」で済みますが、因果 Skill では **「外挿は識別を無効化しうる」**——`consistency` 仮定が破れる可能性があるからです（第0章 §0.6、第2章 §2.4）。

### 4.5.2 契約フィールドの完全定義（4 判定 / 3 状態）

**4 つの判定軸**（Ch8 §8.5.2 と対応、それぞれ異なる失敗モードを捉える相補的チェック——1 つの pass は他の substitute にはなりません）：

| # | 判定軸 | 捉える失敗モード | 契約フィールド |
|---|---|---|---|
| 1 | Mahalanobis 距離（**cluster-conditional 必須**、multimodal 支持では global 単独 unsafe） | 全体的な "遠さ" | `mahalanobis_distance_global` / `mahalanobis_distance_cluster_conditional` / `mahalanobis_threshold` |
| 2 | CATE 予測分散 | 推定不確実性 | `cate_prediction_variance` / `variance_threshold` |
| 3 | k-NN 密度 | 局所支持の欠如（cluster 間の空隙） | `knn_density` / `knn_min` / `knn_radius` |
| 4 | **support envelope** | 周辺（marginal）外挿 | `outside_support_dimensions` / `support_envelope_report_uri` |

```yaml
counterfactual_scope_gate:
  # --- 判定 1: 距離（cluster-conditional 必須） ---
  mahalanobis_check:
    distance_global: <float>
    distance_cluster_conditional: <float>       # multimodal 支持で必須
    cluster_assignment_uri: <string>
    threshold: <float>                          # 学習分布から calibrated（下記 protocol）
    covariance_estimation_uri: <string>

  # --- 判定 2: 不確かさ（CATE 予測分散） ---
  variance_check:
    predicted_cate_variance: <float>
    threshold: <float>                          # 処置群 outcome SD の倍数で定義

  # --- 判定 3: 局所密度 ---
  knn_density_check:
    k: 20
    n_training_points_within_radius: <int>
    knn_min: <int>
    knn_radius: <float>                         # 近傍半径（学習分布中央値等で正規化）

  # --- 判定 4: support envelope（周辺外挿の検出） ---
  support_envelope_check:
    outside_support_dimensions: [...]           # 学習域外に落ちた次元名リスト
    envelope_report_uri: <string>

  # --- 閾値 calibration protocol（閾値の恣意性を防ぐ） ---
  threshold_calibration:
    method: leave_one_out_empirical | conformal | validation_false_extrapolation_rate
    calibration_evidence_uri: <string>
    calibration_approved_at: <timestamp>

  # --- aggregate policy（4 判定の結合ルール、3 状態モデル）---
  aggregate_policy:
    pass_requires: all_four_pass                # 4 判定 全 pass → actionable 予測を返す
    conditional_pass_requires:                  # 一部のみ pass → 診断のみ返す
      - mahalanobis_cluster_conditional_pass
      - variance_pass
      - knn_density_pass
      - support_envelope_pass
    conditional_pass_output: non_actionable_diagnostic_only  # 予測数値は返さず、診断 artifact のみ
    fail_close_action: refuse_prediction_and_notify_reviewer

  # --- 3 状態 ---
  gate_status: pass | conditional_pass | fail
  # pass:              全 4 判定 pass → actionable 予測 + 分散を返す
  # conditional_pass:  一部判定 fail → 予測数値返却禁止、reviewer 用診断のみ
  # fail:              重大 fail → fail-close、silent 戦略切替は estimator_contract_change_gate 必須（Ch6 §6.7 / Ch7 §7.6.1 / Ch8 §8.5.3）

  approver_when_conditional: causal_review_board

  # --- fallback（レビュー起動） ---
  fallback: human_review
  fallback_message_template: |
    Counterfactual point x' = {value} is outside actionable scope (gate_status={status}).
    - Mahalanobis (cluster): {d_mc} (threshold {t_m})
    - Variance: {var} (threshold {t_v})
    - k-NN density: {density} (threshold {t_k})
    - Outside support dimensions: {outside_dims}
```

### 4.5.3 データ型別の distance_metric 選択

| データ型 | 推奨 distance_metric | 補足 |
|---|---|---|
| 表形式（連続共変量） | mahalanobis_distance | 共分散を考慮した距離 |
| 表形式（カテゴリ共変量） | mixed_gower_distance | カテゴリ + 連続の混合 |
| 組成ベクトル（simplex 制約） | aitchison_distance または simplex_projected_mahalanobis | 組成データの制約を考慮 |
| スペクトル・時系列 | embedding_distance（vol-03 第7章の深層特徴） | 潜在空間での距離 |
| 画像・回折・マイクロスコピー特徴マップ | embedding_distance | 同上 |
| 結晶グラフ・分子グラフ | graph_embedding_distance | GNN 埋め込み |
| 打ち切り・区間データ | interval_hausdorff_distance | 区間 vs 区間の距離 |
| マルチモーダル | concatenated_embedding_distance | 各モーダルの embedding を結合 |

**閾値の初期値の決め方（メトリック別）**：

| メトリック | 境界の向き | 初期値の決め方 |
|---|---|---|
| `mahalanobis_threshold`（cluster-conditional） | 上限 | 学習データ内 cluster 内距離分布の **95 パーセンタイル** |
| `variance_threshold` | 上限 | 学習データ内 CV での CATE 予測分散の **95 パーセンタイル** |
| `knn_min`（k-NN density の下限） | **下限** | 学習データ内部での近傍密度分布の **5 パーセンタイル**（または実効サポートの下限） |
| `support_envelope`（各次元の [min, max]） | 各次元 | 学習データ各次元の観測範囲。**外挿検知は "1 次元でも越境で fail"** |

`threshold_calibration.method` は **`leave_one_out_empirical`**（デフォルト）/ **`conformal`**（正確な false-extrapolation rate 保証）/ **`validation_false_extrapolation_rate`**（外挿ラベル付き validation set がある場合）から選択。第8-9章で domain-specific な再校正手順を示します。

### 4.5.4 SMT 応答曲面への適用（Ch11 予告）

SMT の Kriging surrogate は外挿に弱いため（第3章 §3.3）、応答曲面の予測点が **`counterfactual_scope_gate` を通過**しない場合は Skill は最適条件を返しません（fail-close、Table 4.4 の scope gate 逸脱は **fatal**）。Ch11 で具体実装。

---

## 4.6 3 層承認ゲートの完全仕様（新設節）

**vol-04 の中核設計**です。vol-03 第4章の「Agentic 学習権限 3 段階」を、因果的判断向けに **役割分離**します。

### 4.6.1 3 層承認ゲートの位置づけ

> [!NOTE]
> **`gate_level` enum 表記規約（N-9）**：Skill 契約や registry の `gate_level:` フィールドには **`L{n}_{name}` の完全形** を canonical enum 値として書く：`L1_dag_authorization` / `L2_variable_selection_authorization` / `L3_intervention_execution_authorization` / `L4_facility_standard_promotion`（vol-04 全章で一貫、Ch13 §13.4.2 / Ch14 §14.5.2 に back-register 済み）。本節の見出しや本文の説明では読みやすさのため `L1 / L2 / L3 / L4` の短縮形も併記するが、これは **display-format** であり、machine-readable な YAML では常に完全形を使用する。

| ゲート | 承認対象 | 承認者（推奨） | 反応時間目安 |
|---|---|---|---|
| **`dag_authorization`** | DAG 構造の変更、identification 戦略の切替 | Research lead（PI or 上級研究員） | 数時間〜1 日 |
| **`variable_selection_authorization`** | confounder / mediator / collider の判定変更、DoE の design parameter 変更（DiD window、IV 候補、RDD cutoff、blocking factor 等） | Research lead | 数時間 |
| **`intervention_execution_authorization`** | **実際の実験装置を動かす行為** | PI + Facility manager | 実験前の事前承認 |
| **`facility_standard_promotion_gate`**（L4、第14章 §14.5.2 で新設） | 施設標準化への promotion（`facility_scope_escalation.applies_to` enum 全項目） | Facility causal review board | 数日、`audit_manifest_v1` の pass を前提条件とする |

**なぜ 3 層か**：

- **DAG は identification の根本**——変更すれば別の因果的問いになる（第2章 §2.5 パターン 2, 4）
- **変数役割は identification の妥当性**——confounder を collider に間違えれば bias 混入（第2章 §2.5 パターン 1, 3）
- **介入実行は現実世界への影響**——シミュレーションと違い revocable ではない

### 4.6.2 ゲートの発火条件（Skill 契約）

> [!IMPORTANT]
> **`authorization_gates` canonical shape（S-10）**：フィールド名は **複数形 `authorization_gates:`**（singular `authorization_gate:` は非 canonical、Ch5/Ch7 も本節に統一）、値は **dict-shaped**（key = gate 名、value = 発火条件と承認者）。scalar 形式（`authorization_gate: dag_authorization` 等）は Ch4 §4.9 template ⑧ の canonical 形状に違反する。全 downstream Skill（Ch5-Ch12）は本節形状を採用する。

```yaml
authorization_gates:
  dag_authorization:
    required_for:
      - dag_of_record_uri_change        # DAG artifact の変更
      - dag_of_record_sha256_mismatch   # SHA 不一致
      - identification_strategy_change  # backdoor→IV 等
    approver: research_lead              # デフォルト（個別 PI-scope の判断）
    provenance_field: dag_approved_by
    approval_evidence:
      - approver_signature
      - approval_timestamp
      - approval_scope                 # どの変更を承認したか
    # 施設 scope の artifact promotion（dag_of_record の facility 標準化・approved_instruments / approved_donor_pool の登録）
    # は Ch5/Ch7 の approval Skill 経由で `causal_review_board` にエスカレーション（下記 approver_independence の
    # facility_scope_escalation 参照）

  variable_selection_authorization:
    required_for:
      - confounders_declared_change
      - mediators_declared_change
      - colliders_declared_change
      - design_parameter_change        # DiD window / IV / RDD / donor pool / blocking factor
      - adjustment_set_approval_uri_change
    approver: research_lead              # デフォルト（個別 PI-scope）
    provenance_field: variables_approved_by
    approval_evidence:
      - approver_signature
      - approval_timestamp
      - approval_scope
    # 施設共通 approved_instruments / approved_donor_pool の変更は Ch7 §7.5.2 / §7.5.3 の approval Skill 経由で
    # facility_causal_review_board にエスカレーション

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
      - intervention_execution_timestamp   # Ch14 §14.3.4 で back-register
      - egress_channels_declared           # Ch14 §14.3.6 で back-register — 承認された broadcast チャネルの enum
      - authorized_broadcast_targets       # Ch14 §14.3.6 — CATE 推薦の送信先 whitelist
    temporal_ordering:                     # Ch14 §14.3.4 で back-register
      policy: approval_before_execution    # approval_timestamp < intervention_execution_timestamp が必須
      violation_severity: fatal            # fatal `intervention_executed_before_l3_authorization`

  # --- 承認者の独立性（利益相反対応 + facility scope escalation） ---
  approver_independence:
    conflict_policy: independent_reviewer_required_if_approver_is_data_producer
    fallback_approver: facility_causal_review_board
    # facility-scope の artifact promotion は default で causal_review_board にエスカレーション
    facility_scope_escalation:
      applies_to:
        - dag_of_record_promotion_to_facility_standard
        - approved_instruments_registration            # Ch7 §7.5.2
        - approved_donor_pool_registration             # Ch7 §7.5.3
        - approved_meta_learner_covariate_set          # Ch8
        - approved_facility_design_template            # Ch10 §10.7.3 DoE template promotion
        - approved_facility_response_surface_baseline  # Ch11 §11.5/§11.8 GP kernel + hyperparameter template promotion
        - approved_facility_prior_specification        # Ch12 §12.8 Bayesian DoE 事前分布テンプレート promotion（特定材料クラス向け）
        - approved_intervention_becomes_facility_standard  # Ch13 §13.7 承認済み介入の施設標準化（capstone worked example の facility escalation trigger）
      default_approver: facility_causal_review_board
      pre_conditions:                                   # Ch14 §14.5.2 で back-register
        - audit_manifest_v1_pass                        # 全 19 detection check を pass（Ch14 §14.4）
        - audit_manifest_uri_pinned                     # Ch14 §14.4 canonical
        - audit_manifest_sha256_pinned                  # Ch14 §14.4 canonical
      gate_level: L4_facility_standard_promotion        # Ch14 §14.5.2 で新設される L4 gate
      rationale: |
        施設全体に影響する artifact promotion は、個別 PI の research_lead 権限を超える。
        Ch5/Ch7/Ch8 の approval Skill は本ゲートを継承し、default approver として
        causal_review_board を宣言する（Ch4 §4.6.2 の research_lead は "個別 PI-scope の
        decision" に対する default で、facility-scope への escalation はここで規定）。
```

> [!IMPORTANT]
> **利益相反への対処**：`research_lead` や `PI` が **同時にデータ生成者・因果的主張の受益者**である場合、自己 review になり承認の意味が失われます。`approver_independence` フィールドで、**利益相反時は独立レビュアー（例：ARIM 施設の `facility_causal_review_board`）を経由する**ことを契約に明示します。ARIM のような共用施設ではこの governance gap への対処が必須です。
>
> **`research_lead` vs `causal_review_board` の使い分け**：
> - **個別 PI-scope の意思決定**（1 プロジェクト内の DAG 変更・共変量選択・介入設計）→ `research_lead` が default approver
> - **facility-scope の artifact promotion**（施設全体で参照される dag_of_record・approved_instruments・approved_donor_pool・approved_covariate_set の登録・更新）→ `facility_causal_review_board` が default approver（`facility_scope_escalation` 参照）
> - **利益相反時**（approver が同時にデータ生成者）→ どちらの場合も `facility_causal_review_board` に fallback

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

> [!NOTE]
> **E-value の適用スケール**：E-value の元の定義は risk ratio (RR) スケール。**連続 outcome では近似変換（VanderWeele & Ding の推奨変換など）または stratified RR の解釈が必要**——変換手順は Skill 契約に明記します（第9章で ARIM 材料応答向けの変換例）。

### 4.7.2 Skill 契約への組み込み

```yaml
sensitivity_analysis:
  method: e_value                          # or rosenbaum_bounds
  effect_scale: risk_ratio                 # or transformed_continuous with documented mapping
  effect_direction: harmful                # harmful | protective — E-value CI-bound sidedness を決定 (§4.9 canonical)
  ci_bound_closest_to_null: lower_bound    # harmful→lower_bound / protective→upper_bound + RR 反転 (Ch9 §9.2.3)
  smd_to_rr_conversion: vanderweele_2020   # 連続 outcome では required (Ch9 §9.2.3)
  compute_for:
    - point_estimate                       # 点推定の E-value
    - ci_bound_closest_to_null             # 有害効果は下限・保護効果は上限——effect_direction で自動選択（§4.9 canonical）
  threshold:
    minimum_e_value: 1.5                   # デフォルトは "弱頑健フラグ" 用の閾値のみ。研究室・領域で再校正必須
  provenance_field: sensitivity_report_uri  # 感度分析レポート artifact
  fail_action: flag_result_as_low_robustness
```

**閾値 1.5 について**：これは **"弱頑健フラグ" の初期デフォルト**であって普遍的な統計基準ではありません。**信頼区間下限の E-value を主指標とし**、研究室・領域ごとに校正することを推奨します（第9章）。

**E-value と `declared_required_tests` の関係**：

- `declared_required_tests` は **Ch9 の 10 test 概念 enum**（ch09_v0_3）——**別の側面の頑健性**（placebo、random common cause、data subset validation 等、Bayesian DoE では prior_predictive_check / prior_data_alignment を含む）
- `sensitivity_analysis`（E-value）は **unmeasured confounder に対する頑健性**
- **両者は契約上は別ゲートとして扱う相補的チェック**（統計的に完全独立ではない——同じデータ・モデル・identification 仮定に依存する）——片方の pass は他方を代替しません。両方 pass しないと Skill は結果を "high confidence" として返しません

---

## 4.8 因果 × Agentic Skill の禁止事項（severity 付き）

vol-03 第4章の禁止事項リストに、因果推論 × DoE 特有の項目を追加します。

> [!NOTE]
> **Fatal 命名規約（S-2）**：本節の canonical fatal 名は **bare form**（`silent_identification_switch` / `unauthorized_dag_modification` / `refutation_skip` 等、動詞開始 + snake_case）で列挙する。cross-chapter で参照する際は **`Ch{n}.{fatal_name}` prefix 形式**（例：Ch14 が Ch4 fatal を参照する場合 `Ch4.silent_identification_switch`）を使用する（Ch14 §14.6 で確立、Ch4-13 の source-of-truth 章では prefix なし）。この慣習により、audit manifest の cross-reference が明確になる。

> [!NOTE]
> **Canonical enum versioning 規約（S-3）**：canonical enum（`declared_required_tests`, `gate_level`, `action_class` 等）に破壊的変更を加える場合は、enum block に `enum_version: <chapter_id>_<semver>` フィールドを併記する（例：Ch9 §9.7.1 `enum_version: ch09_v0_3`）。silent downgrade は fatal（`downgrade_<enum_name>_enum_version_silently`）。詳細は Ch14 §14.3.3 参照。

### Table 4.4：禁止事項

| # | 禁止事項 | Severity | 検知フィールド | 発火するゲート | 対応する Ch2 パターン |
|---|---|---|---|---|---|
| 1 | Silent identification strategy switch（Skill 内部で backdoor→IV 等） | **fatal** | `identification_strategy` の差分、`estimator_contract_change_gate` の approval なし | `dag_authorization` + `estimator_contract_change_gate`（Ch6 §6.7） | パターン 4 |
| 2 | Unauthorized DAG modification | **fatal** | `dag_of_record_sha256` の差分（旧名 `causal_graph_sha256` は本表 §4.4 alias 参照） | `dag_authorization` | パターン 2 |
| 3 | Adjustment of collider without authorization | **fatal** | `confounders_declared` と DAG の整合、`colliders_declared` 越境、`adjustment_set_approval_uri` の欠如 | `variable_selection_authorization` | パターン 1 |
| 4 | Mediator を confounder に格上げ | **fatal** | `mediators_declared` / `confounders_declared` / `estimand_type` / `mediation_role` の差分 + DAG diff | `dag_authorization` + `variable_selection_authorization` | パターン 3 |
| 5 | Refutation skip | fatal | `declared_required_tests` vs `test_results.<test>.status` の mismatch、`applicability_manifest_uri` の欠如 | — | — |
| 5a | `partial_diagnostic_only` の状態で actionable 数値を release（Ch9 §9.7.1） | **fatal** | `refutation_gate.aggregate_status` = `partial_diagnostic_only` かつ `estimate` release | — | — |
| 5b | Failed required test を post-hoc に `not_applicable` に再分類（Ch9 §9.7.1） | **fatal** | `applicability_manifest_sha256` 変更後の再評価、pre-registration との hash 不一致 | — | — |
| 6 | Positivity 違反を無視して estimation | fatal | `positivity_by_stratum.status` = fail without policy | — | — |
| 7 | `counterfactual_scope_gate` の閾値を Skill 内部で変更 | fatal | 契約 vs 実行時パラメータの差分、`threshold_calibration.calibration_approved_at` なし、`estimator_contract_change_gate` の approval なし | `counterfactual_scope_gate` + `estimator_contract_change_gate` | — |
| 8 | E-value 計算のスキップ | warning | `sensitivity_analysis` の存在 | — | — |
| 8a | E-value の CI-bound sidedness 誤用（protective effect で CI 下限を使う等、Ch9 §9.2.3） | **fatal** | `effect_direction` × `ci_bound_closest_to_null` の整合、SMD→RR 変換方法未記録 | — | — |
| 9 | Unauthorized intervention execution（承認なしで実験装置を動かす） | **fatal** | `intervention_approved_by` の欠如 | `intervention_execution_authorization` | — |
| 10 | Randomization seed の上書き（DoE 章） | fatal | `randomization_seed` の差分 | — | — |
| 11 | DoE の blocking 破綻 | warning | `blocking_factors` と実行データの整合 | — | — |
| 12a | Scope gate 境界近傍での反実仮想（`conditional_pass` 状態、Ch8 §8.5.3） | warning（診断のみ返却） | `counterfactual_scope_gate.gate_status` = `conditional_pass` | `counterfactual_scope_gate` | — |
| 12b | Scope gate 逸脱・SMT 応答曲面の明示的外挿 | **fatal**（fail-close、結果を返さない） | `counterfactual_scope_gate.gate_status` = `fail` | `counterfactual_scope_gate` | — |
| 12c | **`conditional_pass` 状態で actionable CATE 数値を release**（Ch8 §8.5.2） | **fatal** | `gate_status` = `conditional_pass` かつ estimate release | `counterfactual_scope_gate` | — |
| 12d | **4 判定の 1 つを他 pass で substitute**（例：global Mahalanobis pass を cluster-conditional の代替に使用、Ch8 §8.5.2） | **fatal** | `aggregate_policy.pass_requires` = `all_four_pass` 違反 | `counterfactual_scope_gate` | — |
| 12e | **global Mahalanobis 単独 pass** を multimodal 支持データで使用（Ch8 §8.5.2） | **fatal** | `cluster_assignment_uri` の欠如、`mahalanobis_distance_cluster_conditional` の欠如 | `counterfactual_scope_gate` | — |
| 13 | ITE 予測 coverage refutation の未実施（`estimand_type = ite_labeled_prediction` 主張時、Ch9 §9.7.3） | **fatal** | `test_results.ite_prediction_coverage_refutation.status` の欠如、または training data で coverage 検証 | — | — |
| 14 | Estimator contract の silent change（threshold・adjustment set・positivity policy の変更、Ch6 §6.7） | **fatal** | `estimator_contract_sha256` 差分 without L1-L4 approval | `estimator_contract_change_gate` | — |
| 15 | Preregistration manifest の post-hoc 修正（Ch9 §9.7.1） | **fatal** | `preregistration_manifest_sha256` 差分 with approval なし | — | — |
| 16 | Post-treatment variable の adjustment（Ch5 §5.2.2 mediator、Ch14 §14.1.1） | **fatal** | `adjust_for_post_treatment_variable_without_marking_as_mediator` — DAG の temporal ordering チェックで adjustment set 内の変数が処置後に測定されているのに mediator 宣言がない | `variable_selection_authorization` | — |
| 17 | Adjustment set を DAG 承認後に silent 変更（Ch14 §14.3.2） | **fatal** | `modify_adjustment_set_after_downstream_start` — `approved_covariates` と実際の estimator input の symmetric difference が空でない | `variable_selection_authorization` | — |
| 18 | 介入実行が L3 承認より早い（Ch14 §14.3.4） | **fatal** | `intervention_executed_before_l3_authorization` — `intervention_execution_timestamp` < `approval_timestamp` | `intervention_execution_authorization` | — |
| 19 | 介入実行内容が承認と異なる（Ch14 §14.3.4） | **fatal** | `intervention_executed_differs_from_approved` — `approved_intervention_uri` と `executed_intervention_uri` の不一致 | `intervention_execution_authorization` | — |
| 20 | CATE 推薦を L3 承認前に外部送信（Ch14 §14.3.6） | **fatal** | `intervention_recommendation_broadcast_without_l3` — `egress_channels_declared` に無い channel への broadcast、または `authorized_broadcast_targets` 外への送信 | `intervention_execution_authorization` | — |
| 21 | 監査 skill が全 detection check を走らせずに pass 主張（Ch14 §14.5.1） | **fatal** | `claim_audit_pass_without_running_all_checks` — `audit_result_summary.pass_count + fail_count + not_applicable_count ≠ total_checks_expected`（**canonical: `total_checks_expected == 19`**, Ch14 §14.4） | `facility_standard_promotion_gate`（L4） | — |
| 22 | 監査 skill 完了後に audit_manifest を改変（Ch14 §14.5.1） | **fatal** | `modify_audit_manifest_after_completion` — `audit_manifest_sha256` の差分、Ch13 §13.4.4 evidence_chain_sha256 との連鎖破綻 | `facility_standard_promotion_gate`（L4） | — |
| 23 | audit_manifest pass なしで施設標準化 promotion（Ch14 §14.5.2） | **fatal** | `promote_to_facility_standard_without_audit_manifest_pass` — `facility_scope_escalation.pre_conditions.audit_manifest_v1_pass` の欠如 | `facility_standard_promotion_gate`（L4） | — |

**Ch7-9 domain-specific fatals**：本表は master 禁止事項リストですが、Ch6-9 の estimator 別の domain-specific fatals（例：Ch7 §7.3.5 `use_naive_first_stage_f_under_heteroskedastic_errors`、Ch8 §8.4.3 `silently_switch_deep_feature_extractor`、Ch9 §9.4.1 `use_marginal_permutation_without_W_joint_preservation`）は各章の `prohibited_actions` セクションを参照。**本表は cross-estimator な fatal のみを扱います**。

**Severity の意味**：

- **fatal**：Skill 停止、Human 介入必須、監査対象
- **warning**：Skill は数値を返すが `low_confidence` フラグ付き、Human review 推奨
- **flag**：数値を返すが provenance に記録して次回改善（vol-04 では未使用）

---

## 4.9 因果 × Agentic Skill 仕様書テンプレート（**成果物 1**）

以下は本章の**主要成果物**——因果 × Agentic Skill の最小契約テンプレートです。第3章 §3.7 の先取り紹介を **完全版に拡張**しました。

> [!NOTE]
> **ブロック番号規約（B-11 修正）**：§4.2 Table 4.1 の 6 要素（vol-01 起源、⑤=禁止事項 / ⑥=再現性条件）を、vol-04 では **⑤=外部妥当性 / ⑥=禁止事項 / ⑦=再現性条件 / ⑧=承認ゲート / ⑨=agent_action_log / ⑩=egress_control / ⑪=estimator_provenance_reference / ⑫=DoE 拡張** に再割り当てして拡張しています。以降の章（Ch5-Ch14）は本 §4.9 の再番号を **canonical** として引用します。

```yaml
skill:
  name: <skill_name>
  version: <semver>
  purpose: |
    <estimand と causal question Type を明示>

  # === ① 目的 ===
  estimand_type: ate                  # canonical population target enum:
                                       # ate | att | late | 2sls_linear | itt_single_unit |
                                       # cate | ite_labeled_prediction |
                                       # ate_trimmed | att_trimmed |
                                       # ate_clipped_weight | att_clipped_weight |
                                       # ate_via_gformula
                                       # ※ Ch6-8 が実装、Ch9 refutation gate の applicability 判定に使用
  mediation_role: not_applicable      # total | direct | indirect | not_applicable
                                       # ※ mediation 分解を行う場合のみ total/direct/indirect
  causal_question_type: type_a_ate_att # 第1章の 5 Type
                                       # enum: type_a_ate_att / type_b_cate /
                                       #       type_c_dag_discovery / type_d_counterfactual /
                                       #       type_e_doe

  # === ② 入力条件（L1: 識別レイヤ） ===
  identification_strategy: backdoor    # or frontdoor / iv / did / rdd / synthetic_control /
                                       # randomized_experiment (第10-12章, DoE)
  dag_of_record_uri: "artifact://dags/<name>.dot"       # 第5章 dag_approval_skill が発行
  dag_of_record_sha256: "<sha256>"
  adjustment_set_approval_uri: "artifact://approvals/<name>.yaml"  # 第5-8章
  confounders_declared: [<var1>, <var2>, ...]
  mediators_declared: [<var>, ...]
  colliders_declared: [<var>, ...]

  # === データ来歴（audit 用）===
  data_lineage:
    dataset_uri: "artifact://datasets/<name>"
    dataset_sha256: "<sha256>"
    preprocessing_pipeline_sha256: "<sha256>"
    experiment_ids: [<id>, ...]
    arim_facility_id: "<facility>"

  # === 識別仮定（L2） ===
  positivity_by_stratum:               # §4.4.1 canonical shape
    status: checked                    # vol-04 で唯一 checked
    strata:
      - key: device_id
        min_treated: 5
        min_control: 5
        min_effective_sample_size: 20  # cate / ite_labeled_prediction では required
        propensity_epsilon: 0.05       # cate / ite_labeled_prediction では required
        violation_policy: drop_stratum # or fail_close
        stratum_report_uri: "artifact://positivity/<key>.md"
    aggregate_report_uri: "artifact://positivity/aggregate.md"
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
    test_results: dict                 # Ch9 refutation_gate の 10 test の結果 (ch09_v0_3)
    sensitivity_analysis: dict

  # === ④ 成功条件（§4.3 の 6 点セット） ===
  success_criteria:
    identification_validity: required
    refutation_pass: required
    positivity_check: required
    external_validity: required        # counterfactual_scope_gate
    doe_efficiency: not_applicable     # DoE Skill のみ
    randomization_integrity: not_applicable

  identification_validity:
    checker: dowhy_identify_effect     # or strategy_specific_review (IV/DiD/RDD/SC/pgmpy)
    report_uri: "artifact://identification/<name>.md"

  # --- refutation gate (Ch9 §9.7.1 が source of truth) ---
  declared_required_tests:             # Ch9 の 10 test enum (ch09_v0_3) から estimator 別に選択
    - e_value
    - random_common_cause
    - placebo
    - data_subset_validation
    - scope_gate_reverification
    # estimator に応じて rosenbaum_bounds / unobserved_common_cause_strength /
    # ite_prediction_coverage_refutation を追加（Ch9 §9.7.2 表参照）
  applicability_manifest_uri: "artifact://refutation/<name>_applicability.yaml"
  applicability_manifest_sha256: "<sha256>"    # not_applicable claim の事前登録

  sensitivity_analysis:
    method: e_value
    effect_scale: risk_ratio           # or transformed_continuous with documented mapping
    smd_to_rr_conversion: vanderweele_2020  # 連続 outcome では required（Ch9 §9.2.3）
    effect_direction: harmful          # harmful | protective
    ci_bound_closest_to_null: lower_bound   # protective の場合は upper_bound + RR 反転
    compute_for: [point_estimate, ci_bound_closest_to_null]
    threshold:
      minimum_e_value: 1.5             # 弱頑健フラグ用の初期値、研究室で再校正
    provenance_field: sensitivity_report_uri
    fail_action: flag_result_as_low_robustness

  # === ⑤ 外部妥当性（counterfactual_scope_gate） ===
  counterfactual_scope_gate:           # §4.5 で source of truth
    mahalanobis_check:
      threshold: 3.0                   # cluster-conditional distance の 95 percentile
      cluster_assignment_uri: "artifact://scope/<name>_clusters.pkl"
    variance_check:
      threshold: 0.25
    knn_density_check:
      k: 20
      knn_min: 5
    support_envelope_check:
      envelope_report_uri: "artifact://scope/<name>_envelope.md"
    threshold_calibration:
      method: leave_one_out_empirical
      calibration_evidence_uri: "artifact://scope/<name>_calib.md"
      calibration_approved_at: "<timestamp>"
    aggregate_policy:
      pass_requires: all_four_pass
      conditional_pass_output: non_actionable_diagnostic_only
    fallback: human_review
    fallback_message_template: |
      Counterfactual point x' = {value} is outside actionable scope (gate_status={status}).
      - Mahalanobis (cluster): {d_mc} (threshold {t_m})
      - Variance: {var} (threshold {t_v})
      - k-NN density: {density} (threshold {t_k})
      - Outside support dimensions: {outside_dims}

  # === ⑥ 禁止事項（§4.8） ===
  prohibited_actions:
    - silent_identification_switch: fatal
    - unauthorized_dag_modification: fatal
    - collider_adjustment_unauthorized: fatal
    - mediator_to_confounder_upgrade: fatal
    - refutation_skip: fatal
    - release_actionable_estimate_under_partial_diagnostic_only: fatal
    - reclassify_failed_required_test_as_not_applicable_post_hoc: fatal
    - positivity_violation_ignored: fatal
    - scope_gate_threshold_override: fatal
    - e_value_skip: warning
    - e_value_wrong_ci_side_for_protective_effect: fatal
    - unauthorized_intervention_execution: fatal
    - randomization_seed_override: fatal        # DoE Skill
    - doe_blocking_broken: warning              # DoE Skill
    - scope_gate_conditional_pass_action_release: fatal
    - scope_gate_failed_or_smt_extrapolation: fatal  # fail-close
    - scope_gate_substitute_one_check_pass_for_another: fatal
    - use_global_mahalanobis_alone_on_multimodal_support: fatal
    - ite_coverage_refutation_missing_when_ite_labeled_prediction: fatal
    - estimator_contract_silent_change: fatal
    - preregistration_manifest_post_hoc_modification: fatal

  # === ⑦ 再現性条件（L3: 実行レイヤ） ===
  library_stack:                                # 第3章 §3.7
    identification: dowhy==<ver>
    estimation: econml==<ver>
    refutation: dowhy==<ver>
  environment_lock_uri: "artifact://envs/<name>.lock"
  random_seed: 42
  container_sha256: "<sha256>"

  # === ⑧ 承認ゲート（§4.6） ===
  authorization_gates:
    dag_authorization:
      required_for: [dag_of_record_uri_change, dag_of_record_sha256_mismatch, identification_strategy_change]
      approver: research_lead
      provenance_field: dag_approved_by
      approval_evidence: [approver_signature, approval_timestamp, approval_scope]
    variable_selection_authorization:
      required_for: [confounders_declared_change, mediators_declared_change, colliders_declared_change, design_parameter_change, adjustment_set_approval_uri_change]
      approver: research_lead
      provenance_field: variables_approved_by
      approval_evidence: [approver_signature, approval_timestamp, approval_scope]
    intervention_execution_authorization:
      required_for: [physical_experiment_execution]
      approver: pi_and_facility_manager
      provenance_field: intervention_approved_by
      approval_evidence: [pi_signature, facility_manager_signature, approval_timestamp, approved_experiment_id, safety_review_id, intervention_execution_timestamp, egress_channels_declared, authorized_broadcast_targets]
      temporal_ordering:                          # Ch14 §14.3.4 back-register
        policy: approval_before_execution
        violation_severity: fatal

    # 承認者の独立性（利益相反対応 + facility scope escalation、§4.6.2 参照）
    approver_independence:
      conflict_policy: independent_reviewer_required_if_approver_is_data_producer
      fallback_approver: facility_causal_review_board
      facility_scope_escalation:
        applies_to:
          - dag_of_record_promotion_to_facility_standard
          - approved_instruments_registration
          - approved_donor_pool_registration
          - approved_meta_learner_covariate_set
        default_approver: facility_causal_review_board
        pre_conditions:                           # Ch14 §14.5.2 back-register
          - audit_manifest_v1_pass
        gate_level: L4_facility_standard_promotion

  # === ⑨ Agent action log（Ch14 §14.3.3 で back-register） ===
  agent_action_log:                              # Skill が発火した gate 相互作用の監査ログ
    gate_interaction_events:                     # 各 event は list of dict
      - gate_id: <string>                        # e.g., counterfactual_scope_gate | refutation_gate | L1_dag_authorization
        attempt_result: pass | fail | bypass_attempted | not_applicable
        timestamp: <timestamp>
        suppressed: <bool>                        # Skill が result を suppress したか（bypass detection 用）
    log_uri: <string>
    log_sha256: <string>
    log_append_only: true                        # immutability（改竄検出）

  # === ⑩ Egress control（Ch14 §14.3.6 で back-register） ===
  egress_control:                                # CATE 推薦 / 介入 recommendation の外部送信制御
    egress_channels_declared:                    # 事前に enum で宣言（whitelist）
      - internal_dashboard
      - facility_shared_repository
      - external_report_only_after_l3
      - other_project_transfer_only_after_reauth
    authorized_broadcast_targets:                # channel ごとの許可 target
      - target_id: <string>
        channel: <string_from_egress_channels_declared>
        authorized_at: <timestamp>
        authorized_by: <string>
    broadcast_events:                            # 実際の送信イベント log
      - event_id: <string>
        channel: <string>
        target: <string>
        timestamp: <timestamp>
        pre_l3_approval: <bool>                   # true なら fatal 候補

  # === ⑪ Estimator provenance reference（Ch9 refutation_gate 用、B-5 estimator hash chain） ===
  estimator_provenance_reference:                # Ch9 §9.7.1 で refutation_gate.pass の必須条件
    estimator_contract_sha256: <string>          # 本 Skill 全体のハッシュ
    feature_pipeline_sha256: <string>            # Ch8 covariates_deep 使用時に required
    feature_extractor_sha256: <string>           # 同上、深層特徴 provenance
    integration_distribution_sha256: <string>    # g-formula 使用時（Ch8 §8.4）

  # === ⑫ DoE 拡張（DoE Skill のみ、第10-12章） ===
  # DoE Skill の canonical SoT は Ch10 §10.8（本 §4.9 は abbreviated snippet、N-12）
  experimental_design_provenance:
    design_type: full_factorial       # or fractional_factorial / central_composite / ...
    randomization_seed: 42
    blocking_factors: [<factor>, ...]                  # Ch10 では blocking_factors_declared として top-level にも展開
    information_gain_metric: d_efficiency              # d_efficiency | a_efficiency | g_efficiency | v_efficiency (Ch10 §10.7.1)
    information_gain_threshold: 0.8
```

**テンプレートの読み方**：

- **① - ⑦ は vol-01/02/03 と同構造**——読者は既存 Skill 契約を段階的に因果 × DoE 拡張できます
- **⑧ 承認ゲート**は vol-04 で新設——実装は付録 B（MCP 実装パターン）で完成
- **⑨ Agent action log / ⑩ Egress control**は Ch14 §14.3.3 / §14.3.6 で back-register される Agentic 監査ログ
- **⑪ Estimator provenance reference**は Ch9 refutation_gate が消費する estimator ハッシュチェーン
- **⑫ DoE 拡張**は第10-12章の Skill でのみ埋める——因果推論のみの Skill では `not_applicable`

**成果物 2（3 層承認契約）**：⑧ の `authorization_gates` セクションを、Skill から独立した契約書として運用する場合の雛形は付録 A（Skill テンプレート集）に収録。

---

## 章末チェックリスト

- [ ] **vol-04 の 4 つの問い**（identification / DAG 再現 / 反実仮想 scope / エージェント権限）を列挙できる
- [ ] **§4.2 Table 4.1 の 6 要素拡張**を、自分の Skill に照らして各セル埋められる
- [ ] **§4.3 Table 4.2 の 6 点セット**——因果推論 Skill は 1-4、DoE Skill は 1-6——を暗記した
- [ ] **§4.4 Table 4.3 の因果 provenance 3 レイヤ**（L1 識別 / L2 識別仮定 / L3 実行）の区別ができる
- [ ] **§4.4.1 `positivity_by_stratum` canonical shape**（list-of-objects、estimator 別 required/optional）を書き下せる
- [ ] **§4.5 `counterfactual_scope_gate` の 4 判定 / 3 状態**（Mahalanobis cluster-conditional / 分散 / kNN 密度 / support envelope、pass / conditional_pass / fail）と 4 判定の相補性を理解した
- [ ] **§4.6 3 層承認ゲート**——`dag_authorization` / `variable_selection_authorization` / `intervention_execution_authorization`——それぞれの `required_for` と、`research_lead`（個別 PI-scope）vs `causal_review_board`（facility-scope + 利益相反 fallback）の使い分けを言える
- [ ] **§4.7 E-value と refutation は契約上の相補的チェック**（両方 pass で high confidence、統計的には完全独立ではなく同一の identification 仮定に依存）を理解した
- [ ] **§4.8 Table 4.4 禁止事項（12a-12e / 13-15 を含む 20 項目）**を、Severity（fatal / warning）付きで整理でき、Ch7-9 domain-specific fatals は各章参照であることを認識した
- [ ] **§4.9 の Skill 仕様書テンプレート**——① 目的から ⑫ DoE 拡張まで——を自分のテーマで埋められる

---

## 章末演習

### 演習 4.1：自分の Skill 仕様書の起草

第2章 演習 2.1 の DAG と、第3章 演習 3.1 のライブラリ選定を出発点に、**§4.9 のテンプレートを完全に埋めた Skill 仕様書**を作成してください。

必須項目：
- [ ] ① `estimand_type`（canonical population target enum）と `mediation_role` と `causal_question_type`
- [ ] ② `identification_strategy` + `dag_of_record_uri` + `adjustment_set_approval_uri` + `confounders_declared`
- [ ] ② 識別仮定 4 つの status（`checked` は positivity のみ、`positivity_by_stratum` は §4.4.1 canonical shape）
- [ ] ④ `success_criteria` の 6 項目（DoE 該当なしは `not_applicable`）
- [ ] ④ `declared_required_tests`（Ch9 10 test enum ch09_v0_3 から estimator 別に少なくとも 2 つ）+ `applicability_manifest_uri`
- [ ] ④ `sensitivity_analysis.threshold.minimum_e_value`（デフォルト 1.5 か、自研究室基準）+ `effect_direction` + `ci_bound_closest_to_null` + `smd_to_rr_conversion`（連続 outcome の場合）
- [ ] ④ `counterfactual_scope_gate` の 4 判定（Mahalanobis cluster-conditional / 分散 / kNN / support envelope）と 3 状態（pass / conditional_pass / fail）
- [ ] ⑥ `prohibited_actions` の Severity 付き列挙
- [ ] ⑦ `library_stack`（第3章 §3.7）
- [ ] ⑧ 3 層承認の承認者名（自研究室の役割で置換、facility_scope_escalation 含む）
- [ ] ⑪ `estimator_provenance_reference`（Ch9 refutation gate 用の hash chain）

### 演習 4.2：ゲート違反シナリオの検知

以下のシナリオそれぞれについて、**どの禁止事項に該当し、どの承認ゲートで防ぐか**を答えてください：

1. エージェントが Skill 実行中に dag_of_record_uri を上書きした
2. エージェントが後段の工程で得た selection collider を confounder 集合に追加した
3. エージェントが Skill 内部で backdoor → IV に切り替えた
4. エージェントが `counterfactual_scope_gate.distance_threshold` を 3.0 → 5.0 に上書きした
5. エージェントが Human 承認なしで実験装置に SET コマンドを送った

（解答は第14章の失敗パターン集で照合）

### 演習 4.3：3 レイヤ provenance の再現テスト

自分の Skill について、以下の再現テストを設計してください：

- [ ] **L1 再現**：`dag_of_record_sha256` を保存 → 別の環境で SHA を再計算 → 一致確認
- [ ] **L2 再現**：identification 仮定 4 つの assessment レポート（`evidence_uri`）を artifact として保存 → 別レビュアーが独立に評価
- [ ] **L3 再現（estimator 別 tolerance）**：
  - 決定論的推定器（線形・IPW 等）：`random_seed` と `library_stack` を pin し、ATE 点推定の**絶対誤差 < 1e-10**
  - ML-based CATE / DR-Learner / DML：**相対誤差 `rtol=1e-4` または信頼区間の overlap** を再現条件とする
  - MCMC / Bayesian（PyMC / CausalPy）：**posterior summary の tolerance（例：mean の相対誤差 < 1e-2）+ convergence diagnostics（$\hat{R} < 1.01$）** を条件とする

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
- **第13章 (13a/13b)**：本章の 3 層承認ゲートと `counterfactual_scope_gate` を Capstone として operational に統合
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
