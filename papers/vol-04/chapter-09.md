# 第9章　感度分析と Refutation — 因果主張の "検算" を Skill 化

> [!IMPORTANT]
> **本章の位置づけ**：第6-8章で ATE / ATT / LATE / ITT / CATE を推定する Skill を揃えました。しかし **推定値は仮定に依存**——unmeasured confounder / positivity 違反 / DAG misspecification のいずれかが破綻すれば因果主張は無効です。本章では**感度分析と refutation** を Skill 化し、「**refutation pass 未達なら結論を出さない**」契約を導入します。同時に、第8章で operational 化した `counterfactual_scope_gate` を **Ch9 側の再検証**にも組み込み、外挿範囲判定を "推定" と "検算" の両面から二重にガードします。

## 9.1 なぜ感度分析と refutation が必要か

### 9.1.1 因果推論の "3 層の疑い"

| 疑いの層 | 例 | 検算手法 |
|---|---|---|
| **仮定の破綻** | unmeasured confounder、positivity 違反 | **E-value / Rosenbaum bounds**（§9.2-9.3） |
| **モデルの誤指定** | 関数形の間違い、変数の重要な相互作用の欠落 | **placebo test、random common cause**（§9.4） |
| **標本の偶然** | 特殊なサブサンプルへの依存 | **subset validation、bootstrap refutation**（§9.5） |

**AI エージェントに感度分析を skip させると**：ATE の点推定を "確定した効果" として下流に流し、意思決定が破綻します（vol-04 第14章の失敗パターン）。**refutation pass はエージェントが結論を返す前提条件**とします。

### 9.1.2 本章の Skill 骨格

**refutation_gate**：以下の 10 テスト（本章の 5 基本テスト + Ch8 covariates_deep 用 2 テスト + Ch8/Ch12 拡張 3 テスト、canonical enum `ch09_v0_3` — §9.7.1 参照）のうち **契約で必須と宣言したもの全てが pass しない限り、estimator の結論は release できない**：

1. **E-value**（unmeasured confounder への強度スケール、§9.2）
2. **Rosenbaum bounds**（マッチング推定の感度分析、§9.3）
3. **Placebo test**（false effect の非検出、§9.4）
4. **Random common cause**（偽 confounder を加えても効果が保たれるか、§9.4）
5. **Data subset validation**（部分集合で効果が再現されるか、§9.5）
6. **Unobserved common cause strength**（unmeasured confounder の強度パラメータ scan、§9.7 拡張）
7. **Scope gate reverification**（Ch8 CATE / g-formula のみ、§9.7）
8. **ITE prediction coverage refutation**（`ite_labeled_prediction` 主張時のみ、§9.7）
9. **Prior predictive check**（Bayesian DoE のみ、Ch12 §12.2.3 で back-register）
10. **Prior data alignment**（Bayesian DoE のみ、Ch12 §12.7.2 で back-register）

以下では 5 基本テスト（1-5）を §9.2-§9.5 で先に扱い、canonical enum 完全形は §9.7.1 で定義する。

追加で **counterfactual_scope_gate re-verification**（Ch8 から継承、§9.6）を Ch9 側で独立に再検証。

---

## 9.2 E-value — unmeasured confounder への強度スケール

### 9.2.1 定義（VanderWeele & Ding, 2017）

観測された効果 $\widehat{\text{RR}}$（またはリスク比換算値）を **完全に説明消去する unmeasured confounder $U$ の最小強度**：

$$E\text{-value} = \widehat{\text{RR}} + \sqrt{\widehat{\text{RR}} \cdot (\widehat{\text{RR}} - 1)}$$

- $E$-value が **大きい**ほど「$U$ が処置と結果それぞれと $E$-value 倍以上強く関連しなければ効果は消えない」→ **頑健**
- ATE を RR に変換する際は outcome の分布（連続値なら標準化、二値なら直接 RR）に依存

### 9.2.2 vol-03 第9章との cross-reference

vol-03 第9章の **BNN posterior による不確かさ** は「**モデル内側の不確かさ**（parameter uncertainty + data noise）」を扱いますが、E-value は「**モデル外側の不確かさ**（unmeasured confounder）」を扱います：

| 種別 | vol-03 Ch9（BNN posterior） | vol-04 Ch9（E-value） |
|---|---|---|
| 対象 | model parameter の不確かさ | 未観測 confounder の影響 |
| 出力 | credible interval, entropy | E-value（大きいほど頑健） |
| 破綻時 | 予測分散大 → 判断保留 | E-value 小 → confounder 疑い |
| 補完関係 | どちらも "不確かさ" だが**由来が違う**——両方報告が推奨 | 同左 |

> [!WARNING]
> **BNN の狭い credible interval は E-value の大きさを保証しない**——「モデル内で確信度高くても、未観測 confounder で反転しうる」ことをエージェントが混同すると意思決定を誤ります。Skill 契約は両者の同時報告を強制。

### 9.2.3 Skill 契約（E-value）

```yaml
refutation_test: e_value
e_value_config:
  library: dowhy | evalue_r_port | custom
  library_version: <string>
  outcome_scale: risk_ratio | odds_ratio | standardized_mean_difference
  effect_direction: harmful | protective          # RR > 1 (harmful) or RR < 1 (protective)
  rr_oriented_away_from_null: <float>              # protective の場合 1/RR に反転してから E-value 計算
  ci_bound_closest_to_null: <float>                # harmful → CI 下限、protective → 1/CI 上限
  smd_to_rr_conversion:                            # 連続 outcome の場合必須
    method: vanderweele_2020 | chinn_2000 | none_analog_only
    baseline_risk: <float>                          # 変換に必要な baseline
    conversion_evidence_uri: <string>
    conversion_assumption_uri: <string>
  point_estimate_uri: <string>
  point_estimate_sha256: <string>
  confidence_interval_uri: <string>
  e_value_point: <float>                          # 点推定に対する E-value
  e_value_ci_bound: <float>                        # CI の null 側限界に対する E-value
  threshold: <float>                               # 事前登録された "頑健と見なす" 閾値
  threshold_rationale_uri: <string>                # 閾値の設定根拠（ドメイン知識に基づく）
  posterior_cross_reference_uri: <string>          # vol-03 BNN posterior との対比レポート（§9.2.2）
refutation_pass:
  criterion: e_value_ci_bound >= threshold
  status: pass | fail | insufficient_data
prohibited_actions:
  - report_e_value_from_point_estimate_only              # fatal（CI 下限を報告必須）
  - use_wrong_ci_side_for_protective_effect               # fatal（RR<1 の場合は 1/RR 反転 + 上限使用）
  - compute_e_value_for_continuous_outcome_without_conversion_method  # fatal（SMD→RR 変換方法明示必須）
  - report_e_value_pass_when_conversion_metadata_absent   # fatal
  - set_threshold_post_hoc                                # fatal（事前登録必須）
  - report_effect_without_e_value_when_declared_required  # fatal
  - confuse_e_value_with_bnn_posterior_uncertainty        # fatal（意味論混同）
```

> [!WARNING]
> **連続 outcome の E-value は "analog" 扱い**——SMD → RR 変換方法（VanderWeele 2020 / Chinn 2000 等）が事前登録されず、baseline_risk が指定されていない場合、E-value 値は**意味を成しません**。ARIM の破壊靭性のような連続 outcome では、変換方法を明示するか、**変換なしで analog 扱いとして「E-value は使わず placebo + subset を主 refutation にする」ことを事前宣言**します。

### 9.2.4 ARIM ケース

装置プロトコル切替の CATE で $\widehat{\text{RR}} = 1.8$、CI 下限 = 1.3：
- $E$-value（点推定）= 3.0、CI 下限 E-value = 1.9
- 事前登録閾値 = 2.0 → **fail**（CI 下限で頑健性不足）
- 未観測 confounder（オペレータ熟練度など）で 1.9 倍以上の関連が想定されうるため要注意

---

## 9.3 Rosenbaum bounds — マッチング推定の感度分析

### 9.3.1 骨格

**Rosenbaum bounds** はマッチング推定（第6章の PSM）に対して「**マッチした処置群 vs 対照群のオッズ比が $\Gamma$ 倍まで偏っていても、効果が有意である**」という感度パラメータ $\Gamma$ を算出：

- $\Gamma = 1$：完全 unconfounded を仮定
- $\Gamma$ が **大きい**ほど、より強い hidden bias に耐える → 頑健

**Skill での位置づけ**：**PSM 推定の副産物**として必ず出力させる（第6章 estimator 契約と連携）。

### 9.3.2 Skill 契約

```yaml
refutation_test: rosenbaum_bounds
rosenbaum_config:
  library: rbounds_r_port | dowhy_extension
  library_version: <string>
  matched_pairs_uri: <string>                  # PSM の pair 出力
  matched_pairs_sha256: <string>
  gamma_grid: [1.0, 1.5, 2.0, 2.5, 3.0]        # 事前登録
  p_value_per_gamma_uri: <string>
  critical_gamma: <float>                       # p_value < 0.05 が保たれる最大 Γ
  critical_gamma_threshold: <float>             # 事前登録された頑健性下限
  threshold_rationale_uri: <string>
refutation_pass:
  criterion: critical_gamma >= critical_gamma_threshold
  status: pass | fail | insufficient_data
prohibited_actions:
  - run_rosenbaum_on_non_matched_estimator                   # fatal（適用範囲外）
  - expand_gamma_grid_after_seeing_results                   # fatal（fishing）
  - set_critical_gamma_threshold_post_hoc                    # fatal
```

### 9.3.3 適用条件

| 推定器 | Rosenbaum bounds 適用可 |
|---|---|
| PSM（第6章） | ✅ |
| IPW（第6章） | ⚠️（sensitivity analog：Ding-VanderWeele 版が必要） |
| DR / DR-Learner（第6章） | ⚠️（別の framework、E-value + placebo 主体） |
| DiD（第7章） | ❌（parallel trends の sensitivity は placebo/等価性検定） |
| IV（第7章） | ❌（exclusion sensitivity は別途） |
| Synthetic Control（第7章） | ❌（in-space placebo が代替） |
| CATE（第8章） | ⚠️（stratum-wise で近似可、EconML の robustness_value） |

---

## 9.4 Placebo test と Random common cause — DoWhy の refutation ツール

### 9.4.1 Placebo test（false effect の非検出）

**手続き**：処置 $T$ を **無関係な列**（例：ランダム permutation、日付列、ID）に置換して同じ推定を実行 → 効果が 0 に近いことを確認。

> [!WARNING]
> **単純な marginal permutation は adjustment set $W$ との joint 分布を破壊**し、positivity / overlap の artifact を生む——placebo の fail/pass が推定器の妥当性でなく「合成 assignment の異常」を反映してしまう。**必ず `permutation_scheme: marginal | within_stratum | conditional_on_W` を明示** + **placebo assignment 後の positivity 診断** を通す。

**Skill 契約**：
```yaml
refutation_test: placebo
placebo_config:
  library: dowhy | custom
  library_version: <string>
  placebo_method: random_permutation | random_common_cause_variable | date_column
  permutation_scheme: marginal | within_stratum | conditional_on_W   # W-adjusted design 保護
  adjustment_set_uri: <string>
  dag_of_record_uri: <string>
  post_placebo_positivity_check_uri: <string>   # placebo assignment 後の positivity 診断
  n_replications: 100
  seed_uri: <string>                            # 事前登録された乱数 seed リスト
  placebo_effect_distribution_uri: <string>
  placebo_effect_ci_uri: <string>
  original_effect_estimate: <float>
  criterion: placebo_effect_ci_contains_zero_and_original_outside_placebo_ci
refutation_pass:
  status: pass | fail | insufficient_data
prohibited_actions:
  - report_effect_when_placebo_effect_significantly_nonzero   # fatal
  - use_marginal_permutation_without_W_joint_preservation      # fatal（W 保護なし）
  - proceed_when_post_placebo_positivity_fails                 # fatal（合成 assignment 異常）
  - reduce_n_replications_below_declared                       # fatal
  - reroll_placebo_seeds_until_pass                            # fatal（fishing）
```

### 9.4.2 Random common cause（偽 confounder への robustness）

**手続き**：ランダム生成した confounder $U'$ をモデルに加えて再推定 → **効果推定が変わらない**ことを確認。

**Skill 契約**：
```yaml
refutation_test: random_common_cause
rcc_config:
  library: dowhy
  library_version: <string>
  n_replications: 50
  effect_shift_metric: absolute | relative_to_original | standardized   # 明示必須
  effect_shift_threshold: <float>               # metric に応じ設定（例：absolute 0.05, relative 0.05）
  effect_shift_threshold_rationale_uri: <string>
  near_zero_denominator_policy: use_absolute_when_original_below_epsilon  # 相対 metric の 0 割回避
  effect_shift_distribution_uri: <string>
  median_effect_shift: <float>
  p95_effect_shift: <float>                     # tail での不安定性検出
refutation_pass:
  criterion: median_effect_shift <= effect_shift_threshold AND p95_effect_shift <= (1.5 * effect_shift_threshold)
  status: pass | fail | insufficient_data
prohibited_actions:
  - report_effect_when_effect_shift_exceeds_threshold  # fatal
  - report_median_shift_without_p95_tail               # fatal（median 単独は不安定性を隠蔽）
  - use_relative_metric_when_original_effect_near_zero_without_fallback  # fatal（0 割回避必須）
```

### 9.4.3 unmeasured confounder への強度スケール

第4章と第6章で列挙した「観測できない confounder」を **どの程度の強度なら効果が保たれるか** を DoWhy `add_unobserved_common_cause` で試算——**強度スケールは partial R² / observed benchmark に紐付け**（Cinelli-Hazlett omitted variable bias framework）：

```yaml
refutation_test: unobserved_common_cause_strength
uucs_config:
  library: dowhy | sensemakr
  strength_scale: partial_r_squared | observed_confounder_benchmark | absolute
  strength_scale_rationale_uri: <string>
  observed_confounder_benchmark_uri: <string>   # 最強の観測 confounder の強度を benchmark に
  effect_strength_on_treatment_grid: [...]      # partial R² スケール（例：[0.01, 0.05, 0.1, 0.2]）
  effect_strength_on_outcome_grid: [...]        # 同上
  effect_estimate_matrix_uri: <string>          # 2D grid での効果推定
  threshold_of_practical_significance: <float>   # 事前登録
  robustness_value_uri: <string>                 # Cinelli-Hazlett robustness value
refutation_pass:
  criterion: effect_estimate_remains_practically_significant_up_to_declared_strength
  status: pass | fail | insufficient_data | not_applicable   # canonical enum (Ch9 §9.7.1)
prohibited_actions:
  - use_absolute_strength_grid_without_scale_rationale  # fatal
  - benchmark_against_weakest_observed_confounder        # fatal（最強でベンチマーク）
```

**vol-03 BNN posterior との対比**：BNN が示す狭い credible interval が **必ずしも unmeasured confounder への頑健性を意味しない**ことをレポートで明示（§9.2.2）。

---

## 9.5 Data subset validation — 部分集合での再現性

### 9.5.1 骨格

データの部分集合（ランダム / 装置別 / 時期別 / 組成別）で推定を繰り返し、**効果推定が一貫**していることを確認：

```yaml
refutation_test: data_subset_validation
subset_config:
  library: dowhy | custom
  subset_manifest_uri: <string>                  # 事前登録された subset 定義（immutable）
  subset_manifest_sha256: <string>
  subset_definitions:
    - name: random_50pct
      family: random_resample
      method: random_sample
      proportion: 0.5
      n_replications: 20
      seed_uri: <string>                           # seed 事前登録
      pass_criterion: subset_estimates_within_ci_of_original
      pass_threshold: <float>                       # family 別（例：0.8）
      pass_threshold_rationale_uri: <string>
    - name: by_device
      family: grouped_loo
      method: leave_one_group_out
      group_column: device_id
      pass_criterion: subset_estimates_within_ci_of_original
      pass_threshold: <float>                       # grouped は family_wise error 制御
      multiple_testing_policy: bonferroni | holm | none
    - name: by_time_period
      family: grouped_loo
      method: leave_one_group_out
      group_column: quarter
      pass_criterion: subset_estimates_within_ci_of_original
      pass_threshold: <float>
      multiple_testing_policy: bonferroni | holm | none
    - name: by_composition_bin
      family: grouped_loo
      method: leave_one_group_out
      group_column: composition_bin
      pass_criterion: subset_estimates_within_ci_of_original
      pass_threshold: <float>
      multiple_testing_policy: bonferroni | holm | none
  subset_effect_distribution_uri: <string>
  original_effect_estimate: <float>
  ci_level: 0.9
refutation_pass:
  criterion: all_families_meet_own_pass_threshold_with_multiple_testing_correction
  status: pass | fail | insufficient_data
prohibited_actions:
  - exclude_subsets_failing_after_seeing_results             # fatal（fishing）
  - modify_subset_manifest_after_running                      # fatal
  - pool_random_and_grouped_subsets_into_single_pass_fraction # fatal（family 別に評価必須）
  - reduce_n_replications_below_declared                      # fatal
  - report_effect_when_any_family_below_threshold             # fatal
```

### 9.5.2 ARIM での典型的 subset

| Subset | 目的 | 期待される挙動 |
|---|---|---|
| random 50% | 標本の偶然除去 | 効果が CI 内で保たれる |
| 装置別 leave-one-out | 装置固有 confounder 検出 | 大きな shift があれば device-CATE で調べ直し |
| 時期別 leave-one-out | 時間依存 confounder 検出 | 較正刷新期を除いても効果が保たれるか |
| 組成別 leave-one-out | 効果 heterogeneity 検出 | CATE と整合するか（第8章と cross-check） |

---

## 9.6 `counterfactual_scope_gate` の Ch9 側の再検証

### 9.6.1 なぜ再検証か + "独立" の定義

第8章で operational 化した `counterfactual_scope_gate` は **推定段階（Ch8 CATE Skill 内）** での判定でした。Ch9 refutation の一環として、**独立に再検証**します——ただし "独立" の意味を 2 モードに厳密に分けます：

| モード | 意味 | 使い分け |
|---|---|---|
| **`audit_recompute_same_artifacts`** | Ch8 の凍結された preprocessing / feature 変換 / cluster 割当 / 閾値 / 学習分布 hash を **再利用**、実装（コード）だけ独立 | Ch8 の実装バグ / 再現性検証 |
| **`data_update_recompute`** | 新データや再学習で **feature / cluster / 閾値をゼロから再計算** | データ更新後の scope 再確認 |

- **重要**：cluster assignment・covariance 推定・閾値を **勝手に再計算する** と、Ch8 との disagreement が「causal な scope 違反」でなく「metric 定義の差異」を反映するため、gate 判定として意味を成しません。**`audit_recompute_same_artifacts` モードでは Ch8 artifact hash 一致が必須**。
- **`data_update_recompute` モードでは、disagreement は必ず `estimator_contract_change_gate` を通す**——silent に "新しい閾値で pass" にしない。

### 9.6.2 Skill 契約

```yaml
refutation_test: scope_gate_reverification
scope_reverify_config:
  mode: audit_recompute_same_artifacts | data_update_recompute
  cate_prediction_artifacts_uri: <string>       # Ch8 側の scope_gate 判定 artifact
  cate_prediction_artifacts_sha256: <string>
  frozen_ch8_artifacts:                          # audit モードで必須
    preprocessing_pipeline_sha256: <string>
    feature_transform_sha256: <string>
    cluster_assignment_sha256: <string>
    covariance_estimation_sha256: <string>
    threshold_manifest_sha256: <string>
    training_distribution_sha256: <string>
  independent_recomputation:
    mahalanobis_distance_recomputed: <float>
    cluster_conditional_mahalanobis_recomputed: <float>
    knn_density_recomputed: <int>
    support_envelope_recomputed: <string>       # in / out per dimension
    recomputation_code_uri: <string>             # Ch8 と別実装であることの証跡
  agreement_check:
    ch8_gate_status: pass | conditional_pass | fail
    ch9_gate_status: pass | conditional_pass | fail
    audit_mode_disagreement_policy: fail_close_and_flag_ch8_implementation_bug
    data_update_disagreement_policy: escalate_to_estimator_contract_change_gate
refutation_pass:
  criterion: (mode == "audit_recompute_same_artifacts") ? ch8_gate_status == ch9_gate_status AND both == "pass"
            : (mode == "data_update_recompute") ? ch9_gate_status == "pass" AND ecc_gate_status == "approved"
  status: pass | fail | insufficient_data | not_applicable   # canonical enum (§9.7.1)
  disagreement_flag: true | false                            # audit vs Ch8 の一致不能を通知（status とは直交）
  # policy: disagreement_flag=true でも status は必ず canonical 4 値のうち fail (audit mode) または
  #         estimator_contract_change_gate 経由の再承認結果に紐付ける (data_update mode)
prohibited_actions:
  - reuse_ch8_gate_status_without_independent_recomputation      # fatal
  - suppress_disagreement_by_recomputing_with_ch8_thresholds     # fatal
  - switch_modes_after_running_recomputation                     # fatal（mode 事前登録必須）
  - recompute_cluster_assignment_in_audit_mode                    # fatal（audit は artifact 再利用のみ）
  - proceed_data_update_recompute_without_ecc_gate_approval      # fatal
```

### 9.6.3 disagreement の扱い

Ch8 で `pass` が Ch9 で `fail` に：
- データ更新／モデル再学習で分布が変わった可能性
- **fail-close** + reviewer 通知 + `estimator_contract_change_gate` で再承認

---

## 9.7 refutation_gate — Skill 側の "結論を出さない" 契約

### 9.7.1 契約の骨格

推定 Skill（Ch6-8）の release は、**refutation_gate.status == pass を前提条件**とします：

```yaml
refutation_gate:
  enum_version: ch09_v0_3                        # canonical enum version（B-10）
  # 変更履歴:
  #   ch09_v0_1: 初版 8 tests (e_value, rosenbaum_bounds, placebo, random_common_cause,
  #                            unobserved_common_cause_strength, data_subset_validation,
  #                            scope_gate_reverification, ite_prediction_coverage_refutation)
  #   ch09_v0_2: (欠番、Ch12 統合準備)
  #   ch09_v0_3: Ch12 §12.2.3 / §12.7.2 で prior_predictive_check + prior_data_alignment を追加（Bayesian DoE のみ、+2 tests → 計 10 tests）
  # 版数の silent downgrade は Ch14 §14.3.3 の fatal downgrade_declared_required_tests_enum_version_silently に該当。
  preregistration_manifest_uri: <string>         # 事前登録された全設定の manifest
  preregistration_manifest_sha256: <string>
  preregistration_approved_at: <timestamp>
  preregistration_approver: causal_review_board
  applicability_manifest_uri: <string>           # どの test が not_applicable か immutable 宣言
  applicability_manifest_sha256: <string>

  # --- estimator provenance reference（Ch8 → Ch9 hash chain、深層特徴 provenance を継承）---
  # refutation_gate.pass を release する前に、estimation 時と refutation 実行時で
  # estimator 契約が変わっていないことをハッシュで確認する
  estimator_provenance_reference:
    estimator_contract_sha256: <string>          # Ch6/7/8 の estimator 契約全体のハッシュ
    dag_of_record_sha256: <string>               # Ch5 approval Skill が発行
    adjustment_set_approval_uri: <string>        # 承認証跡（本ハッシュに含まれる）
    feature_pipeline_sha256: <string>            # Ch8 covariates_deep 使用時に required
    feature_extractor_sha256: <string>           # Ch8 §8.4.3 深層特徴 provenance（foundation model / DINO 等）
    preprocessing_config_sha256: <string>        # Ch8 §8.4.3
    integration_distribution_sha256: <string>    # Ch8 §8.4 g-formula 使用時に required
    dimensionality_reduction_fit_sha256: <string> # Ch8 §8.4.3 PCA / UMAP fit hash（該当時）

  declared_required_tests:                       # 事前登録された必須 refutation
    - e_value
    - rosenbaum_bounds                            # PSM のみ
    - placebo
    - random_common_cause
    - unobserved_common_cause_strength
    - data_subset_validation
    - scope_gate_reverification                   # Ch8 CATE / g-formula のみ
    - ite_prediction_coverage_refutation          # ite_labeled_prediction 主張時のみ
    - prior_predictive_check                      # Ch12 §12.2.3 — Bayesian DoE のみ (applicability_manifest 要)
    - prior_data_alignment                        # Ch12 §12.7.2 — Bayesian DoE のみ (applicability_manifest 要)
  test_results:
    e_value: {status: pass | fail | insufficient_data | not_applicable, evidence_uri: ...}
    rosenbaum_bounds: {status: ..., evidence_uri: ...}
    placebo: {status: ..., evidence_uri: ...}
    random_common_cause: {status: ..., evidence_uri: ...}
    unobserved_common_cause_strength: {status: ..., evidence_uri: ...}
    data_subset_validation: {status: ..., evidence_uri: ...}
    scope_gate_reverification: {status: ..., evidence_uri: ...}
    ite_prediction_coverage_refutation: {status: ..., evidence_uri: ...}
    prior_predictive_check: {status: ..., evidence_uri: ...}       # Ch12 §12.2.3
    prior_data_alignment: {status: ..., evidence_uri: ...}         # Ch12 §12.7.2
  aggregate_status: pass | fail | partial_diagnostic_only
  aggregate_policy:
    pass_requires: all_declared_required_tests_status_pass
    fail_when_any_required_test_status_is_fail: true
    fail_when_any_required_test_status_is_insufficient_data: true
    not_applicable_requires_preregistered_applicability_manifest: true
    partial_diagnostic_only_when: never_for_required_tests_only_for_supplementary
    partial_output: diagnostic_only              # supplementary test の一部欠落時のみ、actionable 結論なし
    fail_close_action: refuse_release_and_notify_reviewer
    estimator_provenance_pass_requires: all_referenced_hashes_match_at_execution_time
  linkage_to_estimator_contract_change_gate:      # refutation fail 時の re-fit 経路
    on_fail: escalate_to_ecc_gate_L3_or_L4
    silent_re_run_forbidden: true
prohibited_actions:
  - release_estimator_result_when_refutation_gate_fail          # fatal
  - release_estimator_result_when_refutation_gate_partial_diagnostic_only  # fatal（actionable 数値報告禁止）
  - reduce_declared_required_tests_after_running                # fatal（事前登録改変）
  - reclassify_failed_required_test_as_not_applicable_post_hoc  # fatal（applicability manifest は事前登録・凍結）
  - reclassify_failed_required_test_as_insufficient_data_to_evade_fail  # fatal
  - swap_failing_test_for_another_test                          # fatal
  - report_partial_as_pass                                       # fatal
  - modify_preregistration_manifest_after_approval              # fatal
  - release_refutation_pass_when_estimator_provenance_hashes_mismatch  # fatal（estimation 時と refutation 実行時で契約変化）
  - silently_refit_deep_feature_extractor_between_estimation_and_refutation  # fatal
```

> [!IMPORTANT]
> **`partial_diagnostic_only` は required test の一部欠落を許容する状態ではありません**。required test のいずれかが `fail` または `insufficient_data` の場合は必ず `fail` になります。`not_applicable` を宣言できるのは **事前登録された applicability manifest に列挙されているケースのみ**——事後に "この test は適用外" と再分類することは fatal。

### 9.7.2 estimator 別の必須 refutation

| Estimator | E-value | Rosenbaum | Placebo | RCC | UCC Strength | Subset | Scope Re-verify | ITE Coverage |
|---|---|---|---|---|---|---|---|---|
| PSM / Matching（Ch6） | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |
| IPW / DR（Ch6） | ✅ | ⚠️（analog） | ✅ | ✅ | ✅ | ✅ | — | — |
| DR-Learner（Ch6） | ✅ | — | ✅ | ✅ | ✅ | ✅ | — | — |
| DML（Ch6） | ✅ | — | ✅ | ✅ | ✅（partial R²）| ✅ | — | — |
| DiD（Ch7） | ✅（analog） | — | ✅（in-time） | ⚠️ | ⚠️ | ✅ | — | — |
| IV（Ch7） | ⚠️（exclusion sensitivity） | — | ⚠️ | ✅ | ⚠️ | ✅ | — | — |
| Synthetic Control（Ch7） | — | — | ✅（in-space + in-time） | ⚠️ | — | ⚠️（donor LOO） | — | — |
| CATE S-Learner（Ch8） | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️（ITE 主張時） |
| CATE T-Learner（Ch8） | ✅ | ⚠️（stratum） | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| CATE X-Learner（Ch8） | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| CATE DR-Learner（Ch8） | ✅ | — | ✅ | ✅ | ✅（partial R²）| ✅ | ✅ | ⚠️ |
| CATE R-Learner（Ch8） | ✅ | — | ✅ | ✅ | ✅（partial R²）| ✅ | ✅ | ⚠️ |
| g-formula（Ch8） | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| ite_labeled_prediction（Ch8） | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ | **✅ 必須** |

- ✅：必須
- ⚠️：analog / 別 framework で代替（applicability manifest で "analog として代替" を宣言必須）
- —：適用外（applicability manifest で "not_applicable" を宣言必須）

### 9.7.3 ITE coverage refutation（新規）

`ite_labeled_prediction`（Ch8 §8.6.3）を主張する場合、**個体レベル予測区間の empirical coverage** を Ch9 で必ず検証：

```yaml
refutation_test: ite_prediction_coverage_refutation
ite_coverage_config:
  library: mapie | custom
  holdout_dataset_uri: <string>
  holdout_dataset_sha256: <string>
  prediction_intervals_uri: <string>            # Ch8 の ite_labeled_prediction.prediction_interval_uri
  target_coverage: 0.9                           # Ch8 の coverage_target と一致必須
  empirical_coverage: <float>                    # holdout での実測
  coverage_tolerance: 0.02                       # 例：0.88 - 0.92 なら pass
  miscoverage_by_stratum_uri: <string>           # 層別で miscoverage を検出（heterogeneous な信頼性）
  calibration_diagnostic_uri: <string>
refutation_pass:
  criterion: |empirical_coverage - target_coverage| <= coverage_tolerance
             AND max_stratum_miscoverage <= 2 * coverage_tolerance
  status: pass | fail | insufficient_data
prohibited_actions:
  - report_ite_prediction_when_coverage_fails_target       # fatal
  - use_training_data_for_coverage_verification            # fatal（必ず holdout）
  - report_average_coverage_only_without_stratum_check     # fatal
  - loosen_coverage_tolerance_after_seeing_results         # fatal
```

### 9.7.4 refutation checklist（人間側のレビュー用）

Skill が pass を返しても、**Human review 段階**で以下をチェック：

- [ ] `preregistration_manifest_sha256` と `applicability_manifest_sha256` が事前承認済み、`estimator_provenance_reference` の全ハッシュ（estimator_contract_sha256 / dag_of_record_sha256 / feature_pipeline_sha256 等）が estimation 時と一致
- [ ] 全 declared_required_tests が事前登録されていたか（後から追加・削除がないか）
- [ ] 閾値（E-value threshold、Rosenbaum critical_gamma_threshold、subset pass_threshold、coverage_tolerance 等）が事前登録されていたか
- [ ] placebo seed / subset definition / gamma_grid が事前登録されていたか（fishing 防止）
- [ ] Ch8 scope gate と Ch9 再検証で **status 一致**か（audit / data_update モードのどちらか）
- [ ] vol-03 BNN posterior と E-value の対比が報告に含まれているか
- [ ] 失敗した test を "not_applicable" と後付けで再分類していないか（applicability manifest との突合）
- [ ] ITE 主張時に `ite_prediction_coverage_refutation` の empirical coverage が target に一致するか

---

## まとめ — 本章のチェックリスト

- [ ] **§9.1 3 層の疑い**——仮定破綻・モデル誤指定・標本の偶然、それぞれに対応する refutation 手法を挙げられる
- [ ] **§9.2 E-value**——effect_direction / ci_bound_closest_to_null の使い分け、連続 outcome での SMD→RR 変換要件、vol-03 BNN posterior との違いを説明できる
- [ ] **§9.3 Rosenbaum bounds**——PSM / IPW での適用と critical_gamma_threshold の事前登録を説明できる
- [ ] **§9.4 Placebo / RCC / UCC strength**——permutation_scheme の W 保護、RCC の absolute vs relative metric、Cinelli-Hazlett partial R² スケールを説明できる
- [ ] **§9.5 Subset validation**——subset_manifest の凍結、family 別 pass_threshold、multiple testing 制御を書き下せる
- [ ] **§9.6 Scope gate 再検証**——audit_recompute_same_artifacts と data_update_recompute の 2 モード分離と disagreement policy を説明できる
- [ ] **§9.7 refutation_gate**——preregistration_manifest hash、applicability manifest の事前凍結、`partial_diagnostic_only` の限定用途、required test failing → aggregate fail の semantics を理解した
- [ ] **§9.7.3 ITE coverage refutation**——holdout での empirical coverage、stratum 別 miscoverage、tolerance の事前登録を説明できる

---

## 章末演習

### 演習 9.1：E-value 閾値の設定

自分の ARIM 分野で以下を答えてください：

- [ ] 想定される未観測 confounder の強度（例：オペレータ熟練度が処置と outcome にどの程度関連しうるか）
- [ ] その強度に対応する E-value threshold（1.5 / 2.0 / 2.5 / 3.0）
- [ ] 閾値の rationale（ドメイン知識に基づく数値の根拠）
- [ ] BNN posterior（vol-03）との対比を報告書のどこに書くか

### 演習 9.2：Rosenbaum bounds の適用可否

以下の推定シナリオで Rosenbaum bounds を適用すべきか、代替 refutation は何かを答えてください：

1. PSM で装置プロトコル効果を推定
2. DML で高次元 confounder を制御した CATE
3. Synthetic Control で単一装置の較正効果
4. DiD で 2 装置 pre/post 比較

### 演習 9.3：subset validation の設計

以下の設定で subset definition を書き下してください：

- 装置 A/B/C/D の 4 台、2 年間データ、outcome = 破壊靭性 $K_{IC}$、処置 = プロトコル切替
- 組成範囲 $c_1 \in [0.2, 0.6]$
- 期待される heterogeneity：装置別・組成別 CATE

- [ ] 装置 LOO subset の定義
- [ ] 組成 bin LOO subset の定義（bin の切り方も）
- [ ] 時期 LOO subset の定義（何四半期を残すか）
- [ ] `min_subsets_passing` の閾値
- [ ] 各 subset の効果推定を Ch8 CATE map と cross-check する手続き

### 演習 9.4：refutation_gate の設計

第6-8章で作成した estimator のいずれかを選び、以下を設計してください：

- [ ] declared_required_tests（§9.7.2 の表を参考に）
- [ ] 各 test の閾値（事前登録）
- [ ] partial 状態時の対応（診断出力のみ、actionable 結論なし）
- [ ] Ch8 scope gate と Ch9 再検証の disagreement policy
- [ ] Human review checklist（§9.7.3）を運用する reviewer role

---

## 参考資料

### 内部 cross-reference

- **第0章**：4 識別仮定（本章の各 refutation はそれぞれの仮定破綻に対応）
- **第4章**：`counterfactual_scope_gate` 契約定義（本章 §9.6 で Ch8 判定を独立再検証）、3 層承認ゲート
- **第5章**：DAG、DAG proposal / approval Skill（本章 §9.6 で scope gate 再検証時の dag_of_record_sha256 突合に使用）
- **第6章**：PSM / IPW / DR / DML（本章 §9.2-9.3 の主対象、`estimand_type` + `positivity_by_stratum`）、`estimator_contract_change_gate`（本章 refutation fail 時の再承認経路）
- **第7章**：DiD parallel trends assessment、IV exclusion restriction（本章 §9.4 placebo と in-space/in-time で対応）、SC placebo（`in_space_placebo` + `in_time_placebo`）
- **第8章**：CATE / g-formula の `counterfactual_scope_gate` operational 実装（本章 §9.6 で再検証）
- **第13章 (13a)**：Phase 1-2 で本章の refutation を必須通過ステップとして統合
- **第14章**：refutation スキップ・placebo 未実施・random common cause 削除などの失敗パターン
- **vol-03 第9章**：BNN posterior による不確かさ（本章 §9.2.2 で E-value と対比）
- **付録 A**：refutation Skill テンプレート、`refutation_gate` YAML 雛形
- **付録 D**：E-value / Rosenbaum bounds / Placebo test / Random common cause の用語

### 外部参考文献

- VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research: introducing the E-value. *Annals of Internal Medicine*, 167(4), 268-274.
- Ding, P., & VanderWeele, T. J. (2016). Sensitivity analysis without assumptions. *Epidemiology*, 27(3), 368-377.
- Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.). Springer. Chapter 4 (sensitivity bounds).
- Rosenbaum, P. R. (2010). *Design of Observational Studies*. Springer.
- Sharma, A., & Kiciman, E. (2020). DoWhy: An end-to-end library for causal inference. *arXiv:2011.04216*.
- Cinelli, C., & Hazlett, C. (2020). Making sense of sensitivity: Extending omitted variable bias. *Journal of the Royal Statistical Society: Series B*, 82(1), 39-67.
- Ioannidis, J. P. A., Tan, Y. J., & Blum, M. R. (2019). Limitations and misinterpretations of E-values for sensitivity analyses of observational studies. *Annals of Internal Medicine*, 170(2), 108-111.
- DoWhy documentation: https://www.pywhy.org/dowhy/

---

**本章の位置づけ**：第I部（Ch1-3）+ 第II部（Ch4-9）で **観測データからの因果推論 Skill と検算の全パイプライン**が揃いました。次章（Ch10）から**第III部：実験計画（DoE）の Skill 化**に入ります——観測データで得た仮説を、randomization / blocking / factorial design で **検証**する Skill 群を構築します。本章 refutation gate は Ch13a の Phase 1-2（観測データからの CATE → DoE 設計）でも必須通過ステップとして再登場します。
