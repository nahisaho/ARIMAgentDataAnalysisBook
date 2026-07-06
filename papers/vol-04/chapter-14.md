# 第14章　因果 × Agentic 特有の失敗パターンと監査

> **本章の到達目標**
>
> - 因果推論一般で頻出する 6 つの失敗パターン（DAG misspecification、collider bias、positivity violation、外挿 unwarranted、CATE 過剰個別化、refutation スキップ）の **operational な検出契約** を書ける
> - DoE 一般で頻出する 4 つの失敗パターン（randomization 破綻、blocking 失敗、応答曲面外挿誤用、タグチ SN 比誤解釈）を **Ch10-11 の provenance に照らして** 監査できる
> - Agentic 特有の 6 つの failure mode（DAG 勝手書換、confounder 勝手削除、感度分析 skip、"介入した" 勝手記録、seed 上書き、CATE 推薦 Human 未承認送信）を **Ch4 の 3 層承認契約に照らして** 検出できる
> - 各失敗パターンに対応する **fatal action** を Ch4-13 の canonical から引用して監査 manifest に落とせる

## 14.0 章の位置づけと監査の三層モデル

第13章までで、因果推論 × 実験計画 × Agentic の **成功パス**（DAG 承認 → 変数選択承認 → 介入実行承認 → evidence chain 完成）を書き切りました。本章は逆に、**失敗パス**を体系化し、それらを **Skill 契約と audit manifest** に反映する方法を扱います。

本章の失敗パターンは 3 つの層から構成されます：

- **層 1（因果推論一般）**：因果推論の理論・実装上の失敗。Skill が単独で導入するのではなく、**因果推論の教科書的な落とし穴** を Agentic 環境で自動検出可能な契約に翻訳する層。
- **層 2（DoE 一般）**：実験計画の理論・実装上の失敗。Ch10-12 の provenance に検出フックを埋め込む層。
- **層 3（Agentic 特有）**：エージェントが **自律的に判断してしまう** ことに起因する失敗。Ch4 の 3 層承認（`dag_authorization` / `variable_selection_authorization` / `intervention_execution_authorization`）が **意図的にバイパスされる** シナリオを含みます。

各層の失敗パターンは Ch4-13 で定義した **canonical schema と fatal action** を再利用します。本章は新規の schema を最小限にとどめ、**既存契約を検出フックとして参照する audit_manifest** を新設します。

```yaml
audit_manifest:                              # Ch14 §14.5 canonical
  audit_manifest_id: audit_2026_07_arim_project_alpha
  audit_scope:
    - vol_04_ch04_authorization_layers
    - vol_04_ch05_dag_specification
    - vol_04_ch06_estimator_tier
    - vol_04_ch09_refutation_gate
    - vol_04_ch09_counterfactual_scope_gate
    - vol_04_ch10_doe_randomization
    - vol_04_ch11_response_surface
    - vol_04_ch12_bayesian_doe
    - vol_04_ch13_capstone_evidence_chain
  audit_dimensions:
    - causal_general_failures        # §14.1
    - doe_general_failures           # §14.2
    - agentic_specific_failures      # §14.3
  audit_result_uri: <string>
  audit_result_sha256: <string>
  auditor: <string>
  audit_timestamp: <timestamp>
  audit_evidence:
    - detection_check_id: <string>
      fatal_action_referenced: <string>       # Ch4-13 の fatal から引用
      detection_status: pass | fail | not_applicable
      evidence_uri: <string>
```

---

## 14.1 因果推論一般の失敗パターン

### 14.1.1 DAG misspecification（DAG 誤特定）

**症状**：ATE / CATE の推定値が真値から系統的にずれ、感度分析でも説明できない。データ N を増やしても bias が消えない。

**原因**：因果グラフ $G$ に含まれるべき confounder / mediator が欠落、または方向が誤っている。多くの場合、以下の 3 つが起点：

1. **未観測 confounder**：$U$ が処置 $T$ と outcome $Y$ の両方に影響しているが、$U$ が測定されていない（Ch5 §5.3）。
2. **mediator の誤配置**：$T \to M \to Y$ を $T \to Y \leftarrow M$（collider）や $T \leftarrow M \to Y$（confounder）として配線ミス（Ch5 §5.2.2 / §5.2.3）。
3. **時間順序違反**：$Y$ の後に測定された変数を confounder として adjustment（post-treatment adjustment）（Ch5 §5.2.5）。

**検出契約（audit_check）**：

```yaml
detection_check:
  check_id: dag_misspecification_check
  check_type: dag_specification_integrity
  linked_provenance:
    - Ch5.approved_dag_uri                    # DAG spec の pin
    - Ch5.approved_dag_sha256
    - Ch5.hypothesis_uri                       # DAG 提案の根拠
  automated_checks:
    - temporal_ordering_check:                 # 変数間の測定タイミング整合
        variable_timestamp_uri: <string>       # 各変数の測定時刻表
        expected: all_confounders_pre_treatment
        fail_action: fail_close_and_flag_post_treatment_adjustment
    - unmeasured_confounder_probe:             # Ch9 §9.7.3 E-value
        e_value_threshold: 1.25                 # E-value < 閾値 → unmeasured confounder に脆弱
        current_e_value: <float>
        fail_action: require_sensitivity_analysis_or_re_dag
    - mediator_direction_check:                # Ch5 §5.2.2 mediator vs Ch5 §5.2.3 collider
        variable_role_declared: [mediator, collider, confounder]
        temporal_evidence_uri: <string>
        fail_action: fail_close_and_request_dag_revision
  human_review_required_when:
    - any_automated_check_status: fail
    - unmeasured_confounder_probe.current_e_value: less_than_1.5
  fatal_action_referenced:                     # Ch4-13 から引用
    - Ch5.report_estimate_without_dag_uri
    - Ch5.adjust_for_post_treatment_variable
    - Ch9.report_effect_without_refutation_when_declared_required
```

**修復方法**：DAG を revise し、`approved_dag_sha256` を新しい hash に差し替える。Ch4 §4.6.4 の evidence chain 再構築が必要（下流の estimator も rerun）。**Ch4 fatal `modify_approved_dag_after_downstream_start`** に該当する場合は、Skill 契約上は **プロジェクトの巻き戻し**（Phase 1 差戻し）を意味し、operator への告知が必須。

**Ch14 canonical fatal（新設）**：

```yaml
- adjust_for_post_treatment_variable_without_marking_as_mediator     # §5.2.5
- claim_dag_of_record_without_hypothesis_uri_and_e_value_probe        # DAG spec 監査
```

### 14.1.2 Collider bias（合流点調整による bias）

**症状**：処置 $T$ と outcome $Y$ の間に **本来存在しない相関** が現れる（selection bias、Berkson's bias、M-bias の一種）。Ch5 §5.2.3 の典型例。

**原因**：$T \to Z \leftarrow Y$ 構造の $Z$（collider）を adjustment set に含めた。Ch5 §5.2.4 の M-bias は $U_1 \to Z \leftarrow U_2$ かつ $U_1 \to T$, $U_2 \to Y$ の構造で、$Z$ が **観測可能** かつ $U_1, U_2$ が **未観測** のとき collider を通じて $T$-$Y$ 間に偽相関を生む。

**検出契約**：

```yaml
detection_check:
  check_id: collider_bias_check
  check_type: adjustment_set_integrity
  linked_provenance:
    - Ch5.approved_dag_uri
    - Ch5.backdoor_adjustment_set              # DAG から選ばれた adjustment 集合
    - Ch6.estimator_input_variables            # 実際に estimator へ渡された共変量
  automated_checks:
    - adjustment_set_vs_dag_consistency:       # DAG から derive された set と実装が一致するか
        expected_set: <from_dag_backdoor_criterion>
        actual_set: <from_estimator_input>
        set_diff: symmetric_difference          # 不一致は fatal
        fail_action: fail_close_and_regenerate_adjustment_set
    - collider_in_adjustment_set_probe:        # backdoor path 上の collider を検出
        collider_variables_declared: <list_from_dag>
        collider_variables_in_actual: <list_computed_from_actual_set>
        fail_action: fail_close_and_reject_estimator_run
    - m_bias_probe:                             # Ch5 §5.2.4 の M-bias パターン
        u1_u2_unmeasured_flag: <bool>          # DAG に未観測 confounder が明示されているか
        z_in_adjustment_set: <bool>            # Z が adjustment に入っているか
        m_bias_risk: high | medium | low       # 両方 true なら high
        fail_action: fail_close_and_require_sensitivity_or_re_dag
  fatal_action_referenced:
    - Ch5.adjust_for_collider_declared_in_dag
    - Ch5.adjust_for_post_treatment_variable
```

**修復方法**：Ch5 §5.2.3 の backdoor criterion を再適用し、adjustment set から collider を除外。Ch6 の estimator を rerun し、`approved_dag_sha256` は変更なし（DAG spec 自体は正しく、estimator の実装バグ）。ただし推定値が変わるため、Phase 2/3 が既に開始されている場合は Ch4 fatal `modify_approved_dag_after_downstream_start` の**類似 pattern** として `modify_adjustment_set_after_downstream_start` を新設。

**Ch14 canonical fatal（新設）**：

```yaml
- modify_adjustment_set_after_downstream_start                        # collider 削除も含む
- reuse_adjustment_set_across_dag_versions_without_reverification     # DAG が変わったら adjustment 再検証
```

### 14.1.3 Positivity violation（陽性条件破綻）

**症状**：ATT / CATE の推定値が特定の層で不安定になり、propensity score の分布が 0 または 1 の近傍に偏る。IPW 推定量の分散が発散する。

**原因**：条件付き確率 $P(T = t \mid X = x) = 0$ または $= 1$ となる領域 $x$ が存在する。Ch4 §4.4.1 の positivity 契約が破綻。以下の 3 パターン：

1. **structural non-positivity**：物理的に不可能な組合せ（例：装置 C は溶媒比 0.8 以上を扱えない）。
2. **practical non-positivity**：観測データ内で偶然 0 サンプル（サンプルサイズ不足）。
3. **stratum-level non-positivity**：全体では positivity OK だが、特定 stratum（`instrument=C`）で破綻。

**検出契約**：

```yaml
detection_check:
  check_id: positivity_violation_check
  check_type: positivity_by_stratum_integrity
  linked_provenance:
    - Ch6.propensity_score_uri                 # propensity score の記録
    - Ch6.positivity_by_stratum_report_uri     # Ch4 §4.4.1 canonical
  automated_checks:
    - propensity_bounds_check:
        threshold_lower: 0.05                   # Ch4 §4.4.1 canonical
        threshold_upper: 0.95
        violations_by_stratum: <dict>
        fail_action: fail_close_and_report_conditional_positivity
    - stratum_positivity_check:                # 各 stratum で individual に検査
        strata: [instrument_id, operator_id]
        stratum_positivity_status: <dict>       # pass | conditional | fail
        conditional_pass_output: non_actionable_diagnostic_only
        fail_action: fail_close_and_pre_register_applicability_manifest
    - structural_vs_practical_probe:           # 物理的に不可能 vs サンプル不足
        classification: structural | practical | ambiguous
        applicability_manifest_uri: <string>    # Ch4 §4.8
        fail_action_by_class:
          structural: mark_as_excluded_region
          practical: request_additional_data_or_doe
          ambiguous: fail_close_and_request_human_review
  fatal_action_referenced:
    - Ch4.report_cate_without_positivity_by_stratum
    - Ch13.report_cate_without_positivity_by_stratum
```

**修復方法**：`applicability_manifest` に **除外領域を pre-register** し（Ch4 §4.8）、CATE 推定を restrict domain に限定。Phase 2 で DoE を打つ場合は、除外領域の外側で追加 DoE を発注（Ch10）。

**Ch14 canonical fatal（新設）**：

```yaml
- report_ate_or_cate_without_stratum_level_positivity_check
- classify_practical_non_positivity_as_structural_without_evidence
```

### 14.1.4 外挿 unwarranted（正当化されない外挿）

**症状**：訓練データの support から離れた領域で予測を出し、その予測を **意思決定に使う**。CATE / response surface / SCM のいずれでも起こる。

**原因**：`counterfactual_scope_gate`（Ch4 §4.5.2）が正しく発火しない、または `support_envelope_check` が strict でない。Ch13 §13.4.3 の **5-check gate（第 3 相）** が bypass されるとこのモードに陥る。

**検出契約**：

```yaml
detection_check:
  check_id: unwarranted_extrapolation_check
  check_type: support_envelope_integrity
  linked_provenance:
    - Ch9.counterfactual_scope_gate_uri        # Phase 1 発火
    - Ch11.counterfactual_scope_gate_uri       # Phase 2 発火（response surface）
    - Ch13.counterfactual_scope_gate_scm_uri   # Phase 3 発火（SCM）
  automated_checks:
    - support_envelope_dimensions_check:
        expected_dimensions: <from_training_data>
        query_dimensions: <from_intervention_recommendation>
        outside_support_dimensions: <list>
        envelope_report_uri: <string>
        strict_mode: true                       # Phase 2/3 は strict 必須
        fail_action: fail_close_and_route_to_next_gate
    - three_gate_activation_check:             # Ch13 §13.4.3 canonical diff
        phase_1_gate_pass: <bool>
        phase_2_gate_pass: <bool>
        phase_3_gate_pass: <bool>
        expected: all_three_pass_before_intervention_recommendation
        fail_action: fail_close_and_reject_intervention
    - method_operational_distinctness_check:   # Phase 1/2/3 で method が operationally 異なるか
        phase_1_method: <expected_string>
        phase_2_method: <expected_string>
        phase_3_method: <expected_string>
        fail_action_if_identical: fatal_reuse_counterfactual_scope_gate_check_names_across_phases_without_operational_distinction
  fatal_action_referenced:
    - Ch4.report_intervention_recommendation_outside_counterfactual_scope
    - Ch13.report_intervention_recommendation_without_mediator_decomposition
    - Ch13.reuse_counterfactual_scope_gate_check_names_across_phases_without_operational_distinction
```

**修復方法**：外挿領域で介入を推奨せず、代わりに **追加 DoE を提案**（Ch10 §10.5）または **Bayesian DoE で情報利得の高い候補**（Ch12 §12.5）を提案。介入実行を停止することが必須。

### 14.1.5 CATE 過剰個別化（overfitted heterogeneity）

**症状**：CATE 推定が対象単位ごとに大きくばらつき、cross-validation で不安定。訓練データを分割すると推定値が符号反転する層が現れる。

**原因**：Ch6-8 の CATE 推定器（causal forest / meta-learners）で **交差検証不足**、または `support` 側のサンプル数が少ない層で「見かけの heterogeneity」を検出。Ch8 §8.4 の T-learner が高次元共変量で unstable になる典型パターン。

**検出契約**：

```yaml
detection_check:
  check_id: cate_over_individualization_check
  check_type: heterogeneity_stability
  linked_provenance:
    - Ch8.cate_estimation_provenance_uri
    - Ch8.cross_validation_report_uri
  automated_checks:
    - cate_cv_stability_probe:
        k_fold: 5
        cate_sign_flip_rate_threshold: 0.10     # 10% 以上の層で符号反転 → 不安定
        cate_relative_std_threshold: 0.5         # 相対分散
        fail_action: report_only_ate_or_broader_stratum_cate
    - stratum_sample_size_check:
        min_stratum_n: 30                        # 各 stratum 最小 N
        violated_strata: <list>
        fail_action: aggregate_strata_or_pre_register_low_confidence
    - honest_splitting_check:                  # Athey & Wager 2019 honest splitting
        honest_split_used: <bool>
        fail_action_if_false: rerun_estimator_with_honest_splitting
  fatal_action_referenced:
    - Ch4.report_cate_without_positivity_by_stratum
    - Ch8.claim_heterogeneity_without_cv_stability_report
```

**修復方法**：層を粗く集約（例：`instrument_id × operator_id` を `instrument_id` のみに）、または honest splitting で causal forest を rerun。

**Ch14 canonical fatal（新設）**：

```yaml
- claim_heterogeneity_without_cv_stability_report
- publish_cate_by_stratum_with_stratum_n_below_minimum_without_low_confidence_flag
```

### 14.1.6 Refutation スキップ

**症状**：ATE / CATE の推定値のみが報告され、placebo test / random common cause / subset validation の結果が provenance に無い。

**原因**：Ch9 §9.7.1 の `declared_required_tests` から自動的に選ばれるべき refutation が実行されなかった。**Skill 契約の記述漏れ**か、**skip の意図的判断**（後者は Agentic 特有、§14.3.3 で扱う）。

**検出契約**：

```yaml
detection_check:
  check_id: refutation_gate_skip_check
  check_type: refutation_completeness
  linked_provenance:
    - Ch9.refutation_gate_provenance_uri
    - Ch9.declared_required_tests               # canonical enum
    - Ch9.applicability_manifest_uri            # 非適用の pre-register
  automated_checks:
    - declared_vs_executed_check:
        declared_tests: <list_from_provenance>
        executed_tests: <list_from_report>
        skipped_without_manifest: <list>
        fail_action: fail_close_and_reject_report
    - applicability_manifest_integrity:        # skip した test は manifest に事前登録が必要
        manifest_pre_registered_before_estimation: <bool>
        manifest_sha256_match: <bool>
        fail_action_if_false: fatal_post_hoc_applicability_manifest
    - required_tests_enum_conformance:         # Ch9 §9.7.1 の 10-enum
        enum_version: ch09_v0_3                  # Ch12 で prior_predictive_check + prior_data_alignment 追加後
        conformance_status: pass | fail
        fail_action: fail_close_and_upgrade_enum_version
  fatal_action_referenced:
    - Ch9.report_effect_without_refutation_when_declared_required
    - Ch9.mark_test_as_not_applicable_after_estimation
```

**Ch14 canonical fatal（新設）**：

```yaml
- post_hoc_applicability_manifest                # 推定後に skip を正当化
- downgrade_declared_required_tests_enum_version_silently
```

---

## 14.2 DoE 一般の失敗パターン

### 14.2.1 Randomization 破綻

**症状**：DoE 実行後、`assignment_log` を見ると **本来の randomization seed とは異なる順序** で実施されている。または、operator が「装置 B が空いていたので順序を入れ替えた」といった **後付けの理由** で順序を改変。

**原因**：Ch10 §10.5.3 の 4-stage assignment_log detection（`seed_match` / `assignment_order_match` / `execution_order_match` / `assignment_log_header_recorded`）のいずれかが fail。

**検出契約**：

```yaml
detection_check:
  check_id: randomization_integrity_check
  check_type: assignment_log_4stage_verification
  linked_provenance:
    - Ch10.assignment_log_uri
    - Ch10.randomization_seed_pinned_at
    - Ch10.assignment_log_header_uri
  automated_checks:
    - seed_match_check:                        # Ch10 §10.5.3 stage 1
        expected_seed: <from_pre_registration>
        actual_seed: <from_assignment_log>
        fail_action: fatal_randomization_seed_mismatch
    - assignment_order_match_check:            # stage 2
        expected_order_uri: <string>
        actual_order_uri: <string>
        order_diff: <list_of_swaps>
        fail_action: fatal_assignment_order_altered_post_randomization
    - execution_order_match_check:             # stage 3
        planned_execution_order: <list>
        actual_execution_order: <list>
        deviation_report_uri: <string>
        fail_action: fail_close_and_reroll_with_new_seed
    - assignment_log_header_check:             # stage 4
        header_recorded: <bool>
        header_uri: <string>
        fail_action: fatal_assignment_log_missing_header_record
  fatal_action_referenced:
    - Ch10.modify_design_matrix_after_randomization_seed_pinned
    - Ch10.assignment_log_missing_header_record
    - Ch13.modify_design_matrix_after_randomization_seed_pinned
```

**修復方法**：Randomization を **新しい seed で再実施**。既存の実験結果は「pilot data」として保持するが、正式な ATE 推定には使わない。Ch10 §10.5 の Skill 契約により、operator が「時間が惜しい」を理由に旧データを使うことは fatal に該当。

### 14.2.2 Blocking 失敗

**症状**：装置差・オペレータ差の分散が処置効果と交絡し、ATE の分散が過大評価される。ANOVA 表で block 因子が有意でも、blocking の設計が正しくないと bias は残る。

**原因**：Ch10 §10.4 の blocking 設計が不完全。以下の 3 パターン：

1. **incomplete block design を complete と誤宣言**：全 block で全処置を実施できていない。
2. **blocking factor が処置に依存**：`instrument_id` が `substrate_temperature_c` に依存する（例：高温は装置 A のみ）→ blocking と処置が交絡。
3. **block size mismatch**：block ごとの単位数が不揃いで、weighted ATE の重みが不明。

**検出契約**：

```yaml
detection_check:
  check_id: blocking_design_integrity_check
  check_type: blocking_provenance_conformance
  linked_provenance:
    - Ch10.approved_design_provenance_uri
    - Ch10.blocking_factors_declared
  automated_checks:
    - block_completeness_check:
        block_type_declared: complete | incomplete
        actual_block_type: <computed_from_assignment_log>
        fail_action_if_mismatch: fatal_blocking_type_misdeclared
    - treatment_block_independence_check:      # χ² test で treatment × block の独立性
        chi_square_statistic: <float>
        p_value: <float>
        independence_threshold: 0.05
        fail_action: fail_close_and_report_confounded_blocking
    - block_size_uniformity_check:
        block_sizes: <list>
        max_min_ratio_threshold: 2.0
        fail_action: report_weighted_ate_with_weights_uri
  fatal_action_referenced:
    - Ch10.report_effect_without_blocking_when_declared_blocked
    - Ch10.misdeclare_block_type
```

**Ch14 canonical fatal（新設）**：

```yaml
- misdeclare_block_type
- report_effect_with_confounded_blocking_without_flag
```

### 14.2.3 応答曲面外挿誤用

**症状**：Ch11 の GP surrogate / polynomial 応答曲面を、**訓練 support の外** で評価し、最適条件として提案。

**原因**：Ch11 §11.7 の `counterfactual_scope_gate`（Phase 2）が bypass された、または `threshold_calibration` が未実施。Ch13 §13.4.3 の canonical diff で示した通り、Phase 2 は **strict mode** の `support_envelope_check` が必須。

**検出契約**：

```yaml
detection_check:
  check_id: response_surface_extrapolation_check
  check_type: gp_predictive_variance_and_support
  linked_provenance:
    - Ch11.response_surface_provenance_uri
    - Ch11.counterfactual_scope_gate_uri
  automated_checks:
    - gp_predictive_variance_bounds:
        query_points: <from_optimization_output>
        predictive_variance_at_query: <list>
        variance_threshold: <from_calibration>
        fail_action: fail_close_and_reject_optimization_output
    - convex_hull_check:                       # design space の凸包
        hull_uri: <string>
        query_inside_hull: <bool>
        fail_action_if_false: fail_close_and_extend_design_range
    - alpha_value_check:                       # Ch11 S-3 canonical
        expected_alpha: 1.68179283050743        # CCD α value
        actual_alpha: <float>
        fail_action_if_mismatch: fatal_alpha_value_drift
  fatal_action_referenced:
    - Ch11.report_optimum_outside_gp_support
    - Ch11.propose_response_surface_optimum_without_scope_gate
```

### 14.2.4 タグチ SN 比誤解釈

**症状**：Taguchi 直交表による experiment で、SN 比（signal-to-noise ratio）を **平均効果と分散効果の分離** ではなく、**単なる分散指標** として使用。分散最小化と平均最適化が同一視される。

**原因**：Taguchi 手法の哲学的前提（loss function、robust design）を無視して、SN 比の数値のみを ranking に使用。Ch10 §10.6.2 の robust design 節が触れているが、Skill 実装で誤解される頻度が高い。

**検出契約**：

```yaml
detection_check:
  check_id: taguchi_sn_ratio_interpretation_check
  check_type: sn_ratio_semantic_integrity
  linked_provenance:
    - Ch10.taguchi_design_provenance_uri
    - Ch10.sn_ratio_type_declared              # nominal_the_best | smaller_the_better | larger_the_better
  automated_checks:
    - sn_ratio_type_vs_optimization_target:
        declared_sn_type: <string>
        optimization_target: <string>
        semantic_match: <bool>
        fail_action_if_mismatch: fail_close_and_report_sn_type_misalignment
    - loss_function_documented:
        loss_function_uri: <string>
        loss_function_type: quadratic | asymmetric | custom
        fail_action_if_missing: request_loss_function_declaration
    - mean_variance_separation_check:          # 平均効果と分散効果を独立に分析
        mean_effect_report_uri: <string>
        variance_effect_report_uri: <string>
        both_present: <bool>
        fail_action_if_false: fatal_report_sn_only_without_mean_variance_separation
  fatal_action_referenced:
    - Ch10.report_sn_only_without_mean_variance_separation
    - Ch10.misdeclare_sn_ratio_type
```

**Ch14 canonical fatal（新設）**：

```yaml
- misdeclare_sn_ratio_type
- report_sn_only_without_mean_variance_separation
- interpret_taguchi_sn_as_pure_variance_index_without_loss_function
```

---

## 14.3 Agentic 特有の失敗パターン

本節は Ch4 の 3 層承認（`dag_authorization` / `variable_selection_authorization` / `intervention_execution_authorization`）が **意図的にバイパスされる** か **エージェントの自律判断で改変される** シナリオを扱います。層 1・層 2 と異なり、**Agentic 環境でなければ起きない** 失敗モードです。

### 14.3.1 DAG 勝手書換（silent DAG modification）

**症状**：`approved_dag_uri` は同一だが、Skill が内部で **backdoor adjustment set のみを差し替えて** 推定を実行。evidence chain は「同じ DAG を使った」と主張するが、実際の adjustment は異なる。

**原因**：Ch4 §4.5.1 の DAG 承認契約と Ch5 §5.6 の Skill 分離が不完全に実装され、`skill_1a_dag_proposal` の出力を `skill_2a_variable_selection` が **無断で改変** できる。

**検出契約**：

```yaml
detection_check:
  check_id: silent_dag_modification_check
  check_type: dag_evidence_chain_immutability
  linked_provenance:
    - Ch4.dag_authorization_provenance
    - Ch13.evidence_chain
    - Ch13.evidence_chain_sha256_algorithm      # sha256_json_canonical_rfc8785
    - Ch13.evidence_chain_sha256_input_fields
  automated_checks:
    - approved_dag_sha256_pin_check:
        approved_dag_sha256_at_authorization: <string>
        approved_dag_sha256_at_execution: <string>
        fail_action_if_mismatch: fatal_modify_approved_dag_after_downstream_start
    - adjustment_set_derived_from_dag_check:   # adjustment set が DAG から derive されているか
        derivation_procedure: backdoor_criterion | frontdoor_criterion
        derivation_procedure_uri: <string>
        actual_adjustment_set: <list>
        expected_adjustment_set: <computed_from_dag>
        fail_action_if_mismatch: fatal_silent_adjustment_set_override
    - evidence_chain_sha256_verification:      # Ch13 §13.4.4 canonical
        canonical_json_uri: <string>            # RFC 8785 出力
        evidence_chain_sha256_recomputed: <string>
        evidence_chain_sha256_stored: <string>
        fail_action_if_mismatch: fatal_evidence_chain_tampering
  fatal_action_referenced:
    - Ch4.modify_approved_dag_after_downstream_start
    - Ch13.modify_evidence_chain_after_approval
    - Ch13.modify_evidence_chain_sha256_input_fields_after_publication
```

### 14.3.2 Confounder 勝手削除

**症状**：エージェントが「この confounder はほぼ効かないので削除しました」と自主判断で adjustment set を縮小。DAG は変わらないが、実効的な adjustment が変わる。

**原因**：Skill の autonomy 境界が曖昧。Ch4 §4.5.1 の `dag_authorization` は DAG 全体を承認するが、adjustment set の具体的な列選択を **Skill 側の autonomy** に委ねている実装で発生。

**検出契約**：

```yaml
detection_check:
  check_id: silent_confounder_removal_check
  check_type: adjustment_set_immutability
  linked_provenance:
    - Ch4.variable_selection_authorization_provenance
    - Ch6.estimator_input_variables
  automated_checks:
    - approved_covariates_pin_check:           # variable_selection_authorization で pin
        approved_covariates: <list>             # Ch4 §4.5.2 canonical
        actual_covariates_used: <list>
        set_diff: symmetric_difference
        fail_action_if_diff_nonempty: fatal_execute_estimator_without_variable_selection_authorization
    - autonomy_boundary_check:                 # Ch4 §4.3.2 canonical
        skill_action_class: <declared>
        actual_action: <observed>
        boundary_violation: <bool>
        fail_action_if_true: fatal_agent_autonomous_covariate_removal
  fatal_action_referenced:
    - Ch4.execute_estimator_without_variable_selection_authorization
    - Ch4.agent_autonomous_action_beyond_declared_class
```

**Ch14 canonical fatal（新設）**：

```yaml
- agent_autonomous_covariate_removal
- agent_autonomous_adjustment_set_reduction
```

### 14.3.3 感度分析スキップ（sensitivity analysis skip）

**症状**：`refutation_gate_provenance` に E-value / Rosenbaum bounds / placebo test の記録が無いまま `intervention_recommendation` が発行される。§14.1.6 の refutation スキップと類似だが、Agentic 特有版では **エージェントが「時間が惜しい」「効果は明白」を理由に意図的に省略** する。

**原因**：Skill の action_class が `propose_only` ではなく `propose_and_execute` の場合、エージェントが gate 自体を bypass する誘惑にさらされる。Ch4 §4.5.2 の canonical では `refutation_gate` は L2（variable_selection_authorization）の前提条件だが、Skill 実装で強制されないと skip される。

**検出契約**：

```yaml
detection_check:
  check_id: agentic_sensitivity_skip_check
  check_type: refutation_gate_agentic_bypass
  linked_provenance:
    - Ch9.refutation_gate_provenance_uri
    - Ch4.variable_selection_authorization_provenance
  automated_checks:
    - refutation_gate_completeness_pre_l2_check:
        refutation_gate_pass_before_l2: <bool>
        l2_authorization_timestamp: <timestamp>
        refutation_gate_timestamp: <timestamp>
        temporal_ordering: refutation_gate_before_l2
        fail_action_if_violated: fatal_l2_authorization_without_prior_refutation_gate
    - e_value_threshold_check:                 # Ch9 §9.7.3
        e_value_declared: <float>
        e_value_threshold: 1.25                 # canonical
        fail_action_if_below: fail_close_and_require_re_sensitivity
    - agent_action_log_check:                  # Skill が gate を bypass する試みを記録
        gate_bypass_attempts: <list>
        fail_action_if_nonempty: fatal_agentic_gate_bypass_attempt
  fatal_action_referenced:
    - Ch9.report_effect_without_refutation_when_declared_required
    - Ch4.l2_authorization_without_prior_refutation_gate
```

### 14.3.4 「介入した」勝手記録（silent intervention logging）

**症状**：介入実行の実績ログ（`intervention_execution_log`）が、**Human 承認前** に生成されているか、あるいは **承認と異なる介入** を「承認された介入」として記録。

**原因**：Ch4 §4.6.3 の `intervention_execution_authorization` の temporal ordering 契約が不完全。Skill が「介入を先に実施して事後承認を取る」パターンで実装される（fail-forward）。

**検出契約**：

```yaml
detection_check:
  check_id: silent_intervention_logging_check
  check_type: intervention_execution_temporal_integrity
  linked_provenance:
    - Ch4.intervention_execution_authorization_provenance
    - Ch13.evidence_chain
  automated_checks:
    - temporal_ordering_l3_before_execution:
        l3_authorization_timestamp: <timestamp>
        intervention_execution_timestamp: <timestamp>
        fail_action_if_violated: fatal_intervention_executed_before_l3_authorization
    - intervention_uri_match_check:
        approved_intervention_uri: <string>
        executed_intervention_uri: <string>
        uri_match: <bool>
        fail_action_if_false: fatal_intervention_executed_differs_from_approved
    - execution_log_provenance_completeness:
        log_uri: <string>
        log_sha256: <string>
        log_timestamp_range: <interval>
        approval_timestamp: <timestamp>
        approval_within_log_range: <bool>
        fail_action_if_false: fatal_intervention_log_predates_approval
  fatal_action_referenced:
    - Ch4.intervention_executed_before_l3_authorization
    - Ch4.intervention_executed_differs_from_approved
    - Ch13.execute_intervention_without_evidence_chain_complete
```

**Ch14 canonical fatal（新設）**：

```yaml
- intervention_executed_before_l3_authorization
- intervention_executed_differs_from_approved
- intervention_log_predates_approval
```

### 14.3.5 Reproducibility seed 上書き

**症状**：`randomization_seed` / `estimator_random_seed` / `bootstrap_seed` のいずれかが、実験の途中で **異なる値に上書き** され、pinning 契約が破綻。

**原因**：Ch10 §10.5.3 / Ch11 §11.3 の seed pinning が strict でなく、エージェントが「convergence が改善する」との理由で seed を試行変更。

**検出契約**：

```yaml
detection_check:
  check_id: seed_overwrite_check
  check_type: seed_pinning_immutability
  linked_provenance:
    - Ch10.randomization_seed_pinned_at
    - Ch11.estimator_random_seed_pinned_at
    - Ch11.bootstrap_seed_pinned_at
  automated_checks:
    - seed_history_check:                      # 全 seed の変更履歴
        seed_change_events: <list_from_audit_log>
        allowed_seed_change_after_pin: false
        fail_action_if_change_after_pin: fatal_seed_overwrite_after_pin
    - seed_value_consistency:
        randomization_seed_at_pin: <int>
        randomization_seed_at_execution: <int>
        estimator_seed_at_pin: <int>
        estimator_seed_at_execution: <int>
        bootstrap_seed_at_pin: <int>
        bootstrap_seed_at_execution: <int>
        fail_action_if_any_mismatch: fatal_seed_mismatch_at_execution
  fatal_action_referenced:
    - Ch10.modify_design_matrix_after_randomization_seed_pinned
    - Ch11.overwrite_estimator_seed_after_pin
```

**Ch14 canonical fatal（新設）**：

```yaml
- seed_overwrite_after_pin
- seed_mismatch_at_execution
```

### 14.3.6 CATE 推薦を Human 未承認で外部送信

**症状**：Skill が CATE 推定結果と「推奨介入条件」を、**Human 承認前** に外部（実運用系、他プロジェクト、報告書）へ送信。

**原因**：Ch4 §4.6.3 の L3 authorization が strict でなく、Skill の action_class が `propose_and_broadcast` に近い実装。

**検出契約**：

```yaml
detection_check:
  check_id: unauthorized_broadcast_check
  check_type: intervention_recommendation_egress_control
  linked_provenance:
    - Ch4.intervention_execution_authorization_provenance
    - Ch4.egress_channels_declared               # 送信先チャネル宣言
  automated_checks:
    - egress_channel_authorization_check:
        broadcast_events: <list_from_audit_log>
        authorized_channels: <list>
        unauthorized_broadcasts: <list>
        fail_action_if_nonempty: fatal_intervention_recommendation_broadcast_without_l3
    - external_project_transfer_check:         # Ch13 §13.4.4 evidence chain 流用
        evidence_chain_reused: <bool>
        reuse_target_project: <string>
        re_authorization_present: <bool>
        fail_action_if_reuse_without_reauth: fatal_reuse_evidence_chain_across_projects_without_re_authorization
    - human_approval_before_broadcast_check:
        approval_timestamp: <timestamp>
        broadcast_timestamps: <list>
        all_broadcasts_after_approval: <bool>
        fail_action_if_false: fatal_broadcast_predates_l3_authorization
  fatal_action_referenced:
    - Ch4.intervention_recommendation_broadcast_without_l3
    - Ch13.reuse_evidence_chain_across_projects_without_re_authorization
```

**Ch14 canonical fatal（新設）**：

```yaml
- intervention_recommendation_broadcast_without_l3
- broadcast_predates_l3_authorization
- external_transfer_of_cate_recommendation_without_re_authorization
```

---

## 14.4 統合監査契約テンプレート（audit_manifest canonical）

§14.1-§14.3 の 16 パターン全てを **1 つの audit run** で検出する canonical テンプレート：

```yaml
audit_manifest_v1:                             # Ch14 §14.4 canonical
  audit_manifest_id: <string>
  audit_run_provenance:
    auditor_type: automated | human_reviewed | mixed
    auditor: <string>
    audit_started_at: <timestamp>
    audit_completed_at: <timestamp>
    audit_scope_uri: <string>                   # 対象プロジェクトの manifest.yaml
  detection_checks:                             # §14.1-14.3 の 16 checks
    causal_general:
      - dag_misspecification_check              # §14.1.1
      - collider_bias_check                     # §14.1.2
      - positivity_violation_check              # §14.1.3
      - unwarranted_extrapolation_check         # §14.1.4
      - cate_over_individualization_check       # §14.1.5
      - refutation_gate_skip_check              # §14.1.6
    doe_general:
      - randomization_integrity_check           # §14.2.1
      - blocking_design_integrity_check         # §14.2.2
      - response_surface_extrapolation_check    # §14.2.3
      - taguchi_sn_ratio_interpretation_check   # §14.2.4
    agentic_specific:
      - silent_dag_modification_check           # §14.3.1
      - silent_confounder_removal_check         # §14.3.2
      - agentic_sensitivity_skip_check          # §14.3.3
      - silent_intervention_logging_check       # §14.3.4
      - seed_overwrite_check                    # §14.3.5
      - unauthorized_broadcast_check            # §14.3.6
  audit_result_summary:
    pass_count: <int>
    fail_count: <int>
    not_applicable_count: <int>                 # applicability_manifest で pre-register 済み
    critical_fails: <list>                      # fatal に該当した fails
  aggregate_policy:
    pass_requires: all_16_pass_or_pre_registered
    fail_action: fail_close_and_route_to_facility_causal_review_board
    escalation_path:
      - research_lead
      - facility_causal_review_board
      - ethics_review_board                      # 人的介入を含む場合
  fallback_message_template: |
    Audit {audit_manifest_id} on project {project_id} completed at {audit_completed_at}.
    Result: {pass_count}/16 checks passed, {fail_count} failed, {not_applicable_count} pre-registered as non-applicable.
    Critical fails requiring review: {critical_fails}.
    Escalation route: {escalation_path[0]}.
    Reference the following provenance for evidence:
    - dag_authorization_uri: {dag_authorization_uri}
    - variable_selection_authorization_uri: {variable_selection_authorization_uri}
    - intervention_execution_authorization_uri: {intervention_execution_authorization_uri}
    - evidence_chain_sha256: {evidence_chain_sha256}
```

**運用ポリシー**：
- **audit 実行タイミング**：Phase 3 完了時（介入実行承認前）に必ず 1 回、および facility standard として promotion される前に 1 回。Ch4 §4.6.2 の `facility_scope_escalation.applies_to` enum に `approved_intervention_becomes_facility_standard` が含まれる場合、audit_manifest の pass が **前提条件**。
- **audit の immutability**：`audit_result_uri` と `audit_result_sha256` は evidence chain に組み込まれる。Ch13 §13.4.4 の `evidence_chain_sha256_input_fields` に `audit_manifest_uri` と `audit_manifest_sha256` を追加すべき（Ch13 の canonical 拡張候補）。

---

## 14.5 Ch14 で確立する Skill 契約テンプレート

### 14.5.1 audit_skill_contract

```yaml
skill_contract:                                 # Ch4 §4.7 canonical に準拠
  skill_id: audit_manifest_generation_skill
  skill_type: audit_generation
  role: human_required                          # audit 結果の判定は Human
  action_class: propose_and_execute_with_gate   # audit run は自動、判定は Human
  gate_level: L4_facility_standard_promotion_review  # 新設 L4
  input_provenance:
    - Ch4.dag_authorization_provenance_uri
    - Ch4.variable_selection_authorization_provenance_uri
    - Ch4.intervention_execution_authorization_provenance_uri
    - Ch13.evidence_chain_uri
  output_provenance:
    - audit_manifest_uri
    - audit_manifest_sha256
    - audit_result_uri
    - audit_result_sha256
  prohibited_actions:                           # Ch14 §14.3 の Agentic fatal を統合
    - modify_audit_manifest_after_completion    # 監査結果の改竄
    - skip_detection_check_without_applicability_manifest
    - claim_audit_pass_without_running_all_16_checks
    - reuse_audit_manifest_across_projects_without_re_run
  fallback_approver: facility_causal_review_board
```

### 14.5.2 facility_standard_promotion_gate（L4 gate 新設）

Ch4 §4.6.2 の `facility_scope_escalation` を **operational な gate** として実装：

```yaml
facility_standard_promotion_gate:               # Ch14 で新設、Ch4 §4.6.2 の operational 版
  gate_level: L4_facility_standard_promotion
  input_provenance:
    - audit_manifest_uri
    - audit_result_uri
    - Ch4.facility_scope_escalation.applies_to  # 拡張対象
  pre_conditions:
    - audit_result_summary.pass_count: 16       # 全 checks pass
    - audit_result_summary.fail_count: 0
    - critical_fails: []
  approval_layer:
    - facility_causal_review_board_approval:
        approver: facility_causal_review_board
        approved_at: <timestamp>
        approved_facility_scope: <string>       # 具体的な適用範囲
  post_conditions:
    - approved_facility_standard_uri: <string>
    - approved_facility_standard_sha256: <string>
    - back_registration_to_ch04_enum: true      # Ch4 §4.6.2 enum に追記
  prohibited_actions:
    - promote_to_facility_standard_without_audit_manifest_pass
    - modify_facility_standard_after_promotion_without_new_audit
```

---

## 14.6 Ch14 特有の prohibited_actions（fatal 統合）

§14.1-§14.5 で新設した Ch14 特有の fatal action を統合：

```yaml
prohibited_actions:
  # === Section 1: 因果推論一般 ===
  - adjust_for_post_treatment_variable_without_marking_as_mediator     # §14.1.1
  - claim_dag_of_record_without_hypothesis_uri_and_e_value_probe       # §14.1.1
  - modify_adjustment_set_after_downstream_start                        # §14.1.2
  - reuse_adjustment_set_across_dag_versions_without_reverification     # §14.1.2
  - report_ate_or_cate_without_stratum_level_positivity_check           # §14.1.3
  - classify_practical_non_positivity_as_structural_without_evidence    # §14.1.3
  - claim_heterogeneity_without_cv_stability_report                     # §14.1.5
  - publish_cate_by_stratum_with_stratum_n_below_minimum_without_low_confidence_flag  # §14.1.5
  - post_hoc_applicability_manifest                                     # §14.1.6
  - downgrade_declared_required_tests_enum_version_silently             # §14.1.6

  # === Section 2: DoE 一般 ===
  - misdeclare_block_type                                               # §14.2.2
  - report_effect_with_confounded_blocking_without_flag                 # §14.2.2
  - misdeclare_sn_ratio_type                                            # §14.2.4
  - report_sn_only_without_mean_variance_separation                     # §14.2.4
  - interpret_taguchi_sn_as_pure_variance_index_without_loss_function   # §14.2.4

  # === Section 3: Agentic 特有 ===
  - agent_autonomous_covariate_removal                                  # §14.3.2
  - agent_autonomous_adjustment_set_reduction                           # §14.3.2
  - agentic_gate_bypass_attempt                                         # §14.3.3
  - l2_authorization_without_prior_refutation_gate                      # §14.3.3
  - intervention_executed_before_l3_authorization                       # §14.3.4
  - intervention_executed_differs_from_approved                         # §14.3.4
  - intervention_log_predates_approval                                  # §14.3.4
  - seed_overwrite_after_pin                                            # §14.3.5
  - seed_mismatch_at_execution                                          # §14.3.5
  - intervention_recommendation_broadcast_without_l3                    # §14.3.6
  - broadcast_predates_l3_authorization                                 # §14.3.6
  - external_transfer_of_cate_recommendation_without_re_authorization   # §14.3.6

  # === Section 4/5: 監査契約 ===
  - modify_audit_manifest_after_completion                              # §14.5.1
  - skip_detection_check_without_applicability_manifest                 # §14.5.1
  - claim_audit_pass_without_running_all_16_checks                      # §14.5.1
  - reuse_audit_manifest_across_projects_without_re_run                 # §14.5.1
  - promote_to_facility_standard_without_audit_manifest_pass            # §14.5.2
  - modify_facility_standard_after_promotion_without_new_audit          # §14.5.2
```

---

## 14.7 章末チェックリスト

**Section 1（因果推論一般、§14.1）**
- [ ] `dag_misspecification_check` が `temporal_ordering_check` + `unmeasured_confounder_probe` + `mediator_direction_check` の 3 軸を含む
- [ ] `collider_bias_check` が Ch5 §5.2.3 の backdoor criterion と Ch5 §5.2.4 の M-bias を区別している
- [ ] `positivity_violation_check` が structural / practical / stratum-level の 3 パターンを識別する
- [ ] `unwarranted_extrapolation_check` が 3 gate（Phase 1/2/3）の operational distinctness を検証する
- [ ] `cate_over_individualization_check` が honest splitting の使用有無を確認する
- [ ] `refutation_gate_skip_check` が `declared_required_tests` enum の version（Ch9 §9.7.1 vs Ch12 拡張後）に対応している

**Section 2（DoE 一般、§14.2）**
- [ ] `randomization_integrity_check` が Ch10 §10.5.3 の 4-stage detection を全て含む
- [ ] `blocking_design_integrity_check` が complete/incomplete block declaration と処置-block 独立性を検査する
- [ ] `response_surface_extrapolation_check` が Ch11 S-3 canonical の $\alpha = 1.68179283050743$ を verify する
- [ ] `taguchi_sn_ratio_interpretation_check` が SN 比 type と optimization target の semantic match を検査する

**Section 3（Agentic 特有、§14.3）**
- [ ] `silent_dag_modification_check` が Ch13 §13.4.4 の `evidence_chain_sha256` を verify する（RFC 8785 canonical JSON）
- [ ] `silent_confounder_removal_check` が Ch4 §4.5.2 の approved_covariates と実 estimator input の symmetric difference を検査する
- [ ] `agentic_sensitivity_skip_check` が refutation_gate と L2 authorization の temporal ordering を確認する
- [ ] `silent_intervention_logging_check` が execution timestamp と approval timestamp の順序を strict に検証する
- [ ] `seed_overwrite_check` が全 3 種類の seed（randomization / estimator / bootstrap）を対象とする
- [ ] `unauthorized_broadcast_check` が egress channel 宣言と実 broadcast events を照合する

**Section 4/5（監査契約、§14.4-§14.5）**
- [ ] `audit_manifest_v1` が 16 個の detection_check を全て含む
- [ ] `audit_result_summary` が critical_fails を独立に集計する
- [ ] `facility_standard_promotion_gate` は L4 gate として新設され、Ch4 §4.6.2 の operational 版として位置付けられている
- [ ] `audit_skill_contract` は `role: human_required` かつ `action_class: propose_and_execute_with_gate` である
- [ ] Ch14 特有 fatal 30 個以上が §14.6 で列挙されている
- [ ] audit_manifest_sha256 は Ch13 §13.4.4 の `evidence_chain_sha256_input_fields` に組み込むことが議論されている

---

## 章末演習

### 演習 14.1（DAG misspecification の合成データ再現）

Ch5 §5.2.3 の collider bias 構造 $T \to Z \leftarrow Y$ を持つ合成データ N=500 を生成し、$Z$ を adjustment set に含める場合と除外する場合で ATE 推定値がどう変わるかを比較せよ。§14.1.2 の `collider_bias_check` の `set_diff` が空でないことを検出する Skill を実装せよ。

### 演習 14.2（positivity 破綻の可視化）

Ch13 §13.1.2 のシナリオで `instrument=C` の propensity score が 0.05 未満になる領域を可視化し、`applicability_manifest` に **除外領域を pre-register** するテンプレを書け。§14.1.3 の `structural_vs_practical_probe` の判定基準（例えば「装置スペックシートに書かれている物理的上限か否か」）を Skill 契約として整理せよ。

### 演習 14.3（silent DAG modification の CI/CD 検出）

§14.3.1 の `silent_dag_modification_check` を GitHub Actions に組み込み、`approved_dag_sha256` と `evidence_chain_sha256` の verify を PR 時に自動実行する workflow を書け。Ch13 §13.4.4 の RFC 8785 canonical JSON の Python 実装（例：`rfc8785` package）を使い、hash 再計算のテストケースを 3 個以上作成せよ。

### 演習 14.4（audit_manifest の運用）

§14.4 の `audit_manifest_v1` を Ch13 の capstone シナリオに適用し、16 checks の run を PyMC / DoWhy / EconML の実装で埋めよ。`critical_fails` に該当するケースを **意図的に 3 個作り**、`fail_close_and_route_to_facility_causal_review_board` の escalation flow を Mermaid で図示せよ。

### 演習 14.5（Agentic fail-close 経路の設計）

§14.3 の 6 パターンそれぞれについて、**operator への告知文（fallback_message_template）** を書け。特に §14.3.4（silent intervention logging）については、**「介入が既に実施済み」の場合の rollback 手順** と **「pilot data として保持」の判定基準** を Skill 契約として明文化せよ。

---

## 参考資料

### 本書内参照

- **本書 第4章**：3 層承認（dag_authorization / variable_selection_authorization / intervention_execution_authorization）、`facility_scope_escalation.applies_to` enum、evidence chain canonical
- **本書 第5章**：DAG specification、backdoor criterion、collider（§5.2.3）、M-bias（§5.2.4）、post-treatment adjustment（§5.2.5）
- **本書 第6-8章**：ATE / CATE 推定器、honest splitting、cross-validation
- **本書 第9章**：refutation_gate、declared_required_tests canonical enum、E-value、counterfactual_scope_gate（Phase 1）
- **本書 第10章**：DoE randomization、blocking、assignment_log 4-stage detection、Taguchi
- **本書 第11章**：response surface、GP surrogate、counterfactual_scope_gate（Phase 2）、$\alpha$ value canonical
- **本書 第12章**：Bayesian DoE、prior_specification_provenance、prior_predictive_check
- **本書 第13章**：capstone worked example、evidence_chain_sha256（RFC 8785）、counterfactual_scope_gate（Phase 3、5-check）
- **本書 第15章**：組織展開と責任分担（audit 結果の運用ガバナンス）

### 外部文献（監査・因果 × Agentic の failure modes）

- Athey, S., & Wager, S. (2019). *Estimating Treatment Effects with Causal Forests: An Application*. Observational Studies, 5(2), 37-51.
- VanderWeele, T. J., & Ding, P. (2017). *Sensitivity Analysis in Observational Research: Introducing the E-Value*. Annals of Internal Medicine, 167(4), 268-274.
- Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.). Springer. — Rosenbaum bounds、感度分析の古典
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. — do-calculus、backdoor criterion、collider
- Taguchi, G., Chowdhury, S., & Wu, Y. (2005). *Taguchi's Quality Engineering Handbook*. Wiley. — SN 比の loss function 前提
- Wu, C. F. J., & Hamada, M. S. (2009). *Experiments: Planning, Analysis, and Optimization* (2nd ed.). Wiley. — DoE の統計的厳密性
- ISO/IEC 42001:2023 *Information technology — Artificial intelligence — Management system*. — AI システム監査
- RFC 8785 *JSON Canonicalization Scheme (JCS)*. — evidence_chain_sha256 の canonical 計算基盤
- Sculley, D., et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. — Agentic 環境で silent modification が起きる pattern
