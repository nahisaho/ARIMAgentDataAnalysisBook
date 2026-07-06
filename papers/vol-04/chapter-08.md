# 第8章　Heterogeneous Treatment Effect と CATE — 「誰に効くか」の Skill 化（g-formula と外挿範囲を含む）

> [!IMPORTANT]
> **本章の位置づけ**：第6章では集団平均（ATE / ATT）、第7章では準実験による ATT / LATE / ITT を扱いました。本章では **「誰に、どの条件で効くか」＝ Conditional Average Treatment Effect (CATE) / Individual Treatment Effect (ITE)** を Skill 化します。**特に ARIM の材料研究では「どの装置で」「どの組成範囲で」効くかが実践的判断の核**——ATE のみでは "全体平均で効くが自分の条件では効かない" 失敗を招きます。同時に、CATE を用いた個別介入推薦は **反実仮想の外挿を伴う**ため、`counterfactual_scope_gate`（第4章契約、本章 §8.5 で operational 実装）と 3 層承認ゲートを厳格に適用します。

## 8.1 なぜ CATE か — ATE の限界と ARIM での必要性

### 8.1.1 ATE で決まらない意思決定

第6章で装置更新プロトコルが「平均 5% 純度を上げる」という ATE を推定できたとします。しかし研究者の実際の問いは：

- **「自分の装置 A では効くのか？」**（装置 heterogeneity）
- **「合金 X の組成範囲 [0.2, 0.3] では効くのか？」**（組成 heterogeneity）
- **「オペレータ熟練度が高い時と低い時で効き方は違うか？」**（interaction）

これらは **平均ではなく条件付き期待値**：

$$\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]$$

を推定する問題です。$X$ は装置 ID・組成・オペレータ熟練度・環境変数などの共変量。

### 8.1.2 CATE / ITE / uplift の関係

| 概念 | 定義 | 実務での位置づけ |
|---|---|---|
| **ATE** | $\mathbb{E}[Y(1) - Y(0)]$ | 集団平均。政策・予算判断 |
| **CATE** | $\mathbb{E}[Y(1) - Y(0) \mid X = x]$ | サブグループ平均。装置別・組成別 |
| **ITE** | $Y_i(1) - Y_i(0)$（個体） | 反実仮想の個別値。**原則不可観測**、CATE の予測で近似 |
| **uplift** | CATE の応用言い換え | マーケティング由来、材料選抜では「介入する価値の高い試料」を選ぶ |

> [!WARNING]
> **ITE は原理的に観測不可能**（第0章の反実仮想不可観測性）——観測できるのは $Y_i(1)$ か $Y_i(0)$ のどちらか一方のみ。**Skill が ITE を "推定" と称して出力する場合、実体は $\widehat{\tau}(x_i)$（CATE の予測値）**。ラベルを ITE と呼ぶ場合は **母集団平均ではなく個体固有効果を主張していることを明示的に宣言**する契約フィールドが必要（§8.3）。

### 8.1.3 本章で使うツール

| ライブラリ | 用途 | 章内位置 |
|---|---|---|
| **EconML** | Meta-Learners (S/T/X/DR/R-Learner) の実装 | §8.3, §8.4 |
| **DoWhy** | CATE 推定 + refutation | §8.4 |
| **scikit-uplift** | uplift modeling、材料選抜への転用 | §8.6 |
| **causalml**（言及のみ） | Uber の CATE ライブラリ | §8.4 |

---

## 8.2 identification 仮定の再確認（CATE 版）

CATE 推定は ATE と同じ 4 仮定（第0章）に加え、以下の**条件付き版**を要求：

| 仮定 | ATE 版 | CATE 版（$X$ の各層で成立） |
|---|---|---|
| **Positivity** | $0 < P(T = 1) < 1$ | $0 < P(T = 1 \mid X = x) < 1$ **すべての推論対象 $x$ で** |
| **Exchangeability**（unconfoundedness） | $\{Y(0), Y(1)\} \perp T \mid X$ | 同左（$X$ の中に十分な confounders） |
| **Consistency** | $Y = Y(T)$ | 同左 |
| **SUTVA** | 相互作用なし | 同左 |

> [!WARNING]
> **CATE の Positivity は ATE よりも厳しい**——「装置 A × 高組成 × 高熟練オペレータ」というサブグループで処置群が 0 なら、その層での CATE は identify されません。**Skill 契約で `positivity_by_stratum` を CATE の共変量分割で明示**（第6章で必須化した規約）。

### 8.2.1 CATE 特有の落とし穴：過剰個別化と empty stratum

$X$ の次元が高いと、各層のサンプル数が減り、**推定分散が爆発**します。加えて **推論対象 $x^*$ が empty stratum に落ちる**ことがあり、この場合 CATE は identify すら不可能——Skill が予測を返してはいけません。Skill 契約は以下を強制：

```yaml
cate_prediction_variance_threshold: <float>    # 予測分散の上限
overfit_check_uri: <string>                     # 交差検証での out-of-sample R² / MSE
empty_stratum_policy: fail_close                # 対象 x* が min_effective_sample_size 未満の層 → fail-close
min_effective_sample_size_per_stratum: <int>    # 例：20
propensity_overlap_bounds: {epsilon: 0.05}       # e_hat(x) ∈ [ε, 1-ε] を強制
target_x_to_stratum_mapping_uri: <string>       # 予測対象 x* をどの層に割り当てたか記録
```

---

## 8.3 CATE 推定 Skill の契約

### 8.3.1 Meta-Learners の 5 種類

| Meta-Learner | 骨格 | 使い分け |
|---|---|---|
| **S-Learner** | $T$ を feature に混ぜて 1 モデル $\mu(x, t)$、$\widehat{\tau}(x) = \mu(x, 1) - \mu(x, 0)$ | シンプル、$T$ の signal が弱いと regularization で消える |
| **T-Learner** | 処置群・対照群それぞれで別モデル $\mu_1(x), \mu_0(x)$、差分 | 各群サンプル十分なら精度良 |
| **X-Learner** | T-Learner + IPW で欠測処置効果を imputation + reweighting | 処置群の観測が少ないとき有利 |
| **DR-Learner** | doubly-robust score を回帰 target に | 二重ロバスト、Ch6 DML の CATE 版 |
| **R-Learner** | Robinson-style residualization + CATE 回帰 | 交絡が強く、confounding 除去を残差化で行う |

### 8.3.2 estimator 契約（EconML 実装）

```yaml
estimator: cate
estimand_type: cate | ite_labeled_prediction   # ite_labeled_prediction は個体固有主張の宣言
cate_config:
  library: econml | causalml | dowhy
  library_version: <string>
  meta_learner: s_learner | t_learner | x_learner | dr_learner | r_learner
  outcome_model:
    class: <string>                             # e.g., sklearn.ensemble.GradientBoostingRegressor
    hyperparameters: {...}
  treatment_model:                              # DR / R-Learner で使用
    class: <string>
    hyperparameters: {...}
  cross_fitting:
    n_folds: 5                                  # DR/R-Learner は必須
    random_state: <int>
    fold_assignment_uri: <string>               # fold 割当を artifact 化（DR/R-Learner の nuisance leakage 防止に必須）
    fold_assignment_sha256: <string>
    nuisance_model_y_uri: <string>              # outcome nuisance モデル artifact
    nuisance_model_t_uri: <string>              # treatment nuisance モデル artifact
    final_effect_model_uri: <string>
    out_of_fold_prediction_evidence_uri: <string>  # 残差化に使った予測が out-of-fold であることの証跡
  covariates: [X1, X2, ...]                     # CATE を条件付ける $X$
  effect_modifiers: [...]                       # 非交絡だが効果を変える変数（EconML の "W" と "X" の区別）
  confounders: [...]                             # 交絡（EconML の "W"）
identification_validity:
  strategy: backdoor_conditional                 # 通常は backdoor + $X$ 層別
  positivity_by_stratum:                         # Ch6 で必須化、CATE では層別が細かい
    - {key: <string>, min_treated: <int>, min_control: <int>, min_effective_sample_size: <int>, propensity_epsilon: 0.05, violation_policy: fail_close}
  positivity_stratum_definition_uri: <string>    # 層の切り方の承認証跡
  target_x_to_stratum_mapping_uri: <string>      # 予測対象 x* → 層のマッピング
  dag_of_record_uri: <string>
  dag_of_record_sha256: <string>
  adjustment_set_approval_uri: <string>
cate_quality:
  cate_prediction_variance_threshold: <float>    # 各予測点での分散上限
  cross_validated_mse: <float>
  overfit_check_uri: <string>
  calibration_plot_uri: <string>                 # 予測 CATE vs 実測差の校正
  external_validity_report_uri: <string>         # 学習ドメイン外の $x$ への外挿判定（§8.5）
estimator_contract_change_gate:                  # Ch6 §6.7 継承
  meta_learner_change_policy: L4_approval_required
  outcome_model_class_change_policy: L4_approval_required
  covariate_set_change_policy: L3_notify + variable_selection_authorization
prohibited_actions:
  - silently_change_meta_learner                # fatal
  - silently_expand_covariates_beyond_approved  # fatal
  - relabel_cate_as_ite_without_declaration     # fatal（個体固有主張は明示宣言）
  - relabel_cate_as_ate                          # fatal
  - report_cate_at_x_outside_learning_domain_without_scope_gate  # fatal（§8.5）
  - report_point_prediction_without_variance    # fatal（CATE は分散必須）
  - use_ml_model_without_cross_fitting_for_dr_or_r_learner  # fatal
  - use_in_fold_nuisance_predictions_for_residualization  # fatal（fold leakage）
  - predict_for_target_x_in_empty_or_below_min_stratum  # fatal（empty stratum policy）
  - predict_when_propensity_outside_epsilon_bounds  # fatal
```

### 8.3.3 5 Meta-Learners 選択の early warning matrix

| 状況 | 第一選択 | 代替 | 避けるべき |
|---|---|---|---|
| サンプル十分、$T$ balanced、交絡弱 | T-Learner | S-Learner | — |
| $T$ imbalanced（処置群少数） | X-Learner | DR-Learner | T-Learner（対照群過学習） |
| 交絡強い、共変量高次元 | R-Learner | DR-Learner | S-Learner（$T$ が正則化で消える） |
| ML モデルの efficient inference が必要 | DR-Learner（Ch6 DML の CATE 版） | R-Learner | S/T-Learner |
| interpretability 重視 | S-Learner（線形） | Causal Tree/Forest（EconML） | R-Learner |

---

## 8.4 ARIM ケース：装置別 × 組成別 CATE

### 8.4.1 設定

- 処置 $T$：新プロトコル採用（0/1）
- outcome $Y$：合金の破壊靭性 $K_{IC}$
- 共変量 $X$：装置 ID（A/B/C/D）、組成比 $c_1$、熱処理温度 $T_h$、オペレータ熟練度
- confounders $W$：バッチ番号、環境湿度、前処理時間

**問い**：装置 A × 高組成 ($c_1 > 0.5$) × 高熟練オペレータのサブグループでは効果があるか？

### 8.4.2 Skill 実行フロー

1. **Positivity 診断（層別）**：装置 × 組成 bin × 熟練度で $P(T=1 \mid X)$ を計算、各層 min_treated, min_control 確認
2. **Meta-Learner 選定**：交絡強・共変量中次元 → **R-Learner または DR-Learner**
3. **cross-fitting**：5-fold、outcome model = GradientBoostingRegressor、treatment model = LogisticRegression
4. **CATE 予測 + 分散**：EconML `effect_interval` で信頼区間
5. **外部妥当性チェック**：訓練ドメイン外の $x$（例：極端組成）は §8.5 で fail-close
6. **artifact**：`cate_map_uri`（$X$ 空間での CATE ヒートマップ）、`calibration_plot_uri`、`overfit_check_uri`

### 8.4.3 vol-03 深層特徴を CATE 共変量に

**vol-03 第8章**で紹介した Foundation Model 由来の深層特徴（SEM 画像・回折パターンの embedding）を **CATE の $X$ に含める**ことができます。ただし：

| 課題 | 対応 |
|---|---|
| 深層特徴は解釈性低 | `effect_modifier_semantics_uri` で "この embedding 次元が組成の何を捉えるか" を人間可読ドキュメント化 |
| 高次元 → 過剰個別化リスク | `dimensionality_reduction_policy`（PCA / UMAP に制約、`cate_prediction_variance_threshold` を厳格化） |
| Foundation Model の provenance | vol-03 の `foundation_model_version` + `feature_extractor_sha256` を CATE 契約に継承 |

```yaml
cate_config:
  covariates_deep:
    foundation_model_version: <string>
    feature_extractor_sha256: <string>
    feature_pipeline_uri: <string>              # 前処理・tokenizer・normalization 含む pipeline 全体
    feature_pipeline_sha256: <string>
    preprocessing_config_sha256: <string>
    embedding_training_artifact_uri: <string>   # embedding 学習時の artifact（凍結の証跡）
    dimensionality_reduction:
      method: pca | umap | none
      n_components: <int>
      fit_sha256: <string>                       # PCA/UMAP の fit 済みオブジェクト hash
      justification_uri: <string>
    train_predict_pipeline_equality_check_uri: <string>  # train と predict の pipeline hash 一致検証
    effect_modifier_semantics_uri: <string>      # embedding の解釈ドキュメント
prohibited_actions:
  - use_deep_features_without_foundation_model_provenance   # fatal
  - use_deep_features_when_train_predict_pipeline_hashes_differ  # fatal
  - refit_dimensionality_reduction_on_prediction_data       # fatal
  - increase_deep_feature_dimensions_silently               # fatal（過剰個別化）
```

---

## 8.5 g-formula と外挿範囲 — `counterfactual_scope_gate` の operational 実装

### 8.5.1 g-formula とは

CATE を統合して集団の反実仮想を推定：

$$\mathbb{E}[Y(t)] = \int \mu(x, t) \, dF(X)$$

$\mu(x, t)$ は outcome model、$F(X)$ は共変量分布。**Skill が「介入 $T = 1$ を集団に適用したら $Y$ がどうなるか」を推定する場合、g-formula を明示的に呼ぶ**。

**注意**：g-formula は **既存共変量分布 $F(X)$ に対する** 反実仮想——**統合分布そのものが承認・凍結されていないと、エージェントが grid・合成分布・平滑化分布を勝手に注入して外挿を隠蔽できる**。従って g-formula 契約では統合分布と per-point scope gate を必須にします：

```yaml
gformula_config:
  integration_distribution_uri: <string>         # 統合に使う F(X) の実体（凍結）
  integration_distribution_sha256: <string>
  distribution_type: empirical | approved_synthetic | reweighted_empirical
  synthetic_distribution_approval_uri: <string>  # approved_synthetic の場合必須
  per_point_scope_gate_report_uri: <string>      # 統合対象 x ごとの gate 結果
  weighted_support_fail_close: true              # 支持域外の重みがあれば全体を fail-close
prohibited_actions:
  - integrate_over_unapproved_synthetic_distribution     # fatal
  - integrate_over_grid_including_out_of_support_x       # fatal
  - report_gformula_ate_when_any_point_gate_fails        # fatal
  - silently_smooth_or_reweight_integration_distribution # fatal
```

### 8.5.2 `counterfactual_scope_gate` の operational 定義

第4章で契約定義した `counterfactual_scope_gate` を、本章で**判定可能な閾値**に落とし込みます。**4 判定は異なる失敗モードを捉える相補的チェック**であり、1 つの pass は他の substitute にはなりません：

| 判定軸 | 捉える失敗モード | 閾値 | 契約項目 |
|---|---|---|---|
| **Mahalanobis 距離（cluster-conditional）**：学習データがクラスタ化していれば所属クラスタ重心との距離、非クラスタ化なら全体重心 | 全体的な "遠さ"（ただし multimodal 支持では単独では不十分） | $D_M(x^*) \leq d_{\max}$ | `mahalanobis_distance_global`, `mahalanobis_distance_cluster_conditional`, `mahalanobis_threshold` |
| **CATE 予測分散**：$\widehat{\text{Var}}[\widehat{\tau}(x^*)]$ | 推定不確実性 | $\leq v_{\max}$ | `cate_prediction_variance`, `variance_threshold` |
| **k-NN 密度**：$x^*$ 近傍 $k$ 個の学習点存在 | 局所的支持の欠如（cluster 間の空隙） | $\geq k_{\min}$ | `knn_density`, `knn_min`, `knn_radius` |
| **support envelope**：各共変量の学習域内 | 周辺（marginal）外挿 | $x^* \in [\min_i x_i, \max_i x_i]$ | `support_envelope_report_uri` |

> [!WARNING]
> **Global Mahalanobis は multimodal 支持（例：装置ごと・組成レジームごと）で unsafe**——2 つのクラスタの中間点は Mahalanobis 距離が小さくても支持が無い場合がある。**ARIM のような装置別・組成別レジームを持つデータでは cluster-conditional Mahalanobis（所属クラスタ内での距離）と k-NN 密度の併用が必須**。global Mahalanobis 単独 pass は "外挿でない" 証拠にはならない。

```yaml
counterfactual_scope_gate:
  gate_status: pass | fail | conditional_pass
  mahalanobis_check:
    distance_global: <float>
    distance_cluster_conditional: <float>       # 所属クラスタ内での距離（ARIM のような multimodal 支持で必須）
    cluster_assignment_uri: <string>
    threshold: <float>                          # 学習分布の calibrated 閾値（下記 calibration protocol）
    covariance_estimation_uri: <string>
  variance_check:
    predicted_cate_variance: <float>
    threshold: <float>                          # 処置群 outcome SD の倍数で定義
  knn_density_check:
    k: 20
    n_training_points_within_radius: <int>
    knn_min: <int>
    knn_radius: <float>                         # 近傍半径の定義（学習分布の中央値等で正規化）
  support_envelope_check:
    outside_support_dimensions: [...]
    envelope_report_uri: <string>
  threshold_calibration:                        # 閾値の恣意性を防ぐ承認済み protocol
    method: leave_one_out_empirical | conformal | validation_false_extrapolation_rate
    calibration_evidence_uri: <string>
    calibration_approved_at: <timestamp>
  aggregate_policy:                             # 4 チェックの結合ルール
    pass_requires: all_four_pass                # 4 判定は相補的、単独 pass は substitute にならない
    conditional_pass_requires: [mahalanobis_cluster_conditional_pass, variance_pass, knn_density_pass, support_envelope_pass]
    conditional_pass_output: non_actionable_diagnostic_only  # actionable な予測数値は返さず、review 用の diagnostic のみ返す
    fail_close_action: refuse_prediction_and_notify_reviewer
approver_when_conditional: causal_review_board
prohibited_actions:
  - predict_at_x_with_gate_fail                                # fatal
  - return_actionable_cate_estimate_under_conditional_pass     # fatal（conditional_pass は診断出力のみ）
  - substitute_one_check_pass_for_another                       # fatal（4 判定は相補的）
  - use_global_mahalanobis_alone_on_multimodal_support          # fatal（cluster-conditional 必須）
  - relax_thresholds_without_estimator_contract_change_gate     # fatal
  - report_ate_using_gformula_outside_learning_support          # fatal
  - silently_extrapolate_deep_features                          # fatal
```

### 8.5.3 gate の 3 状態

- **pass**：全 4 チェック pass → **actionable な予測値と分散を返す**
- **conditional_pass**：**全 4 判定が pass しなかった場合の中間状態**（例：Mahalanobis + variance のみ pass、kNN 密度または support envelope fail）→ **actionable な予測数値は返さず、"外挿注意 + Human review 用の診断 artifact"（分散推定・近傍点リスト・境界超過次元）のみ返す**。予測数値の release には Human 承認が必須。
- **fail**：**予測を返さない**（fail-close）。**別戦略への silent 切替禁止**——Ch7 §7.6.1 と同じ規約：`estimator_contract_change_gate` を通す

> [!IMPORTANT]
> **`conditional_pass` は "外挿注意ラベル付きで予測を返す" 状態ではなく**、"予測数値を返さず、レビュー用診断のみ返す" 状態です。これは agent が conditional_pass を silent path として悪用することを防ぎます。

### 8.5.4 ARIM での運用

例：組成範囲 $c_1 \in [0.2, 0.6]$ で訓練した CATE モデル。研究者が $c_1 = 0.75$（訓練域外）での介入効果を尋ねた：

1. cluster-conditional Mahalanobis：$D_M = 8.2$（閾値 3.5）→ **fail**
2. gate → **予測返却拒否 + reviewer 通知**
3. reviewer が「$c_1 = 0.75$ の追加データ取得」または「外挿の理論的正当化」で承認 → `estimator_contract_change_gate` L4 承認

別の例：$c_1 = 0.62$（境界近傍）で global Mahalanobis は pass するが cluster-conditional で fail、k-NN 密度 fail：
→ **conditional_pass**（診断のみ）→ Human review、actionable な CATE 数値は返さない

---

## 8.6 個別介入推薦の 3 層承認と uplift 応用

### 8.6.1 3 層承認ゲート（CATE 特有）

第4章の 3 層承認を CATE 個別介入推薦に適用：

| 層 | CATE での意味 | Skill 契約項目 |
|---|---|---|
| **`dag_authorization`** | CATE の DAG が第5章承認済み | `dag_of_record_uri`, `dag_of_record_sha256` |
| **`variable_selection_authorization`** | 共変量 / effect modifier / confounder の切り分け承認 | `covariate_role_partitioning_approval_uri` |
| **`intervention_execution_authorization`** | **actionable な個別介入推薦の release 承認**——外部送信だけでなく、**内部実験計画・robot/API payload・ranked protocol リスト・lab ticket・human-facing 実行指示のいずれの形式でも actionable な個別推薦を生成する前**に必要 | `individual_recommendation_release_approval_uri`, `counterfactual_scope_gate.gate_status: pass` を前提 |

> [!IMPORTANT]
> **`intervention_execution_authorization` は "外部送信" のみでは不十分**——エージェントが「内部実験計画」「robot queue draft」「ランキング付き protocol list」「lab-ready recipe」等の**形式で actionable 化した瞬間**に発動する。CATE 予測に基づく個別介入を実験に送るのは、`counterfactual_scope_gate` pass + Human 承認の両方が必須。エージェントが「CATE 予測分散が閾値内だから自動送信」「内部プランだから承認不要」と判断することは fatal（第14章の失敗パターン）。

### 8.6.2 scikit-uplift による材料選抜

材料研究では **「介入する価値の高い試料を選ぶ」= uplift** を CATE の応用として利用：

```yaml
skill_type: uplift_material_screening_skill
authorization_level: agent_autonomous              # 提案までは自律
pre_registration:
  candidate_universe_uri: <string>                 # 事前登録された候補プール
  candidate_universe_frozen_at: <timestamp>
  ranking_rule_uri: <string>                       # ranking 手続きを事前固定
  top_k_policy_uri: <string>                       # k の決め方 + 変更条件
  evaluation_split_uri: <string>                   # uplift@k 評価用の holdout split
inputs:
  - candidate_materials_uri
  - cate_model_uri
  - cate_model_sha256
  - cate_contract_uri                              # CATE Skill 契約（§8.3.2）参照必須
  - cate_contract_sha256                           # 契約 hash で以下の内容を担保：
                                                    #   estimand_type, positivity_by_stratum,
                                                    #   dag_of_record_uri + sha256,
                                                    #   adjustment_set_approval_uri
  - counterfactual_scope_gate_config
outputs:
  - uplift_scores_per_candidate_uri
  - top_k_candidates_uri
  - scope_gate_status_per_candidate_uri            # 各候補について gate pass/conditional/fail
downstream_gate: intervention_execution_authorization  # 実際に実験する候補選定は Human 承認
prohibited_actions:
  - submit_candidate_with_scope_gate_fail_to_experiment   # fatal
  - reorder_top_k_after_seeing_experimental_outcomes      # fatal（look-ahead bias）
  - adaptively_change_k_or_ranking_rule_after_seeing_scope_gate_results  # fatal（adaptive top-k bias）
  - filter_candidates_after_seeing_interim_results        # fatal
  - modify_candidate_universe_after_freeze                 # fatal
  - use_uplift_score_from_stale_cate_model                # fatal（provenance 尊重）
  - consume_cate_model_without_verifying_cate_contract_fields  # fatal（Ch6/Ch8 必須項目確認）
```

**scikit-uplift の使いどころ**：qini curve、uplift@k 評価、multi-treatment uplift（複数プロトコルからどれを試料に適用するか）。

### 8.6.3 estimand の宣言：CATE か ITE か

Skill 契約で必須：

| ラベル | 意味 | 使ってよい場面 | 必須付随フィールド |
|---|---|---|---|
| `estimand_type: cate` | 条件 $X = x$ での**平均処置効果** | サブグループ意思決定、政策 | `cate_prediction_variance`, `calibration_plot_uri` |
| `estimand_type: ite_labeled_prediction` | **個体固有効果の予測**主張 | 特定試料の推薦、ただし予測値 ≠ 真の ITE を明示 | 下記 ITE-specific 契約項目すべて |

**`ite_labeled_prediction` を主張する場合の追加必須項目**：

```yaml
ite_labeled_prediction:
  declaration_uri: <string>                        # "真の ITE は不可観測、これは τ̂(x_i) の近似" を明示
  prediction_interval_uri: <string>                # 個体レベルの予測区間（点推定単独は禁止）
  uncertainty_method: conformal | bayesian_posterior | bootstrap
  coverage_target: 0.9                              # 予測区間の目標カバレッジ
  coverage_empirical_evidence_uri: <string>        # holdout で target coverage を達成した証跡
  individual_decision_threshold: <float>            # この閾値を超える場合のみ actionable
  output_text_disclaimer_uri: <string>             # ユーザー向け "CATE prediction at x_i, not identified ITE" 文言
prohibited_actions:
  - report_ite_labeled_prediction_without_interval        # fatal
  - report_ite_labeled_prediction_without_declaration     # fatal
  - use_point_estimate_alone_for_individual_recommendation  # fatal
  - claim_identified_ite                                    # fatal（原理的不可観測）
```

> [!WARNING]
> **`ite_labeled_prediction` は "個体推薦時に CATE モデル出力を ITE と呼び替えるための fig-leaf" ではありません**。宣言・予測区間・カバレッジ検証・disclaimer 文言のすべてが揃わない ITE 主張は fatal（§8.3.2 `relabel_cate_as_ite_without_declaration` + `use_point_estimate_alone_for_individual_recommendation`）。

---

## 8.7 CATE / g-formula / uplift の位置づけ表

| Skill | 主要 estimand | 出力粒度 | 承認レベル | scope gate 必須 |
|---|---|---|---|---|
| CATE Skill（§8.3-8.4） | cate | サブグループ | dag + variable_selection | 予測時（per-point） |
| g-formula Skill（§8.5） | ate_via_gformula | 集団平均 | dag + variable_selection + integration distribution 承認 | **統合対象の全 x で per-point 判定**（1 点でも fail なら全体 fail-close） |
| 個別介入推薦（§8.6.1） | ite_labeled_prediction | 個体 | 3 層すべて（actionable 化の瞬間に発動） | pass 必須（conditional_pass では actionable 出力不可） |
| uplift 材料選抜（§8.6.2） | uplift_ranking | 個体 ranking | dag + variable_selection（提案）+ intervention_execution（release） | 各候補で per-point 判定、fail は除外 |

---

## まとめ — 本章のチェックリスト

- [ ] **§8.1 ATE と CATE と ITE**——集団平均・条件付き平均・個体効果の違いと、ITE の原理的不可観測性を説明できる
- [ ] **§8.2 CATE 特有の positivity**——`min_effective_sample_size`、`propensity_epsilon`、empty stratum の fail-close policy を説明できる
- [ ] **§8.3 5 Meta-Learners**——S / T / X / DR / R-Learner の使い分けを状況別に判断できる。DR/R-Learner の cross-fitting fold assignment 記録・out-of-fold nuisance 予測の必要性を説明できる
- [ ] **§8.3.2 CATE Skill 契約**——`estimand_type: cate | ite_labeled_prediction`、`positivity_by_stratum`、`fold_assignment_uri` などを書き下せる
- [ ] **§8.4.3 深層特徴 + CATE**——vol-03 の Foundation Model 特徴を CATE 共変量に使う際の `feature_pipeline_sha256` と train-predict pipeline equality check を理解した
- [ ] **§8.5 g-formula の統合分布 governance**——`integration_distribution_uri` と per-point scope gate、`weighted_support_fail_close` の意味を説明できる
- [ ] **§8.5.2 `counterfactual_scope_gate`**——4 判定の相補的失敗モード（distance / uncertainty / density / envelope）を区別し、multimodal 支持での cluster-conditional Mahalanobis の必要性を理解した
- [ ] **§8.5.3 gate 3 状態**——`conditional_pass` は actionable 予測を返さず診断のみ返す状態であり、silent fallback は禁止であることを理解した
- [ ] **§8.6.1 `intervention_execution_authorization` の発動タイミング**——外部送信のみならず内部プラン・robot payload・ranked list など actionable 化の瞬間に必要
- [ ] **§8.6.2 uplift の pre-registration**——candidate_universe / ranking_rule / top_k_policy の事前登録と adaptive top-k bias 防止を理解した
- [ ] **§8.6.3 ITE 主張の必須項目**——declaration + prediction interval + coverage 実証 + disclaimer 文言のセット

---

## 章末演習

### 演習 8.1：Meta-Learner の選択

以下のシナリオそれぞれで、5 Meta-Learners のどれを第一選択にするか、その理由と、EconML でどう設定するかを答えてください：

1. 処置群 n=1500 / 対照群 n=1500、共変量 5 次元、交絡は観測されているが弱い
2. 処置群 n=80 / 対照群 n=2000、共変量 20 次元、交絡強
3. 処置群 n=800 / 対照群 n=800、共変量 100 次元（うち 90 次元が深層特徴）、交絡は複雑
4. interpretability 重視、変数 3 次元、線形回帰で十分と判断

### 演習 8.2：`counterfactual_scope_gate` の閾値設定

自分の ARIM データ（または合成データ）で以下を設計してください：

- [ ] Mahalanobis 距離の閾値（学習分布の 90 or 95 パーセンタイルなど）
- [ ] CATE 予測分散閾値（処置群 outcome SD の何倍か）
- [ ] k-NN 密度閾値（$k$ と $k_{\min}$）
- [ ] support envelope の per-covariate 境界
- [ ] 4 チェックの結合ポリシー（all_pass? conditional_pass の要件は？）

### 演習 8.3：個別介入推薦のワークフロー設計

以下の要素を含む個別介入推薦フローを設計してください：

1. CATE モデル（vol-03 深層特徴を使うか使わないか選択）
2. uplift ranking の top-k 選定（scikit-uplift）
3. 各候補への `counterfactual_scope_gate` 判定
4. `intervention_execution_authorization` の Human 承認プロトコル
5. 実験実行後の反省（推薦したが失敗した候補への provenance 記録）

### 演習 8.4：過剰個別化の検出

以下の CATE 出力を見て、過剰個別化を疑う理由と対策を挙げてください：

- 訓練 R² = 0.95、CV R² = 0.32
- 予測 CATE 値の分散が処置群 outcome SD の 5 倍
- 各共変量層でのサンプル数 min = 3
- Mahalanobis 距離の分布が学習域を大きく超える予測点で使われている

対策として、以下を検討してください：
- [ ] 共変量の次元削減
- [ ] Meta-Learner の変更（例：DR → S-Learner）
- [ ] `cate_prediction_variance_threshold` の厳格化
- [ ] scope gate の 4 判定の再設定

---

## 参考資料

### 内部 cross-reference

- **第0章**：4 識別仮定（positivity の CATE 版が本章 §8.2）、declared / assessed / checked
- **第4章**：`counterfactual_scope_gate` 契約定義（本章 §8.5 で operational 実装）、3 層承認ゲート
- **第5章**：DAG、DAG proposal / approval Skill（本章 §8.6.1 の `dag_authorization` で継承）
- **第6章**：backdoor + DR-Learner / DML（本章 §8.3 の Meta-Learner の集団平均版）、`estimator_contract_change_gate`（本章 CATE で継承）、`positivity_by_stratum`
- **第7章**：DiD / IV / SC（本章の CATE は backdoor 系だが、DiD × CATE の staggered は Callaway-Sant'Anna との統合が Ch8 拡張）
- **第9章**：refutation、E-value、`counterfactual_scope_gate` の Ch9 側の再検証
- **第11章**：応答曲面 → CATE の続き（DoE + CATE のハイブリッド）
- **第13a章**：Phase 1-2 で CATE 推定 → DoE 設計
- **第13b章**：Phase 3 で SCM ベース反実仮想 + g-formula
- **第14章**：CATE の過剰個別化、scope gate 未通過での推薦などの失敗パターン
- **付録 A**：CATE Skill、g-formula Skill、uplift Skill のテンプレート
- **付録 D**：Meta-Learner / Mahalanobis / support envelope / uplift の用語

### 外部参考文献

- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *PNAS*, 116(10), 4156-4165.
- Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2), 299-319.
- Athey, S., & Wager, S. (2019). Estimating treatment effects with causal forests: An application. *Observational Studies*, 5, 37-51.
- Robins, J. M. (1986). A new approach to causal inference in mortality studies with a sustained exposure period. *Mathematical Modelling*, 7, 1393-1512. (g-formula 原論文)
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. Ch13-14 (g-formula).
- EconML documentation: https://econml.azurewebsites.net/
- scikit-uplift documentation: https://www.uplift-modeling.com/
- CausalPy documentation: https://causalpy.readthedocs.io/

---

**本章の位置づけ**：CATE / g-formula / uplift / `counterfactual_scope_gate` の operational 実装が揃いました。次章（Ch9）では、**因果主張の "検算"**——refutation、E-value、Rosenbaum bounds、placebo、subset validation を Skill 化し、"refutation pass 未達なら結論を出さない" 契約を導入します。本章の scope gate は Ch9 の refutation でも再検証されます。
