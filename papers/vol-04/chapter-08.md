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

### 8.2.1 CATE 特有の落とし穴：過剰個別化

$X$ の次元が高いと、各層のサンプル数が減り、**推定分散が爆発**します。Skill 契約は **CATE 予測分散の上限**を強制：

```yaml
cate_prediction_variance_threshold: <float>  # 予測分散の上限
overfit_check_uri: <string>                   # 交差検証での out-of-sample R² / MSE
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
  covariates: [X1, X2, ...]                     # CATE を条件付ける $X$
  effect_modifiers: [...]                       # 非交絡だが効果を変える変数（EconML の "W" と "X" の区別）
  confounders: [...]                             # 交絡（EconML の "W"）
identification_validity:
  strategy: backdoor_conditional                 # 通常は backdoor + $X$ 層別
  positivity_by_stratum:                         # Ch6 で必須化、CATE では層別が細かい
    - {key: <string>, min_treated: <int>, min_control: <int>, violation_policy: fail_close}
  positivity_stratum_definition_uri: <string>    # 層の切り方の承認証跡
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
    dimensionality_reduction:
      method: pca | umap | none
      n_components: <int>
      justification_uri: <string>
    effect_modifier_semantics_uri: <string>      # embedding の解釈ドキュメント
prohibited_actions:
  - use_deep_features_without_foundation_model_provenance  # fatal
  - increase_deep_feature_dimensions_silently             # fatal（過剰個別化）
```

---

## 8.5 g-formula と外挿範囲 — `counterfactual_scope_gate` の operational 実装

### 8.5.1 g-formula とは

CATE を統合して集団の反実仮想を推定：

$$\mathbb{E}[Y(t)] = \int \mu(x, t) \, dF(X)$$

$\mu(x, t)$ は outcome model、$F(X)$ は共変量分布。**Skill が「介入 $T = 1$ を集団に適用したら $Y$ がどうなるか」を推定する場合、g-formula を明示的に呼ぶ**。

**注意**：g-formula は **既存共変量分布 $F(X)$ に対する** 反実仮想。**分布外の $x$（外挿）を扱う場合は別途 gate 必須**。

### 8.5.2 `counterfactual_scope_gate` の operational 定義

第4章で契約定義した `counterfactual_scope_gate` を、本章で**判定可能な閾値**に落とし込みます：

| 判定軸 | 閾値 | 契約項目 |
|---|---|---|
| **Mahalanobis 距離**：予測点 $x^*$ と学習データ $\{x_i\}$ の重心との距離を共分散で正規化 | $D_M(x^*) \leq d_{\max}$（例：$d_{\max}$ = 学習分布の 95 パーセンタイル） | `mahalanobis_distance`, `mahalanobis_threshold` |
| **CATE 予測分散**：$\widehat{\text{Var}}[\widehat{\tau}(x^*)]$ | $\leq v_{\max}$（例：処置群 outcome の SD の 2 倍以下） | `cate_prediction_variance`, `variance_threshold` |
| **k-NN 密度**：$x^*$ 近傍 $k$ 個の学習点存在 | $\geq k_{\min}$（例：$k_{\min} = 20$） | `knn_density`, `knn_min` |
| **support envelope**：各共変量の学習域内 | $x^* \in [\min_i x_i, \max_i x_i]$ | `support_envelope_report_uri` |

```yaml
counterfactual_scope_gate:
  gate_status: pass | fail | conditional_pass
  mahalanobis_check:
    distance: <float>
    threshold: <float>                     # 学習分布の 95 パーセンタイル距離
    covariance_estimation_uri: <string>
  variance_check:
    predicted_cate_variance: <float>
    threshold: <float>                     # 処置群 outcome SD の倍数で定義
  knn_density_check:
    k: 20
    n_training_points_within_radius: <int>
    knn_min: <int>
  support_envelope_check:
    outside_support_dimensions: [...]
    envelope_report_uri: <string>
  aggregate_policy:                        # 4 チェックの結合ルール
    pass_requires: all_four_pass
    conditional_pass_requires: [mahalanobis_pass, variance_pass]  # 部分 pass 時のみ
    fail_close_action: refuse_prediction_and_notify_reviewer
approver_when_conditional: causal_review_board
prohibited_actions:
  - predict_at_x_with_gate_fail                              # fatal
  - relax_thresholds_without_estimator_contract_change_gate  # fatal
  - report_ate_using_gformula_outside_learning_support       # fatal
  - silently_extrapolate_deep_features                       # fatal
```

### 8.5.3 gate の 3 状態

- **pass**：全 4 チェック pass → 予測を返す
- **conditional_pass**：Mahalanobis + variance のみ pass → **予測を返すが、"外挿注意" ラベル + Human review 通知**
- **fail**：予測を **返さない**（fail-close）。**別戦略への silent 切替禁止**——Ch7 §7.6.1 と同じ規約：`estimator_contract_change_gate` を通す

### 8.5.4 ARIM での運用

例：組成範囲 $c_1 \in [0.2, 0.6]$ で訓練した CATE モデル。研究者が $c_1 = 0.75$（訓練域外）での介入効果を尋ねた：

1. Mahalanobis 距離：$D_M = 8.2$（閾値 3.5）→ **fail**
2. gate → **予測返却拒否 + reviewer 通知**
3. reviewer が「$c_1 = 0.75$ の追加データ取得」または「外挿の理論的正当化」で承認 → `estimator_contract_change_gate` L4 承認

---

## 8.6 個別介入推薦の 3 層承認と uplift 応用

### 8.6.1 3 層承認ゲート（CATE 特有）

第4章の 3 層承認を CATE 個別介入推薦に適用：

| 層 | CATE での意味 | Skill 契約項目 |
|---|---|---|
| **`dag_authorization`** | CATE の DAG が第5章承認済み | `dag_of_record_uri`, `dag_of_record_sha256` |
| **`variable_selection_authorization`** | 共変量 / effect modifier / confounder の切り分け承認 | `covariate_role_partitioning_approval_uri` |
| **`intervention_execution_authorization`** | **個別介入推薦を外部（実験系）に送信する承認** | `individual_recommendation_release_approval_uri`, `counterfactual_scope_gate.gate_status: pass` を前提 |

> [!IMPORTANT]
> **CATE 予測に基づく個別介入を実験に送るのは、`counterfactual_scope_gate` pass + Human 承認の両方が必須**。エージェントが「CATE 予測分散が閾値内だから自動送信」することは fatal（第14章の失敗パターン）。

### 8.6.2 scikit-uplift による材料選抜

材料研究では **「介入する価値の高い試料を選ぶ」= uplift** を CATE の応用として利用：

```yaml
skill_type: uplift_material_screening_skill
authorization_level: agent_autonomous              # 提案までは自律
inputs:
  - candidate_materials_uri
  - cate_model_uri
  - cate_model_sha256
  - counterfactual_scope_gate_config
outputs:
  - uplift_scores_per_candidate_uri
  - top_k_candidates_uri
  - scope_gate_status_per_candidate_uri            # 各候補について gate pass/conditional/fail
downstream_gate: intervention_execution_authorization  # 実際に実験する候補選定は Human 承認
prohibited_actions:
  - submit_candidate_with_scope_gate_fail_to_experiment   # fatal
  - reorder_top_k_after_seeing_experimental_outcomes      # fatal（look-ahead bias）
  - use_uplift_score_from_stale_cate_model                # fatal（provenance 尊重）
```

**scikit-uplift の使いどころ**：qini curve、uplift@k 評価、multi-treatment uplift（複数プロトコルからどれを試料に適用するか）。

### 8.6.3 estimand の宣言：CATE か ITE か

Skill 契約で必須：

| ラベル | 意味 | 使ってよい場面 |
|---|---|---|
| `estimand_type: cate` | 条件 $X = x$ での**平均処置効果** | サブグループ意思決定、政策 |
| `estimand_type: ite_labeled_prediction` | **個体固有効果の予測**主張 | 特定試料の推薦、ただし予測値 ≠ 真の ITE を明示 |

> [!WARNING]
> **同一 CATE モデルの出力を「個体推薦」の場面で ITE と呼ぶ場合、`ite_labeled_prediction_declaration_uri` で "個体固有主張であること、そして真の ITE は不可観測なので $\widehat{\tau}(x_i)$ は近似であること" を明示宣言する。** 宣言なしの ITE 主張は fatal（§8.3.2 `relabel_cate_as_ite_without_declaration`）。

---

## 8.7 CATE / g-formula / uplift の位置づけ表

| Skill | 主要 estimand | 出力粒度 | 承認レベル | scope gate 必須 |
|---|---|---|---|---|
| CATE Skill（§8.3-8.4） | cate | サブグループ | dag + variable_selection | 予測時 |
| g-formula Skill（§8.5） | ate_via_gformula | 集団平均 | dag + variable_selection | 分布外統合時 |
| 個別介入推薦（§8.6.1） | ite_labeled_prediction | 個体 | 3 層すべて | pass 必須 |
| uplift 材料選抜（§8.6.2） | uplift_ranking | 個体 ranking | dag + variable_selection（提案）+ intervention_execution（送信） | 各候補で判定 |

---

## まとめ — 本章のチェックリスト

- [ ] **§8.1 ATE と CATE と ITE**——集団平均・条件付き平均・個体効果の違いと、ITE の原理的不可観測性を説明できる
- [ ] **§8.2 CATE 特有の positivity**——共変量層別で positivity が必要な理由、過剰個別化のリスクを理解した
- [ ] **§8.3 5 Meta-Learners**——S / T / X / DR / R-Learner の使い分けを状況別に判断できる
- [ ] **§8.3.2 CATE Skill 契約**——`estimand_type: cate | ite_labeled_prediction`、`positivity_by_stratum`、`cross_fitting`、`cate_prediction_variance_threshold` を書き下せる
- [ ] **§8.4.3 深層特徴 + CATE**——vol-03 の Foundation Model 特徴を CATE 共変量に使う際の provenance 要件を理解した
- [ ] **§8.5 g-formula と `counterfactual_scope_gate`**——Mahalanobis / variance / kNN / support envelope の 4 判定を operational に運用できる
- [ ] **§8.5.3 gate 3 状態**——pass / conditional_pass / fail の意味と、fail 時の silent fallback 禁止を理解した
- [ ] **§8.6 3 層承認 + uplift**——個別介入推薦を実験系に送るための承認プロセスと scikit-uplift の使いどころを説明できる

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
