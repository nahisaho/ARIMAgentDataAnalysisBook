# 第11章　応答曲面法とタグチメソッドの Skill 化

> **学習目標**：
> - Screening（Ch10）から **optimization** へと DoE の目的が変わったとき、CCD / Box-Behnken / OA L8-L18 の使い分けを Skill 契約として書ける
> - 応答曲面フィッティングを **linear → GBM → GP** の段階的パスとして設計し、モデル選択自体を Skill 契約に組み込める
> - smt / GPy による **Kriging surrogate** に予測不確かさ（`prediction_variance`）を出力させ、`counterfactual_scope_gate` に食わせる
> - タグチメソッドの **SN 比** を、 単なる「損失関数の対数」ではなく Larger-The-Better / Smaller-The-Better / Nominal-The-Best の 3 契約として書ける
> - **応答曲面の外挿失敗**——エージェントが最適解を実験域外に置くパターン——を Ch4/Ch8/Ch9 の `counterfactual_scope_gate` で fail-close できる

---

## 11.1 Screening から optimization へ — DoE の第 2 相

第10章までは **screening**——「どの因子が効くか」を分別する段階でした。**要因の有無を知る**ことが目的で、線形モデル（主効果 + 一部 interaction）で十分でした。

本章は **optimization**——「最適な水準はどこか」を探る段階です。この段階では：

- **2 次以上の応答曲面**が必要（screening の線形近似では極値は捉えられない）
- **中心点（center point）** で curvature（曲率）の検定が必要
- **surrogate model** で連続因子空間上の任意点を予測（GP / Kriging）
- **外挿**が最大の失敗リスク——実験域を離れた最適解を返すエージェントは、実験を破綻させる

| 段階 | 目的 | 代表設計 | モデル | 本書での位置付け |
|---|---|---|---|---|
| screening | 要因分別 | Full / Fractional factorial, PB | linear + few interactions | 第10章 |
| **optimization** | 最適水準探索 | **CCD, Box-Behnken, OA L18** | **linear→GBM→GP** | **本章** |
| sequential | one-at-a-time exploitation | Bayesian D-optimal (one-shot) | GP + acquisition | 第12章（one-shot）+ vol-05（逐次） |

### 11.1.1 応答曲面のモデル階層

エージェントは「どのモデルで応答を近似するか」を **一気に決めない**——3 段階のモデル階層で提案し、各段で Ch4 §4.6.2 の approver に承認を求めます。

> **命名注意（Ch6 との用語衝突回避）**：本章では **モデル階層** を `T1 / T2 / T3`（Tier）と表記します。Ch6 §6.7.1 `estimator_contract_change_gate` の `L1 / L2 / L3 / L4` は **再 fit の重大度**を表す gate レベルであり、意味が異なります。両者を混同すると、algorithm family change（本章の T1→T2 / T2→T3 昇格）を誤って Ch6 L2（gate なし）で処理する重大な governance ミスを招きます（対応: 全 T1→T2 と T2→T3 昇格は **Ch6 §6.7.1 L4 event**）。

$$
\underbrace{y = \beta_0 + \sum \beta_i x_i + \sum \beta_{ij} x_i x_j + \sum \beta_{ii} x_i^2 + \varepsilon}_{\text{T1: 2 次多項式}}
\;\longrightarrow\;
\underbrace{y = f_{\text{GBM}}(x) + \varepsilon}_{\text{T2: 非線形}}
\;\longrightarrow\;
\underbrace{y = f_{\text{GP}}(x) + \varepsilon,\; \varepsilon \sim \mathcal{GP}(0, k)}_{\text{T3: 不確かさ付き surrogate}}
$$

**契約含意**（Ch6 §6.7.1 L4: algorithm family change に該当）：

- **T1 → T2 昇格**（statsmodels OLS → LightGBM）：**Ch6 L4 event**（algorithm family change）。以下を **すべて** 要求：
  - `variable_selection_authorization`（factor 集合含意の変更）+ `research_lead` 承認
  - `estimator_contract_change_gate`（approval type）+ new Skill version の発行
  - `estimator_diff_report_uri`（点推定 / CI / CV-RMSE の差分を旧 Skill 対比で記録）
- **T2 → T3 昇格**（LightGBM → smt Kriging）：同じく **Ch6 L4 event**。上記に加えて：
  - `counterfactual_scope_gate` の再検証（GP は補外域で分散が爆発するため）
- **モデル選択を silent に切り替える** は **fatal**（`response_surface_model_switch_without_approval`）
- **T1/T2/T3 を `L1/L2/L3` と混同呼称する** は **fatal**（Ch6 との用語衝突による gate 誤発火リスク）

### 11.1.2 識別仮定は Ch10 から変わらない

Ch10 §10.1 で確認したとおり、**randomization + design freeze** が守られていれば：

| 仮定 | 状態 | 応答曲面での意味 |
|---|---|---|
| exchangeability | **checked** | design 内で治療割当は独立 |
| positivity | **checked** | 設計行列に定義された cell 内 |
| consistency | **checked** | 実行プロトコル pin |
| SUTVA | **assessed** | 依然として spillover / carryover はゼロと仮定 |

**追加される仮定**：

| 仮定 | 状態 | 応答曲面での意味 |
|---|---|---|
| **response model correctness** | *assessed* | 2 次多項式・GBM・GP が真の response surface を近似できるという仮定 |
| **experimental region validity** | *checked*（設計内）／ **fail-close**（外挿域） | 応答面は設計 convex hull 内でのみ信頼できる |

本章の Skill 契約はこの **response model correctness** を明示的に宣言し、外挿要求を検知して停止させます。

---

## 11.2 Central Composite Design (CCD)

### 11.2.1 CCD の構造

CCD は **factorial + axial（star） + center** の 3 成分から構成される、応答曲面のための古典的設計です：

$$
N_{\text{CCD}} = \underbrace{2^k}_{\text{factorial}} + \underbrace{2k}_{\text{axial}} + \underbrace{n_c}_{\text{center}}
$$

| 成分 | 目的 | 座標例（$k=3$） |
|---|---|---|
| **factorial**（$2^k$）| 主効果 + 2-way interaction | $(\pm 1, \pm 1, \pm 1)$ |
| **axial**（$2k$）| 2 次項（$\beta_{ii}$）の推定 | $(\pm \alpha, 0, 0), (0, \pm \alpha, 0), (0, 0, \pm \alpha)$ |
| **center**（$n_c$）| curvature 検定 + 純誤差推定 | $(0, 0, 0)$ を $n_c$ 反復 |

$k=3$ のとき $N = 8 + 6 + n_c$、通常 $n_c = 3 \sim 5$ で **17 〜 19 runs**。

### 11.2.2 axial distance $\alpha$ の選択

$\alpha$ には設計思想に応じて 3 種類：

| 種類 | $\alpha$ の値 | 特性 |
|---|---|---|
| **Rotatable** | $\alpha = (2^k)^{1/4}$（$k=3$ で $\approx 1.68179283050743$、$2^{0.75}$ の倍精度値）| 全方向で予測分散が等しい（推奨） |
| **Orthogonal** | $k$ と $n_c$ に依存する複雑な式 | 効果推定が直交、interaction 検定に向く |
| **Face-centered (CCF)** | $\alpha = 1$ | axial 点も factorial の面上——各因子の水準を 3 に制限したい場合 |

**契約項目**：

```yaml
experimental_design_provenance:
  design_type: central_composite_design
  ccd_variant: rotatable                              # rotatable | orthogonal | face_centered
  factors: [...]                                      # §10.2.2 と同構造
  factorial_component:
    n_runs: 8
    generator: full_factorial | fractional_factorial
  axial_component:
    n_runs: 6
    alpha_value: 1.68179283050743                     # $(2^k)^{1/4}$ for k=3、倍精度で pin
                                                       # 短縮値（例：1.682）は design_matrix_sha256 の
                                                       # 環境依存不一致を招くため fatal
    alpha_rationale: rotatability
  center_component:
    n_runs: 5
    replicate_purpose: [curvature_test, pure_error_estimation]
  n_runs_total: 19
  design_matrix_sha256: <string>

prohibited_actions:
  - modify_alpha_after_design_freeze                  # fatal
  - drop_center_runs_if_curvature_test_fails          # fatal（curvature を隠す）
  - use_face_centered_alpha_and_claim_rotatability    # fatal（設計思想の詐称）
```

### 11.2.3 curvature test

中心点の平均 $\bar{y}_c$ と factorial 部の平均 $\bar{y}_f$ の差が有意なら、**2 次項が必要**：

$$
\text{curvature test statistic} = \frac{\bar{y}_c - \bar{y}_f}{s \sqrt{\frac{1}{n_c} + \frac{1}{n_f}}} \sim t
$$

**Skill 契約**：`curvature_test_uri` を出力必須、$p < 0.05$ なら 2 次項推定へ進む、そうでなければ **linear model で完結宣言**（過剰モデル禁止）。

---

## 11.3 Box-Behnken Design (BBD)

### 11.3.1 BBD の特徴

Box-Behnken は **3 水準・要因ペアごとの完全要因 + 中心点**の設計です。**極端な corner（$(\pm 1, \pm 1, \pm 1)$）を使わない**のが CCD との違いで、危険物・破損リスクのある因子組合せを避けたいときに有効です。

$k=3$ の場合、runs 数：

- ペアごとの 2 因子 factorial（残り 1 因子は中央）$\times$ 3 pairs $\times$ 4 corners = 12
- center runs $n_c = 3 \sim 5$
- 合計 **15 〜 17 runs**

**CCD との比較**：

| 特性 | CCD | Box-Behnken |
|---|---|---|
| 因子水準数 | 5（factorial ± 1、axial ± $\alpha$、center 0） | 3（±1, 0） |
| 極端な角点 | あり（$(\pm 1, \pm 1, \pm 1)$）| **なし** |
| 実験域 | 立方体 | 球状に近い |
| 推奨場面 | 一般 | 危険物制限・装置限界近傍 |

### 11.3.2 Skill 契約

```yaml
experimental_design_provenance:
  design_type: box_behnken_design
  factor_levels: [-1, 0, 1]                            # 3 水準固定
  corner_exclusion_rationale: |                        # BBD 選択の根拠を明記
    温度 600°C × 圧力 10 bar × 前処理時間 30min の corner は
    装置定格を超えるため、Box-Behnken で excluded
  n_runs_total: 15
  center_runs: 3
  design_matrix_sha256: <string>

prohibited_actions:
  - add_corner_runs_post_hoc_to_bbd                    # fatal（BBD は corner-free が定義）
  - use_bbd_with_less_than_3_factors                   # fatal（BBD は k ≥ 3 が必須）
```

---

## 11.4 応答曲面フィッティング — linear → GBM → GP のパス

### 11.4.1 3 段階モデルの選択基準

エージェントは応答曲面フィッティングを **1 発でモデル選択しない**——3 段階（T1/T2/T3、§11.1.1 参照）で spec を書き、各段で承認を求めます：

| 段階 | モデル | 使うタイミング | ライブラリ | Ch6 §6.7.1 |
|---|---|---|---|---|
| **T1** | 2 次多項式回帰 | curvature test 有意、変数 3-5 個、runs 15-30 | statsmodels.ols | 初期 fit |
| **T2** | GBM（LightGBM/XGBoost）| T1 の residual に非線形パターン、runs 30+ | lightgbm | **L4 event**（T1→T2 昇格） |
| **T3** | GP（Kriging）| **予測不確かさ**が必要（Bayesian DoE 引継ぎ、外挿検知）| smt.surrogate_models.KRG | **L4 event**（T2→T3 昇格） |

**契約含意**：**「モデル選択の自由度」自体を Skill 契約で pin**する。以下のような Silent switch は fatal：

- 応答曲面 fit の結果が悪い → GBM に切替（内部で silent 実行）
- GBM でも合わない → GP に切替（同）
- 途中で切ったモデルの結果は捨てて最後のモデルの結論のみ報告

これらはすべて **HARKing**（Hypothesizing After Results are Known）の変種で、**変数選択の隠れた自由度**にあたります。

### 11.4.2 モデル選択 Skill（proposal / approval 分離）

Ch10 §10.7.3 で確立した **proposal / approval Skill 分離パターン** を応答曲面モデル選択に適用します。T1→T2 / T2→T3 昇格は Ch6 §6.7.1 L4 event（algorithm family change）であり、**agent 自律による選択は許されず、承認 Skill 経由でしか実行できません**。

**Skill A: `response_surface_model_selection_proposal_skill`（agent-autonomous）**

```yaml
skill_type: response_surface_model_selection_proposal_skill
authorization_level: agent_autonomous                        # proposal のみ、承認は Skill B
inputs:
  - design_matrix_uri
  - design_matrix_sha256
  - execution_results_uri
  - curvature_test_uri                                       # from §11.2.3
  - candidate_models: [linear_second_order, gbm, gp_kriging]
  - current_tier: T1 | T2 | T3
  - proposed_tier: T1 | T2 | T3

outputs:
  # --- HARKing 防止：immutable candidate timeline（S-1 operational detector）---
  candidate_model_timeline_uri: <string>                     # 全候補モデルの fit 履歴（append-only）
  candidate_model_timeline_sha256: <string>
  candidate_model_timeline_schema:
    entries:                                                 # 各エントリは immutable
      - timestamp: <ISO8601>
      - model_family: linear_second_order | gbm | gp_kriging
      - cv_score: <float>
      - cv_metric: rmse | mae | log_predictive_density
      - rejection_reason: <string> | selected
  selected_model: <one_of_candidate_models>                  # timeline の leaf でなければ fatal
  estimator_diff_report_uri: <string>                        # Ch6 L4 要件：旧 Skill 対比
    # 内容: point-estimate / CI / CV-RMSE の差分、feature importance の変化
  extrapolation_use_case_declaration: <string>               # T2→T3 昇格時必須
  counterfactual_scope_gate_result: <object>

prohibited_actions:
  - report_only_final_model_after_silent_switching           # fatal（HARKing）
  - selected_model_not_matching_candidate_timeline_leaf      # fatal（S-1 operational detector）
  - modify_curvature_test_p_value_threshold_after_seeing_results  # fatal
  - use_gp_for_screening_only_experiment                     # fatal（over-modeling）
  - select_gbm_or_gp_without_l1_residual_diagnostics         # fatal
  - tier_naming_l1_l2_l3_collision_with_ch6_gate_levels      # fatal（本章 §11.1.1 命名規約違反）
```

**Skill B: `response_surface_model_selection_approval_skill`（human-required）**

```yaml
skill_type: response_surface_model_selection_approval_skill
authorization_level: human_required_l4_gate                  # Ch6 §6.7.1 L4 event
inputs:
  # Skill A の全出力 + 追加証拠
  - candidate_model_timeline_uri
  - candidate_model_timeline_sha256
  - estimator_diff_report_uri
  - proposed_tier_upgrade: T1_to_T2 | T2_to_T3

outputs:
  - approval_status: approved | conditional_approved | rejected
  - approver_signature: <string>                             # research_lead 署名
  - approved_at: <timestamp>
  - new_skill_version_issued: required                       # Ch6 L4 要件：新 Skill version 発行
  - new_skill_version: <semver>

authorization_gate:
  gate_type: estimator_contract_change_gate                  # Ch6 §6.7.1 L4 approval type
  required_approver: research_lead
  required_evidence:
    - candidate_model_timeline_sha256
    - estimator_diff_report_uri                              # 旧 Skill との差分
    - cv_scores_comparison                                   # 候補モデル間の CV score
    - residual_diagnostics_uri                               # T1 の residual プロット
    - extrapolation_use_case_declaration                     # T2→T3 昇格時
  variable_selection_authorization_required: true            # factor 集合含意の変更として

prohibited_actions:
  - approve_tier_upgrade_without_estimator_diff_report       # fatal
  - approve_tier_upgrade_without_new_skill_version           # fatal（Ch6 L4 要件違反）
  - approve_own_proposal                                     # fatal（Ch4 §4.6.2 approver_independence）
```

---

## 11.5 Kriging / GP surrogate — 予測不確かさを Skill 契約に

### 11.5.1 GP の予測分散が必要な理由

T3 の Kriging / GP は、点推定 $\hat{y}(x)$ に加えて **予測分散 $\hat{\sigma}^2(x)$** を返します。応答曲面設計では、これが 3 つの用途で **必須**：

1. **外挿検知**：$\hat{\sigma}^2(x)$ は設計点から離れるほど爆発——`counterfactual_scope_gate` の判定に使用
2. **Bayesian DoE**（第12章）：次候補点選択の acquisition function の入力
3. **意思決定信頼区間**：「温度 550°C、前処理 20 分で最適」と Skill が主張するとき、95% CI $[\hat{y}(x^*) \pm 1.96 \hat{\sigma}(x^*)]$ を必須表示

### 11.5.2 smt による Kriging fitting

```python
from smt.surrogate_models import KRG
import numpy as np

# design matrix (N x k), y (N,)
X_train = np.array([...])
y_train = np.array([...])

model = KRG(
    corr="squar_exp",           # squared exponential kernel
    theta0=[1e-2] * X_train.shape[1],
    print_prediction=False,
)
model.set_training_values(X_train, y_train)
model.train()

# prediction at query point x_star
y_pred = model.predict_values(x_star.reshape(1, -1))
y_var = model.predict_variances(x_star.reshape(1, -1))
y_std = np.sqrt(y_var)
```

### 11.5.3 GP Skill 契約

```yaml
skill_type: response_surface_gp_skill
authorization_level: human_required_l4_gate            # T2→T3 昇格は Ch6 §6.7.1 L4 event

surrogate_model_provenance:                             # §10.8 ⑨ に加える block
  model_family: gaussian_process_kriging
  library: smt
  library_version: 2.4.0
  kernel:
    name: squared_exponential                           # or matern52 | matern32
    theta_optimizer: cobyla
    theta_bounds:                                       # length-scale 範囲
      min: 1e-6
      max: 1e2
    theta0: [1.0, 1.0, 1.0]                            # coded 空間で単位長さを default（S-2）
    theta_initialization_rationale: |                  # S-2: theta0 の根拠を pin
      Coded design 空間（-1, +1 に正規化済み）では、length-scale の初期値は 1.0 が中立的な出発点。
      因子スケールが大きく異なる場合は per-factor で調整（例：温度と時間の変動幅比が 10 倍以上なら
      preliminary scan で per-factor theta0 を決めて `theta_initialization_scan_uri` に pin）。
    theta_initialization_scan_uri: <optional>          # per-factor 調整時
  noise_variance:
    estimation: mle | fixed
    fixed_value: <optional>                             # fixed 時は必須
  fit_report_uri: <string>                              # log-marginal-likelihood, θ 推定値
  fit_report_sha256: <string>
  cross_validation:
    method: leave_one_out                               # LOO-CV は GP で解析的に計算可能
    rmse: <float>
    log_predictive_density: <float>
  hyperparameter_frozen_at: <timestamp>                 # fit 後は immutable

# --- 予測時の必須出力 ---
prediction_output_contract:
  point_estimate: y_pred                                # scalar or array
  prediction_variance: y_var                            # scalar or array（必須、None 不可）
  confidence_interval:
    level: 0.95
    lower: y_pred - 1.96 * sqrt(y_var)
    upper: y_pred + 1.96 * sqrt(y_var)
  # --- N-1: noisy GP での分散下限 ---
  variance_lower_bound_at_training_points:
    # noise-free GP（nugget=0）は training 点で分散 0 になる。
    # noisy GP（noise_variance.estimation ∈ {mle, fixed_value > 0}）では
    # y_var[x_train] >= noise_variance の下限を必ず満たす必要がある。
    check_required: true
    check_rule: |
      For every x within epsilon of a training point:
        y_var(x) >= noise_variance.fixed_value (or fitted MLE nugget)
    epsilon: 1e-6

prohibited_actions:
  - report_gp_point_estimate_without_variance           # fatal（外挿検知が破綻）
  - refit_gp_hyperparameters_between_prediction_calls   # fatal（不整合な予測）
  - use_gp_prediction_outside_convex_hull_without_scope_gate_pass  # fatal（§11.7）
  # --- N-1: noisy GP 用の operational 検出 ---
  - report_training_point_ci_below_estimated_noise_variance  # fatal（noisy GP）
    # noise-free GP（nugget=0）は training 点で分散 0 が正常挙動。
    # noisy GP でσ_n² = 0.02 と fit されているのに y_var[training_pt]=0 と報告するのは
    # aleatoric uncertainty の misrepresentation で fatal。
```

---

## 11.6 タグチメソッド — SN 比と品質工学

### 11.6.1 タグチメソッドの位置付け

タグチメソッド（品質工学）は **応答曲面法とは異なる哲学**——「平均を最適化する」のではなく **「ばらつきを含めた損失を最小化する」** を目的とします。ARIM 材料実験では：

- **平均値** = 材料性能（V_OC、収率）
- **ばらつき** = 装置差・環境差・オペレータ差による不均一性

タグチは **loss function** $L(y) = k(y - m)^2$ を SN 比（signal-to-noise ratio）に変換して最大化します。

### 11.6.2 SN 比の 3 種類

**契約含意**：目的関数によって SN 比の定義が異なる——エージェントが「SN 比を計算した」と主張するとき、**どの SN 比か**を明示必須：

| 種類 | 目的 | 定義 | 例 |
|---|---|---|---|
| **Larger-The-Better (LTB)** | 最大化 | $\text{SN} = -10 \log_{10}\!\left(\frac{1}{n}\sum \frac{1}{y_i^2}\right)$ | V_OC, 変換効率, 収率 |
| **Smaller-The-Better (STB)** | 最小化 | $\text{SN} = -10 \log_{10}\!\left(\frac{1}{n}\sum y_i^2\right)$ | 欠陥数, リーク電流 |
| **Nominal-The-Best (NTB)** | 目標値追従 | $\text{SN} = 10 \log_{10}\!\left(\frac{\bar{y}^2}{s^2}\right)$ | 膜厚 100nm 追従, 組成比 1:1 |

**警告**：SN 比の解釈と選択は **目的関数の宣言**に依存する。**目的関数を変えずに SN 比の種類を切り替える** は post-hoc 選択の隠れた自由度で、fatal。

### 11.6.3 直交表 L8 / L12 / L18

タグチは **標準直交表（Standard Orthogonal Arrays）** を使います：

| 直交表 | runs | 因子数 | 水準 | 用途 |
|---|---|---|---|---|
| **L8** | 8 | up to 7 | 2 水準 | small screening |
| **L12** | 12 | up to 11 | 2 水準（Plackett-Burman 型）| main effects only |
| **L18** | 18 | 1 × 2-level + 7 × 3-level | 混合水準（$2^1 \times 3^7$）| 混合水準 screening + optimization |
| **L27** | 27 | up to 13 | 3 水準 | full 3-level fractional |

**inner array / outer array** の 2 層構造がタグチの特徴：

- **inner array**：制御因子（設計者が選べる）
- **outer array**：ノイズ因子（環境・使用時変動）

各 inner cell を outer array の全条件で反復——**環境変動に頑健な水準を選ぶ**。

### 11.6.4 タグチ Skill 契約

```yaml
skill_type: taguchi_optimization_skill
authorization_level: human_required

experimental_design_provenance:
  design_type: taguchi_orthogonal_array
  oa_name: L18                                          # L8 | L12 | L18 | L27
  oa_source: taguchi_standard | oapackage_generated
  oa_lookup_uri: <string>                               # 標準表の source
  inner_array:
    factors_count: 5
    factors: [...]
  outer_array:
    factors_count: 2
    factors: [batch_number, ambient_humidity]
    runs_per_inner_cell: 4
  n_runs_total: 72                                      # 18 × 4

sn_ratio_declaration:
  ratio_type: larger_the_better                         # または smaller_the_better | nominal_the_best
  objective_function: v_oc_maximization
  frozen_at: <timestamp>                                 # SN 比選択は design_freeze 時に凍結
  rationale: |
    V_OC は太陽電池の性能指標であり最大化が目的。STB / NTB は該当しない。

interpretation_contract:
  main_effects_plot_uri: <string>                       # 各因子水準ごとの SN 比の平均
  optimal_condition_prediction:
    conditions: {factor_A: 3, factor_B: 2, ...}
    predicted_sn: <float>
    confirmation_run_required: true                     # タグチ流：最適条件を必ず確認実験
    # --- N-5: confirmation run conditions を hash で pin ---
    confirmation_run_conditions_uri: <string>           # 確認実験の水準セット定義
    confirmation_run_conditions_sha256: <string>        # 承認前に pin、実行時 verify
    confirmation_run_authorization_gate: intervention_execution_authorization

prohibited_actions:
  - switch_sn_ratio_type_after_execution                # fatal（HARKing）
  - report_optimal_condition_without_confirmation_run   # fatal（タグチ流に反する）
  - use_ltb_sn_for_bidirectional_target                 # fatal（SN 選択誤り）
  - conflate_control_and_noise_factors                  # fatal（inner/outer array 意義喪失）
  - claim_taguchi_analysis_without_outer_array_declaration  # warning → fatal if published
  - modify_confirmation_run_conditions_between_approval_and_execution  # fatal（N-5、silent switch）
```

> **Ch12 への forward reference（S-6）**：タグチ SN 比は「事前宣言された outer array のノイズ因子」に対する頑健性を扱います。一方、第12章の **Bayesian DoE** は「事後に学習されるノイズ分布」に対する頑健性を acquisition function で逐次最適化します。両者は補完的で、Bayesian DoE は「未見のノイズに対する robustness」を subsume できます（詳細は §12.4）。

### 11.6.5 応答曲面 vs タグチ の使い分け

| 場面 | 応答曲面法 | タグチメソッド |
|---|---|---|
| **目的が最適水準の点推定** | ✅ | △ |
| **目的が頑健性（環境変動に強い水準）** | △ | ✅ |
| **予測不確かさが必要** | ✅（GP）| ✗ |
| **runs 数を抑えたい** | △（CCD 15-19）| ✅（L8 で 8 runs）|
| **多水準連続因子** | ✅ | △（3 水準に離散化） |
| **多ノイズ因子への robustness** | ✗ | ✅（outer array） |

---

## 11.7 応答曲面の外挿失敗 — Agentic の最頻失敗パターン

### 11.7.1 失敗シナリオ

応答曲面フィッティング後、エージェントは以下のような主張をよくします：

> **「応答曲面によれば、温度 720°C、圧力 15 bar、前処理 45 分で V_OC 最大 1.28 V」**

しかし実験域は **温度 400-600°C、圧力 1-10 bar、前処理 15-30 分**——**外挿**です。多項式回帰は外挿域で任意に外挿し、GBM は境界付近で不連続、GP は不確かさが爆発します——それでも点推定は返してしまう。

**これは Ch4 §4.5.2 の `counterfactual_scope_gate` が fail-close すべき典型例**。応答曲面 Skill は以下を必須化：

### 11.7.2 応答曲面での `counterfactual_scope_gate` 契約

```yaml
counterfactual_scope_gate:
  # Ch8-9 で導入された 4 checks を応答曲面用に operational 化
  mahalanobis_check:
    training_points_uri: <string>                       # 設計行列の点群
    threshold: 3.0                                       # Mahalanobis d²  < 3σ 圏内
    cluster_conditional: true                            # blocking factor 別に cluster
  variance_check:
    method: gp_predictive_variance                       # GP のときは自然に計算
    threshold: <float>                                   # 訓練時 max variance の 2 倍等
  knn_density_check:
    k: 5                                                 # 訓練データ N=15-20 なので小さめ
    knn_min: 2
  support_envelope_check:
    method: convex_hull_or_ellipsoid
    envelope_report_uri: <string>
    strict: true                                         # 応答曲面では strict（凸包外は fail）

  # --- B-3: threshold_calibration（Ch4 §4.5.2 canonical / Ch10 §10.8 と parity）---
  threshold_calibration:
    method: leave_one_out_empirical                     # or conformal（推奨、RSM N≈15-20 では conformal が有効）|
                                                         # validation_false_extrapolation_rate
    calibration_evidence_uri: <string>                  # threshold 選択の根拠エビデンス
    calibration_approved_at: <timestamp>                # 承認時刻（silent 上書き禁止）

  # --- 応答曲面特有の状態 ---
  aggregate_policy:
    pass_requires: all_four_pass
    conditional_pass_output: non_actionable_diagnostic_only
    fail_action: fail_close_and_reject_optimum_recommendation
  # --- S-4: 3-state gate_status を明示出力 ---
  gate_status: pass | conditional_pass | fail           # Ch4 §4.5.2 3-state canonical
  fallback: human_review
  fallback_approver: research_lead
  fallback_message_template: |
    Response surface Skill proposes optimum at x* = {x_star}.
    Design region (convex hull): {design_hull_summary}.
    Failing checks: {failing_checks}.
    Predicted variance at x*: {y_var}.
    Extrapolation from design region: manual review by {approver} required.
    Options: (a) reject the recommendation, (b) add confirmation runs at x* to extend the design region.

prohibited_actions:
  - report_optimum_outside_convex_hull_without_scope_gate_pass  # fatal
  - silently_extend_prediction_range_beyond_design_matrix       # fatal
  - use_polynomial_extrapolation_as_actionable_recommendation   # fatal
  - suppress_gp_predictive_variance_in_optimum_report           # fatal
  - modify_scope_gate_threshold_without_calibration_evidence    # fatal（B-3、Ch4 §4.5.2 item 7）
```

### 11.7.3 「境界最適解」の追加リスク

最適解が **設計境界上**にあるとき、応答曲面は「その方向に更に伸びる」ことを示唆します——実際にはその外は未観測。エージェントは以下を **必ず** 追加宣言：

```yaml
# --- N-4: operational な boundary 検出 ---
optimum_on_design_boundary:
  detection_rule: max_coded_component_absolute_value_greater_equal_alpha_tolerance
  alpha_boundary_tolerance: 0.95            # coded α の 95%（= 5% 以内で境界とみなす）
                                             # 例: rotatable CCD で α=1.68、threshold = 0.95 * 1.68 = 1.596
                                             # optimum の coded component の |max| がこれ以上なら boundary
  distance_to_convex_hull_face_threshold: 0.05  # 代替: 凸包面までの相対距離（設計スパン正規化）
                                                # どちらかの検出で true 判定
  detected: true | false
  detected_axis: <string>                    # boundary trigger となった因子名（複数可）
```

- boundary optimum の場合 → **追加確認実験の推奨**（`recommended_confirmation_runs_uri`）を必須出力
- 追加実験を **エージェントが勝手に発注しない**——`intervention_execution_authorization` を必須通過
- 確認実験の水準セットは §11.6.4 と同じく `confirmation_run_conditions_sha256` で pin

---

## 11.8 応答曲面 Skill 全体契約テンプレート

Ch10 §10.8 のテンプレートに応答曲面固有項目を追加：

```yaml
skill:
  name: response_surface_perovskite_voc_skill
  version: 0.1.0
  purpose: |
    ペロブスカイト太陽電池の V_OC を組成 × 温度 × 前処理時間の応答曲面として fit し、
    最適水準を提案する（外挿禁止、GP 予測不確かさ必須）。

  # === ① 目的 ===
  estimand_type: ate                                    # 応答曲面上の各点 ATE
  mediation_role: not_applicable
  causal_question_type: type_e_doe

  # === ② 入力条件（L1: 識別レイヤ） ===
  identification_strategy: randomized_experiment
  dag_of_record_uri: "artifact://dags/rsm_voc.dot"
  dag_of_record_sha256: <sha256>
  confounders_declared: []
  mediators_declared: []
  colliders_declared: []
  blocking_factors_declared:
    - instrument_id
    - operator_id

  # === ②' データ来歴 ===
  data_lineage:
    dataset_uri: <string>
    dataset_sha256: <string>
    arim_facility_id: <string>
    experiment_ids: [<id>, ...]

  # === ②'' 識別仮定（L2, Ch4 §4.4.1 canonical）===
  positivity_by_stratum:
    status: checked
    strata:
      - key: ccd_cell
        min_treated: 1
        min_control: not_applicable
        violation_policy: fail_close
        stratum_report_uri: <string>
  sutva_declared:
    status: assessed
  consistency_declared:
    status: checked
  exchangeability_declared:
    status: checked

  # --- 応答曲面固有：L2 追加仮定 ---
  response_model_correctness_declared:
    status: assessed
    model_family: gaussian_process_kriging                # 選択されたモデル
    kernel: squared_exponential
    residual_diagnostics_uri: <string>
    cv_report_uri: <string>
  experimental_region_validity_declared:
    status: checked_within_design | fail_close_outside
    convex_hull_uri: <string>
    convex_hull_sha256: <string>

  # === ③ 出力形式 ===
  outputs:
    response_surface_uri: <string>                       # fitted model artifact
    optimum_recommendation:
      x_star: {temperature: 512, pressure: 4.8, pretreatment_time: 22}
      y_star_point: 1.14
      y_star_variance: 0.003                              # GP からの予測分散（必須）
      y_star_ci_95: [1.03, 1.25]
      optimum_on_design_boundary: false
      recommended_confirmation_runs_uri: <string>         # boundary optimum の場合必須
    test_results: dict
    # --- B-5: sensitivity_analysis を Ch10 §10.8 と parity で完全展開 ---
    sensitivity_analysis:
      method: e_value
      effect_direction: harmful | protective              # harmful は lower_bound、protective は upper_bound
      effect_scale: transformed_continuous_with_smd_to_rr_conversion
                                                          # RSM outcome は V_OC / 収率 / 膜厚などの連続量
      ci_bound_closest_to_null: lower_bound               # effect_direction に応じて明示
      smd_to_rr_conversion: vanderweele_2020
      threshold:
        minimum_e_value: 1.5
      provenance_field: sensitivity_report_uri

  # === ④ 成功条件 ===
  success_criteria:
    identification_validity: required
    refutation_pass: required
    positivity_check: required
    external_validity: required                           # 応答曲面では特に重要
    doe_efficiency: required
    randomization_integrity: required
    response_surface_fit_quality: required                # 応答曲面固有
    counterfactual_scope_gate_pass: required              # 外挿禁止（応答曲面固有）

  # --- refutation gate (Ch9) ---
  declared_required_tests:
    - random_common_cause
    - data_subset_validation                              # design 内 subset での fit 再検証
    - scope_gate_reverification                           # optimum の外挿性
    - ite_prediction_coverage_refutation                  # GP の predictive coverage
    - e_value                                             # B-5: RSM outcome は連続量、E-value を parity で必須
  applicability_manifest_uri: <string>
  applicability_manifest_sha256: <string>
  preregistration_manifest_uri: <string>
  preregistration_manifest_sha256: <string>

  # --- counterfactual scope gate (§11.7.2) を完全展開（B-3: threshold_calibration 含む）---
  counterfactual_scope_gate:
    mahalanobis_check:
      threshold: 3.0
      cluster_conditional: true
    variance_check:
      method: gp_predictive_variance
      threshold: <float>
    knn_density_check:
      k: 5
      knn_min: 2
    support_envelope_check:
      strict: true
      envelope_report_uri: <string>
    threshold_calibration:                                # B-3: Ch4 §4.5.2 canonical 必須
      method: leave_one_out_empirical                     # or conformal | validation_false_extrapolation_rate
      calibration_evidence_uri: <string>
      calibration_approved_at: <timestamp>
    aggregate_policy:
      pass_requires: all_four_pass
      conditional_pass_output: non_actionable_diagnostic_only
      fail_action: fail_close_and_reject_optimum_recommendation
    gate_status: pass | conditional_pass | fail          # S-4: 3-state 出力
    fallback: human_review
    fallback_approver: research_lead

  # === ⑤ 禁止事項 ===
  prohibited_actions:
    - modify_factors_or_levels_after_design_freeze
    - modify_alpha_after_design_freeze                    # CCD 固有
    - drop_center_runs_if_curvature_test_fails            # CCD 固有
    - response_surface_model_switch_without_approval      # §11.4 固有
    - report_only_final_model_after_silent_switching      # §11.4 固有（HARKing）
    - selected_model_not_matching_candidate_timeline_leaf # §11.4.2 S-1 operational detector
    - tier_naming_l1_l2_l3_collision_with_ch6_gate_levels # §11.1.1 命名衝突（B-1）
    - report_gp_point_estimate_without_variance           # §11.5 固有
    - report_training_point_ci_below_estimated_noise_variance  # §11.5.3 noisy GP（N-1）
    - report_optimum_outside_convex_hull_without_scope_gate_pass  # §11.7 固有
    - silently_extend_prediction_range_beyond_design_matrix       # §11.7 固有
    - use_polynomial_extrapolation_as_actionable_recommendation   # §11.7 固有
    - suppress_gp_predictive_variance_in_optimum_report           # §11.7 固有
    - modify_scope_gate_threshold_without_calibration_evidence    # §11.7.2 B-3（Ch4 §4.5.2 item 7）
    - switch_sn_ratio_type_after_execution                # タグチ固有（§11.6）
    - modify_confirmation_run_conditions_between_approval_and_execution  # §11.6.4 N-5
    - preregistration_manifest_post_hoc_modification
    - assignment_log_row_reorder_after_execution
    - execution_record_missing_design_matrix_sha256
    # --- B-4: Ch10 §10.5.3 4 段 detection 由来の 2 fatal ---
    - assignment_log_missing_header_record                # fatal（Ch10 §10.5.3、B-5 由来）
    - silent_deletion_of_failed_runs_without_design_matrix_recompute  # fatal（同上）
    # --- B-6: mixed model（blocking 宣言時必須）2 fatal ---
    - swap_mixed_model_specification_between_design_and_analysis  # fatal（Ch10 N-5）
    - random_effects_missing_declared_blocking_factor     # fatal（同上）

  # === ⑥ 再現性条件 ===
  library_stack:
    design_generation: pyDOE2==1.3.0
    surrogate: smt==2.4.0                                  # or GPy | scikit-learn.GaussianProcess
    analysis: statsmodels==0.14.0
  environment_lock_uri: <string>
  random_seed: 42
  container_sha256: <string>

  # === ⑦ 承認ゲート ===
  authorization_gates:
    dag_authorization:
      required_for: [dag_of_record_uri_change, dag_of_record_sha256_mismatch]
      approver: research_lead
    variable_selection_authorization:
      required_for:
        - factor_universe_change
        - design_parameter_change                          # CCD alpha / BBD center runs 含む
        - response_surface_model_family_change             # 応答曲面固有
        - sn_ratio_type_change                             # タグチ固有
      approver: research_lead
    intervention_execution_authorization:
      required_for:
        - physical_experiment_execution
        - confirmation_run_execution_at_optimum            # §11.7.3 boundary optimum
      approver: pi_and_facility_manager
    approver_independence:
      conflict_policy: independent_reviewer_required_if_approver_is_data_producer
      fallback_approver: facility_causal_review_board
      facility_scope_escalation:
        applies_to:
          - approved_facility_design_template
          - approved_facility_response_surface_baseline    # N-3: 応答曲面 baseline の facility 標準化
                                                            # 定義: 特定材料クラス向けの GP kernel + hyperparameter 範囲テンプレート
                                                            #       ≥N の approved 実験の後に facility 標準として promotion 可能
                                                            # Ch4 §4.6.2 enum に back-register 済み
        default_approver: facility_causal_review_board

  # === ⑧ Estimator provenance reference ===
  estimator_provenance_reference:
    estimator_contract_sha256: <string>
    dag_of_record_sha256: <string>
    design_matrix_sha256: <string>
    surrogate_model_sha256: <string>                       # 応答曲面固有

  # === ⑨ DoE 拡張（Ch10 §10.8 と feature parity）===
  experimental_design_provenance:
    design_type: central_composite_design                  # or box_behnken_design | taguchi_orthogonal_array
    factors: [...]                                         # N-6: 各エントリは Ch4 §4.9 / Ch10 §10.2.2 に従い
                                                            # role: primary|nuisance|blocking, levels, coded_mapping を必須
    factors_frozen_at: <timestamp>
    n_runs_total: 19
    ccd_variant: rotatable
    alpha_value: 1.68179283050743                          # S-3: 倍精度で pin（$2^{0.75}$）
    center_runs: 5
    curvature_test_uri: <string>
    randomization_seed_provenance: {...}
    blocking_factors_declared: [instrument_id, operator_id]  # 宣言時は下記 mixed model block も必須
    design_matrix_uri: <string>
    design_matrix_sha256: <string>
    assignment_log_uri: <string>
    assignment_log_sha256: <string>                        # header + records の hash
                                                            # 4-stage detection per Ch10 §10.5.3:
                                                            # seed_match / design_hash_match / permutation_reproducibility / execution_records_binding
    information_gain_metric: g_efficiency                  # 応答曲面では G-eff（worst-case 予測分散）が推奨
    information_gain_threshold: 0.80
    information_gain_baseline:                             # N-7: Ch10 §10.8 と parity で 4 フィールド完全展開
      baseline_design_type: d_optimal_with_same_run_budget
      baseline_efficiency: 1.00
      efficiency_ratio_achieved: 0.85                      # 実際に達成した相対効率
      n_runs_ratio: 0.50                                   # baseline に対する run 数比
    # --- B-6: blocking 宣言に対応する mixed_model_specification（Ch10 N-5 fatal 防止）---
    mixed_model_specification_uri: <string>                # 分析時の random effect 指定ファイル
    mixed_model_specification_sha256: <string>             # design 段 ↔ analysis 段 swap 検知
    mixed_model_specification_schema: statsmodels_mixedlm  # 仕様形式の pin
    mixed_model_random_effects_declared: [instrument_id, operator_id]  # blocking factors と一致必須
    mixed_model_fixed_effects_declared: [temperature, pressure, pretreatment_time]

  # === ⑩ 応答曲面固有：surrogate model provenance ===
  surrogate_model_provenance:
    model_family: gaussian_process_kriging
    library: smt
    library_version: 2.4.0
    kernel: {name: squared_exponential, theta_optimizer: cobyla}
    fit_report_uri: <string>
    fit_report_sha256: <string>
    cross_validation:
      method: leave_one_out
      rmse: <float>
      log_predictive_density: <float>
    hyperparameter_frozen_at: <timestamp>
```

---

## 11.9 実装スニペット — smt / statsmodels / pyDOE2

### 11.9.1 CCD design 生成

```python
from pyDOE2 import ccdesign
import numpy as np
import hashlib
import json

# k=3 factors, rotatable, 5 center runs
design_coded = ccdesign(3, center=(0, 5), alpha='r', face='ccc')
# shape (19, 3), coded as -alpha, -1, 0, +1, +alpha
# alpha for rotatable k=3: 2^0.75 = 1.68179283050743 (double precision)

ALPHA_K3_ROTATABLE = 2 ** 0.75  # S-3: 倍精度で computed、byte-exact 再現に必須

# actual level mapping（S-3: coded axis は倍精度 α を使う）
level_maps = {
    "temperature":       {-ALPHA_K3_ROTATABLE: 350, -1: 400, 0: 500, 1: 600, ALPHA_K3_ROTATABLE: 650},
    "pressure":          {-ALPHA_K3_ROTATABLE: 0.3, -1: 1.0, 0: 5.0, 1: 10.0, ALPHA_K3_ROTATABLE: 12.0},
    "pretreatment_time": {-ALPHA_K3_ROTATABLE:   5, -1:  15, 0:  22, 1:  30, ALPHA_K3_ROTATABLE:  35},
}
# note: 実運用では coded → actual mapping を YAML で pin（α も倍精度で）

# deterministic hash (Ch10 §10.9.1 パターン踏襲)
design_f64 = design_coded.astype(np.float64)
schema_header = json.dumps({
    "design_type": "ccd_rotatable",
    "factors": ["temperature", "pressure", "pretreatment_time"],
    "alpha": ALPHA_K3_ROTATABLE,                 # S-3: 倍精度で pin（1.682 の短縮値は fatal）
    "center_runs": 5,
    "dtype": "float64",
    "shape": list(design_f64.shape),
    "byte_order": "little",
}, sort_keys=True).encode("utf-8")
design_sha256 = hashlib.sha256(schema_header + b"|" + design_f64.tobytes()).hexdigest()
```

### 11.9.2 Box-Behnken

```python
from pyDOE2 import bbdesign

design = bbdesign(3, center=3)  # k=3, 3 center runs → 15 runs
```

### 11.9.3 2 次多項式応答曲面 fit（statsmodels）

```python
import statsmodels.formula.api as smf
import pandas as pd

df = pd.DataFrame({
    "T": T_values, "P": P_values, "t": t_values, "y": y_observed
})

# 2 次多項式：main + 2-way + squared terms
model = smf.ols(
    "y ~ T + P + t + T:P + T:t + P:t + I(T**2) + I(P**2) + I(t**2)",
    data=df
).fit()

print(model.summary())
# curvature test: I(T**2), I(P**2), I(t**2) の t 検定を統合
```

### 11.9.4 Kriging surrogate（smt）

```python
from smt.surrogate_models import KRG
import numpy as np

X_train = df[["T", "P", "t"]].values
y_train = df["y"].values

model = KRG(
    corr="squar_exp",
    theta0=[1e-2, 1e-2, 1e-2],
    hyper_opt="Cobyla",
    print_prediction=False,
)
model.set_training_values(X_train, y_train)
model.train()

# grid prediction for surface visualization
T_grid, P_grid = np.meshgrid(np.linspace(400, 600, 20), np.linspace(1, 10, 20))
X_query = np.column_stack([T_grid.ravel(), P_grid.ravel(), np.full(400, 22.5)])
y_pred = model.predict_values(X_query)
y_var = model.predict_variances(X_query)

# LOO CV — diagnostic-only（S-5: hyperparameter は各 fold で再 fit するが、
# これは held-out 汎化性能の測定用のみ。deployed Skill は §11.5.3 契約に従い
# 単一 fit + hyperparameter_frozen_at pin で運用され、予測呼出間で refit しない
# （prohibited: refit_gp_hyperparameters_between_prediction_calls）。
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
rmse_scores = []
for train_idx, test_idx in loo.split(X_train):
    m = KRG(corr="squar_exp", theta0=[1e-2] * 3, print_prediction=False)
    m.set_training_values(X_train[train_idx], y_train[train_idx])
    m.train()
    y_hat = m.predict_values(X_train[test_idx])
    rmse_scores.append((y_train[test_idx] - y_hat.ravel())**2)
loo_rmse = np.sqrt(np.mean(rmse_scores))
```

### 11.9.5 タグチ SN 比計算

```python
import numpy as np

def sn_ratio(y, target="ltb", nominal=None):
    """
    y : array of replicates (or per-cell average of outer array)
    target : 'ltb' | 'stb' | 'ntb'
    """
    y = np.asarray(y, dtype=float)
    if target == "ltb":
        return -10 * np.log10(np.mean(1.0 / y**2))
    elif target == "stb":
        return -10 * np.log10(np.mean(y**2))
    elif target == "ntb":
        return 10 * np.log10(np.mean(y)**2 / np.var(y, ddof=1))
    else:
        raise ValueError(f"Unknown target: {target}")

# per-cell SN across outer array replicates
cell_sn = np.array([sn_ratio(cell_y, target="ltb") for cell_y in outer_replicates_per_cell])

# main effects: 各因子水準ごとの SN 平均
```

---

## 11.10 章末チェックリスト

- [ ] **§11.1 screening → optimization の pivot**を、識別仮定と `response_model_correctness_declared` の追加として説明できる
- [ ] **§11.1.1 モデル階層 T1/T2/T3**（linear → GBM → GP）と、Ch6 §6.7.1 との用語衝突（`L1/L2/L3` 誤用）を回避しつつ、全昇格が Ch6 **L4 event** であることを説明できる
- [ ] **§11.2 CCD の 3 成分**（factorial + axial + center）と rotatable/orthogonal/CCF の使い分けを説明できる
- [ ] **§11.2.3 curvature test** で 2 次項の必要性を判定し、post-hoc に center runs を落とす禁止行為を fail-close できる
- [ ] **§11.3 Box-Behnken** の corner-free 性と、危険物制限下での使用場面を説明できる
- [ ] **§11.4.2 モデル選択 Skill** の「silent switch 禁止」「HARKing 防止」契約を書ける
- [ ] **§11.5.2 smt Kriging** で `predict_values` + `predict_variances` を組で返し、変数を suppress しない契約を書ける
- [ ] **§11.6.2 SN 比の 3 種類**（LTB / STB / NTB）と、目的関数固定下で post-hoc 切替禁止を説明できる
- [ ] **§11.6.3 直交表 L8/L12/L18/L27** の使い分けと、inner array / outer array の意味を説明できる
- [ ] **§11.7 応答曲面の外挿失敗**を、`counterfactual_scope_gate` 4-check + `fail_action: fail_close_and_reject_optimum_recommendation` で fail-close できる
- [ ] **§11.7.3 boundary optimum** の場合、`recommended_confirmation_runs_uri` を必須出力にできる
- [ ] **§11.8 応答曲面 Skill 契約テンプレート**を、自研究テーマで完全に埋められる
- [ ] **§11.9 smt / statsmodels / pyDOE2** の実装スニペット（ccdesign / bbdesign / ols / KRG）と決定論的 hash を実装できる

---

## 章末演習

### 演習 11.1：CCD 設計と curvature test の spec

第10章 演習 10.1 で立てた full factorial（screening）を発展させ、以下を作成：

1. curvature test の統計仮説と検定手順
2. CCD rotatable で $k$ 因子・center runs $n_c$ を決定した根拠
3. $\alpha$ の値と rotatability を保証する計算
4. §11.8 テンプレートを埋めた Skill 仕様書

### 演習 11.2：Kriging surrogate の外挿検知

以下のシナリオで `counterfactual_scope_gate` の 4-check を operational に設計：

- 訓練点：温度 400-600°C, 圧力 1-10 bar, 前処理 15-30 分 の CCD 19 点
- クエリ点 1：$x^* = (500, 5.5, 22)$（training convex hull 内）
- クエリ点 2：$x^* = (720, 15, 45)$（明確に外挿）
- クエリ点 3：$x^* = (610, 10.5, 31)$（境界すれすれ）

各点で mahalanobis / variance / knn density / support envelope をどう計算し、pass/conditional_pass/fail を判定するかを書き下せ。

### 演習 11.3：タグチメソッドの SN 比選択

以下 3 つのシナリオで、適切な SN 比を選び、`sn_ratio_declaration` block を書く：

1. ペロブスカイト太陽電池の V_OC 最大化
2. 電池内部リーク電流の最小化
3. 薄膜の厚さを 100 nm ± 5 nm に制御

さらに、実行後に SN 比タイプを切り替える「post-hoc HARKing」を fail-close する `prohibited_actions` を具体化。

### 演習 11.4：応答曲面 vs タグチの選択

自研究テーマで、以下 2 パスのどちらを選ぶか根拠を書く：

- Path A：CCD + GP surrogate → 最適点の点推定 + CI
- Path B：L18 直交表 + SN 比 → 頑健水準の選定

**§11.6.5 の比較表**を援用し、目的（点推定 vs 頑健性）・runs 制約・多ノイズ因子の有無で判断。

---

## 参考資料

### 内部 cross-reference

- **第0章**：4 識別仮定（本章 §11.1.2 で response_model_correctness と experimental_region_validity を追加宣言）
- **第4章**：`experimental_design_provenance` 契約定義、Ch4 §4.5.2 counterfactual_scope_gate 4-check（本章 §11.7 で応答曲面用に operational 化）、Ch4 §4.6.2 3 層承認（本章 §11.7.3 boundary optimum の confirmation run authorization）
- **第6章**：`estimator_contract_change_gate` L1-L4（本章 §11.1.1 のモデル階層 T1/T2/T3 は Ch6 の L1-L4 とは別の概念——用語衝突を避けるため T1/T2/T3 を使用。全 T1→T2 / T2→T3 昇格は Ch6 **L4 event** に該当）
- **第8章**：g-formula と外挿範囲チェック（本章 §11.7 応答曲面外挿の operational 実装源泉）
- **第9章**：refutation_gate 8-concept enum、`ite_prediction_coverage_refutation`（GP predictive coverage、本章 §11.8 で応答曲面 refutation gate に採用）
- **第10章**：full factorial / fractional factorial（screening）、`design_matrix_sha256` immutability chain、`randomization_seed_provenance` パターン（本章で CCD / BBD / OA に踏襲）
- **第12章**：Bayesian D-/A-optimality、one-shot Bayesian DoE（本章 §11.5 GP と接続）
- **第13章**：capstone で応答曲面と反実仮想を統合、§11.7 の外挿禁止契約を実践
- **第14章**：応答曲面の外挿誤用・タグチ SN 比の誤解釈の失敗パターン
- **vol-03 第8章**：深層特徴を GP の入力として活用する応用（本章 §11.5 GP と接続可能）

### 外部参考文献

- Myers, R. H., Montgomery, D. C., & Anderson-Cook, C. M. (2016). *Response Surface Methodology* (4th ed.). Wiley. — 応答曲面法の標準教科書
- Box, G. E. P., & Draper, N. R. (2007). *Response Surfaces, Mixtures, and Ridge Analyses* (2nd ed.). Wiley. — 幾何学的直観と ridge analysis
- Taguchi, G., Chowdhury, S., & Wu, Y. (2005). *Taguchi's Quality Engineering Handbook*. Wiley. — タグチ本人の総合ハンドブック
- Roy, R. K. (2010). *A Primer on the Taguchi Method* (2nd ed.). SME. — 実務向け導入
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. https://gaussianprocess.org/gpml/ — GP の理論書（無償公開）
- smt (Surrogate Modeling Toolbox): https://smt.readthedocs.io/ — Python Kriging 実装
- pyDOE2 GitHub: https://github.com/clicumu/pyDOE2 — `ccdesign` / `bbdesign` / `lhs`

### API チートシート（詳細は付録 B）

| ライブラリ | 主要関数 | 用途 |
|---|---|---|
| pyDOE2 | `ccdesign(n, center, alpha, face)` | Central Composite |
| pyDOE2 | `bbdesign(n, center)` | Box-Behnken |
| smt | `KRG(corr, theta0)` | Kriging surrogate |
| smt | `.predict_variances(X)` | 予測分散（外挿検知に必須） |
| GPy | `GPy.models.GPRegression` | GP 回帰（代替） |
| statsmodels | `smf.ols("y ~ T + I(T**2) + T:P", ...)` | 2 次多項式応答曲面 |
| OApackage | `oa_from_txt("L18.oa")` | タグチ L18 表 |
