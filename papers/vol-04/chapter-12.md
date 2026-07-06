# 第12章　Bayesian 実験計画（one-shot）— 事前分布と情報利得で Skill 化

> **学習目標**：
> - 第10章の screening、第11章の optimization に続き、**事前分布と事後分布を Skill 契約に組み込む** Bayesian DoE の位置付けを説明できる
> - **Bayesian D-optimality / A-optimality** の目的関数を書き下し、pyDOE2 の D-efficiency との違いを Skill 契約項として明示できる
> - PyMC を用いた **「事前分布 → 実験計画一括生成 → 事後分布」の one-shot ループ**を Skill として書ける
> - vol-02 第11章の **階層モデル**（階層事前）を DoE 事前分布として再利用し、装置差・オペレータ差を組み込める
> - **情報利得（KL / mutual information）** を Skill の意思決定変数として `information_gain_metric` に登録できる
> - 第11章 T3 の GP surrogate と Bayesian DoE を接続し、事前 GP + 事後 GP のワークフローを構築できる
> - **逐次 Bayesian DoE（acquisition function-based BO）は vol-05 に完全委譲**——本章の one-shot 契約が「なぜ逐次と切り離せるか」を橋渡し節で説明できる

---

## 12.1 Bayesian DoE の位置付け — screening / optimization / prior-informed のラダー

第10章・第11章で構築した DoE の Skill 契約は、いずれも **頻度主義的**——設計行列は「情報利得の期待値」を metric（D-efficiency / G-efficiency 等）で最大化しますが、事前知識は明示的にモデル化されません。

しかし ARIM の材料実験は、**過去の類似実験・文献値・シミュレーションデータ**が豊富にあります。「温度 500°C 付近が最適」という事前信念を持ちながらも、CCD で 400-600°C を等価に扱うのはもったいない——**事前分布**を明示すれば、事後の情報利得（データを見た後の不確かさ縮小）を最大化する設計が可能です。これが **Bayesian DoE** の出発点です。

| 段階 | 目的 | 代表章 | 事前分布の扱い |
|---|---|---|---|
| **screening** | 因子分別 | 第10章 | 明示しない（uninformative） |
| **optimization** | 最適水準探索 | 第11章 | 明示しない（GP は non-informative の一種） |
| **prior-informed one-shot** | **事前信念を組み込んだ一括計画** | **本章（第12章）** | **明示的（Bayesian）** |
| sequential BO | 次候補逐次探索 | **vol-05 完全委譲** | 事前分布 + 逐次更新 |

### 12.1.1 one-shot の scope 限定（v0.2 での再定義）

本章は **one-shot Bayesian DoE** に scope を限定します：

- **入力**：事前分布 $p(\theta)$、モデル $p(y \mid x, \theta)$、run 予算 $N$
- **出力**：**一括生成された** $N$-run 実験計画 $\{x_1, \ldots, x_N\}$、事後分布 $p(\theta \mid y_{1:N})$
- **スコープ外**：$x_{n+1}$ を $y_{1:n}$ に条件付けて逐次選択する **acquisition function ベースの BO**（vol-05）

この scope 限定の根拠：

1. **契約複雑度の管理**：逐次 BO は「停止条件」「stopping rule の HARKing」「acquisition function の post-hoc 切替」等、one-shot にない失敗モードを持つ。これらは Skill 契約項として独立の章立てが必要（vol-05 §5-8）
2. **Human-in-the-loop の粒度**：one-shot は 1 回の approval で完結し、`intervention_execution_authorization` の粒度と自然に整合。逐次は run ごと承認 vs 一括承認の選択が必要で、独立した設計判断
3. **本書の学習曲線**：Ch10-11 → Ch12 で「頻度主義 → Bayesian」を学び、vol-05 で「one-shot → 逐次」を学ぶ 2 段階カリキュラム

**§12.11 の橋渡し節**（1 ページ）で、one-shot 契約が vol-05 の逐次契約にどう接続するかを示します。

### 12.1.2 識別仮定と事前分布の関係

第0章の 4 識別仮定（exchangeability / positivity / consistency / SUTVA）に加え、Bayesian DoE では **事前分布の妥当性**が新たな仮定になります：

| 仮定 | 状態 | Bayesian DoE での意味 |
|---|---|---|
| exchangeability | **checked** | randomization で保証（第10章） |
| positivity | **checked** | 事前分布の support が実験域を含む |
| consistency | **checked** | プロトコル pin（第10章） |
| SUTVA | *assessed* | 依然として carryover ゼロ仮定 |
| **prior specification correctness** | *assessed* | **事前分布が事前知識と整合、かつ well-calibrated** |
| **prior-data alignment** | *checked_after_execution* | 事後分布が事前分布から大きく乖離する場合は事前 misspecification を疑う（§12.7） |

**エージェントが勝手に事前分布を「wide にして中立を装う」ことは fatal**——事前分布の選択は `dag_of_record` と同格の記録対象として `prior_specification_provenance` に pin されます。

---

## 12.2 事前分布の設計 — vol-02 第11章の階層モデルとの接続

### 12.2.1 事前分布の 3 層構造

Bayesian DoE の事前分布は 3 層で設計します（vol-02 第11章の階層モデルの直系継承）：

$$
\underbrace{\theta_{ij} \sim \mathcal{N}(\mu_j, \tau_j^2)}_{\text{group-level: 装置 } j \text{ の因子 } i}
\;;\;
\underbrace{\mu_j \sim \mathcal{N}(\mu_0, \sigma_0^2), \; \tau_j \sim \text{HalfCauchy}(0, s)}_{\text{hyper-prior: 装置間分散}}
\;;\;
\underbrace{\mu_0, \sigma_0, s \text{ fixed}}_{\text{hyperparameter}}
$$

ARIM 材料実験の典型的マッピング：

| 階層レベル | 変動源 | 事前分布の例 |
|---|---|---|
| **individual**（$\theta_{ij}$）| 個別 run の効果 | $\mathcal{N}(\mu_j, \tau_j^2)$ |
| **group**（$\mu_j$）| 装置 $j$ 固有のバイアス | $\mathcal{N}(\mu_0, \sigma_0^2)$ |
| **population**（$\mu_0$）| 施設全体の平均効果 | 文献値・過去実験から informative prior |
| **variance**（$\tau_j, \sigma_0$）| 変動の広がり | HalfCauchy / HalfNormal |

### 12.2.2 事前分布の provenance

**契約項目**：

```yaml
prior_specification_provenance:
  prior_type: hierarchical                          # or non_hierarchical | improper（後者は fatal）
  prior_family:
    theta: normal
    mu: normal
    tau: half_cauchy
  prior_hyperparameters:                            # 数値で pin（leaf は必ずスカラー型のハイパーパラメータ）
    mu_0_mu: 0.0                                    # μ_0 の平均（超パラメータ）
    mu_0_sigma: 1.0                                 # μ_0 の標準偏差
    tau_scale: 2.5                                  # τ_j (HalfCauchy) のスケール
    sigma_0_scale: 1.0                              # σ_0 (HalfCauchy) のスケール
  prior_source: literature_review | previous_experiment | expert_elicitation | simulation
  prior_source_uri: <string>                        # 文献 DOI や過去実験の manifest
  prior_source_sha256: <string>                     # 事前分布定義ファイルのハッシュ
  elicitation_report_uri: <optional>                # expert_elicitation の場合必須
  prior_predictive_check_uri: <string>              # 事前予測分布が「もっともらしいか」の診断
  prior_predictive_check_sha256: <string>
  approved_by: research_lead                        # 事前分布は Ch4 §4.6.2 承認対象
  approved_at: <timestamp>
  frozen_at: <timestamp>                            # 実験実行前に凍結（post-hoc 変更禁止）

prohibited_actions:
  - modify_prior_after_execution_started            # fatal（HARKing の Bayesian 版）
  - use_improper_prior_without_documented_rationale # fatal（sampling 破綻 or bias 隠蔽）
  - suppress_prior_predictive_check_failure         # fatal（事前 misspecification 隠蔽）
  - swap_prior_family_between_design_and_analysis   # fatal（design と analysis の swap）
```

### 12.2.3 事前予測チェック（Prior Predictive Check）

**契約含意**：エージェントが事前分布を提案したら、**実際にデータを生成してみて「もっともらしい」か診断**しなければなりません（Gelman et al. 2020 "Bayesian workflow"）。これは事前 misspecification を Skill 契約時点で捕捉する gate です。

```python
# 事前予測サンプル：観測モデルからサンプルを引く
with pymc.Model() as model:
    # 事前分布
    theta = pymc.Normal("theta", mu=0.0, sigma=1.0)
    # 観測モデル
    y = pymc.Normal("y", mu=theta * x, sigma=0.1, observed=None)
    prior_pred = pymc.sample_prior_predictive(draws=1000)

# 事前予測分布の diagnostic
# もし y の 99%CI が [-1000, +1000] のように極端に広ければ、事前が wide すぎる
# もし [-0.01, +0.01] のように狭ければ、事前が strong すぎる
```

**Skill 契約**：`prior_predictive_check_uri` に以下 3 項目を必須記録：

1. 事前予測分布の 95%CI
2. 実験域における事前予測の spread（設計内での事前不確かさ）
3. 事前サンプルの物理的妥当性（例：V_OC が [-100, +100] V に散らばっていたら物理外）

---

## 12.3 Bayesian D-optimality / A-optimality

### 12.3.1 D-optimality（頻度主義）の復習

第10章 §10.7 で扱った **頻度主義 D-optimality** の目的関数：

$$
\phi_D^{\text{freq}}(\mathcal{D}) = \det\!\left(X^\top X\right)
$$

これは Fisher 情報行列の行列式で、パラメータ推定の分散楕円体の体積を最小化します。**事前分布は登場しません**。

### 12.3.2 Bayesian D-optimality

Bayesian 版では事前分布の情報を Fisher 情報に **加算**します：

$$
\phi_D^{\text{Bayes}}(\mathcal{D}) = \det\!\left(X^\top X + \Sigma_0^{-1}\right)
$$

ここで $\Sigma_0$ は事前分布 $p(\theta) = \mathcal{N}(\mu_0, \Sigma_0)$ の共分散。**事前分布が既に多くの情報を持っていれば $\Sigma_0^{-1}$ が大きくなり、実験計画は追加情報を得るための配置に集中できます**。

### 12.3.3 Bayesian A-optimality

A-optimality は分散の **和**を最小化——事後共分散のトレース：

$$
\phi_A^{\text{Bayes}}(\mathcal{D}) = \operatorname{tr}\!\left((X^\top X + \Sigma_0^{-1})^{-1}\right)
$$

**D vs A の使い分け**：

| Criterion | 意味 | 使う場面 |
|---|---|---|
| **D-optimality** | 分散楕円体の体積 | 全パラメータを同程度に精密に推定したい |
| **A-optimality** | 分散和 | 個別パラメータの分散を等しく小さくしたい |
| **G-optimality** | worst-case 予測分散 | 予測に使いたい（Ch11 応答曲面向け） |
| **I-optimality** | 積分予測分散 | 実験域全体の予測精度 |

### 12.3.4 事前 → 事後 の情報利得

事前 $p(\theta)$ から事後 $p(\theta \mid y_{1:N})$ への情報利得は **KL divergence** で測定：

$$
\text{IG}(\mathcal{D}) = \mathbb{E}_{y \sim p(y \mid \mathcal{D})}\!\left[\, \text{KL}\!\left(\,p(\theta \mid y, \mathcal{D})\,\|\,p(\theta)\,\right)\right]
$$

これは **期待情報利得（Expected Information Gain, EIG）** で、Bayesian DoE の意思決定変数として最も直接的です。しかし解析的計算は困難で、通常は近似（Laplace / MC / nested MC）を使います。

**Skill 契約**：

```yaml
information_gain_metric: bayesian_d_optimality       # or bayesian_a_optimality | eig_kl | eig_mc
information_gain_threshold: 0.80                     # baseline に対する相対効率
information_gain_baseline:                           # Ch10 N-4 / Ch11 N-7 と parity で 4 フィールド完全展開
  baseline_design_type: uniform_random_over_experimental_region
  baseline_efficiency: 1.00
  efficiency_ratio_achieved: <float>
  n_runs_ratio: <float>

information_gain_estimation:
  method: laplace_approximation | monte_carlo | nested_monte_carlo
  n_mc_samples: 1000                                 # MC の場合
  convergence_diagnostic_uri: <string>               # ESS / MCSE の記録
  seed: 42
```

**注意**：EIG の MC 推定は分散が大きい。**MC 分散を報告せずに EIG 点推定のみを Skill 契約に載せる**のは fatal（`report_eig_point_estimate_without_mc_variance`）。

---

## 12.4 情報利得を Skill の意思決定変数に

### 12.4.1 意思決定変数としての情報利得

頻度主義 DoE では、Skill は「D-efficiency ≥ 0.80」のような閾値で設計を承認します。Bayesian DoE では：

- **意思決定変数**：期待情報利得 EIG（bits / nats）
- **閾値**：事前不確かさに対する相対削減率（例：「事後 entropy が事前の 50% 以下」）
- **fail-close**：EIG が閾値未満なら実験を実行しない（run 予算の無駄を防ぐ）

### 12.4.2 EIG 計算の実装（Laplace 近似）

Laplace 近似では、事後分布を事後モード周辺の正規分布として近似：

$$
p(\theta \mid y) \approx \mathcal{N}(\hat{\theta}_{\text{MAP}}, \hat{H}^{-1})
$$

ここで $\hat{H}(y, \mathcal{D})$ は候補設計 $\mathcal{D}$ と観測 $y$ の下での事後負対数密度の Hessian（**事後精度行列** = prior 精度 $\Sigma_0^{-1}$ + データ寄与）。この Laplace 近似の下で EIG は：

$$
\text{EIG}(\mathcal{D}) \approx \frac{1}{2} \mathbb{E}_{y \sim p(y \mid \mathcal{D})}\!\left[\log\det \hat{H}(y, \mathcal{D})\right] - \frac{1}{2} \log\det \Sigma_0^{-1}
$$

（第 2 項は $y$ に依存しないため期待値の外に出せる。第 1 項は非線形モデルでは $y$ に依存するため MC が必要）。

> [!NOTE]
> **線形 Gaussian の場合、$\hat{H}(y, \mathcal{D}) = X^\top X / \sigma^2 + \Sigma_0^{-1}$ は $y$ に依存しない**ため $\mathbb{E}_y[\cdot]$ は自明に消え、EIG は決定論的に計算できます（§12.9.3 実装参照）。この場合 `eig_estimation_contract.mc_variance_estimate = 0.0` を報告してください。**非線形の場合のみ MC が必要**で、§12.4.3 nested MC / §12.9.3 `eig_nested_mc` を使用します。

`scipy.optimize.minimize` で $\hat{\theta}_{\text{MAP}}$ を求め、`autograd` や `jax.hessian` で Hessian を計算します。

### 12.4.3 EIG の nested MC 推定

Laplace 近似が信頼できない（非正規事後）場合は **nested MC**：

$$
\text{EIG}(\mathcal{D}) \approx \frac{1}{N_{\text{outer}}} \sum_{i=1}^{N_{\text{outer}}} \log p(y_i \mid \theta_i, \mathcal{D}) - \frac{1}{N_{\text{outer}}} \sum_{i=1}^{N_{\text{outer}}} \log \left[ \frac{1}{N_{\text{inner}}} \sum_{j=1}^{N_{\text{inner}}} p(y_i \mid \theta_j', \mathcal{D}) \right]
$$

これは $\theta$ について marginalize した predictive log-density の差。**計算コストが $N_{\text{outer}} \times N_{\text{inner}}$ で爆発**するため、$N_{\text{outer}} \leq 1000$, $N_{\text{inner}} \leq 100$ 程度で運用し、収束診断を必須化します。

**Skill 契約**：

```yaml
eig_estimation_contract:
  method: nested_mc
  n_outer: 1000
  n_inner: 100
  mc_variance_estimate: <float>                      # MC 分散（必須報告）
  mc_variance_ratio: <float>                         # 点推定 EIG に対する相対分散
  convergence_check:
    method: mcse_over_estimate                       # MCSE / estimate 比
    threshold: 0.10                                  # 10% 以下で pass
    achieved: <float>
    pass: true | false
  seed: 42
prohibited_actions:
  - report_eig_point_estimate_without_mc_variance    # fatal
  - reduce_n_inner_after_seeing_eig_result           # fatal（MC 分散を post-hoc に減らして見せる）
  - use_laplace_approximation_when_prior_is_multimodal  # fatal（近似が破綻）
```

---

## 12.5 PyMC 実装 — 事前分布 → 実験計画 → 事後分布 の one-shot ループ

### 12.5.1 一括ワークフロー

Bayesian DoE の one-shot は、以下 5 ステップを **一括で** 実行します（$N$ 回の逐次呼出ではない）：

1. **事前分布定義**：`prior_specification_provenance` を pin
2. **事前予測チェック**：`prior_predictive_check_uri` を出力
3. **候補設計行列の生成**：CCD / BBD / D-optimal-with-prior 等から候補集合
4. **EIG 計算 + 最大化**：各候補について EIG を計算、最大の 1 つを選択
5. **実行 + 事後分布サンプリング**：MCMC / VI で事後 $p(\theta \mid y_{1:N})$ を得る

### 12.5.2 PyMC でのコード骨格

```python
import pymc as pm
import numpy as np
import arviz as az

# --- Step 1: 事前分布定義 ---
def build_model(design_matrix: np.ndarray, y_observed=None):
    with pm.Model() as m:
        # 階層事前（vol-02 §11.3 パターン踏襲）
        mu_0 = pm.Normal("mu_0", mu=0.0, sigma=1.0)
        sigma_0 = pm.HalfCauchy("sigma_0", beta=1.0)
        beta = pm.Normal("beta", mu=mu_0, sigma=sigma_0, shape=design_matrix.shape[1])
        sigma_obs = pm.HalfCauchy("sigma_obs", beta=1.0)
        mu_y = pm.math.dot(design_matrix, beta)
        y = pm.Normal("y", mu=mu_y, sigma=sigma_obs, observed=y_observed)
    return m

# --- Step 2: 事前予測チェック ---
with build_model(candidate_design_1) as m:
    prior_pred = pm.sample_prior_predictive(draws=1000, random_seed=42)
# → prior_pred を診断: 事前予測の CI が物理範囲内か

# --- Step 3-4: EIG 計算（各候補設計について）---
def compute_eig_nested_mc(design_matrix, n_outer=1000, n_inner=100, seed=42):
    rng = np.random.default_rng(seed)
    with build_model(design_matrix) as m:
        prior_samples = pm.sample_prior_predictive(draws=n_outer, random_seed=seed)
    # nested MC の実装（省略、§12.5.3 の関数を呼ぶ）
    eig, mc_var = _nested_mc_eig(prior_samples, design_matrix, n_inner, rng)
    return eig, mc_var

candidates = [candidate_design_1, candidate_design_2, ...]
eigs = [compute_eig_nested_mc(c, seed=42) for c in candidates]
best_idx = int(np.argmax([e[0] for e in eigs]))
best_design = candidates[best_idx]

# --- Step 5: 実行後、事後分布サンプリング ---
# y_observed = 実験実行後のデータ
with build_model(best_design, y_observed=y_observed) as m:
    trace = pm.sample(
        draws=2000, tune=1000, chains=4, cores=4,
        random_seed=42, return_inferencedata=True,
    )
az.summary(trace)
```

### 12.5.3 決定論的 hash の pin

第10章 §10.9.1 / 第11章 §11.9.1 と同じく、事前分布・設計行列・事後 trace すべてに **decisive hash** を pin：

```python
import hashlib, json

# 事前分布の hash（数値パラメータ + schema）
prior_spec = {
    "prior_type": "hierarchical",
    "families": {"beta": "normal", "mu_0": "normal", "sigma_0": "half_cauchy"},
    "hyperparameters": {"mu_0_mu": 0.0, "mu_0_sigma": 1.0, "sigma_0_scale": 1.0},
    "dtype": "float64",
}
prior_sha256 = hashlib.sha256(json.dumps(prior_spec, sort_keys=True).encode("utf-8")).hexdigest()

# 事後 trace の hash（chain × draw × parameter tensor）
posterior_array = trace.posterior["beta"].values.astype(np.float64)  # (chains, draws, k)
schema_header = json.dumps({
    "variable": "beta",
    "shape": list(posterior_array.shape),
    "dtype": "float64",
    "byte_order": "little",
    "n_chains": posterior_array.shape[0],
    "n_draws": posterior_array.shape[1],
    "seed": 42,
}, sort_keys=True).encode("utf-8")
posterior_sha256 = hashlib.sha256(schema_header + b"|" + posterior_array.tobytes()).hexdigest()
```

---

## 12.6 第11章 T3（GP surrogate）との接続 — 事前 GP + 事後 GP

### 12.6.1 GP における事前と事後

第11章 §11.5 で扱った GP は、実は **すでに Bayesian**——kernel と hyperparameter が事前分布を定義しており、`predict_variances` は事後予測分散を返しています。**Ch12 の Bayesian DoE と Ch11 T3 GP は連続的**です：

| 章 | 事前分布 | 事後分布 | 設計選択の目的関数 |
|---|---|---|---|
| Ch11 T3 | GP kernel（$k(x, x')$、θ は MLE で fit） | GP posterior（$\hat{y}$, $\hat{\sigma}^2$） | 予測（optimum 探索） |
| **Ch12** | **GP kernel + θ の Bayesian prior** | **fully Bayesian GP posterior** | **情報利得（EIG）** |

Ch11 T3 は θ を **fit（MAP または MLE）** しますが、Ch12 は θ を **integrate out**（marginal likelihood サンプリング）します。

### 12.6.2 Bayesian GP の Skill 契約

```yaml
skill_type: bayesian_doe_gp_skill
authorization_level: human_required_l4_gate          # Ch11 T3 継承（Ch6 §6.7.1 L4）

surrogate_model_provenance:                          # Ch11 §11.5.3 継承
  model_family: gaussian_process_bayesian            # ← Ch11: gaussian_process_kriging
  library: pymc                                      # または gpflow | numpyro
  library_version: 5.10.0
  kernel:
    name: squared_exponential
    theta_prior:                                     # ← Ch11 は theta_optimizer、Ch12 は事前分布
      length_scale: half_normal(sigma=1.0)
      output_variance: half_normal(sigma=1.0)
      noise_variance: half_normal(sigma=0.1)
  theta_marginalization: mcmc                        # or vi | laplace
  posterior_diagnostic_uri: <string>                 # R̂, ESS, divergences
  posterior_diagnostic_sha256: <string>

# --- Ch11 T3 との比較（cross-reference）---
inherits_from_chapter_11_section: 11.5.3
distinguishes_from_chapter_11_by:
  - theta_marginalization_instead_of_fitting
  - eig_based_design_selection_instead_of_optimum_seeking
  - prior_specification_provenance_required
```

### 12.6.3 Ch11 T4 昇格としての Bayesian GP

**用語規約**（Ch11 §11.1.1 の T1/T2/T3 命名を拡張）：

- Ch11 §11.1.1: T1（linear）→ T2（GBM）→ T3（GP MLE）
- Ch12: **T3 → T4（Bayesian GP、θ marginalized）** は **Ch6 §6.7.1 L4 event**

**契約**：T3→T4 昇格は Ch11 §11.4.2 の proposal / approval Skill パターンをそのまま踏襲。追加の approval エビデンス：

- `prior_specification_provenance`（事前分布の justification）
- `prior_predictive_check_uri`（事前予測診断）
- `eig_estimation_contract`（EIG 推定の MC diagnostics）
- `estimator_diff_report_uri` + `_sha256`（**Ch6 §6.7.1 L4 mandatory**）
  - 内容: T3 MLE-GP と T4 marginalized-GP の点推定・95% CI・予測分散を **同一設計格子で比較**した差分レポート
  - フィールド: `point_estimate_delta`, `ci_width_delta`, `predictive_variance_delta`, `worst_case_grid_point`, `grid_sha256`

---

## 12.7 counterfactual_scope_gate と Bayesian posterior support

### 12.7.1 Bayesian 版の 4-check

第4章 §4.5.2 / 第11章 §11.7.2 で導入した `counterfactual_scope_gate` の 4-check を Bayesian DoE に operational 化：

| Check | 頻度主義版（Ch11） | Bayesian 版（Ch12） |
|---|---|---|
| **mahalanobis** | 訓練点分布に対する Mahalanobis 距離 | **事後予測分布**に対する Mahalanobis 距離 |
| **variance** | GP 予測分散（MLE hyperparameter） | **事後予測分散**（θ marginalize 済み、hyperparameter uncertainty 込み） |
| **knn density** | 訓練点近傍密度 | **事前分布 support** の density（事前不確かさが強い領域は fail） |
| **support envelope** | 凸包 | **事後 credible region**（例：95% posterior mass の hull） |

**Bayesian 版が有利な点**：hyperparameter の uncertainty を予測分散に組み込むため、Ch11 MLE-GP よりも **保守的（分散が広め）**——外挿検知が敏感になります。

### 12.7.2 事前-事後 alignment チェック

`prior_data_alignment_check` は Bayesian DoE 特有：

> [!NOTE]
> **命名規約（N-3）**：`prior_data_alignment` は Ch9 §9.7.1 の `declared_required_tests` canonical enum 名。ここで扱う `prior_data_alignment_check` はその **operational check function 名**（YAML block と関数名に接尾辞 `_check` を付ける）。両者は 1:1 対応する。

```yaml
prior_data_alignment_check:
  method: prior_predictive_p_value                   # 事前予測と実データの整合性 p-value
  threshold: 0.05                                    # p < 0.05 は事前 misspecification 疑い
  fail_action: flag_prior_misspecification_and_notify_research_lead
  detection_rule: |
    実データの分布統計量が、事前予測分布の 5-95 percentile 帯から外れる run が
    総 run 数の 20% を超える場合、事前 misspecification と判定
  fallback_message_template: |
    Prior-data alignment failed for design {design_id}.
    Prior predictive p-value: {p_value}.
    Runs outside prior 90% CI: {n_outside} / {n_total}.
    Possible causes: (a) prior mean/scale misspecified, (b) observation noise underestimated,
    (c) new physical regime discovered.
    Manual review by {approver} required before publishing results.
```

**注意**：事前-事後 alignment が pass しなかったからといって、**事前分布を post-hoc に変更するのは fatal**——alignment failure は「事前分布が事後を過度に引っ張っている」ではなく「事前分布 vs 観測分布」の乖離であって、fix は **新実験の計画**であって「事前を data に合わせる」ことではない。

### 12.7.3 counterfactual_scope_gate 契約

```yaml
counterfactual_scope_gate:
  mahalanobis_check:
    method: posterior_predictive_mahalanobis
    training_points_uri: <string>
    threshold: 3.0
    cluster_conditional: true
  variance_check:
    method: posterior_predictive_variance            # θ marginalized
    threshold: <float>
  knn_density_check:
    method: prior_support_knn
    k: 5
    knn_min: 2
  support_envelope_check:
    method: posterior_credible_region
    credible_level: 0.95
    envelope_report_uri: <string>
    strict: true

  # --- Ch11 継承の threshold_calibration ---
  threshold_calibration:
    method: prior_predictive_conformal              # Bayesian 特有: 事前予測で conformal calibration
    calibration_evidence_uri: <string>
    calibration_approved_at: <timestamp>

  # --- Ch12 特有：prior-data alignment ---
  prior_data_alignment_check:
    method: prior_predictive_p_value
    threshold: 0.05
    p_value: <float>                                # 実データ観測後に埋まる
    pass: true | false

  aggregate_policy:
    pass_requires: all_five_pass                    # ← Ch11 は 4-check、Ch12 は 5-check
    conditional_pass_output: non_actionable_diagnostic_only
    fail_action: fail_close_and_reject_posterior_action
  gate_status: pass | conditional_pass | fail       # Ch11 §11.7.2 3-state 継承
  fallback: human_review
  fallback_approver: research_lead
  fallback_message_template: |                       # Ch4 §4.5.2 canonical 必須（Ch11 §11.7.2 と parity）
    Bayesian DoE Skill が posterior action を x* = {x_star}（design {design_id}）で提案しました。
    設計領域: {design_hull_summary}
    posterior 95% credible region: {credible_region_summary}
    failing_checks: {failing_checks}
    prior-data alignment p-value: {alignment_p_value}（閾値 {alignment_threshold}）
    posterior predictive variance at x*: {posterior_var}
    posterior_trace_sha256: {posterior_trace_sha256}
    prior_specification_sha256: {prior_specification_sha256}
    選択肢:
      (a) posterior action を却下する
      (b) 設計領域を拡張するため追加実験を発注する
      (c) prior 再エリシテーションのため {approver} にエスカレートする
    ⚠️ 事後の prior 変更は fatal（§12.7.2 参照）。alignment_check fail 後の prior 変更は禁止。

prohibited_actions:
  - modify_prior_after_alignment_check_failure       # fatal（§12.7.2 の HARKing の Bayesian 版）
  - report_posterior_action_without_alignment_pass  # fatal
  - suppress_alignment_p_value_in_final_report       # fatal
```

---

## 12.8 Bayesian DoE Skill 全体契約テンプレート

Ch10 §10.8 / Ch11 §11.8 のテンプレートに Bayesian 固有項目を追加：

```yaml
skill:
  name: bayesian_doe_perovskite_voc_skill
  version: 0.1.0
  purpose: |
    ペロブスカイト太陽電池の V_OC について、過去 3 バッチの事後分布を事前として、
    情報利得最大となる 20-run 実験計画を one-shot で生成する。

  # === ① 目的 ===
  estimand_type: ate
  mediation_role: not_applicable
  causal_question_type: type_e_doe

  # === ② 入力条件（L1: 識別レイヤ）===
  identification_strategy: randomized_experiment
  dag_of_record_uri: "artifact://dags/bayesian_doe_voc.dot"
  dag_of_record_sha256: <sha256>
  confounders_declared: []
  mediators_declared: []
  colliders_declared: []
  blocking_factors_declared:
    - instrument_id
    - operator_id

  # === ②' データ来歴 ===
  data_lineage:
    prior_dataset_uri: <string>                     # 事前分布を作るのに使った過去バッチ
    prior_dataset_sha256: <string>
    arim_facility_id: <string>

  # === ②'' 識別仮定（L2, Ch4 §4.4.1 canonical）===
  positivity_by_stratum:
    status: checked
    strata:
      - key: prior_support_bin
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

  # --- Bayesian DoE 特有：追加仮定 ---
  prior_specification_correctness_declared:
    status: assessed
    prior_source: previous_experiment
    prior_predictive_check_passed: true
  prior_data_alignment_declared:
    status: checked_after_execution                 # 事後にしか check できない
    alignment_p_value: <float>

  # === ③ 出力形式 ===
  outputs:
    design_matrix_uri: <string>
    design_matrix_sha256: <string>
    posterior_trace_uri: <string>
    posterior_trace_sha256: <string>
    posterior_summary_uri: <string>                 # arviz.summary
    eig_report_uri: <string>                        # EIG 推定と MC diagnostics
    prior_data_alignment_report_uri: <string>
    test_results: dict
    sensitivity_analysis:                           # Ch10 N-4 / Ch11 B-5 と parity
      method: e_value
      effect_direction: harmful | protective
      effect_scale: transformed_continuous_with_smd_to_rr_conversion
      ci_bound_closest_to_null: lower_bound
      smd_to_rr_conversion: vanderweele_2020
      threshold:
        minimum_e_value: 1.5
      provenance_field: sensitivity_report_uri

  # === ④ 成功条件 ===
  success_criteria:
    identification_validity: required
    refutation_pass: required
    positivity_check: required
    external_validity: required
    doe_efficiency: required
    randomization_integrity: required
    prior_predictive_check_pass: required           # Ch12 特有
    prior_data_alignment_pass: required             # Ch12 特有（事後 gate）
    counterfactual_scope_gate_pass: required
    posterior_convergence_pass: required            # Ch12 特有（R̂, ESS）

  # --- refutation gate (Ch9) ---
  declared_required_tests:
    - random_common_cause
    - data_subset_validation
    - scope_gate_reverification
    - ite_prediction_coverage_refutation
    - e_value                                        # Ch11 B-5 継承
    - prior_predictive_check                         # Ch9 §9.7.1 canonical enum に back-register 済み（Ch12 §12.2.3）
    - prior_data_alignment                           # Ch9 §9.7.1 canonical enum に back-register 済み（Ch12 §12.7.2）
  applicability_manifest_uri: <string>
  applicability_manifest_sha256: <string>
  preregistration_manifest_uri: <string>
  preregistration_manifest_sha256: <string>

  # --- counterfactual scope gate (§12.7)、Ch11 4-check + prior_data_alignment ---
  counterfactual_scope_gate:
    mahalanobis_check:
      method: posterior_predictive_mahalanobis
      threshold: 3.0
    variance_check:
      method: posterior_predictive_variance
      threshold: <float>
    knn_density_check:
      method: prior_support_knn
      k: 5
      knn_min: 2
    support_envelope_check:
      method: posterior_credible_region
      credible_level: 0.95
      strict: true
    threshold_calibration:
      method: prior_predictive_conformal
      calibration_evidence_uri: <string>
      calibration_approved_at: <timestamp>
    prior_data_alignment_check:                      # Ch12 特有
      method: prior_predictive_p_value
      threshold: 0.05
    aggregate_policy:
      pass_requires: all_five_pass
      conditional_pass_output: non_actionable_diagnostic_only
      fail_action: fail_close_and_reject_posterior_action
    gate_status: pass | conditional_pass | fail
    fallback: human_review
    fallback_approver: research_lead
    fallback_message_template: |                     # Ch4 §4.5.2 canonical
      Bayesian DoE Skill が posterior action を x* = {x_star}（design {design_id}）で提案しました。
      設計領域: {design_hull_summary}
      posterior 95% credible region: {credible_region_summary}
      failing_checks: {failing_checks}
      prior-data alignment p-value: {alignment_p_value}（閾値 {alignment_threshold}）
      posterior predictive variance at x*: {posterior_var}
      posterior_trace_sha256: {posterior_trace_sha256}
      prior_specification_sha256: {prior_specification_sha256}
      選択肢:
        (a) posterior action を却下する
        (b) 設計領域を拡張するため追加実験を発注する
        (c) prior 再エリシテーションのため {approver} にエスカレートする
      ⚠️ 事後の prior 変更は fatal（§12.7.2 参照）。

  # === ⑤ 禁止事項 ===
  prohibited_actions:
    # DoE 一般（Ch10 継承）
    - modify_factors_or_levels_after_design_freeze
    - reduce_n_runs_after_partial_execution
    - randomization_seed_override
    - ignore_blocking_factor_in_analysis
    - assignment_log_row_reorder_after_execution
    - execution_record_missing_design_matrix_sha256
    - assignment_log_missing_header_record            # Ch10 B-5
    - silent_deletion_of_failed_runs_without_design_matrix_recompute  # Ch10 B-5
    - swap_mixed_model_specification_between_design_and_analysis      # Ch10 N-5
    - random_effects_missing_declared_blocking_factor                 # Ch10 N-5
    # 応答曲面継承（Ch11）
    - report_optimum_outside_convex_hull_without_scope_gate_pass
    - modify_scope_gate_threshold_without_calibration_evidence        # Ch11 B-3
    - tier_naming_l1_l2_l3_collision_with_ch6_gate_levels             # Ch11 §11.1.1
    # Bayesian DoE 特有（Ch12）
    - modify_prior_after_execution_started            # fatal（HARKing の Bayesian 版）
    - use_improper_prior_without_documented_rationale
    - suppress_prior_predictive_check_failure
    - swap_prior_family_between_design_and_analysis
    - modify_prior_after_alignment_check_failure      # §12.7.2
    - report_posterior_action_without_alignment_pass  # §12.7.2
    - report_eig_point_estimate_without_mc_variance   # §12.4.3
    - reduce_n_inner_after_seeing_eig_result          # §12.4.3
    - use_laplace_approximation_when_prior_is_multimodal  # §12.4.3
    - report_posterior_summary_without_convergence_diagnostic  # R̂, ESS 未添付
    - preregistration_manifest_post_hoc_modification

  # === ⑥ 再現性条件 ===
  library_stack:
    design_generation: pyDOE2==1.3.0
    bayesian_inference: pymc==5.10.0                 # or numpyro | gpflow
    diagnostic: arviz==0.16.0
    surrogate: smt==2.4.0                            # Ch11 T3 との接続
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
        - design_parameter_change
        - prior_family_change                        # Ch12 特有
        - prior_hyperparameter_change                # Ch12 特有
      approver: research_lead
    intervention_execution_authorization:
      required_for:
        - physical_experiment_execution
      approver: pi_and_facility_manager
    approver_independence:
      conflict_policy: independent_reviewer_required_if_approver_is_data_producer
      fallback_approver: facility_causal_review_board
      facility_scope_escalation:
        applies_to:
          - approved_facility_design_template
          - approved_facility_response_surface_baseline
          - approved_facility_prior_specification    # Ch4 §4.6.2 enum に back-register 済み（Ch12 §12.8）
                                                       # 定義: 特定材料クラス向けの Bayesian DoE 事前分布テンプレート
                                                       # ≥N 回の approved 実験で prior calibration が safe と検証された後に promotion
        default_approver: facility_causal_review_board

  # === ⑧ Estimator provenance reference ===
  estimator_provenance_reference:
    estimator_contract_sha256: <string>
    dag_of_record_sha256: <string>
    design_matrix_sha256: <string>
    prior_specification_sha256: <string>              # Ch12 特有
    posterior_trace_sha256: <string>                  # Ch12 特有

  # === ⑨ DoE 拡張（Ch10 §10.8 / Ch11 §11.8 と feature parity）===
  experimental_design_provenance:
    design_type: bayesian_d_optimal                   # or bayesian_a_optimal | eig_maximizing_ccd
    factors: [...]                                    # Ch4 §4.9 / Ch10 §10.2.2: role: primary|nuisance|blocking
    factors_frozen_at: <timestamp>
    n_runs_total: 20
    randomization_seed_provenance: {...}
    blocking_factors_declared: [instrument_id, operator_id]
    design_matrix_uri: <string>
    design_matrix_sha256: <string>
    assignment_log_uri: <string>
    assignment_log_sha256: <string>                   # 4-stage detection per Ch10 §10.5.3
    information_gain_metric: bayesian_d_optimality    # § 12.3.2
    information_gain_threshold: 0.80
    information_gain_baseline:                        # Ch10 N-4 / Ch11 N-7 parity 4 フィールド
      baseline_design_type: uniform_random_over_experimental_region
      baseline_efficiency: 1.00
      efficiency_ratio_achieved: <float>
      n_runs_ratio: <float>
    # blocking 宣言に伴う mixed_model_specification（Ch10 N-5）
    mixed_model_specification_uri: <string>
    mixed_model_specification_sha256: <string>
    mixed_model_specification_schema: pymc_hierarchical_model  # ← Ch10 は statsmodels_mixedlm
    mixed_model_random_effects_declared: [instrument_id, operator_id]
    mixed_model_fixed_effects_declared: [temperature, pressure, pretreatment_time]

  # === ⑩ Bayesian DoE 特有：prior_specification_provenance ===
  prior_specification_provenance:
    prior_type: hierarchical
    prior_family:
      beta: normal
      mu_0: normal
      sigma_0: half_cauchy
    prior_hyperparameters:
      mu_0_mu: 0.0
      mu_0_sigma: 1.0
      sigma_0_scale: 1.0
    prior_source: previous_experiment
    prior_source_uri: <string>                        # 過去バッチの manifest
    prior_source_sha256: <string>
    elicitation_report_uri: <optional>
    prior_predictive_check_uri: <string>
    prior_predictive_check_sha256: <string>
    approved_by: research_lead
    approved_at: <timestamp>
    frozen_at: <timestamp>

  # === ⑪ Bayesian DoE 特有：eig_estimation_contract ===
  eig_estimation_contract:
    method: nested_mc                                 # or laplace_approximation | monte_carlo
    n_outer: 1000
    n_inner: 100
    mc_variance_estimate: <float>
    mc_variance_ratio: <float>                        # < 0.10 で pass
    convergence_check:
      method: mcse_over_estimate
      threshold: 0.10
      achieved: <float>
      pass: true | false
    seed: 42

  # === ⑫ Bayesian DoE 特有：posterior_convergence_provenance ===
  posterior_convergence_provenance:
    sampler: nuts                                     # or hmc | vi | smc
    n_chains: 4
    n_draws: 2000
    n_tune: 1000
    r_hat_max: <float>                                # < 1.01 で pass
    ess_bulk_min: <float>                             # > 400 で pass
    ess_tail_min: <float>                             # > 400 で pass
    divergences: <int>                                # 0 が理想
    convergence_pass: true | false
    diagnostic_report_uri: <string>                   # arviz.summary の詳細
    diagnostic_report_sha256: <string>
```

---

## 12.9 実装スニペット — PyMC / numpy による one-shot ループ

### 12.9.1 階層事前モデルの構築

```python
import pymc as pm
import numpy as np
import arviz as az

def build_hierarchical_model(design_matrix: np.ndarray, y_observed=None,
                              n_instruments: int = 2, instrument_id=None,
                              seed: int = 42):
    """
    Hierarchical Bayesian DoE model:
      beta ~ Normal(mu_0, sigma_0)  # per-factor effect
      y = X @ beta + instrument_effect + noise
    """
    with pm.Model() as m:
        # Population-level
        mu_0 = pm.Normal("mu_0", mu=0.0, sigma=1.0)
        sigma_0 = pm.HalfCauchy("sigma_0", beta=1.0)

        # Factor effects
        beta = pm.Normal("beta", mu=mu_0, sigma=sigma_0,
                         shape=design_matrix.shape[1])

        # Instrument random effect (blocking)
        tau_instrument = pm.HalfCauchy("tau_instrument", beta=0.5)
        instrument_effect = pm.Normal("instrument_effect", mu=0.0,
                                       sigma=tau_instrument,
                                       shape=n_instruments)

        # Observation noise
        sigma_obs = pm.HalfCauchy("sigma_obs", beta=1.0)

        # Likelihood
        mu_y = pm.math.dot(design_matrix, beta)
        if instrument_id is not None:
            mu_y = mu_y + instrument_effect[instrument_id]

        y = pm.Normal("y", mu=mu_y, sigma=sigma_obs, observed=y_observed)
    return m
```

### 12.9.2 事前予測チェック

```python
def prior_predictive_check(model, physical_range: tuple[float, float],
                            seed: int = 42, n_samples: int = 1000):
    """
    Sample prior predictive and diagnose plausibility.
    Returns (pass: bool, diagnostic: dict).

    `physical_range` は **必須引数**（object domain の物理的許容範囲）。
    例: V_OC [0.5, 1.5] V、film thickness [10, 1000] nm。
    デフォルト値を許すと prior misspecification を silent に見逃すため意図的に必須化。
    """
    if physical_range is None or len(physical_range) != 2:
        raise ValueError("physical_range must be a (low, high) tuple; "
                          "no default is allowed (see §12.9.2 rationale).")
    with model:
        prior_pred = pm.sample_prior_predictive(draws=n_samples, random_seed=seed)
    y_prior = prior_pred.prior_predictive["y"].values.flatten()

    p_lower, p_upper = np.percentile(y_prior, [2.5, 97.5])
    n_in_range = int(np.sum((y_prior >= physical_range[0]) &
                             (y_prior <= physical_range[1])))
    fraction_in_range = n_in_range / len(y_prior)

    diagnostic = {
        "prior_predictive_95ci": (float(p_lower), float(p_upper)),
        "fraction_within_physical_range": fraction_in_range,
        "physical_range": physical_range,
        "n_samples": n_samples,
        "seed": seed,
    }
    # Pass if at least 80% of prior samples are physically plausible
    passed = fraction_in_range >= 0.80
    return passed, diagnostic
```

### 12.9.3 EIG 推定（Laplace 近似）

```python
from scipy.optimize import minimize
import numpy as np

def eig_laplace_approximation(design_matrix: np.ndarray,
                               prior_precision: np.ndarray,
                               noise_variance: float,
                               seed: int = 42) -> tuple[float, float]:
    """
    Compute EIG under Laplace approximation for linear Gaussian model.

    Returns (eig_estimate, mc_variance).
    For linear-Gaussian, EIG is deterministic (does not depend on y),
    so mc_variance = 0.0 by construction. For non-linear models, use
    `eig_nested_mc` (below) which returns a proper MC variance.
    """
    _ = seed  # reserved for API parity with eig_nested_mc
    X = design_matrix
    k = X.shape[1]

    # For linear Gaussian: posterior precision = X.T @ X / sigma^2 + prior_precision
    posterior_precision = X.T @ X / noise_variance + prior_precision

    # EIG = 0.5 * (log|posterior_precision| - log|prior_precision|)
    # (does not depend on y for linear-Gaussian; MC needed for nonlinear)
    sign_post, logdet_post = np.linalg.slogdet(posterior_precision)
    sign_prior, logdet_prior = np.linalg.slogdet(prior_precision)
    if sign_post <= 0 or sign_prior <= 0:
        raise ValueError("Non-positive-definite precision detected")

    eig = 0.5 * (logdet_post - logdet_prior)

    # Linear-Gaussian の EIG は決定論的なので mc_variance = 0.0
    # eig_estimation_contract.method = "laplace" のとき、この値をそのまま
    # mc_variance_estimate に格納すること（bootstrap の設計行分散を混同しない）
    mc_variance = 0.0

    return float(eig), mc_variance


def eig_nested_mc(design_matrix: np.ndarray,
                   prior_sampler,                     # callable: rng -> theta サンプル
                   likelihood_sampler,                # callable: (theta, X, rng) -> y サンプル
                   log_marginal_likelihood,           # callable: (y, X) -> log p(y|X)
                   log_conditional_likelihood,        # callable: (y, theta, X) -> log p(y|theta,X)
                   n_outer: int = 200,
                   n_inner: int = 50,
                   seed: int = 42) -> tuple[float, float]:
    """
    Non-linear モデル用 nested MC EIG 推定子（stub — vol-05 で完成）。

    EIG = E_theta E_y[log p(y|theta,X) - log p(y|X)]
    - n_outer: theta サンプル数（外側 MC）
    - n_inner: log p(y|X) 推定用の theta サンプル数（内側 MC）
    Returns (eig_estimate, mc_variance) where mc_variance is the sample
    variance of per-outer-sample EIG contributions divided by n_outer
    (i.e., a proper Monte Carlo standard error of the mean squared).
    """
    rng = np.random.default_rng(seed)
    contributions = []
    for _ in range(n_outer):
        theta = prior_sampler(rng)
        y = likelihood_sampler(theta, design_matrix, rng)
        log_p_y_theta = log_conditional_likelihood(y, theta, design_matrix)
        log_p_y = log_marginal_likelihood(y, design_matrix)  # 内側で n_inner 使用
        contributions.append(log_p_y_theta - log_p_y)
    eig = float(np.mean(contributions))
    mc_variance = float(np.var(contributions, ddof=1) / n_outer)  # MCSE^2
    return eig, mc_variance
```

### 12.9.4 事後サンプリング + 収束診断

```python
def sample_posterior_with_diagnostics(model, seed=42,
                                        draws=2000, tune=1000, chains=4):
    """
    Sample posterior with NUTS and check convergence.
    """
    with model:
        trace = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=chains,
            random_seed=seed, return_inferencedata=True,
            target_accept=0.95,  # for hierarchical stability
        )

    summary = az.summary(trace, hdi_prob=0.95)
    r_hat_max = float(summary["r_hat"].max())
    ess_bulk_min = float(summary["ess_bulk"].min())
    ess_tail_min = float(summary["ess_tail"].min())
    divergences = int(trace.sample_stats["diverging"].sum())

    convergence_pass = (
        r_hat_max < 1.01
        and ess_bulk_min > 400
        and ess_tail_min > 400
        and divergences == 0
    )
    diagnostic = {
        "r_hat_max": r_hat_max,
        "ess_bulk_min": ess_bulk_min,
        "ess_tail_min": ess_tail_min,
        "divergences": divergences,
        "convergence_pass": convergence_pass,
        "sampler": "nuts",
        "n_chains": chains,
        "n_draws": draws,
        "seed": seed,
    }
    return trace, diagnostic
```

### 12.9.5 決定論的 hash の実装

```python
import hashlib, json

def compute_prior_specification_sha256(prior_spec: dict) -> str:
    """
    Deterministic hash of prior specification.
    prior_spec must contain: prior_type, prior_family, prior_hyperparameters, dtype.
    """
    canonical = json.dumps(prior_spec, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def compute_posterior_trace_sha256(trace, variable: str = "beta",
                                     seed: int = 42) -> str:
    """
    Deterministic hash of posterior trace (chains x draws x parameter).
    """
    arr = trace.posterior[variable].values.astype(np.float64)
    schema = {
        "variable": variable,
        "shape": list(arr.shape),
        "dtype": "float64",
        "byte_order": "little",
        "n_chains": int(arr.shape[0]),
        "n_draws": int(arr.shape[1]),
        "seed": seed,
    }
    header = json.dumps(schema, sort_keys=True).encode("utf-8")
    return hashlib.sha256(header + b"|" + arr.tobytes()).hexdigest()
```

### 12.9.6 事前-事後 alignment チェック

```python
def prior_data_alignment_check(model, y_observed: np.ndarray,
                                 seed: int = 42, n_samples: int = 1000
                                 ) -> tuple[bool, dict]:
    """
    Compute prior predictive p-value: fraction of prior samples with
    test statistic more extreme than observed.
    """
    with model:
        prior_pred = pm.sample_prior_predictive(draws=n_samples, random_seed=seed)
    y_prior = prior_pred.prior_predictive["y"].values  # (n_samples, n_obs)

    # Test statistic: mean (can be extended to variance, quantiles, etc.)
    obs_mean = float(np.mean(y_observed))
    prior_means = np.mean(y_prior, axis=-1).flatten()

    # Two-sided p-value with MC floor (rule of three-style)
    # 全事前サンプルより obs_mean が極端な場合、p_lower or p_upper = 0 となるが、
    # これを floor = 1/n_samples に置き換え、「MC 限界での fail」と「真に極端」を区別する。
    p_lower = float(np.mean(prior_means <= obs_mean))
    p_upper = float(np.mean(prior_means >= obs_mean))
    p_raw = 2 * min(p_lower, p_upper)
    mc_floor = 1.0 / n_samples
    p_value = max(p_raw, mc_floor)
    p_value_is_at_mc_floor = bool(p_raw < mc_floor)

    passed = p_value >= 0.05
    diagnostic = {
        "prior_predictive_p_value": p_value,
        "p_value_raw": p_raw,
        "p_value_is_at_mc_floor": p_value_is_at_mc_floor,
        "mc_floor": mc_floor,
        "obs_mean": obs_mean,
        "prior_mean_ci_95": (
            float(np.percentile(prior_means, 2.5)),
            float(np.percentile(prior_means, 97.5)),
        ),
        "pass": passed,
        "threshold": 0.05,
        "n_samples": n_samples,
        "seed": seed,
    }
    return passed, diagnostic
```

---

## 12.10 章末チェックリスト

- [ ] **§12.1 Bayesian DoE の位置付け**を、頻度主義 DoE との識別仮定の追加（`prior_specification_correctness_declared`）で説明できる
- [ ] **§12.1.1 one-shot と逐次の分離**を、契約複雑度・Human-in-the-loop 粒度・学習曲線の 3 観点で正当化できる
- [ ] **§12.2 事前分布 3 層構造**（individual / group / population）を vol-02 §11.3 の階層モデルの直系として書ける
- [ ] **§12.2.2 prior_specification_provenance** の 11 フィールド全てを埋められる
- [ ] **§12.2.3 事前予測チェック** を Skill 契約に組み込み、3 項目（95%CI / 実験域内 spread / 物理妥当性）を必須出力にできる
- [ ] **§12.3 Bayesian D/A-optimality** の目的関数を頻度主義版との違い（$\Sigma_0^{-1}$ の加算）を示して書ける
- [ ] **§12.3.4 EIG（KL divergence 期待）** の定義と、Laplace / MC / nested MC の使い分けを説明できる
- [ ] **§12.4.3 eig_estimation_contract** で MC 分散と MCSE/estimate < 0.10 を必須にできる
- [ ] **§12.5 PyMC one-shot ループ** を 5 ステップ（prior 定義 → prior pred → 候補生成 → EIG 最大化 → 事後）で書ける
- [ ] **§12.6 Ch11 T3 との接続**を、θ の fit（Ch11）vs marginalize（Ch12）で対比できる
- [ ] **§12.7 counterfactual_scope_gate 5-check**（Ch11 の 4-check + prior_data_alignment）と 3-state `gate_status` を実装できる
- [ ] **§12.7.2 事前-事後 alignment failure 時**に「事前分布を post-hoc 変更するのは fatal」であることを説明できる
- [ ] **§12.8 Bayesian DoE Skill 契約テンプレート**を、自研究テーマで完全に埋められる
- [ ] **§12.9 PyMC 実装スニペット**（build_hierarchical_model / prior_predictive_check / eig_laplace / sample_posterior / hash / alignment_check）を実装できる
- [ ] **§12.11 橋渡し節**で、one-shot 契約がなぜ vol-05 逐次契約と切り離せるかを説明できる

---

## 12.11 橋渡し節：逐次 Bayesian DoE への発展（vol-05 完全委譲）

本章の one-shot Bayesian DoE と、vol-05 で扱う **逐次 Bayesian DoE（acquisition function ベースの BO）** の違いを、Skill 契約の観点から整理します。

### 12.11.1 何が同じで何が違うか

| 側面 | one-shot（本章） | 逐次（vol-05） |
|---|---|---|
| **事前分布** | 一度定義、実行前に frozen | 各 iteration で事後を次の事前として更新 |
| **設計選択** | 候補集合 $\{ \mathcal{D}_c \}$ から EIG 最大の 1 つ | 各 step で 1 点 $x_{n+1}$ を acquisition 関数で選ぶ |
| **停止条件** | 予算 $N$ で自然停止 | **stopping rule** が独立の契約項目（vol-05 §5-6） |
| **Human-in-the-loop** | 1 回の approval で完結 | run ごと承認 vs 一括承認の選択（vol-05 §7） |
| **HARKing 面** | 事前分布 post-hoc 変更 | + acquisition function post-hoc 切替 + stopping rule HARKing |

### 12.11.2 vol-05 に完全委譲される項目

以下は本章では **意図的にスコープ外**とし、vol-05 で完全に扱います（章番号は vol-05 outline 執筆時に確定、現状は暫定）：

- `sequential_experiment_stop_condition`（停止条件の provenance）→ vol-05 §5 想定
- `acquisition_function_provenance`（EI / UCB / TS / KG 等の pin）→ vol-05 §6 想定
- `sequential_authorization_pattern`（run-by-run vs batch approval）→ vol-05 §7 想定
- `optimization_regret_bound`（逐次 BO の理論保証）→ vol-05 §8 想定
- `restart_policy`（stuck 状態からの脱出、fatal region の学習）→ vol-05 §7 想定

> [!NOTE]
> vol-05 章構成は vol-04 完成後に確定するため、上記の章番号は執筆時に更新されます。cross-reference の SHA pin は vol-05 完成時に doi/manifest ハッシュへ差し替えます。

本章の Skill 契約は、vol-05 のこれらの項目が **追加で必要**になるだけで、本章の項目は **すべて継承**されます——つまり **one-shot 契約は逐次契約の proper subset**。

### 12.11.3 いつ one-shot、いつ逐次を選ぶか

| 状況 | 推奨 |
|---|---|
| **run 単価が高い**（1 run = 数百万円、DFT 数週間） | 逐次（vol-05）——1 run ごとに情報最大化 |
| **run 予算が事前に決まっている**（施設予算・卒論期限） | **one-shot（本章）** |
| **事前知識が豊富**（同材料系で 100+ runs） | **one-shot（本章）**——事前分布を informative に |
| **事前知識が乏しい**（新規材料系） | 逐次（vol-05）——探索と活用のバランスを学習 |
| **Human approval のコストが高い** | **one-shot（本章）**——1 回で完結 |
| **並列実行可能性が高い**（複数装置、batch size $N$ 固定） | **one-shot（本章）**——batch size = $N$ の "parallel one-shot" は 1 意思決定・1 承認で完結 |

> [!TIP]
> **"parallel one-shot" と "batched sequential" の切り分け**：
> - Batch size を **事前に固定**し 1 回だけ実行 → **one-shot（本章）**（承認 1 回、事後分析 1 回）
> - Batch size を **観測後に調整**、または batch を複数回反復 → **batched sequential（vol-05）**（batch ごとに acquisition 更新 + 承認）
>
> 操作的判定基準は「**batch 後に acquisition function を再評価するか否か**」——再評価するなら sequential、しないなら one-shot。

ARIM の典型ケース（run 予算固定、事前知識あり、並列装置あり）では **one-shot が第 1 選択**——本章の Skill 契約で十分カバーされます。

---

## 章末演習

### 演習 12.1：事前分布の設計と事前予測チェック

自研究テーマで、以下を作成：

1. 過去 3 バッチの実験結果から事前分布 $p(\theta)$ を階層モデルで定義（§12.2.1 の 3 層構造）
2. `prior_specification_provenance` の 11 フィールドを埋める
3. §12.9.2 の `prior_predictive_check` を実行、fraction_within_physical_range ≥ 0.80 を満たすか確認
4. 満たさなければ `prior_hyperparameters` を調整（**実験実行前に**、post-hoc は fatal）

### 演習 12.2：EIG 最大化と MC 分散報告

以下 3 候補設計について EIG を計算：

- 候補 A：CCD rotatable（k=3, α=1.68179283050743, center=5）
- 候補 B：Bayesian D-optimal（`variable_selection_authorization` 済み事前分布）
- 候補 C：Uniform random over experimental region（baseline）

§12.9.3 で Laplace 近似 EIG と MC 分散を計算し、`eig_estimation_contract` の `mc_variance_ratio < 0.10` を満たすか判定。満たさなければ nested MC の `n_outer` / `n_inner` を増やす。

### 演習 12.3：事前-事後 alignment failure の処理

以下シナリオで、`prior_data_alignment_check` が fail した場合の対応を書き下せ：

- 事前分布：$\beta_{\text{temperature}} \sim \mathcal{N}(0.5, 0.1)$（過去バッチから 500°C 付近が最適と信念）
- 実験結果：500°C run では低性能、650°C run で高性能——実際は事前が misspecified
- $p$-value = 0.02（< 0.05、alignment failure）

**要点**：事前を post-hoc に変更する（$\mathcal{N}(0.7, 0.2)$ にする）は fatal。適切な対応は？

### 演習 12.4：Ch11 T3 と Ch12 の使い分け

自研究テーマで、Ch11 T3（MLE-GP）と Ch12（fully Bayesian GP）のどちらを選ぶか根拠を書く。特に：

- 事前分布の妥当性の証拠が **あるか / ないか**
- **計算予算**：Ch12 の MCMC は Ch11 T3 の 100 倍程度のコスト
- **予測不確かさの重要度**：意思決定に posterior variance が必須か、点推定で十分か

### 演習 12.5：one-shot vs 逐次の選択

§12.11.3 の 6 状況について、自研究テーマがどれに該当するかを判定し、one-shot（本章）と逐次（vol-05）どちらを選ぶかを書く。特に「並列実行可能性が高い」場合の batched sequential（vol-05）との比較を述べよ。

---

## 参考資料

### 内部 cross-reference

- **第0章**：4 識別仮定（本章 §12.1.2 で `prior_specification_correctness_declared` と `prior_data_alignment_declared` を追加宣言）
- **第4章**：`experimental_design_provenance` 契約定義、Ch4 §4.5.2 counterfactual_scope_gate（本章 §12.7 で 5-check に拡張、`prior_data_alignment_check` 追加）、Ch4 §4.6.2 3 層承認（本章 §12.8 で `prior_family_change` / `prior_hyperparameter_change` を `variable_selection_authorization` に含む）
- **第6章**：`estimator_contract_change_gate` L1-L4（本章 §12.6.3 で Ch11 T3 → T4 昇格は **L4 event**）
- **第9章**：refutation_gate 8-concept enum（本章 §12.8 で `prior_predictive_check` / `prior_data_alignment` を `declared_required_tests` に追加）
- **第10章**：`design_matrix_sha256` immutability chain、`randomization_seed_provenance`、`assignment_log_sha256` 4-stage detection、`mixed_model_specification` 5 フィールド（本章 §12.8 で `pymc_hierarchical_model` schema として踏襲）
- **第11章**：モデル階層 T1/T2/T3（本章 §12.6.3 で T3 → T4 に拡張、命名衝突回避継承）、GP surrogate（本章 §12.6 で fully Bayesian GP に発展）、応答曲面 counterfactual_scope_gate 4-check（本章 §12.7 で 5-check に）
- **第13章**：capstone Phase 3 で Bayesian DoE を反実仮想シミュレーションに接続、事後分布から SCM の noise 分布を推定
- **第14章**：Bayesian DoE 特有の失敗パターン（事前 post-hoc 変更、EIG MC 分散隠蔽、alignment failure の隠蔽）
- **vol-02 第11章**：階層モデル（本章 §12.2 事前分布 3 層構造の直系継承）
- **vol-02 第10-12章**：PyMC / MCMC 診断（本章 §12.9.4 sample_posterior_with_diagnostics で R̂ / ESS / divergences）
- **vol-05**：逐次 Bayesian DoE（本章 §12.11 で橋渡し、`sequential_experiment_stop_condition` / `acquisition_function_provenance` / stopping rule HARKing は vol-05 完全委譲）

### 外部参考文献

- Chaloner, K., & Verdinelli, I. (1995). *Bayesian experimental design: A review*. Statistical Science, 10(3), 273–304. — Bayesian DoE の古典 review
- Ryan, E. G., Drovandi, C. C., McGree, J. M., & Pettitt, A. N. (2016). *A review of modern computational algorithms for Bayesian optimal design*. International Statistical Review, 84(1), 128–154. — 現代的計算手法（Laplace / MC / nested MC）
- Rainforth, T., Foster, A., Ivanova, D. R., & Bickford Smith, F. (2024). *Modern Bayesian experimental design*. Statistical Science, 39(1), 100–114. — nested MC EIG と variational EIG の比較
- Gelman, A., Vehtari, A., Simpson, D., et al. (2020). *Bayesian workflow*. arXiv:2011.01808. — 事前予測チェックと prior-data alignment の workflow
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. https://gaussianprocess.org/gpml/ — GP の理論書
- Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). *Probabilistic programming in Python using PyMC3*. PeerJ CS, 2:e55. — PyMC の設計思想
- PyMC docs: https://www.pymc.io/ — 実装 API
- ArviZ docs: https://python.arviz.org/ — 事後診断

### API チートシート（詳細は付録 B）

| ライブラリ | 主要関数 | 用途 |
|---|---|---|
| pymc | `pm.Normal / HalfCauchy` | 事前分布定義 |
| pymc | `pm.sample_prior_predictive(draws=1000)` | 事前予測サンプル |
| pymc | `pm.sample(draws=2000, tune=1000, chains=4)` | NUTS 事後サンプリング |
| arviz | `az.summary(trace)` | R̂ / ESS / MCSE の一括診断 |
| arviz | `az.plot_trace(trace)` | trace / density の視覚化 |
| numpy | `np.linalg.slogdet(precision)` | Bayesian D-optimality（数値安定な log det） |
| scipy | `scipy.optimize.minimize` | Laplace 近似の MAP 探索 |
| smt | `KRG(corr, theta0)` | Ch11 T3 GP との接続（比較用） |
