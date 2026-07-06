# 付録B　Scikit-learn / PyMC チートシートと PyMC↔Stan 対応表

> **本付録の目的**：本文で登場した API・prior・診断コード・トラブル対処を **リファレンス** としてまとめる。加えて、**論文中の Stan コードを PyMC で読み解く**ための同一モデル対応表を用意する。

> [!NOTE]
> **cmdstanpy のセットアップは扱いません**。Stan コードは**読解用**で、実行は PyMC 側で行う想定です。

---

## B.1 Scikit-learn チートシート

### B.1.1 頻出 API 早見表

| 用途 | API | 主な引数 | 注意 |
|---|---|---|---|
| 線形回帰 + 正則化 | `Ridge`, `Lasso`, `ElasticNet` | `alpha` | 特徴量スケール前提 |
| ロジスティック回帰 | `LogisticRegression` | `C`, `penalty`, `class_weight` | `C = 1/alpha` |
| PLS | `PLSRegression` | `n_components` | 内部で中心化・スケーリング |
| RandomForest | `RandomForestRegressor`, `-Classifier` | `n_estimators`, `max_depth`, `min_samples_leaf` | OOB は `bootstrap=True` 時に各サンプルを OOB trees の集約予測で評価 |
| Gradient Boosting | `GradientBoostingRegressor`, `HistGradientBoostingRegressor` | `learning_rate`, `n_estimators`, `max_depth` | Hist は欠損対応 |
| PCA | `PCA` | `n_components` | `explained_variance_ratio_` を確認 |
| k-means | `KMeans` | `n_clusters`, `init`, `n_init` | 初期化で結果変動 |
| HDBSCAN | `HDBSCAN` | `min_cluster_size`, `min_samples` | ノイズ点 = -1 |
| Isolation Forest | `IsolationForest` | `contamination`, `n_estimators` | 閾値の選び方注意 |
| One-Class SVM | `OneClassSVM` | `nu`, `gamma`, `kernel` | 特徴量スケール敏感 |
| SHAP | `shap.TreeExplainer`, `shap.KernelExplainer` | — | サンプル数抑制 |

### B.1.2 Pipeline 頻出パターン

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 数値列 + カテゴリ列
preproc = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])
pipe = Pipeline([
    ("prep", preproc),
    ("model", Ridge(alpha=1.0)),
])
pipe.set_output(transform="pandas")   # 特徴量名維持
```

### B.1.3 CV 選択フローチャート

| 状況 | CV | 引数 |
|---|---|---|
| iid・シャッフル可 | `KFold` | `shuffle=True, random_state=<int>` |
| 群単位で予測 | `GroupKFold` | `groups=group_key`（sklearn 1.4+ で metadata routing 有効時は `params={"groups": g}`） |
| クラス不均衡 + 群 | `StratifiedGroupKFold` | `groups=, y=` |
| 時系列 | `TimeSeriesSplit` | `n_splits=, gap=<int>` |
| Nested CV | `GridSearchCV` inside `cross_validate` | outer/inner を別々に |

### B.1.4 交差検証の返り値

```python
res = cross_validate(pipe, X, y, cv=cv, groups=g,
                     scoring=("neg_root_mean_squared_error", "r2"),
                     return_train_score=True,
                     return_estimator=True,
                     return_indices=True)
# res["test_neg_root_mean_squared_error"], res["train_r2"],
# res["estimator"], res["indices"]["train"], res["indices"]["test"]
```

### B.1.5 スコアリング指標対応表

| task_type | 指標 | scoring 文字列 |
|---|---|---|
| regression | RMSE | `neg_root_mean_squared_error` |
| regression | MAE | `neg_mean_absolute_error` |
| regression | R² | `r2` |
| classification | ROC-AUC | `roc_auc` |
| classification | F1 | `f1`, `f1_macro`, `f1_weighted` |
| classification | LogLoss | `neg_log_loss` |
| classification | Balanced Acc | `balanced_accuracy` |

**注意**：`neg_*` は「大きいほど良い」に統一するため負号。**RMSE は sklearn 1.4+ で `root_mean_squared_error` が導入**。

### B.1.6 特徴量重要度の 3 系統

| 系統 | API | 特徴 |
|---|---|---|
| **モデル固有** | `.coef_`, `.feature_importances_` | 高速だが定義依存 |
| **Permutation** | `sklearn.inspection.permutation_importance` | モデル非依存、CV スコア変化を測る |
| **SHAP** | `shap.TreeExplainer(model).shap_values(X)` | 個別試料の寄与を分解 |

---

## B.2 PyMC チートシート

### B.2.1 頻出 API

| 用途 | API | 例 |
|---|---|---|
| モデル定義 | `pm.Model()` | `with pm.Model(coords=coords) as m:` |
| Normal prior | `pm.Normal(name, mu, sigma)` | `pm.Normal("beta", 0, 5)` |
| HalfNormal | `pm.HalfNormal(name, sigma)` | `pm.HalfNormal("sigma", 1)` |
| Sum-to-zero | `pm.ZeroSumNormal(name, sigma, dims)` | 階層モデル identifiable 化 |
| Deterministic | `pm.Deterministic(name, expr, dims)` | 事後保存対象 |
| データ差替え | `pm.Data(name, value, dims)` + `pm.set_data({name: new})` | out-of-sample 予測 |
| サンプリング | `pm.sample(draws, tune, chains, cores, nuts_sampler, target_accept, random_seed)` | — |
| Prior predictive | `pm.sample_prior_predictive(random_seed=)` | Ch10 §11.3 |
| Posterior predictive | `pm.sample_posterior_predictive(idata, predictions=True/False, random_seed=)` | `predictions=True` で out-of-sample |

### B.2.2 事前分布の選び方

| データ範囲・意味 | 推奨 prior | 例 |
|---|---|---|
| 実数（正負両方、スケール既知） | `Normal(mu, sigma)` | `Normal(0, 5)` |
| 正のスケールパラメータ | `HalfNormal(sigma)` or `HalfCauchy(beta)` | `HalfNormal(1)` |
| 相関行列（コレスキー分解） | `LKJCholeskyCov(name, eta, n, sd_dist=...)` | `chol, corr, sigmas = pm.LKJCholeskyCov("chol", eta=2, n=K, sd_dist=pm.HalfNormal.dist(1.0, size=K))` — `sd_dist` は `.dist()` 経由、`n` は次元数 |
| Group effect（sum-to-zero） | `ZeroSumNormal(sigma, dims=...)` | identifiable |
| Weakly informative（標準化済み特徴量の回帰係数、特にロジスティック） | `Normal(0, 2.5)` | Gelman recommendation（対象は必ず標準化後） |
| Wide "flat" | `Normal(0, 100)` | **避けるべき**（Ch14 §15.5） |
| 制約付き（[0, 1]） | `Beta(alpha, beta)` | `Beta(2, 2)` は中央寄り |
| 分散比 | `HalfStudentT(nu=3, sigma)` | 重い裾で robust |

**推奨**（第11章 §11.3）：**必ず prior predictive check** を実行し、範囲がドメイン外でないことを確認。

### B.2.3 診断指標のしきい値

| 指標 | しきい値（既定） | 意味 | 対処 |
|---|---|---|---|
| $\hat{R}$ (rhat) | ≤ 1.01 | chain 間の一致 | `tune` 増、初期値散開、`chains` 増 |
| ESS bulk | ≥ 400（chain あたり 100） | 有効サンプルサイズ（中心） | `draws` 増やす |
| ESS tail | ≥ 400 | 裾のサンプル効率 | non-centered 化 |
| divergences | **0**（非ゼロは保留・再フィット） | HMC の発散 | 発散点の localize → scale/標準化 → 非中心化 → 最終手段で `target_accept` を上げる |
| BFMI | ≥ 0.3 | エネルギー適応度 | reparameterize |
| max treedepth | `reached_max_treedepth == False` | tree 深さ天井到達 | まず幾何を確認、その後 `max_treedepth=15` 検討 |

> **divergences のしきい値**：Ch12 §13.9 では認定要件として **非ゼロは不許可**。「〜% までなら OK」ではなく、原因（曲率・スケール・パラメータ化）を必ず特定してから再サンプルする。

### B.2.4 事後予測チェック（Ch12 §13.5）

```python
import arviz as az
az.plot_ppc(idata, num_pp_samples=100)   # 分布比較
az.plot_bpv(idata, kind="p_value")       # Bayesian p-value
az.plot_bpv(idata, kind="u_value")       # PIT （probability integral transform）
```

**判定**：BPV が [0.05, 0.95] に収まる、PIT が一様に近い → 適合。

### B.2.5 InferenceData の永続化

```python
# path は Skill ディレクトリからの相対（.github/skills/<name>/artifacts/、付録a §A.1.1）
idata.to_netcdf("artifacts/posterior_v1.0.0.nc")     # 保存
idata = az.from_netcdf("artifacts/posterior_v1.0.0.nc")  # 読み込み

# hash for provenance
import hashlib
with open("artifacts/posterior_v1.0.0.nc", "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()
```

---

## B.3 バックエンド比較

### B.3.1 サンプラー選択マトリクス

| 状況 | バックエンド | 引数 |
|---|---|---|
| 小規模・PyMC 標準 | PyTensor NUTS | `nuts_sampler="pymc"`（明示推奨） |
| 中〜大規模階層 | NumPyro NUTS on CPU | `nuts_sampler="numpyro"` |
| GPU 利用可 | NumPyro NUTS on GPU | `+ jax.config.update("jax_platform_name", "gpu")` |
| Rust NUTS | NutPie | `nuts_sampler="nutpie"` |

> **`nuts_sampler` を必ず明示する**：PyMC の内部デフォルトはバージョンや導入済みパッケージによって変わり得る（例: `nutpie` が入っていれば自動選択されるケースがある）。再現性・provenance のために **`nuts_sampler=` を省略しない**。認定実行では固定必須（Ch14 §15.7）。

### B.3.2 環境設定（NumPyro/JAX）

```python
import jax
jax.config.update("jax_enable_x64", True)         # ★ float64 必須
print(jax.devices())                              # 実行デバイス確認
```

**注意**（第11章 §11.9, 第15章 §15.7）：
- FP32 だと収束不良・divergences 増加
- CPU/GPU で乱数結果は完全一致しない
- **認定は 1 バックエンド固定**、比較は tolerance 内

---

## B.4 MCMC トラブル対処早見表

**対処の優先順位（Ch12 §13.6 の梯子）**：
`(1) 発散点の localize` → `(2) スケール・標準化見直し` → `(3) 非中心化・reparameterize` → `(4) 最終手段で target_accept を上げる`。
いきなり `target_accept=0.99` にすると根本原因が隠れる。

| 症状 | 原因候補 | 対処（優先順） |
|---|---|---|
| `divergences > 0` | 事後の曲率が急峻、centered parameterization、スケール不整合 | localize（`az.plot_pair` で発散点確認）→ 標準化 → 非中心化 → `target_accept=0.99` |
| `rhat > 1.01` | chain が混ざっていない | `tune` 増、初期値散開、`chains` 増 |
| `ESS_tail` 小 | 裾のサンプル効率悪 | 非中心化、`draws` 増 |
| `reached_max_treedepth == True` | tree が浅すぎ／幾何が悪い | まず幾何・スケールを確認、それでもダメなら `max_treedepth=15` |
| `NaN` in log_prob | 数値不安定、prior 極端 | prior 見直し、`jitter` 増 |
| `BFMI < 0.3` | エネルギー分布が悪い | reparameterize、scale 調整 |
| 予測が prior に張り付く | prior が狭すぎ | Prior sensitivity（Ch14 §15.5） |
| PyMC で通ったが NumPyro で違う結果 | float32 / 乱数実装差 | `jax_enable_x64`、tolerance 検証（Ch14 §15.7） |

**非中心化の書換パターン**：

```python
# Centered（避ける）
mu = pm.Normal("mu", 0, sigma_group, dims="group")

# Non-centered（推奨）
mu_raw = pm.Normal("mu_raw", 0, 1, dims="group")
mu = pm.Deterministic("mu", mu_raw * sigma_group, dims="group")
```

---

## B.5 PyMC ↔ Stan 対応表

論文で Stan コード（`.stan` ファイル）を見たとき、**同じモデルを PyMC でどう書くか**を対応させます。

### B.5.1 プリミティブ対応

| 概念 | Stan | PyMC |
|---|---|---|
| モデル宣言 | `model { ... }` | `with pm.Model() as m:` |
| データ | `data { int N; vector[N] y; }` | `y_data = np.array(...)` |
| パラメータ宣言 | `parameters { real mu; real<lower=0> sigma; }` | `mu = pm.Normal("mu", ...)` / `sigma = pm.HalfNormal("sigma", ...)` |
| 事前分布 | `mu ~ normal(0, 5);` | `mu = pm.Normal("mu", 0, 5)` |
| 尤度 | `y ~ normal(mu, sigma);` | `pm.Normal("y", mu, sigma, observed=y_data)` |
| Deterministic | `transformed parameters { real mu2 = mu * 2; }` | `mu2 = pm.Deterministic("mu2", mu * 2)` |
| 制約付き | `real<lower=0, upper=1> p;` | `pm.Beta("p", 1, 1)` or `pm.Uniform("p", 0, 1)` |
| ベクトル化 | `y ~ normal(alpha + beta * x, sigma);` | `pm.Normal("y", alpha + beta * x, sigma, observed=y_data)` |

### B.5.2 分布名対応

| Stan | PyMC |
|---|---|
| `normal(mu, sigma)` | `pm.Normal(name, mu, sigma)` |
| `normal_lpdf(x \| mu, sigma)` | `pm.logp(pm.Normal.dist(mu, sigma), x)` |
| `student_t(nu, mu, sigma)` | `pm.StudentT(name, nu, mu, sigma)` |
| `cauchy(mu, sigma)` | `pm.Cauchy(name, mu, sigma)` |
| `gamma(alpha, beta)` | `pm.Gamma(name, alpha, beta)` |
| `inv_gamma(alpha, beta)` | `pm.InverseGamma(name, alpha, beta)` |
| `beta(alpha, beta)` | `pm.Beta(name, alpha, beta)` |
| `bernoulli(p)` | `pm.Bernoulli(name, p)` |
| `binomial(n, p)` | `pm.Binomial(name, n, p)` |
| `poisson(lambda)` | `pm.Poisson(name, lambda)` |
| `lkj_corr_cholesky(eta)` | `pm.LKJCorr(name, eta, n)`（相関のみ）／ `pm.LKJCholeskyCov(name, eta, n, sd_dist=...)`（共分散、SD 付き） |

### B.5.3 モデル対応例：線形回帰

**Stan**:
```stan
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;      // half-normal は「制約 + normal」で表現
}
model {
  alpha ~ normal(0, 5);
  beta ~ normal(0, 5);
  sigma ~ normal(0, 1);     // 制約 <lower=0> により half-normal になる
  y ~ normal(alpha + beta * x, sigma);
}
```

> **Stan には `half_normal` という分布関数はない**。`real<lower=0>` の制約付きパラメータに `normal(0, sigma)` を書くと自動的に half-normal として扱われる。

**PyMC**:
```python
with pm.Model() as m:
    alpha = pm.Normal("alpha", 0, 5)
    beta = pm.Normal("beta", 0, 5)
    sigma = pm.HalfNormal("sigma", 1)
    pm.Normal("y", mu=alpha + beta * x, sigma=sigma, observed=y)
```

### B.5.4 モデル対応例：非中心化階層モデル

**Stan**:
```stan
data {
  int<lower=0> N;
  int<lower=0> J;              // number of groups
  array[N] int<lower=1, upper=J> group;
  vector[N] y;
}
parameters {
  real mu_grand;
  real<lower=0> sigma_group;
  real<lower=0> sigma_y;
  vector[J] eta;               // non-centered
}
transformed parameters {
  vector[J] mu_j = mu_grand + eta * sigma_group;
}
model {
  mu_grand ~ normal(0, 5);
  sigma_group ~ normal(0, 1);  // <lower=0> により half-normal
  sigma_y ~ normal(0, 1);      // 同上
  eta ~ normal(0, 1);
  for (n in 1:N)
    y[n] ~ normal(mu_j[group[n]], sigma_y);
}
```

**PyMC**:
```python
coords = {"group": group_names, "obs": np.arange(N)}
with pm.Model(coords=coords) as m:
    mu_grand = pm.Normal("mu_grand", 0, 5)
    sigma_group = pm.HalfNormal("sigma_group", 1)
    sigma_y = pm.HalfNormal("sigma_y", 1)

    eta = pm.Normal("eta", 0, 1, dims="group")
    mu_j = pm.Deterministic("mu_j", mu_grand + eta * sigma_group,
                            dims="group")

    pm.Normal("y", mu=mu_j[group_idx], sigma=sigma_y,
              observed=y, dims="obs")
```

### B.5.5 モデル対応例：ロジスティック回帰

**Stan**:
```stan
data {
  int<lower=1> N;
  int<lower=1> K;                       // 特徴量数
  matrix[N, K] X;
  array[N] int<lower=0, upper=1> y;
}
parameters { vector[K] beta; real alpha; }
model {
  beta ~ normal(0, 2.5);
  alpha ~ normal(0, 5);
  y ~ bernoulli_logit(alpha + X * beta);
}
```

**PyMC**:
```python
with pm.Model() as m:
    beta = pm.Normal("beta", 0, 2.5, shape=K)
    alpha = pm.Normal("alpha", 0, 5)
    logit_p = alpha + X @ beta
    pm.Bernoulli("y", logit_p=logit_p, observed=y)
```

### B.5.6 主要な違い（読解時の注意）

| 論点 | Stan | PyMC |
|---|---|---|
| 変数宣言 | 型と制約を最初に宣言 | Distribution オブジェクトが型を持つ |
| ループ | `for` を書くか vectorize | vectorize が原則 |
| Index | 1-based | 0-based |
| `generated quantities` | 別ブロック | `pm.Deterministic` または `pm.sample_posterior_predictive` |
| `target +=` | 直接 log-posterior に加算 | `pm.Potential` |
| Data 差替え | 別実行 | `pm.set_data` |
| 変分推論 | `stan_variational` | `pm.ADVI`, `pm.fit` |

### B.5.7 Stan 独自機能で PyMC に直接対応がないもの

| Stan 機能 | PyMC での代替 |
|---|---|
| `ODE integrator` (`integrate_ode_*`) | `sunode` or `pymc-experimental` |
| `algebraic_solver` | scipy と手動連携 |
| `map_rect`（並列化） | `pymc-experimental` の一部機能、または backend 側で並列 |
| ADVI の細かい制御 | `pm.fit(method="advi", ...)` |

---

## B.6 ArviZ プロット早見表

| プロット | API | 用途 |
|---|---|---|
| 事後分布 | `az.plot_posterior(idata, var_names=...)` | HDI・平均・中央値 |
| Trace | `az.plot_trace(idata)` | chain 混ざり確認 |
| Forest | `az.plot_forest(idata, ...)` | パラメータ比較 |
| Pair | `az.plot_pair(idata, ...)` | 相関確認 |
| PPC | `az.plot_ppc(idata)` | 事後予測適合 |
| BPV | `az.plot_bpv(idata, kind="p_value")` | Bayesian p-value |
| PIT | `az.plot_bpv(idata, kind="u_value")` | 予測校正 |
| Energy | `az.plot_energy(idata)` | BFMI 診断 |
| Rank | `az.plot_rank(idata)` | chain 混ざり（rank-normalized） |
| LOO-PIT | `az.plot_loo_pit(idata, y=...)` | leave-one-out 校正 |

---

## B.7 まとめ

- **sklearn**：Pipeline + CV は基本、CV は task に応じて選択。scoring は `neg_*` に注意
- **PyMC**：モデル定義 → prior predictive → sample → diagnostics → PPC の 5 ステップ
- **バックエンド**：NumPyro CPU が階層モデルのデフォルト、`jax_enable_x64=True` 必須
- **Stan → PyMC 変換**：分布名・vectorize・0-based index に注意。ODE / map_rect は直接対応なし
- **診断**：R̂ ≤ 1.01, ESS ≥ 400, divergences = 0 が最低ライン

---

## 参考資料

### 本書内の該当章

- 第5-8章：sklearn 実装
- 第8章：CV 詳細
- 第10-12章：PyMC 実装・診断
- 第12章：階層モデル・非中心化
- 付録A：Skill テンプレート

### 外部参考

- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [PyMC docs](https://www.pymc.io/) — 公式ドキュメント
- [ArviZ docs](https://python.arviz.org/) — 診断・可視化
- [Stan User's Guide](https://mc-stan.org/docs/stan-users-guide/) — Stan 側の参照
- Betancourt, M. *A Conceptual Introduction to HMC*. arXiv:1701.02434 (2017). — HMC の直観
- Gelman, A., et al. *Prior Choice Recommendations* (Stan Wiki). — prior 選択の指針
- Vehtari, A., et al. *Rank-Normalization, Folding, and Localization: An Improved R̂*. Bayesian Analysis (2021). — R̂ 改良版
