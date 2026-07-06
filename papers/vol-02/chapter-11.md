# 第11章　PyMC 入門と Skill 化

> [!IMPORTANT]
> **本章の到達目標**
>
> - `pm.Model` の基本構文で、**prior → likelihood → posterior** の 3 層を書き分けられる
> - **prior predictive check** で「観測前に prior が生む世界」を確認できる
> - **posterior predictive check** で「事後がデータを再現できるか」を診断できる
> - **収束診断の「見る場所」**（$\hat{R}$、ESS、divergences、tree depth、energy）を知り、警告に対処できる
> - **校正曲線のベイズ版** を最小限の PyMC コードで組み立てられる
> - PyMC Skill を、**`sampler_config` / `backend_config` / `posterior_artifact` / `diagnostics_summary`** を含む provenance と共に凍結できる
> - 章末で **NumPyro/JAX に移る前の準備**（`jax_enable_x64`, 乱数キー, PyTensor fallback）を整える

## 本章で扱わないこと

- **階層モデル**：第12章で扱う（本章は非階層 = 完全プーリングまで）
- **MCMC 実務の判断論**（chain 数打切り基準、reparameterization 選択、divergences の再パラメータ化）：第13章
- **失敗パターン全般**：第15章
- **Stan コードの併記**：付録 B の対応表のみ
- **深層ベイズ（BNN、VI）**：本書のスコープ外

---

## 11.1 章の位置づけ

第10章で **CI vs CrI、事前分布、`uncertainty_scheme`** を言葉で整理しました。本章はそれを**動かす**回です。

- vol-01 の Skill 仕様書 6 要素はそのまま使う
- Skill の**主成果物**は「点推定 + 頻度論 CI」から「事後分布 + CrI + 事後予測分布」に置き換わる
- provenance には **`sampler_config` / `backend_config` / `posterior_artifact` / `diagnostics_summary`** が加わる（付録 A で正式化）

### 5 ステップ・ワークフロー

本章では以下の 5 ステップを 1 サイクルとして繰り返します：

| ステップ | 内容 | 対応節 |
|---|---|---|
| 1. モデル記述 | `pm.Model` 内で prior と尤度を書く | §11.2 |
| 2. Prior predictive check | 観測を使わず prior だけでデータを生成、物理的に妥当か目視 | §11.3 |
| 3. サンプリング | NUTS で事後分布からサンプル | §11.4 |
| 4. 収束診断 | $\hat{R}$、ESS、divergences、tree depth、energy | §11.4 |
| 5. Posterior predictive check | 事後からデータを再生成、観測と比較 | §11.5 |

> [!NOTE]
> **順序は厳格**：ステップ 2 を飛ばしていきなり 3→5 に行くのが最も多い失敗です。「観測を見ずに prior だけで妥当な世界が出るか」を確認しないと、事後の妥当性を評価する基準が持てません。

---

## 11.2 `pm.Model` の基本構文

### 3 層構造

PyMC のモデルは常に **prior → deterministic transformation → likelihood** の 3 層で書きます。

```python
import pymc as pm
import numpy as np

with pm.Model() as model:
    # ① 事前分布（parameters）
    alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
    beta  = pm.Normal("beta",  mu=0.0, sigma=1.0)
    sigma = pm.HalfNormal("sigma", sigma=1.0)

    # ② 決定的変換（linear predictor など）
    mu = pm.Deterministic("mu", alpha + beta * x_obs)

    # ③ 尤度（likelihood, observed が付く）
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)
```

**3 層ルール**：

- `observed=` が**付かない**確率変数 = パラメータ（推定対象）
- `observed=` が**付く**確率変数 = 尤度（データが実現値）
- `pm.Deterministic` は事後保存対象にしたい派生量にのみ使う（すべてを Deterministic にしない）

> [!IMPORTANT]
> **上記は in-sample 専用**：`x_obs` が plain NumPy array のままだと、**同じモデルを新規データの予測に再利用できません**（`x_obs` がフィット時の値に固定されるため）。out-of-sample 予測を Skill として提供する場合は、入力を `pm.Data` に包み、`pm.set_data` で差し替えます：
>
> ```python
> with pm.Model(coords={"sample": np.arange(len(y_data))}) as model:
>     x = pm.Data("x", x_obs, dims="sample")
>     alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
>     beta  = pm.Normal("beta",  mu=0.0, sigma=1.0)
>     sigma = pm.HalfNormal("sigma", sigma=1.0)
>     mu = pm.Deterministic("mu", alpha + beta * x, dims="sample")
>     y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data, dims="sample")
>
> # 予測時
> with model:
>     pm.set_data({"x": x_new}, coords={"sample": np.arange(len(x_new))})
>     idata_pred = pm.sample_posterior_predictive(
>         idata, predictions=True, random_seed=42,
>     )   # predictions=True で観測 y との整合を要求せず、in/out-of-sample を分離
> ```
>
> **`predictions=True` を付ける理由**：`observed=y_data` は fit 時の値に固定されており、`dims="sample"` を通じて `sample` 座標に紐付いています。`predictions=True` を付けないと、`set_data` で `sample` 長を変えたときに **観測 y（長さ n）と新座標（長さ len(x_new)）の shape 不整合**で `CoordinateValidationError` になります。`predictions=True` は out-of-sample 予測専用のパスに切り替え、結果は `idata_pred.predictions` に格納されます（`posterior_predictive` ではありません）。

> [!TIP]
> **命名規則**：Skill 仕様書 ⑤（禁止事項）と揃えて、パラメータ名は **snake_case・単位付きコメント**を推奨します（`alpha  # eV, offset`）。ArviZ の可視化で名前がそのまま軸ラベルになるため、後から書き換えるコストが大きい。

### 座標（coords）と次元名

`pm.Model(coords=...)` で軸に名前を付けておくと、事後の可視化・保存が段違いに楽になります。

```python
coords = {"sample": np.arange(len(y_data)), "feature": ["temp", "pressure"]}
with pm.Model(coords=coords) as model:
    beta = pm.Normal("beta", mu=0.0, sigma=1.0, dims="feature")
    ...
```

**階層モデルで必須**（第12章）：`dims="lot"`, `dims="instrument"` の名前が付くと partial pooling の解釈が容易。本章の非階層モデルでも習慣化しておきます。

---

## 11.3 Prior predictive check：観測前の世界を見る

### なぜ観測前にチェックするか

事前分布は **「物理的に妥当な世界を生む分布」でなければなりません**。強すぎる prior は観測を無視した結論を生み、弱すぎる prior は非物理的な予測を生みます。**Prior predictive check は、観測を見る前に prior を評価する唯一の方法**です。

```python
with model:
    prior_pred = pm.sample_prior_predictive(draws=500, random_seed=42)
```

`prior_pred.prior_predictive["y_obs"]` が **prior のみから生成された仮想観測**（500 × n_samples 行列）。これを ArviZ でプロットして、**物理的に不可能な値**（負の bandgap、100% を超える収率など）が高頻度で出ていないかを確認します。

### 診断の基準

| 症状 | 診断 | 対応 |
|---|---|---|
| 予測範囲が物理的閾値を大きく超える | Prior が弱すぎる | `sigma` を狭める、`Truncated` を追加 |
| 予測範囲が実測レンジより明らかに狭い | Prior が強すぎる | `sigma` を広げる、または情報事前分布の正当化を再確認 |
| 予測分布が実測平均から大きくずれる | 中心（`mu`）が誤っている | 単位・スケール・変換の確認 |
| 予測分布に極端な多峰・裾 | 尤度・変換が不適切 | パラメータ変換（log、standardized）を検討 |

> [!WARNING]
> **Prior predictive を「観測と一致させる」のは目的外**：目的は「物理的に許容可能な世界を prior が生むか」であり、観測を再現することではありません。観測を再現するのは posterior predictive です（§11.5）。

### Skill 仕様書との対応

prior predictive check の結果は、`prior_specification` の `justification` に**具体的な数値サマリ**として記録します：

```yaml
prior_specification:
  ...
  prior_predictive_summary:
    check_date:     2026-XX-XX
    n_prior_draws:  500
    quantile_2.5:   -0.8
    quantile_97.5:  0.9
    physical_bound_violations: 0     # 例：負の bandgap を生んだ割合
    reviewed_by:    "PI name"
```

---

## 11.4 サンプリングと収束診断

### NUTS でのサンプリング

```python
with model:
    idata = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        random_seed=42,
        target_accept=0.90,     # divergences が出たら 0.95 → 0.99 と上げる
        return_inferencedata=True,
    )
```

**デフォルトで押さえておく点**：

- `chains=4` 未満は診断が甘くなる（`chains=1` は $\hat{R}$ が意味を持たない）
- `tune` は warmup 期間。診断に含めない
- `return_inferencedata=True` は PyMC 5 の既定。ArviZ の `InferenceData` オブジェクトが返る
- `target_accept` は 0.80 から上げていく。0.95 でも divergences が消えないなら**モデル構造の問題**（第13章の reparameterization）

### 収束診断の「見る場所」

**5 つのシグナル + 1 目視**を、常に同じ順序で見ます：

| 順序 | 診断量 | 見方 | 警告の閾値 |
|---|---|---|---|
| 0（目視） | Trace plot | `az.plot_trace(idata, var_names=["alpha", "beta", "sigma"])` | chain が"毛虫状"に重なっているか（分離・段差は不整合） |
| 1 | `divergences` | `int(idata.sample_stats["diverging"].sum().item())` | > 0 で要調査、> 10 で対処必須 |
| 2 | $\hat{R}$（R-hat） | `az.summary(idata)["r_hat"]` | > 1.01 で要確認、> 1.05 は明確な未収束の強い警告 |
| 3 | ESS bulk / ESS tail | `az.summary(idata)[["ess_bulk", "ess_tail"]]` | 4 chains の総 ESS で 400 が最低ライン。tail quantile / threshold probability / rare-event 推定はより大きな ESS を要求 |
| 4 | Tree depth | `sample_stats["tree_depth"]` の最大 | `max_treedepth`（デフォルト 10）に張り付いていたら reparameterize |
| 5 | Energy（BFMI） | `az.bfmi(idata)` | < 0.3 で階層構造・スケーリング・重い裾・prior 広すぎを疑う |

> [!IMPORTANT]
> **$\hat{R}$ の推奨閾値は 1.01**：かつては 1.1 が使われましたが、Vehtari et al. (2021) 以降 **1.01** が標準です [[10-1]](#ref-10-1)。ArviZ もこの閾値をベースに実装されています。「> 1.05 対処必須」は本書の運用目安であり、Skill 認定の合否ラインは組織で調整してください。

> [!TIP]
> **YAML 保存時のスカラ変換**：`idata.sample_stats["diverging"].sum()` は xarray の scalar DataArray を返します。provenance にそのまま書くと型互換で事故ります。`int(x.item())` / `float(x.item())` で Python scalar に変換してから記録します（後述の `diagnostics_summary` テンプレはこれを想定した値）。

### divergences への一次対応

1. `target_accept` を 0.90 → 0.95 → 0.99 と上げる
2. それでも残る場合はモデル構造（階層モデルの中心化 → 非中心化パラメータ化。第12章）
3. Prior が広すぎて事後の裾を探索できていないケースも多い（狭める）

### `diagnostics_summary` フィールド

Skill provenance に**必ず記録**する診断量：

```yaml
diagnostics_summary:
  diagnostics_pass:  true               # 下記 thresholds を全て満たしたかの派生ブール
  thresholds:
    max_rhat:        1.01
    min_ess_bulk:    400                # 全パラメータの最小値に対する下限
    min_ess_tail:    400
    max_divergences: 0
    min_bfmi:        0.3
  n_divergences:     0
  max_rhat:          1.003
  min_ess_bulk:      1250               # 全パラメータ中の最小値
  min_ess_tail:      980                # 同上
  max_tree_depth:    8                  # max_treedepth 未満なら健全
  bfmi_min:          0.85
  sampler:           pymc.NUTS
  target_accept:     0.90
  chains:            4
  draws_per_chain:   1000
  tune_per_chain:    1000
  runtime_seconds:   47.2
  arviz_version:     0.17.x
  pymc_version:      5.x.x
  random_seed:       42
```

> [!NOTE]
> `min_ess_bulk` / `min_ess_tail` は **モデル内の全パラメータの最小値**として運用します。特定パラメータ（例：関心量）だけの ESS を別途記録したい場合は `key_parameters: {x_unknown: {ess_bulk: 1450, ess_tail: 1100}}` を追加。

---

## 11.5 Posterior predictive check：事後で観測を再現できるか

### 実行

```python
with model:
    pm.sample_posterior_predictive(
        idata, random_seed=42, extend_inferencedata=True,
    )

import arviz as az
az.plot_ppc(idata, num_pp_samples=100)
```

`az.plot_ppc` は**観測データの分布**（デフォルトでは KDE）に、事後予測分布からのサンプルを重ねます。**分布形が一致するか**、**外れ値の位置・量が現実的か**を見ます。

### 診断の観点

| 観察 | 診断 |
|---|---|
| 事後予測と観測がほぼ重なる | モデルはデータを再現できている（妥当性の必要条件） |
| 事後予測の裾が観測より短い | 分散が過小推定（誤差モデルの見直し、Student-t への変更検討） |
| 事後予測が観測より広く分散 | 情報が反映されていない（likelihood の再検討、prior 弱化） |
| 特定領域の観測が事後予測に現れない | モデルの構造的な欠落（非線形性、階層構造の欠如） |

> [!WARNING]
> **PPC は「一致すれば正しい」ではない**：観測を再現できるモデルは**無数にあります**。PPC が失敗すればモデルが誤りである十分条件ですが、成功しても正しい保証はありません（過学習・特定化不足）。PPC は「明らかな不整合を検出する」使い方が本筋です。

### 統計量ベースの PPC

視覚だけでなく、**要約統計量の一致**も確認します：

```python
az.plot_bpv(idata, kind="p_value")       # Bayesian p-value
```

Bayesian p-value が **0 または 1 に極端に近い場合**、観測の特徴を事後予測が説明できていないサインです。**0.05〜0.95 は粗い目安**であり、頻度論の仮説検定のような厳密な合否ラインではありません。**モデル不整合の検出用シグナル**として使います。

---

## 11.6 ハンズオン：校正曲線のベイズ版

### 頻度論版（vol-01 復習）

vol-01 では校正曲線を線形回帰でフィットし、`np.polyfit` + `numpy.linalg.lstsq` で係数を推定しました。予測は点推定、CI は残差からの近似でした。

### ベイズ版：モデル

未知濃度 $x_i$ で観測される信号 $y_i$ が

$$
y_i = \alpha + \beta x_i + \varepsilon_i, \quad \varepsilon_i \sim \mathrm{Normal}(0, \sigma)
$$

**Skill 仕様書の目的欄**：既知標準からの検量線を学習し、未知試料の濃度を **予測分布**（点推定ではない）で返す。

```python
with pm.Model(coords={"sample": np.arange(len(y_obs))}) as calibration:
    # ① priors（弱情報）
    alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)          # offset
    beta  = pm.Normal("beta",  mu=1.0, sigma=1.0)          # slope（1 付近を弱く期待）
    sigma = pm.HalfNormal("sigma", sigma=1.0)              # 観測誤差

    # ② deterministic
    mu = alpha + beta * x_std                              # x_std は既知濃度

    # ③ likelihood
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs, dims="sample")

# --- 5 ステップ ---
with calibration:
    prior_pred = pm.sample_prior_predictive(draws=500, random_seed=42)     # ② prior predictive
    idata      = pm.sample(draws=1000, tune=1000, chains=4,
                           target_accept=0.90, random_seed=42)             # ③ サンプリング
    pm.sample_posterior_predictive(idata, random_seed=42,
                                   extend_inferencedata=True)              # ⑤ posterior predictive
```

### 未知試料の濃度予測

未知信号 $y^\star$ が観測されたときの $x^\star$ の**事後分布**は、通常は逆問題として別途モデル化します（信号→濃度の逆写像）。最も透明な方法は、**未知濃度 $x^\star$ 自体をパラメータとして扱う**ことです（Bayesian inverse regression）：

```python
with pm.Model(coords={"std": np.arange(len(y_std_obs))}) as inverse_cal:
    alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
    beta  = pm.Normal("beta",  mu=1.0, sigma=1.0)
    sigma = pm.HalfNormal("sigma", sigma=1.0)

    # 既知標準の観測
    mu_std = alpha + beta * x_std
    pm.Normal("y_std", mu=mu_std, sigma=sigma, observed=y_std_obs, dims="std")

    # 未知試料の濃度（推定対象パラメータ、物理的に非負・校正範囲内に制約）
    x_unknown = pm.TruncatedNormal(
        "x_unknown", mu=0.0, sigma=5.0,
        lower=x_std.min(), upper=x_std.max(),   # 校正範囲外への外挿を抑制
    )
    mu_unk = alpha + beta * x_unknown
    pm.Normal("y_unknown", mu=mu_unk, sigma=sigma, observed=y_unknown_obs)

    idata_inverse = pm.sample(
        draws=1000, tune=1000, chains=4,
        target_accept=0.90, random_seed=42,
    )

az.summary(idata_inverse, var_names=["x_unknown"])
```

`az.summary(...)` が **95% CrI 付きの未知濃度の事後分布**を返します。頻度論の外挿は逆行列演算で近似 CI を出しますが、ベイズ版はそのまま逆問題として自然に扱えます。

> [!WARNING]
> **物理制約を prior で入れる**：`Normal(0, 5)` のように制約なしにすると、**負の濃度・校正範囲外の値**が事後に混入し得ます。上記の `TruncatedNormal(..., lower=x_std.min(), upper=x_std.max())` は最小限の物理制約。ドメインに応じて `LogNormal`, `Uniform(0, C_max)`, `HalfNormal` なども検討してください。**校正範囲外の外挿は prior 依存が急激に強まる**ため、Skill 仕様書の適用範囲節で明示的に禁止します。

> [!NOTE]
> **仕様書の `uncertainty_scheme.posterior_quantity`** をここで使い分けます：
>
> - パラメータ CrI（$\alpha, \beta$ の不確かさ） = `parameter`
> - 未知濃度 CrI（$x^\star$ の不確かさ、上記モデル） = `parameter`（$x^\star$ もパラメータ）
> - 新規試料の信号予測 = `prediction`（posterior predictive）

---

## 11.7 事前分布への感度分析

### 目的

第10章で凍結した `sensitivity_alternatives` を実際に走らせ、**事後の結論が prior に依存しすぎていないか**を診断します。

### 手順（3 分岐が実務のミニマム）

1. **メイン prior**（弱情報）でサンプリング → 事後の点推定 + CrI を記録
2. **弱い prior**（`sigma` を 2〜5 倍）で再サンプリング
3. **強い prior**（`sigma` を 1/2〜1/5 倍）で再サンプリング
4. 3 つの事後を並べ、**意思決定が変わるか**を確認

### 診断基準

| 観察 | 判定 | 対応 |
|---|---|---|
| 3 分岐で CrI が実質重なる | 事後はデータ支配、prior 選択に頑健 | 報告時に「robust to prior choice」と記録 |
| 弱い prior と主 prior はほぼ同じ、強い prior で結論が変わる | 主 prior が強すぎる疑い | prior の正当化を再確認 |
| 弱い prior で CrI が発散、強い prior でしか安定しない | データが情報不足、prior に依存した結論 | データ追加 or 意思決定閾値を厳しく |

### Skill provenance への記録

```yaml
sensitivity_analysis:
  performed:    true
  branches:
    - {label: main,     posterior_mean: 2.42, cri_95: [2.10, 2.75]}
    - {label: weaker,   posterior_mean: 2.41, cri_95: [2.02, 2.83]}
    - {label: stronger, posterior_mean: 2.44, cri_95: [2.20, 2.68]}
  decision_stable_across_branches: true
  reviewed_by:  "PI name"
```

`decision_stable_across_branches: false` の場合、Skill 仕様書 ④（成功条件）の閾値を再検討します。

---

## 11.8 PyMC Skill の provenance テンプレ

vol-01 の provenance を PyMC 用に拡張したフィールド（付録 A で正式化）：

```yaml
provenance:
  # vol-01 基本フィールド
  library_versions:   {pymc: 5.x.x, arviz: 0.17.x, numpy: 1.x.x}
  data_hash:          sha256:...
  random_seed:        42

  # 第10章で導入済み（Skill 実行前に凍結）
  prior_specification:
    parameter:                       bandgap_offset
    distribution:                    Normal
    hyperparameters:                 {mu: 0.0, sigma: 0.5}
    units:                           eV
    parameter_scale:                 original
    support:                         [-3.0, 3.0]
    likelihood_context:              "Normal observation model with sigma_obs estimated"
    justification:                   "内部レポート XX (2024)"
    literature_ref:                  [ref-lab-report-2024]
    justification_document:          docs/priors/bandgap_offset.md
    data_cutoff_for_prior:           2025-12-31
    prior_predictive_check_required: true
    sensitivity_alternatives:
      - {distribution: Normal, sigma: 1.0, label: weaker}
      - {distribution: Normal, sigma: 0.2, label: stronger}
    reviewed_by:                     "PI name"
    review_date:                     2026-XX-XX

  # 第11章で追加
  sampler_config:
    sampler:          NUTS
    chains:           4
    draws:            1000
    tune:             1000
    target_accept:    0.90
    max_treedepth:    10
    init_strategy:    jitter+adapt_diag

  backend_config:
    backend:          pytensor       # numpyro / nutpie / blackjax も可（§11.9）
    jax_x64:          null           # numpyro 使用時のみ true

  posterior_artifact:
    path:             artifacts/posterior_v1.0.0.nc     # .github/skills/<name>/artifacts/ 配下（付録A §A.1.1）
    format:           netCDF
    sha256:           ...

  diagnostics_summary:
    # §11.4 の diagnostics_summary をそのまま貼る（thresholds + diagnostics_pass を含む）

  prior_predictive_summary:
    # §11.3 の記録（prior_specification.prior_predictive_check_required=true の実行結果）
  posterior_predictive_summary:
    bayesian_p_value:      0.42
    kolmogorov_smirnov_p:  0.35    # 参考、閾値ではない

  sensitivity_analysis:
    # §11.7 の記録（prior_specification.sensitivity_alternatives の実行結果）

  uncertainty_scheme:
    # 第10章 §10.5 のスキームをそのまま
```

**エージェント時代の運用**：この provenance を **Skill 実行の出力に強制的に含めさせる**設計にします（テンプレを出力しない Skill は監査違反）。第15章の失敗パターンで再登場します。

> [!NOTE]
> **依存関係**：`prior_predictive_summary` は `prior_specification` の子、`sensitivity_analysis` は `prior_specification.sensitivity_alternatives` の実行結果、`diagnostics_summary.diagnostics_pass` が `false` なら Skill は成功条件④を満たしません。この 4 者は**独立に書き換えてはいけない**ため、Skill 実行時にワンショットで生成します。

---

## 11.9 NumPyro/JAX に移る前の 5 ページ

第12章（階層モデル）ではサンプリング速度がボトルネックになりやすく、**NumPyro（JAX バックエンド）** を推奨します。本章末で移行準備を整えます。

### 準備 1：JAX の 64-bit モード

**必ず有効化**します。デフォルトの 32-bit では数値精度不足で $\hat{R}$ が悪化することがあります。

```python
import jax
jax.config.update("jax_enable_x64", True)
```

これは **Python プロセスの先頭で 1 回だけ実行**し、JAX を使う他のコードより前に置きます。

### 準備 2：PyMC からの NumPyro 呼び出し

PyMC 5 では**同じモデル定義**で NumPyro NUTS を使えます：

```python
with calibration:
    idata_np = pm.sample(
        1000, tune=1000, chains=4,
        nuts_sampler="numpyro",         # PyTensor の代わりに NumPyro を使用
        random_seed=42,
        target_accept=0.90,
    )
```

`backend_config.backend: numpyro` を provenance に記録します。

### 準備 3：乱数キー管理

JAX の乱数は **状態を持たない `PRNGKey`** で扱います。PyMC 経由なら `random_seed` を渡せば内部で処理されますが、**NumPyro を直接使う場合**（第12章の一部）は：

```python
import jax.random as jr
key = jr.PRNGKey(42)
key, subkey = jr.split(key)     # 使うたびに分岐
```

**アンチパターン**：同じ `PRNGKey(0)` を複数箇所で使うと**乱数系列が重複**します。必ず `split` で分岐させます。

### 準備 4：PyTensor fallback

NumPyro でエラーが出た場合、**同じモデル・同じシードで PyTensor に戻す**ことができます。Skill 仕様書に **fallback ルート**を明記：

```yaml
backend_config:
  primary:   {backend: numpyro, jax_x64: true}
  fallback:  {backend: pytensor}
  fallback_reason_logged: true
```

### 準備 5：環境の凍結

vol-01 の `pip freeze > requirements.txt` に加えて、**JAX のプラットフォーム記録**を追加：

```python
print(jax.devices())              # 例: [CpuDevice(id=0)] or [gpu(id=0)]
```

これを Skill provenance の `backend_config` に記録します。同じ Skill が **CPU/GPU で数値が微差ぶれる**ため、再現性の必要条件です。

> [!TIP]
> **GPU での再現性は本質的に難しい**：`XLA_FLAGS` の設定・ドライバのバージョン・CUDA バージョンで数値が変わり得ます。**Skill 認定の最終検証は CPU で行う**のがミニマム安全策です（第15章）。

---

## 11.10 章末ワーク

1. **`pm.Model` の 3 層を書き分ける**：vol-01 で使った線形回帰データを、prior → deterministic → likelihood の 3 層で書く。`pm.Deterministic` を使うべき変数と使うべきでない変数を判別する
2. **Prior predictive check の目視**：bandgap や Raman ピーク位置を対象に prior を書き、`sample_prior_predictive` で 500 サンプル生成、**物理的閾値違反率**を計算する
3. **収束診断の 5 シグナルを 1 モデルで確認**：divergences / $\hat{R}$ / ESS bulk / ESS tail / tree depth / BFMI を `az.summary` と `sample_stats` から抽出、`diagnostics_summary` YAML を埋める
4. **校正曲線ベイズ版のハンズオン**：既知標準 5 点 + 未知試料 3 点で `x_unknown` を推定、CrI 幅と頻度論外挿の CI 幅を比較する
5. **感度分析 3 分岐**：主 prior・弱い prior・強い prior で同じデータをフィットし、CrI が重なるかを比較する。`decision_stable_across_branches` を判定する
6. **NumPyro に移し替える**：同じモデルで `nuts_sampler="numpyro"` を有効化、サンプリング速度と $\hat{R}$/ESS を比較する。`jax.devices()` を provenance に記録する

---

## 11.11 本章のまとめ

- **PyMC は prior → deterministic → likelihood の 3 層**で書く。この順序を崩さない
- **5 ステップ（記述 → prior predictive → サンプリング → 診断 → posterior predictive）** を Skill の骨格とする
- **収束診断は 5 シグナル**：divergences → $\hat{R}$ (>1.01 で警告) → ESS bulk/tail → tree depth → BFMI
- **PPC は妥当性の必要条件、十分条件ではない**：一致で安心せず、不一致では棄却
- **感度分析 3 分岐**（main / weaker / stronger）を Skill に組み込み、`decision_stable_across_branches` を出す
- PyMC Skill の provenance には **`sampler_config` / `backend_config` / `posterior_artifact` / `diagnostics_summary`** が必須
- 第12章の階層モデル前に **JAX 64-bit・NumPyro 経路・PyTensor fallback・乱数キー分岐**を凍結

---

## 参考資料

### 本書内の該当章

- 第5章 §5.3：`task_type: bayesian_inference` の位置づけ
- 第10章：CI vs CrI、事前分布、`uncertainty_scheme` の言語化
- 第12章：階層モデル・partial pooling・非中心化パラメータ化（NumPyro 推奨）
- 第13章：MCMC 実務の判断論（打切り基準、divergences 対処、reparameterization）
- 第15章：事前分布の暴走、MCMC 未収束、バックエンド差の失敗
- 付録 A：provenance スキーマ拡張（本章の追加フィールドの正式定義）
- 付録 B：PyMC ↔ Stan 対応表

### 外部参考

<a id="ref-10-1">[10-1]</a> Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-Normalization, Folding, and Localization: An Improved $\hat{R}$ for Assessing Convergence of MCMC. *Bayesian Analysis*, 16(2), 667–718. — $\hat{R}$ の改良版と 1.01 閾値の根拠
<a id="ref-10-2">[10-2]</a> PyMC Development Team. *PyMC Documentation*. [https://www.pymc.io/](https://www.pymc.io/) — 本章コードの一次出典
<a id="ref-10-3">[10-3]</a> ArviZ Contributors. *ArviZ Documentation*. [https://python.arviz.org/](https://python.arviz.org/) — 診断可視化 API
<a id="ref-10-4">[10-4]</a> Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. *arXiv:1701.02434*. — NUTS の背景と divergences の意味
<a id="ref-10-5">[10-5]</a> Gabry, J., Simpson, D., Vehtari, A., Betancourt, M., & Gelman, A. (2019). Visualization in Bayesian workflow. *Journal of the Royal Statistical Society: Series A*, 182(2), 389–402. — prior/posterior predictive check のワークフロー
<a id="ref-10-6">[10-6]</a> NumPyro Team. *NumPyro Documentation*. [https://num.pyro.ai/](https://num.pyro.ai/) — JAX バックエンド、`PRNGKey`
<a id="ref-10-7">[10-7]</a> JAX Team. *JAX Documentation: 64-bit precision*. [https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html) — `jax_enable_x64` の使い方
