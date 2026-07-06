# 第12章　階層モデル：反復測定・ロット差・測定誤差

> [!IMPORTANT]
> **本章の到達目標**
>
> - **完全プーリング / 完全非プーリング / 部分プーリング**の 3 体制を区別し、材料研究のどの場面でどれを選ぶかを説明できる
> - **反復測定モデル**（試料 × 繰り返し）を PyMC で書ける
> - **ロット差モデル**（試料 × ロット、切片ランダム効果）を書け、`sigma_lot` の事後分布から**「ロット差は無視できるか」**を判断できる
> - **測定誤差モデル**（観測 = 真値 + 誤差）で、**真値を潜在変数**として扱える
> - **中心化 vs 非中心化パラメータ化**の違いを理解し、divergences が出たときに reparameterize できる
> - 階層モデルで **NumPyro/JAX を推奨バックエンドとして使う設定**を組める
> - Advanced Capstone（第14章）に向けた **階層モデル Skill** の provenance テンプレを完成できる

## 本章で扱わないこと

- **完全な混合効果モデル理論**（統計学教科書）：本書はエージェント時代の実装ワークフローに絞る
- **多変量ランダム効果 / LKJ prior**：スカラのランダム効果に絞る（相関構造は付録の参考文献に委ねる）
- **時系列の階層モデル / 状態空間**：本章のスコープ外
- **カテゴリカル反応の階層モデル**：連続反応（回帰系）に絞る
- **収束診断の判断論**：第13章
- **失敗パターン集**：第15章

---

## 12.1 なぜ階層モデルか：材料研究の 3 体制

材料研究のデータは、**ネストされた構造**を持つのが普通です：

- 試料 → その中の繰り返し測定
- 合成バッチ（ロット） → その中の試料
- 装置 → その装置で測った試料
- 研究室 → その研究室の装置

このネスト構造を無視して**「独立同分布」として扱うと**、頻度論では標準誤差の過小推定、Bayesian では過信した CrI が出ます。

### 3 つの選択肢

同じデータに対して、次の 3 つのモデリング体制があります。

| 体制 | モデル | 情報の借用 | 使う場面 |
|---|---|---|---|
| **完全プーリング**（complete pooling） | グループを無視、全データを 1 つの母集団と仮定 | すべて共有 | ロット差・装置差が明確に無視できるとき |
| **完全非プーリング**（no pooling） | グループごとに独立にモデル化 | しない | 各グループのデータが十分多く、共通の母構造がないとき |
| **部分プーリング**（partial pooling） | グループ固有効果 + 母集団分布 | 母集団経由で借用 | データが不均等 or 少ないグループがあるとき（実務のデフォルト） |

**部分プーリングは階層モデルの中核概念**です。データが少ないグループは全体平均に「縮む（shrinkage）」ため、極端な推定を避けられます。

### 材料研究の判断表

| 場面 | 推奨 | 理由 |
|---|---|---|
| 単一装置・単一ロット・n=100+ | 完全プーリング | 階層構造がそもそもない |
| 5 ロット × 20 試料/ロット、ロット差の分散に関心 | **部分プーリング** | `sigma_lot` の事後分布を意思決定に使える |
| 3 装置 × 装置ごとに 200 試料 | 完全非プーリング or 部分プーリング | データが多ければ非プーリング、少ないなら部分プーリング |
| 10 研究室 × 各研究室で 3 試料のみ | **部分プーリング** | データ少 + 情報借用が必要 |

---

## 12.2 Partial pooling の直感

部分プーリングは、**「各グループ固有の値」と「全体の平均」の折衷**を、データ量に応じて自動調整します。

$$
\alpha_g \sim \mathrm{Normal}(\mu_\alpha, \sigma_\alpha), \quad g = 1, \dots, G
$$

- $\mu_\alpha$：全体平均（母集団パラメータ、ハイパーパラメータ）
- $\sigma_\alpha$：グループ間のばらつき（母集団パラメータ）
- $\alpha_g$：グループ $g$ 固有の値

**縮み（shrinkage）の起き方**：

- グループ $g$ のデータが多い → $\alpha_g$ は自身のデータに強く引かれる
- グループ $g$ のデータが少ない → $\alpha_g$ は $\mu_\alpha$ に強く引かれる
- $\sigma_\alpha$ が小さい → 全グループが $\mu_\alpha$ に近い
- $\sigma_\alpha$ が大きい → 各グループが独立に近い

**このメカニズムを手作業でチューニングするのは非現実的**です。階層モデルは Bayesian の枠組みで自動的にこれを行います [[11-1]](#ref-11-1) [[11-2]](#ref-11-2)。

---

## 12.3 反復測定モデル（試料 × 繰り返し）

### 場面

同じ試料に対して、同じ装置で 3〜5 回測定を繰り返す。**繰り返し内のばらつき**（測定精度）と、**試料間のばらつき**（試料の真の分散）を分けたい。

### モデル

$$
y_{s, r} \sim \mathrm{Normal}(\alpha_s, \sigma_{\text{obs}}), \quad \alpha_s \sim \mathrm{Normal}(\mu, \sigma_{\text{sample}})
$$

- $y_{s,r}$：試料 $s$ の $r$ 回目の測定値
- $\alpha_s$：試料 $s$ の真値（潜在）
- $\sigma_{\text{obs}}$：測定装置のノイズ
- $\mu, \sigma_{\text{sample}}$：試料集団の平均と分散

```python
import pymc as pm
import numpy as np

# 合成階層データ（リポジトリルート data/synthetic-hierarchy/ から、付録A §A.1.1）
# sample_id: 各測定がどの試料か、shape=(N,)
# y_obs:     測定値、shape=(N,)

n_samples = len(np.unique(sample_id))

with pm.Model(coords={"sample": np.arange(n_samples),
                       "obs":    np.arange(len(y_obs))}) as repeated:
    # 母集団パラメータ
    mu           = pm.Normal("mu", mu=0.0, sigma=5.0)
    sigma_sample = pm.HalfNormal("sigma_sample", sigma=2.0)
    sigma_obs    = pm.HalfNormal("sigma_obs",    sigma=1.0)

    # 試料ごとの真値（中心化パラメータ化：後で非中心化に変更）
    alpha = pm.Normal("alpha", mu=mu, sigma=sigma_sample, dims="sample")

    # 尤度
    y = pm.Normal("y", mu=alpha[sample_id], sigma=sigma_obs,
                  observed=y_obs, dims="obs")

    idata = pm.sample(
        draws=1000, tune=1000, chains=4,
        nuts_sampler="numpyro",         # §11.9 の準備を活用
        target_accept=0.90,             # 中心化 & divergences なしなら 0.90 で開始
        random_seed=42,
    )
```

> [!TIP]
> **`sample_id` は 0-based int**：PyMC の fancy indexing は 0-based の整数 ndarray を想定しています。文字列やカテゴリ型をそのまま渡すと失敗します。
>
> ```python
> sample_id = np.asarray(sample_id, dtype="int64")
> assert sample_id.min() == 0
> assert sample_id.max() == n_samples - 1
> ```
>
> `lot_id`, `inst_id` も同様。

### 読み方

- `sigma_obs`：装置ノイズ（繰り返し測定 3〜5 回で 1 試料の平均を出す価値の判断材料）
- `sigma_sample`：試料集団の真の分散
- `sigma_obs / sigma_sample` 比：**この比が 1 に近ければ、繰り返し測定がノイズに埋もれるサイン**。試料を増やす方が情報効率が高い

> [!TIP]
> **設計への逆流**：試料 20 個 × 繰り返し 5 回 = 100 測定と、試料 50 個 × 繰り返し 2 回 = 100 測定は、**情報量が異なります**。反復測定モデルを走らせると、どちらの設計が母平均・母分散の推定に効率的かを事後的に評価できます（Advanced Capstone の実例、第14章）。

---

## 12.4 ロット差モデル（試料 × ロット）

### 場面

合成バッチ（ロット）ごとにわずかな差がある可能性がある。**「ロット差は無視できるか？」**を Skill で判定したい。

### モデル

$$
y_{s} \sim \mathrm{Normal}(\alpha_{l(s)}, \sigma_{\text{obs}}), \quad \alpha_l \sim \mathrm{Normal}(\mu, \sigma_{\text{lot}})
$$

- $l(s)$：試料 $s$ が属するロット
- $\sigma_{\text{lot}}$：ロット間ばらつき（**関心量**）

```python
with pm.Model(coords={"lot":    np.arange(n_lots),
                       "sample": np.arange(len(y_obs))}) as lot_effect:
    mu        = pm.Normal("mu", mu=0.0, sigma=5.0)
    sigma_lot = pm.HalfNormal("sigma_lot", sigma=2.0)
    sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)

    # ロットごとの平均
    alpha_lot = pm.Normal("alpha_lot", mu=mu, sigma=sigma_lot, dims="lot")

    y = pm.Normal("y", mu=alpha_lot[lot_id], sigma=sigma_obs,
                  observed=y_obs, dims="sample")

    idata_lot = pm.sample(
        draws=1000, tune=1000, chains=4,
        nuts_sampler="numpyro",
        target_accept=0.90, random_seed=42,
    )
```

> [!WARNING]
> **G が小さいと `sigma_lot` は prior 依存**：ロット数 G < 5 のとき、`sigma_lot` の事後分布は**強く prior に依存**します（データ 1 個から母集団分散は推定できないのと同じ理屈）。以下を必須運用にします：
>
> - `sensitivity_alternatives` で `sigma_lot` の prior を 2〜3 種類試す（第11章 §11.7）
> - Skill 仕様書 ⑤（禁止事項）に「G < 5 の階層モデルで sigma_group を単独証拠にしない」を追加
> - G が増やせるならデータ設計側で追加を検討

### 「ロット差は無視できるか」判定

`sigma_lot` の**事後分布**を、閾値 $\tau$（例：0.05 eV）と比較して 3 判定：

| 判定 | 条件 | 結論 |
|---|---|---|
| **無視できる** | 95% CrI の**上端** < $\tau$ | ロット差は実務上小さい、完全プーリング化を検討 |
| **無視できない** | 95% CrI の**下端** > $\tau$ | ロット差は明確、階層モデルを維持 |
| **保留** | 95% CrI が $\tau$ を跨ぐ | データ追加、prior 感度分析、意思決定閾値の再検討 |

> [!IMPORTANT]
> **`sigma_lot ≈ 0` を「無視できる」と誤読しない**：Bayesian では**事後分布そのもの**を判断材料にします。「点推定が 0.02 だから小さい」は不十分。閾値ベースで **確率的な意思決定**を行い、`P(sigma_lot > threshold | data)` を Skill 出力に含めます：
>
> ```python
> threshold = 0.05  # eV
> prob_significant = (idata_lot.posterior["sigma_lot"] > threshold).mean().item()
> ```
>
> これは第10章 `uncertainty_scheme.posterior_quantity: threshold_probability` の実装。

---

## 12.5 測定誤差モデル（観測 = 真値 + 誤差）

### 場面

装置校正のばらつき・環境ノイズ・オペレーター差などで、**観測値は真値の推定であり、真値自体が不確か**。Errors-in-Variables 型の問題。

### モデル

$$
y_i = x_i^{\text{true}} + \epsilon_i, \quad x_i^{\text{true}} \sim \mathrm{Normal}(\mu, \sigma_x), \quad \epsilon_i \sim \mathrm{Normal}(0, \sigma_{\text{obs}})
$$

観測 $y_i$ とは別に、$x_i^{\text{true}}$ を**潜在変数**として推定します。第10章 §10.3 で挙げた「観測誤差の不確かさ」を実装するのがこの型です。

```python
with pm.Model(coords={"obs": np.arange(len(y_obs))}) as meas_error:
    mu         = pm.Normal("mu", mu=0.0, sigma=5.0)
    sigma_x    = pm.HalfNormal("sigma_x", sigma=2.0)          # 真値の分布
    sigma_obs  = pm.HalfNormal("sigma_obs", sigma=1.0)         # 装置ノイズ

    # 潜在真値：観測ごとに独立の真値（=単発測定・n 試料 = n 観測のケース）
    x_true = pm.Normal("x_true", mu=mu, sigma=sigma_x, dims="obs")

    # 観測モデル
    y = pm.Normal("y", mu=x_true, sigma=sigma_obs,
                  observed=y_obs, dims="obs")

    idata_me = pm.sample(
        draws=1000, tune=1000, chains=4,
        nuts_sampler="numpyro",
        target_accept=0.90, random_seed=42,
    )
```

> [!IMPORTANT]
> **反復測定がある場合は `x_true` を `sample` 次元に**：同じ試料を複数回測っているなら、真値は**観測ごと**ではなく**試料ごと**です：
>
> ```python
> with pm.Model(coords={"sample": np.arange(n_samples),
>                        "obs":    np.arange(len(y_obs))}) as meas_error_repeated:
>     ...
>     x_true = pm.Normal("x_true", mu=mu, sigma=sigma_x, dims="sample")
>     y = pm.Normal("y", mu=x_true[sample_id], sigma=sigma_obs,
>                   observed=y_obs, dims="obs")
> ```
>
> 前者と後者を混同すると、装置ノイズ `sigma_obs` の推定が歪みます（前者は `sigma_obs` を過小、後者は正しく分離）。

> [!TIP]
> **スケール選択**：反応速度定数・拡散係数・強度比などは**対数スケールで階層モデルを組む方が自然**な場合があります（例：`x_true` を `Normal` ではなく `LogNormal`、あるいはあらかじめ `y_log = np.log(y_obs)` で対数化）。物理量がゼロ・負を取りうるかを確認してから決めます。

### 識別性の注意

**`sigma_x` と `sigma_obs` は識別可能である必要**があります。両方が推定対象で、繰り返し測定などの追加情報がないと、**モデルは 2 つの分散を区別できません**（`sigma_x^2 + sigma_obs^2` は決まるが分解できない）。

| 状況 | 識別可能か | 対応 |
|---|---|---|
| 装置ノイズ `sigma_obs` の外部情報あり（メーカー仕様など） | ✅ | `sigma_obs` を固定 or 強い prior |
| 繰り返し測定がある | ✅ | 反復測定モデルで両分散を分離 |
| 単発測定のみ、外部情報なし | ❌ | 測定誤差モデルは適用不可、完全プーリングに戻す |

> [!WARNING]
> **識別不能な測定誤差モデルは、divergences・低 ESS を大量発生させます**。§12.6 の非中心化パラメータ化で解決すると思い込みやすいですが、根本原因はモデル構造です。まず `sigma_obs` に**強い情報事前分布**を置けるか検討します。

---

## 12.6 中心化 vs 非中心化パラメータ化：divergences への処方箋

### Neal's funnel と divergences

階層モデルで最も頻発する問題が **funnel geometry** です。$\sigma_\alpha$ が小さいとき、$\alpha_g$ の事後分布が漏斗状になり、NUTS は探索に失敗して **divergences** を大量に出します [[11-3]](#ref-11-3) [[11-4]](#ref-11-4)。

### 中心化（centered）パラメータ化

これまで書いてきた形式：

```python
alpha = pm.Normal("alpha", mu=mu, sigma=sigma_sample, dims="sample")
```

$\sigma_\alpha$ が中〜大のときは**この形が効率的**。しかし $\sigma_\alpha$ が小さいと funnel が発生します。

### 非中心化（non-centered）パラメータ化

**標準正規からの再パラメータ化**：

$$
\alpha_g = \mu + \sigma_\alpha \cdot z_g, \quad z_g \sim \mathrm{Normal}(0, 1)
$$

```python
with pm.Model(coords={"sample": np.arange(n_samples),
                       "obs":    np.arange(len(y_obs))}) as repeated_nc:
    mu           = pm.Normal("mu", mu=0.0, sigma=5.0)
    sigma_sample = pm.HalfNormal("sigma_sample", sigma=2.0)
    sigma_obs    = pm.HalfNormal("sigma_obs",    sigma=1.0)

    # 非中心化：標準正規オフセット
    alpha_offset = pm.Normal("alpha_offset", mu=0.0, sigma=1.0, dims="sample")
    alpha        = pm.Deterministic(
        "alpha", mu + sigma_sample * alpha_offset, dims="sample",
    )

    y = pm.Normal("y", mu=alpha[sample_id], sigma=sigma_obs,
                  observed=y_obs, dims="obs")
```

### いつどちらを使うか

**根本原則**は「funnel geometry を避ける」ことです。実務的には、**グループごとの尤度がどれだけ強いか**で決まります [[11-3]](#ref-11-3)：

| 状況 | 推奨パラメータ化 | 理由 |
|---|---|---|
| **グループごとのデータが少ない**（1〜5 観測/群） | **非中心化** | 尤度が弱く、事後が prior 由来の漏斗状になる |
| **グループごとのデータが多い**（30+ 観測/群） | 中心化 | 尤度が強く、事後が funnel から離れる |
| $\sigma_\alpha$ の prior が広い（弱情報） | **非中心化** | 事後 $\sigma_\alpha$ が小さい領域に潜り込みやすい |
| $\sigma_\alpha$ の事前情報が確度高く中〜大 | 中心化 | funnel geometry は発生しにくい |
| **迷ったら** | **両方走らせて比較** | divergences・ESS・BFMI・runtime を比較 |

> [!NOTE]
> **「事後の $\sigma_\alpha$ が小さい / 大きい」は結果**、原因ではありません。実装時点では**データの情報量と prior の広さ**で判断します。事後を見て切り替えるのは 2 回目のフィット以降。**両バリアントを Skill として保持**し、`switched_from` に切り替え履歴を残すのが本書のスタイルです。

### 実務のフロー

1. **中心化で書く**（読みやすさ優先、divergences があっても初期実験としてOK）
2. `target_accept=0.90` でサンプリング
3. divergences > 0 → 非中心化に切替 + 再サンプリング
4. まだ divergences → `target_accept=0.95` に引き上げ
5. まだ divergences → **モデル構造・prior・識別性**を疑う（第13章の判断論）

> [!IMPORTANT]
> **非中心化にすれば必ず解決するわけではない**：識別性の問題（§12.5）、prior の広すぎ、data の情報不足による本質的な難しさは reparameterization では解決しません。第13章の実務判断論で扱います。

---

## 12.7 NumPyro をバックエンドに使う

### なぜ NumPyro

階層モデルは中〜大規模になりやすく、**JIT コンパイル + XLA による高速化**が効きます。同じモデルで、PyTensor バックエンドの数倍〜数十倍速くなることも珍しくありません。

### 最小設定

第11章 §11.9 で凍結した手順：

```python
import jax
jax.config.update("jax_enable_x64", True)   # 必ず先頭で 1 回

import pymc as pm

with repeated_nc:
    idata = pm.sample(
        draws=1000, tune=1000, chains=4,
        nuts_sampler="numpyro",     # ここだけ変える
        target_accept=0.95,
        random_seed=42,
    )
```

### provenance への追加記録

第11章の `backend_config` は本章以降**ネスト形が canonical**です。第11章のフラット形（`backend_config.backend`）は入門用の簡易表現で、本章から下記の primary/fallback 構造を正とします：

```yaml
backend_config:
  primary:
    backend:                numpyro
    jax_x64:                true
    jax_platform:           cpu             # jax.devices() の結果を要記録
    numpyro_chain_method:   parallel        # or sequential / vectorized
  fallback:
    backend:                pytensor
    fallback_reason_logged: true
  parameterization:
    strategy:               non_centered    # centered / non_centered
    switched_from:          centered        # divergences で切り替えた記録
    switch_reason:          "n_divergences=143 with centered"
```

**`parameterization.switched_from`** を書く理由：Skill 認定後に他者が読んだとき、「なぜ非中心化を選んだか」の履歴がないと、モデル修正時に判断が引き継がれません。

---

## 12.8 応用例：装置間差

### 場面

3〜5 装置で同じ試料を測る。**装置間差**が試料差やロット差より大きいと、Skill の**適用範囲**を装置ごとに切り分ける必要があります。

### モデル（試料 × 装置の 2 階層）

$$
y_{s,i} \sim \mathrm{Normal}(\alpha_s + \gamma_i, \sigma_{\text{obs}}), \quad \alpha_s \sim \mathrm{Normal}(\mu_\alpha, \sigma_{\text{sample}}), \quad \gamma_i \sim \mathrm{Normal}(0, \sigma_{\text{inst}})
$$

- $\gamma_i$：装置 $i$ のオフセット（sum-to-zero 制約 or 弱い prior で識別）
- $\sigma_{\text{inst}}$：装置間ばらつき

```python
with pm.Model(coords={"sample": np.arange(n_samples),
                       "inst":   np.arange(n_instruments),
                       "obs":    np.arange(len(y_obs))}) as inst_effect:
    mu_alpha     = pm.Normal("mu_alpha", mu=0.0, sigma=5.0)
    sigma_sample = pm.HalfNormal("sigma_sample", sigma=2.0)
    sigma_inst   = pm.HalfNormal("sigma_inst",   sigma=1.0)
    sigma_obs    = pm.HalfNormal("sigma_obs",    sigma=1.0)

    # 試料効果（非中心化）
    alpha_offset = pm.Normal("alpha_offset", 0.0, 1.0, dims="sample")
    alpha = pm.Deterministic("alpha",
                             mu_alpha + sigma_sample * alpha_offset, dims="sample")

    # 装置効果：sum-to-zero を明示（mu_alpha との交絡回避）
    gamma_raw       = pm.Normal("gamma_raw", 0.0, 1.0, dims="inst")
    gamma_zero_mean = gamma_raw - gamma_raw.mean()
    gamma = pm.Deterministic("gamma",
                             sigma_inst * gamma_zero_mean, dims="inst")

    y = pm.Normal("y", mu=alpha[sample_id] + gamma[inst_id],
                  sigma=sigma_obs, observed=y_obs, dims="obs")
```

> [!IMPORTANT]
> **sum-to-zero が必要な理由**：`gamma_i ~ Normal(0, sigma_inst)` だけでは、`mu_alpha` と `gamma` の実現平均が**識別不能に交絡**します（`mu_alpha + gamma_i = (mu_alpha + c) + (gamma_i - c)` が同じ尤度）。装置数 3〜5 の少 G では特に事後が発散しやすくなります。上記のように `gamma_raw - gamma_raw.mean()` で **`sum(gamma) = 0`** を強制するか、PyMC の `pm.ZeroSumNormal`（対応バージョンで）を使います。
>
> **少 G（G < 5）の警告**：装置数が少ないと `sigma_inst` は prior 依存が強くなります（§12.4 と同じ理屈）。`sensitivity_alternatives` を必須運用にします。

### 装置間差の意思決定

- $\sigma_{\text{inst}}$ の CrI が実務閾値以下 → 装置差は無視、Skill は全装置共通
- 閾値を超える → **装置ごとに別 Skill として認定**、`applicability_domain` に装置ID を明記

これは Skill の**適用範囲仕様**（vol-01 第7章 §8.5）を Bayesian で駆動する典型パターンです。

---

## 12.9 階層モデル Skill の provenance テンプレ

第11章 §11.8 の provenance に、階層モデル特有の情報を追加します。

```yaml
provenance:
  # 第11章のフィールドをすべて継承
  ...

  # 第12章で追加
  hierarchical_structure:
    levels:
      - {name: sample,     n: 20,  data_per_level: 5}
      - {name: lot,        n: 4,   data_per_level: 25}
      - {name: instrument, n: 3,   data_per_level: 33}
    structure_type: crossed             # nested / crossed / partially_crossed
    partial_pooling_targets:  [alpha, gamma]
    complete_pooling_targets: [mu_alpha]
    model_formula:
      response:         y
      likelihood:       Normal
      linear_predictor: "alpha[sample_id] + gamma[inst_id]"
      observation_dim:  obs
    index_mapping:
      sample_id: "obs -> sample"
      inst_id:   "obs -> instrument"
    zero_sum_constraints: [gamma]       # sum-to-zero を課した効果

  parameterization:
    strategy:            non_centered
    switched_from:       centered
    switch_reason:       "n_divergences=143 with centered"
    verified_via:        "prior predictive check + posterior predictive check"

  identifiability_check:
    performed:           true
    method:              "prior predictive check + external calibration data"
    concerns:            []
    reviewed_by:         "PI name"

  shrinkage_summary:
    # 標準化群偏差（各群が母平均から何 σ_α 離れているか）
    standardized_group_deviation:
      metric: "|alpha_partial[g] - mu_alpha| / sigma_alpha"
      max:    1.8
      min:    0.1
    # 縮み係数（no-pooling 推定 → partial-pooling 推定でどれだけ母平均に寄ったか）
    shrinkage_factor:
      metric: "1 - |alpha_partial[g] - mu_alpha| / |alpha_no_pooling[g] - mu_alpha|"
      max:    0.72        # データ少 → 強い縮み
      min:    0.08        # データ多 → ほぼ動かない
    n_shrunk_groups:      3    # shrinkage_factor > 0.5 の群数
    groups_with_low_ess:  []

  applicability_domain:
    instruments:         [inst-A, inst-B, inst-C]
    lot_range:           [lot-2024-01, lot-2025-12]
    sample_composition:  "半導体単結晶、n=100 まで検証済"
    exclusions:          ["装置 D は sigma_inst の CrI 外、別 Skill 認定必要"]
```

### 合成階層データ

本章の例は **リポジトリルートの `data/synthetic-hierarchy/`**（配置規約は付録A §A.1.1）の合成データを想定しています（第3章 §3.9 で予告）。実データ候補は付録 C。

- **なぜ合成か**：階層構造が制御可能で「真値」が既知、Skill が正しく回収できるかを検証できる
- **提供される階層**：試料 × 繰り返し / 試料 × ロット / 試料 × 装置 / 3 階層フル
- **意図的な難所**：小サンプル群、識別性ギリギリ、divergences を誘発する設定

第14章 Advanced Capstone では、この合成データで **階層モデル Skill の完成を認定**します。

---

## 12.10 章末ワーク

1. **反復測定モデルを 5 分で走らせる**：リポジトリルート `data/synthetic-hierarchy/repeated/` から `sample_id` と `y_obs` を読み込み、§12.3 のモデルを NumPyro で回す。`sigma_obs / sigma_sample` 比を計算し、繰り返し数の設計判断を書く
2. **ロット差の意思決定**：§12.4 のモデルで `P(sigma_lot > 0.05)` を計算し、Skill 仕様書の④成功条件に**閾値ベース確率**を書く
3. **識別性チェック**：§12.5 の測定誤差モデルで、`sigma_obs` に**強い prior**（例：`HalfNormal(0.1)`）と**弱い prior**（例：`HalfNormal(5.0)`）で走らせ、`sigma_x` の事後がどう変わるかを比較する
4. **中心化 → 非中心化への切替**：§12.6 の両バリアントで同じデータを走らせ、`n_divergences` を比較。差が出たら `switched_from` を provenance に記録
5. **装置間差の Skill 分割判断**：§12.8 のモデルで `sigma_inst` の 95% CrI を出し、閾値と比較して Skill の `applicability_domain` を決める
6. **NumPyro 高速化の確認**：同じモデルを `pytensor` と `numpyro` で走らせ、`runtime_seconds` と `max_rhat`、`min_ess_bulk` を比較する

---

## 12.11 本章のまとめ

- **3 体制**：完全プーリング / 完全非プーリング / 部分プーリング。実務のデフォルトは部分プーリング
- **反復測定・ロット差・測定誤差**は材料研究の 3 大階層シナリオ。それぞれの `sigma_*` を意思決定に接続する
- **`P(sigma_lot > threshold | data)`** のような**閾値ベース確率**を Skill 出力に含める（`uncertainty_scheme.posterior_quantity: threshold_probability`）
- **測定誤差モデルは識別性が要**。繰り返し測定 or 外部校正情報がないと解けない
- **中心化 vs 非中心化**：グループごとのデータが少ない・prior が広いときは非中心化。divergences が出たら切替、両バリアントを持ち、`switched_from` を記録。`target_accept` は 0.90 で開始し、必要に応じて 0.95 → 0.99 へ
- **NumPyro を推奨**：`jax_enable_x64` + `nuts_sampler="numpyro"` + `jax.devices()` を凍結
- **装置間差**は Skill の適用範囲決定に直結。`applicability_domain` を Bayesian で駆動
- 階層モデル Skill の provenance には **`hierarchical_structure` / `parameterization` / `identifiability_check` / `shrinkage_summary` / `applicability_domain`** を追加

---

## 参考資料

### 本書内の該当章

- 第3章 §3.9：合成階層データ `data/synthetic-hierarchy/`（リポジトリルート、付録A §A.1.1）の位置づけ
- 第5章 §5.3：`task_type: bayesian_inference`
- 第10章：`uncertainty_scheme`（`posterior_quantity: threshold_probability` を本章で活用）
- 第11章 §11.6：非階層の Bayesian（本章はその拡張）
- 第11章 §11.9：JAX/NumPyro 準備（本章のバックエンド）
- 第13章：MCMC 実務判断（divergences の扱い、reparameterization 選択）
- 第14章：Advanced Capstone（本章の階層モデル Skill を統合）
- 付録 A：provenance スキーマ拡張（本章追加フィールドの正式定義）
- 付録 C：階層データの実データ候補

### 外部参考

<a id="ref-11-1">[11-1]</a> Gelman, A., & Hill, J. (2006). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press. — 階層モデルの実務的教科書
<a id="ref-11-2">[11-2]</a> McElreath, R. (2020). *Statistical Rethinking*, 2nd ed. — partial pooling の直感的解説（Ch 13–14）
<a id="ref-11-3">[11-3]</a> Betancourt, M., & Girolami, M. (2015). Hamiltonian Monte Carlo for Hierarchical Models. In *Current Trends in Bayesian Methodology with Applications*. — funnel geometry と非中心化パラメータ化
<a id="ref-11-4">[11-4]</a> Betancourt, M. (2017). Diagnosing Biased Inference with Divergences. [https://betanalpha.github.io/assets/case_studies/divergences_and_bias.html](https://betanalpha.github.io/assets/case_studies/divergences_and_bias.html) — divergences の実務対処
<a id="ref-11-5">[11-5]</a> Stan Development Team. *Stan User's Guide: Hierarchical Models*. [https://mc-stan.org/docs/stan-users-guide/](https://mc-stan.org/docs/stan-users-guide/) — 対応する Stan 実装（付録 B 参照）
<a id="ref-11-6">[11-6]</a> NumPyro Team. *NumPyro Examples*. [https://num.pyro.ai/en/stable/examples/](https://num.pyro.ai/en/stable/examples/) — JAX バックエンドでの階層モデル実装例
