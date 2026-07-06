# 第13章　MCMC の実務と限界：判断と修正

> [!IMPORTANT]
> **本章の到達目標**
>
> - 収束診断を **合格 / 保留 / 不合格** の 3 判定に落とし込み、Skill 認定の合否を機械的に決められる
> - **打切り基準・chain 数・warmup 長**の決め方を、モデルサイズ・診断量に応じて選べる
> - **divergences のエスカレーション梯子**を持ち、発生箇所の局在化・尺度確認・非中心化などの幾何側対処を先に置き、`target_accept` 引き上げはサンプラ緩和として後段に扱える
> - **非中心化以外の reparameterization**（対数変換、標準化、シフト）を場面ごとに使い分けられる
> - **事後予測チェック**を、マージナル・条件付き・要約統計・グループ別の 4 視点で読める
> - **PyMC ↔ NumPyro のバックエンド差**が生む数値差を許容範囲として設計できる
> - Skill 認定の**実務チェックリスト**（診断 10 + provenance 4 の計 14 項目）を持ち帰れる

## 本章で扱わないこと

- **NUTS / HMC の内部理論**：直感で足りる範囲に絞る（Betancourt の参考文献に委ねる）
- **変分推論（ADVI / SVI）・EP / INLA**：本書は NUTS 中心
- **新しいモデル種類**：本章は Ch10-11 のモデルを「動かし切る」ための章
- **失敗パターン集**：第15章
- **アーキテクチャレベルの Skill 運用**：第16章

---

## 13.1 章の位置づけ

第11章と第12章で階層モデルまで組みました。**動く**モデルは書けます。しかし実運用では：

- 「$\hat{R}$ が 1.03 なのに再サンプリングしないと通らない」
- 「divergences が 3 個残る。無視していい？」
- 「PyMC で通ったのに NumPyro で結果が微妙に違う」

こうした**判断ローカルな問題**が Skill 認定を阻みます。第13章は **診断結果 → 実行決定** の橋渡しに集中します。

| 章 | 責務 | 本章との違い |
|---|---|---|
| 第11章 | どこを見るか（5 シグナル） | 見る場所を教える |
| 第12章 | 構造的な処方（非中心化） | パラメータ化を変える |
| **第13章** | **判断論** | 診断量を「合否」に翻訳 |
| 第15章 | 失敗パターン集 | 「してはいけないこと」の網羅 |

---

## 13.2 収束診断の 3 判定

Skill が使ってよいか / 再サンプリングが必要か / モデル改修が必要かを、**同じルールで**判定します。

### 判定ルール

| 判定 | 条件（すべて満たす） | 対応 |
|---|---|---|
| ✅ **合格**（Skill 実行結果として使用可） | divergences=0 & max $\hat{R}$ < 1.01 & min ESS bulk ≥ 400 & min ESS tail ≥ 400 & `reached_max_treedepth` フラグが False & BFMI ≥ 0.3 | provenance に `diagnostics_pass: true` を記録して使用 |
| ⚠️ **保留**（再サンプリング） | 1 以上 divergences（非ゼロは無条件で保留以下）, $\hat{R}$ が 1.01 をわずかに超える程度, または ESS が 400 前後で不安定 | サンプリング設定変更（§13.3、§13.4）→ 再フィット |
| ❌ **不合格**（モデル改修） | divergences が再フィットしても消えない, $\hat{R}$ が 1.01 を大きく超える or 再フィット後も持続, ESS < 100, BFMI < 0.3 が持続 | 再パラメータ化・prior 見直し・データ確認（§13.4、§13.5） |

> [!IMPORTANT]
> **$\hat{R}$ の運用境界は 1.01**（Vehtari 2021）：1.01 を超えた時点で合格ではありません。ここでは「1.01 をわずかに超える」までを保留、大きく超える／再フィット後も持続する場合を不合格として扱います。「1.05 まで許容」といった古い基準は Skill 認定では採用しません。

> [!IMPORTANT]
> **「合格まで再フィット」を無限に許容しない**：3 回再フィットしても合格しないケースは、**サンプリング設定では解けない問題**です。§13.4 の梯子でモデル側の修正に進みます。Skill 仕様書 ⑤（禁止事項）に「診断合格まで乱数を変えて再フィットし続ける」を必ず入れます。

### `diagnostics_pass` の自動判定コード

```python
def check_diagnostics(idata, thresholds=None):
    """
    診断量を機械的に集約して diagnostics_pass を返す。
    返り値は Ch10 の diagnostics_summary の「診断コア」部分に相当し、
    provenance 側で sampler_config・実行環境・runtime を別途付与して統合する。
    """
    import arviz as az
    th = thresholds or dict(
        max_rhat=1.01, min_ess_bulk=400, min_ess_tail=400,
        max_divergences=0, min_bfmi=0.3, max_treedepth=10,
    )
    # round_to=None でバージョン依存の丸めを避け、原値で比較
    summary = az.summary(idata, round_to=None)

    ss = idata.sample_stats
    n_diverge = int(ss["diverging"].sum().item())
    bfmi_min  = float(az.bfmi(idata).min())

    # tree depth は 2 系統をどちらでも拾えるようにする
    if "reached_max_treedepth" in ss:
        reached_max_treedepth = bool(ss["reached_max_treedepth"].any().item())
        max_tree_depth = None
    elif "tree_depth" in ss:
        max_tree_depth = int(ss["tree_depth"].max().item())
        reached_max_treedepth = max_tree_depth >= th["max_treedepth"]
    else:
        max_tree_depth = None
        reached_max_treedepth = False

    checks = dict(
        divergences_ok = n_diverge <= th["max_divergences"],
        rhat_ok        = float(summary["r_hat"].max())    <= th["max_rhat"],
        ess_bulk_ok    = float(summary["ess_bulk"].min()) >= th["min_ess_bulk"],
        ess_tail_ok    = float(summary["ess_tail"].min()) >= th["min_ess_tail"],
        bfmi_ok        = bfmi_min                          >= th["min_bfmi"],
        treedepth_ok   = not reached_max_treedepth,
    )
    return {
        "diagnostics_pass": all(checks.values()),
        "checks":           checks,
        "n_divergences":    n_diverge,
        "max_rhat":         float(summary["r_hat"].max()),
        "min_ess_bulk":     float(summary["ess_bulk"].min()),
        "min_ess_tail":     float(summary["ess_tail"].min()),
        "bfmi_min":         bfmi_min,
        "max_tree_depth":   max_tree_depth,
        "reached_max_treedepth": reached_max_treedepth,
        "thresholds":       th,
    }
```

**この関数は診断コアを返します**。エージェントに合否判断を委ねません。ただし Ch10 の `diagnostics_summary` にはこれ以外に `sampler`, `chains`, `draws_per_chain`, `tune_per_chain`, `target_accept`, `runtime_sec`, `random_seed`, `pymc_version`, `arviz_version` 等の実行メタデータが必要です。provenance を書く側で本関数の返り値にそれらをマージします。

> [!NOTE]
> ArviZ の `az.ess(idata)` は xarray オブジェクトを返すため、単純な `.min()` はスカラーになりません。上のコードでは `az.summary(...)` の `ess_bulk`/`ess_tail`（pandas Series）から `.min()` を取ることで、`float` 化を安全にしています。BFMI（別名 E-BFMI）は `az.bfmi(idata)` が chain 別 numpy 配列を返すため `.min()` で最悪 chain の値になります。

---

## 13.3 打切り基準・chain 数・warmup の決め方

### chain 数

| モデルの複雑さ | 推奨 chain 数 |
|---|---|
| 非階層・パラメータ 10 個未満 | 4 |
| 階層 1 層・パラメータ 10-100 | 4-8 |
| 階層 2+ 層・パラメータ 100+ | **8+** |
| Skill 認定用の最終検証 | **8**（chain 分散を検出しやすい） |

- **4 未満は `$\hat{R}$` の意味が弱くなる**（chain 内・chain 間分散比が信用できない）
- **並列実行のコスト差は少ない**（NumPyro なら chain 並列は無料に近い）

### warmup（tune）長

| 状況 | 推奨 tune |
|---|---|
| 非階層・弱情報 prior | 1000（PyMC デフォルト） |
| 階層モデル | 2000-3000 |
| divergences が残る | 5000 まで引き上げ |
| tree depth が張り付く | tune 引き上げ + `max_treedepth=15` |

**警告**：`tune` を増やすのは **step size 適応が不十分**なときの一次対応。**モデル構造の問題は tune では治りません**。

### draws（本サンプリング）

- 分位点・CrI 幅の推定：**ESS 400 以上**を目安（min ESS bulk / tail）
- 極値・rare event の推定：ESS 1000-4000 以上
- 「draws 数」ではなく「達成された ESS」で判断（draws 2000 でも自己相関が高ければ ESS 200）

### `random_seed` の運用

```python
# 完全再現性のため chain ごとに seed を分岐
idata = pm.sample(
    draws=1000, tune=2000, chains=8,
    random_seed=[42, 43, 44, 45, 46, 47, 48, 49],
    nuts_sampler="numpyro",
    target_accept=0.90,
)
```

**Skill 認定の合否判定**は、seed を変えて 2〜3 回走らせても同じ判定になることを確認します（`diagnostics_pass` の seed ロバスト性）。

---

## 13.4 divergences のエスカレーション梯子

**divergences は「事後の探索に失敗した」サイン**です。ゼロを目指し、**次の順序で対処**します。「まず `target_accept` を上げる」から始めるとモデル幾何の問題を隠しがちなので、**先に発生箇所と入力尺度を確認**します。

| 段 | 対処 | 例 | 効果の目安 |
|---|---|---|---|
| 1 | **発生箇所の局在化** | `az.plot_pair` / `az.plot_energy` で divergences が集中するパラメータを特定 | 幾何 or 尺度 or 識別性かを切り分け |
| 2 | **prior predictive と入力尺度の確認** | prior predictive が突飛でないか、入力の scale がパラメータ間で桁違いでないか | データ側・尺度側の単純ミスを排除 |
| 3 | **入力データを標準化** | `x → (x - mean) / std` | 尺度不一致による探索困難 |
| 4 | **非中心化パラメータ化** | 中心化 → 非中心化 | funnel geometry の直接対処（第12章 §12.6）。階層モデルは基本ここに来る |
| 5 | 対数・対数正規スケール | `y → log(y)` | 正値・right-skewed 変数 |
| 6 | `target_accept` を上げる（**サンプラ緩和**） | 0.90 → 0.95 → 0.99 | 幾何をいじれない場面の最終手段的な緩和 |
| 7 | `tune` を長くする | 1000 → 2000 → 5000 | step size 適応が甘い場合 |
| 8 | `max_treedepth` を上げる | 10 → 15 | tree depth 張り付きの併発時のみ |
| 9 | **prior を引き締める** | 弱情報 → 情報寄り（科学的な根拠がある場合のみ） | 事後が広すぎで漏斗発生時 |
| 10 | **モデル構造の簡略化** | 階層を減らす、識別性の悪い項を落とす | 構造的な識別性欠如 |
| 11 | データを増やす / 群を統合 | 少 G を統合 | 情報量不足の解決 |

> [!IMPORTANT]
> **段 6-8 は「サンプラ設定の緩和」**：`target_accept=0.99` に張り付けたまま合格させるのは、幾何の問題を隠すだけで根治しません。段 6-8 に落ち着いた場合も、**なぜ段 1-5 で解決しなかったか**を provenance にメモします。
>
> **Skill バージョンの上げ方**：
> - `target_accept`, `tune`, `max_treedepth` の変更 → `sampler_config` パッチ（Skill バージョン据え置き可）
> - 標準化・非中心化のように**統計モデルが数学的に等価**な実装変更 → 実装パッチ（マイナーバージョンで記録）
> - Prior 変更・尤度変更・構造簡略化・群統合 → **モデル仕様マイナーバージョン**（`skill_version: 1.0 → 1.1`）
>
> 「動かない Skill を数字だけ上げる」を防ぐため、`sampler_config` の変更だけでバージョンを上げるのは避けます。

### 「消えない divergences」の見分け方

- 段 1〜3 で解決 → データ・尺度側の問題
- 段 4〜5 で残る → 識別性 or データ不足（段 10〜11）
- **段 10 で残る → モデルが問題設定を反映できていない**（ドメイン専門家に戻る）

---

## 13.5 非中心化以外の reparameterization

第12章 §12.6 は funnel 対策の非中心化を扱いました。ここでは**それ以外のよくあるパターン**を並べます。

### パターン 1：対数変換（正値・right-skewed 変数）

**症状**：正値パラメータ（濃度、速度定数、強度）で分布が右裾を引く／divergences が右側で頻発。

**対処**：

```python
# Before: 発散しやすい
rate = pm.HalfNormal("rate", sigma=10.0)
# After: 対数正規で自然なスケール
log_rate = pm.Normal("log_rate", mu=0.0, sigma=1.0)
rate     = pm.Deterministic("rate", pm.math.exp(log_rate))
```

または尤度側で `LogNormal` を直接使う。第12章 §12.5 の TIP を発展させたもの。

### パターン 2：入力の標準化（多スケール）

**症状**：温度（100-1000 K）と圧力（0-10 MPa）を同じモデルに入れて divergences。

**対処**：

```python
x_std = (x_raw - x_raw.mean()) / x_raw.std()
# モデル内では x_std を使い、事後解釈時に元スケールに戻す
```

**注意**：Skill として運用するときは、`x_raw.mean()` と `x_raw.std()` をフィット時の値で凍結し、予測時に同じ変換を適用します（vol-01 の contract 章の再掲）。

### パターン 3：切片オフセット（中心化）

**症状**：切片と傾きが強く相関し、Trace plot でも両者が同時に揺れる。

**対処**：

```python
x_c = x - x.mean()   # 平均で中心化するだけ
mu  = alpha + beta * x_c
```

これで `alpha` と `beta` の事後相関が下がり、収束が速くなります。

### パターン 4：Cholesky 分解された多変量 prior

多変量ランダム効果（相関ありの複数群パラメータ）が扱いにくいときは、共分散行列の Cholesky 因子で prior を張ります。**本書ではスコープ外**（付録の参考文献参照）ですが、第14章 Advanced Capstone で必要になる場合に備えて存在だけ覚えておきます。

### パターン選択表

| 症状 | パターン |
|---|---|
| Funnel、階層モデル | 非中心化（第12章 §12.6） |
| 正値・右裾 | 対数変換（パターン 1） |
| 多スケール入力 | 標準化（パターン 2） |
| 切片×傾きの強相関 | 切片オフセット（パターン 3） |
| 相関ある多変量ランダム効果 | Cholesky（付録 B） |

---

## 13.6 事後予測チェック（PPC）の 4 視点

第11章 §11.5 は「PPC は必要条件・十分条件ではない」を示しました。**具体的に何を見るか**を 4 視点で整理します。

### 視点 1：マージナル分布

```python
az.plot_ppc(idata, num_pp_samples=100)
```

観測 $y$ の分布と事後予測分布を重ねる。**目視で 3 点**：

- ピーク位置
- 裾の伸び方
- 外れ値の存在感

### 視点 2：要約統計

```python
az.plot_bpv(idata, kind="t_stat", t_stat="mean")
az.plot_bpv(idata, kind="t_stat", t_stat="std")
az.plot_bpv(idata, kind="p_value")
```

観測の平均・標準偏差を Bayesian p-value で診断。**モデルが要約統計を再現できるか**を数値化。

> [!NOTE]
> `az.plot_bpv` はバージョンごとに配置・引数名が変わるため、Skill 認定では `arviz_version` を `backend_config` に固定して記録します（第11章 §11.9）。呼び出し時に `NotImplementedError` が出た場合は、当該 ArviZ バージョンにピンし直し、リリースノートを確認します。

### 視点 3：条件付き分布（サブグループ別）

**階層モデルで必須**：全体は合っても、特定ロット・特定装置で外れることがあります。観測次元（例：`obs`）に沿って群のインデックスを絞り、`plot_ppc` の `coords` で渡します。

```python
import numpy as np

# 装置 0 に該当する観測インデックス（inst_id は fit 時に使った長さ N の配列）
obs_idx = np.where(inst_id == 0)[0]

az.plot_ppc(
    idata,
    var_names=["y"],
    coords={"obs": obs_idx},    # モデル定義時の観測次元名に合わせる
    num_pp_samples=100,
)

# あるいは手動で取り出して比較
y_pp_g0 = idata.posterior_predictive["y"].isel(obs=obs_idx)
y_ob_g0 = idata.observed_data["y"].isel(obs=obs_idx)
```

**全体一致・部分不整合**を検出するのが階層モデル PPC の主目的です。観測次元名（`obs` / `sample` など）はモデル定義側の `dims` に合わせます。

### 視点 4：残差 vs 予測値・vs 入力

```python
# ArviZ に頼らず簡潔に
y_pred_mean = idata.posterior_predictive["y"].mean(dim=["chain", "draw"]).values
residuals   = y_obs - y_pred_mean

# scatter: residuals vs y_pred_mean
```

- 残差にパターン（U 字、フラリング） → **モデルが取り逃しているシグナル**
- 残差の分散が非一定（heteroscedasticity） → 誤差モデルの見直し

> [!TIP]
> **平均残差は Gaussian 尤度想定**：LogNormal・カウント・heavy-tailed 尤度では、posterior predictive の **median** や PIT / quantile residual の併用が安定します。Gaussian 以外を尤度に使うモデルでは、`y_pred_median = idata.posterior_predictive["y"].median(dim=["chain", "draw"])` を主に使い、平均残差は補助的に扱います。

### PPC 合否ルール

| 視点 | 合格 | 保留 | 不合格 |
|---|---|---|---|
| マージナル | 分布形が概ね一致 | 裾が一部乖離 | ピーク位置が明確にズレ |
| 要約統計 | Bayesian p 値が 0.1-0.9 | 0.05-0.1 or 0.9-0.95 | 0-0.05 or 0.95-1 |
| 条件付き | 全グループで一致 | 少数群で乖離 | 一貫した群依存の外れ |
| 残差 | ランダム散布 | 弱いパターン | 明確な系統偏差 |

Skill 認定は **すべての視点で「合格 or 保留」**を最低ラインにします。

---

## 13.7 PyMC ↔ NumPyro のバックエンド差

同じモデル・同じ seed でも、**バックエンドで数値がわずかに異なる**のは正常です。原因は：

- JIT コンパイル後の演算順序
- 乱数生成器の内部実装差
- 浮動小数点の丸め順

### 許容範囲の設計

```yaml
backend_reproducibility:
  reference_backend:      numpyro
  tolerance:
    # posterior 事後標準偏差で正規化した差（単位に依存しない）
    posterior_mean_diff_sd_units: 0.1     # |mean_a - mean_b| / posterior_sd < 0.1
    cri_endpoint_diff_sd_units:   0.2     # |q_a - q_b| / posterior_sd < 0.2
    # 上が定義できない場合のフォールバック（例：分散がほぼ 0 のパラメータ）
    relative_diff_fallback:       0.01    # |a - b| / max(|a|, |b|)
    absolute_diff_floor:          1.0e-8  # 数値ノイズの吸収下限
  cross_backend_tested:   true
  cpu_vs_gpu_tested:      false           # GPU は本 Skill では認定範囲外
```

**Skill 認定の最終検証は CPU + 参照バックエンド 1 つ**（第11章 §11.9 の再掲）。事後標準偏差で正規化するのは、材料・ナノテクの量が桁違いに広がる（濃度・エネルギー・時間定数）ためで、生の絶対誤差 0.01 は単位・尺度によって過大にも過少にもなります。

### バックエンド差が「異常」なとき

| 観察 | 診断 | 対応 |
|---|---|---|
| 事後平均が 5% 以上ズレる | モデルの識別性・数値精度の問題 | `jax_enable_x64=True` を確認、prior を引き締め、標準化を検討 |
| どちらか片方だけ divergences 大量 | JIT で暗黙の再パラメータ化が破綻 | fallback を pytensor に切替、原因調査 |
| CPU/GPU で 10%+ 差 | 数値精度・並列演算差 | Skill 認定を CPU に限定、GPU は開発時のみ |

---

## 13.8 診断ダッシュボード

Skill 実行の**単一ファイル出力**（Markdown or HTML）に、次を必ず含めます。

| 領域 | 内容 |
|---|---|
| メタ情報 | `random_seed`, backend, chains, draws, tune, target_accept, runtime |
| 3 判定 | ✅ / ⚠️ / ❌ とその理由 |
| 収束サマリ | `az.summary(idata)` のうち $\hat{R}$, ESS bulk, ESS tail, mcse_mean |
| Trace plot | 主要パラメータのみ |
| PPC 4 視点 | マージナル・要約統計・条件付き・残差 |
| Sensitivity | main / weaker / stronger 3 分岐の並記 |
| provenance | 第11章＋第12章のフィールドすべて |

**このダッシュボードが出力されない Skill は「診断が終わっていない」扱い**にします。

---

## 13.9 実務チェックリスト（Skill 認定合否）

Skill 認定前に、次の 14 項目を機械的に確認します。**すべて ✅ でなければ認定不可**。項目 1–10 は「診断が終わっているか」、項目 11–14 は「provenance が Ch9–12 の要件を満たしているか」を見ます。

### 診断（10 項目）

1. ☐ `random_seed` を chain ごとに分岐、複数 seed で `diagnostics_pass` が安定
2. ☐ chain 数がモデル複雑さに対して適切（階層は 8+）
3. ☐ `divergences = 0`（非ゼロは認定不可。**「無視できる」判断で通さない**。§13.4 の梯子でモデル・尺度側に戻る）
4. ☐ max $\hat{R}$ < 1.01
5. ☐ min ESS bulk ≥ 400 かつ min ESS tail ≥ 400
6. ☐ `reached_max_treedepth` フラグが False（tree depth 張り付きなし）
7. ☐ BFMI（別名 E-BFMI）≥ 0.3
8. ☐ PPC 4 視点すべてで「合格 or 保留」
9. ☐ Prior sensitivity（main/weaker/stronger）で意思決定が変わらない
10. ☐ 診断ダッシュボードが出力されている

### provenance（4 項目）

11. ☐ 第10章：`uncertainty_scheme` / `prior_specification` が記録済み
12. ☐ 第11章：`sampler_config` / `backend_config` / `posterior_artifact` / `diagnostics_summary`（`check_diagnostics()` 返り値 + 実行メタデータ）が記録済み
13. ☐ 階層モデルの場合：第12章の `hierarchical_structure` / `parameterization` / `identifiability_check` / `shrinkage_summary` / `applicability_domain` が記録済み
14. ☐ 第13章：`backend_reproducibility`（tolerance を sd 正規化で定義）が記録済み

> [!TIP]
> **14 項目をコードで自動化**：`check_diagnostics()` を拡張し、項目 1–10 のブール配列と `all()` を返す関数にします。項目 11–14 は provenance YAML のキー存在チェックで自動化できます。Skill 実行の末尾で必ず呼び、`certification_pass: true/false` を provenance に記録します。

---

## 13.10 章末ワーク

1. **`check_diagnostics()` を実装し、3 モデルに適用**：第11章の校正曲線・第12章のロット差・装置間差モデルに適用、YAML 化する
2. **divergences 梯子を実験**：第12章の中心化モデルで target_accept を 0.80 → 0.99 に変えながら divergences 数を記録、どの段で消えたかをレポート
3. **標準化の効果測定**：多スケール入力データを、標準化前後で走らせ、ESS bulk / 実行時間を比較
4. **PPC 4 視点を階層モデルで実施**：装置間差モデルに 4 視点すべての PPC を出し、装置別 PPC で外れ群があるか確認
5. **バックエンド差の測定**：同じ seed で pytensor と numpyro を走らせ、事後平均と CrI 端点の差を記録、`tolerance` を経験的に決める
6. **チェックリストを自作 Skill に適用**：既存の Bayesian Skill 1 つに 14 項目（診断 10 + provenance 4）を通し、⚠️/❌ を列挙して修正

---

## 13.11 本章のまとめ

- 収束診断は **3 判定**（合格 / 保留 / 不合格）で運用。エージェントに合否を委ねない
- **chain 数はモデル複雑さで決める**：階層 2+ 層なら 8+ chains
- **warmup は step size 適応の話**、モデル構造の問題は tune で解けない
- **divergences のエスカレーション梯子**（局在化 → 尺度 → 幾何 → サンプラ緩和 → prior → 構造 → データ）を持ち、prior 以降はモデル仕様バージョンを上げる
- **非中心化以外の reparameterization**：対数変換・標準化・切片オフセットを場面で使い分け
- PPC は **4 視点**（マージナル・要約統計・条件付き・残差）で読む。階層モデルは条件付き（グループ別）が要
- バックエンド差は正常、**tolerance を設計**し、認定は CPU + 1 バックエンドに絞る
- **10 項目チェックリスト**を Skill 認定の合否条件にする（診断 10 + provenance 4 = 計 14 項目）

---

## 参考資料

### 本書内の該当章

- 第11章 §11.4：5 シグナル収束診断（本章はその判断論）
- 第11章 §11.5：PPC の必要条件・十分条件（本章はその実装）
- 第11章 §11.9：バックエンド設定（本章 §13.7 の前提）
- 第12章 §12.6：非中心化パラメータ化（本章 §13.5 の派生元）
- 第14章：Advanced Capstone（本章の 10 項目チェックリストを適用）
- 第15章：失敗パターン（本章の判断論を破ったとき何が起きるか）
- 付録 A：`diagnostics_pass`, `diagnostics_summary` の正式定義
- 付録 B：PyMC ↔ Stan 対応表

### 外部参考

<a id="ref-12-1">[12-1]</a> Betancourt, M. (2017). Diagnosing Biased Inference with Divergences. [https://betanalpha.github.io/assets/case_studies/divergences_and_bias.html](https://betanalpha.github.io/assets/case_studies/divergences_and_bias.html) — divergences の実務対処（本章 §13.4 の主要出典）
<a id="ref-12-2">[12-2]</a> Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-Normalization, Folding, and Localization: An Improved $\hat{R}$ for Assessing Convergence of MCMC. *Bayesian Analysis*, 16(2), 667–718. — $\hat{R}$ 1.01 閾値の根拠
<a id="ref-12-3">[12-3]</a> Gabry, J., Simpson, D., Vehtari, A., Betancourt, M., & Gelman, A. (2019). Visualization in Bayesian workflow. *JRSS Series A*, 182(2), 389–402. — PPC 4 視点の背景
<a id="ref-12-4">[12-4]</a> Betancourt, M. (2020). Towards a Principled Bayesian Workflow. [https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html) — 診断 → 修正 → 再診断のワークフロー
<a id="ref-12-5">[12-5]</a> Stan Development Team. *Stan Reference Manual: Diagnostics*. [https://mc-stan.org/docs/reference-manual/](https://mc-stan.org/docs/reference-manual/) — 診断量の正式定義（付録 B と対応）
<a id="ref-12-6">[12-6]</a> ArviZ Contributors. *ArviZ API: diagnostic functions*. [https://python.arviz.org/en/stable/api/diagnostics.html](https://python.arviz.org/en/stable/api/diagnostics.html) — `az.rhat`, `az.ess`, `az.bfmi` の API
