# 第13章　総合ハンズオン（Advanced Capstone）：sklearn → PyMC → 階層モデル

> [!IMPORTANT]
> **本章の到達目標**
>
> - **実データ（MatBench または RRUFF）**で sklearn / PyMC の基本層 Skill を組み、実データ特有の前処理・分割・診断まで通す
> - **合成階層データ**で階層モデル capstone Skill を完成させ、装置差・ロット差を分離して推定する
> - **3 層の Skill**（sklearn 回帰 → PyMC 校正 → 階層モデル）が **同じ provenance 系列でチェイン**して動くことを確認する
> - 第4-12章で導入した provenance 拡張（`cv_scheme` / `sampler_config` / `backend_config` / `posterior_artifact` / `diagnostics_summary` / `hierarchical_structure` / `parameterization` / `identifiability_check` / `shrinkage_summary` / `applicability_domain` / `backend_reproducibility`）を**すべてひとつのアーティファクトに集約**する
> - 第12章の **14 項目チェックリスト**を capstone に適用し、機械的に合否判定できる

## 本章で扱わないこと

- **新しいモデル種類**：本章は Ch5-12 の統合。新規手法は導入しない
- **深層学習・生成モデル**：vol-03 以降
- **実データの階層構造分析**：ライセンス・階層メタの入手困難ゆえに**合成データで扱う**。実データ候補の入手先は付録 C
- **本番運用のインフラ**：第15章

---

## 13.1 章の位置づけ：3 層の capstone

第4章から第12章まで、Skill の**部品**を順に組んできました。本章は**組み立て**です。

| 層 | データ | Skill | 主要章 | 出力の使い道 |
|---|---|---|---|---|
| Layer 1 | 実データ（MatBench または RRUFF） | **sklearn 回帰** or **PLS** | 第5-8章 | 物性の点予測 + 交差検証済み評価指標 |
| Layer 2 | 実データ + Layer 1 の予測 | **PyMC 校正曲線**（点予測 → 不確実性つき） | 第9-10章 | 予測値の事後分布 + CrI |
| Layer 3 | **合成階層データ** | **階層モデル**（装置差・ロット差の分離） | 第11-12章 | 階層構造のパラメータ推定 + PPC 診断 |

> [!NOTE]
> **実データはこう、階層構造は合成で学ぶ**：ARIM 相当の階層構造（装置×ロット×実験室）を明示的に持つ公開データは稀です。本章は Layer 1-2 で実データの前処理・診断の勘所を掴み、Layer 3 で合成階層データを使って階層モデル設計を安全に練習します。実データで階層構造を扱う場合の候補は付録 C にまとめます。

### 3 層が provenance で繋がる意味

各層の出力は**次層の入力**です：

```
実データ → Layer 1 → predicted_y  ─┐
                                   ├─→ Layer 2 → posterior_y ─┐
真値 y  ────────────────────────────┘                          ├─→ Layer 3 → 階層効果
                                                                │
合成階層データ ────────────────────────────────────────────────┘
```

各層の **`input_sha256` と `posterior_artifact` を次層に埋め込む**ことで、「どの Skill の出力を使ったか」がハッシュで追跡できます。これを本章で実装します。

---

## 13.2 データ準備

### 実データ層：MatBench または RRUFF

本章は次のどちらか一方を選びます。**両方やる必要はありません**。

| データ | 主用途 | サイズの目安 | 難所 |
|---|---|---|---|
| **MatBench**（`matbench_expt_gap` など） | 物性値回帰 | 4,000〜10,000 点 | ライセンス確認、単位、外れ値 |
| **RRUFF**（Raman/IR スペクトル） | スペクトル → 物質同定 or ピーク特性回帰 | 数百〜数千スペクトル | 波数軸の統一、ベースライン補正、正規化 |

> [!TIP]
> **どちらを選ぶか**：物性値の scalar 回帰から入りたければ MatBench、スペクトル前処理も含めて練習したければ RRUFF。以降のコード例は MatBench の `matbench_expt_gap` を仮定します（実測バンドギャップ回帰）。RRUFF を選んだ場合は、Layer 1 の `X` に PLS 用のスペクトル行列を、`y` にピーク面積などの物性ラベルを差し込む形で読み替えてください。

### データ契約と 3 分割（train / calibration / test）

Layer 2 はベイズ校正で「Layer 1 の予測 → 真値」を推定します。この校正モデルを **test set で fit すると評価用データが漏洩**します。そのため本章では 3 分割を採用します：

| 分割 | 用途 |
|---|---|
| `train` | Layer 1（sklearn）を fit |
| `calibration` | Layer 2（PyMC 校正曲線）を fit |
| `test` | 3 層すべてを一切触らない**最終評価専用** |

第4章の anti-leakage 契約は「fit 前に凍結」を要求します。ここでは 3 分割それぞれのハッシュを凍結します：

```python
from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True,
)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=43, shuffle=True,  # 0.25 * 0.8 = 0.2
)

data_split = dict(
    method="random_three_way",
    fractions={"train": 0.6, "calibration": 0.2, "test": 0.2},
    random_states=[42, 43],
    train_sha256=sha256_of(X_train, y_train),
    calib_sha256=sha256_of(X_calib, y_calib),
    test_sha256 =sha256_of(X_test,  y_test),
)
```

> [!IMPORTANT]
> **Layer 2 は calibration set で fit、test set は最後まで触らない**：これは **training-time calibration** の設計です。deployment 段階で新しい観測が来たときに再校正する **deployment recalibration** の運用は別で、その場合は運用中の別 hold-out を用意します。本章では training-time calibration を採用します。

### 合成階層データ層

Layer 3 用の合成階層データセット（`data/synthetic-hierarchy/`、本書 2 章 §2.9 参照）を使います。想定構造は 3 階層：

- **実験室** `lab_id` ∈ {0, 1, 2}（3 lab）
- **装置** `inst_id` ∈ 各 lab に 2〜3 台
- **ロット** `lot_id` ∈ 各実験×装置あたり 5〜10 lot
- **反復** 1 lot につき 3〜5 反復測定

真のパラメータは既知（合成データなので）で、**Layer 3 の推定が真値に戻ってくるか**を確認できます。合成データ生成スクリプトは付録 A に置きます（本章では読み込むだけ）。

---

## 13.3 Layer 1：sklearn 回帰スキル

第4-8章の統合。**pipeline + CV + evaluate まで一気通貫**、と provenance 記録。

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  Ridge(alpha=1.0, random_state=42)),
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
cv_rmse_mean = -scores.mean()
cv_rmse_std  = scores.std()

pipe.fit(X_train, y_train)
y_pred_calib = pipe.predict(X_calib)   # Layer 2 へ渡す
y_pred_test  = pipe.predict(X_test)    # 最終評価まで開封しない
```

### Layer 1 の出力ハッシュ（次層 upstream 用）

```python
import hashlib, numpy as np
def sha256_of_array(*arrs):
    h = hashlib.sha256()
    for a in arrs: h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

y_pred_calib_sha256 = sha256_of_array(y_pred_calib)
y_pred_test_sha256  = sha256_of_array(y_pred_test)
```

### Layer 1 の provenance（Ch4-8 の集約）

```yaml
layer_1_regression:
  skill_id:      sklearn-regression-matbench-expt-gap
  skill_version: 1.0.0
  input_sha256:  <hash of X_train,y_train>
  data_split:    {method: random_three_way, fractions: {train: 0.6, calibration: 0.2, test: 0.2}, random_states: [42, 43], train_sha256: ..., calib_sha256: ..., test_sha256: ...}
  cv_scheme:     {type: KFold, n_splits: 5, shuffle: true, random_state: 42, group_key: null}
  model_config:  {estimator: Ridge, alpha: 1.0, scaler: StandardScaler}
  metrics:       {cv_rmse_mean: <value>, cv_rmse_std: <value>}   # test_* は最終評価まで空
  output_artifacts:
    y_pred_calib: {split: calibration, sha256: <y_pred_calib_sha256>, shape: [<n_calib>], dtype: float64}
    y_pred_test:  {split: test,        sha256: <y_pred_test_sha256>,  shape: [<n_test>],  dtype: float64}
    model_sha256: <fit 済み pipeline artifact hash>
  applicability_domain: {feature_ranges: {...}, y_range: [<min>,<max>]}
  package_versions:     {sklearn: "1.5.x", numpy: "1.26.x", ...}
  random_seed:          42
  run_datetime_utc:     ...
```

> [!IMPORTANT]
> **Layer 1 の失敗パターン**：`cross_val_score` を pipeline **外の**スケーリングと組み合わせるとリーク（第14章で詳述）。ここでは `Pipeline` を作って CV に渡すことで自動的に fold 内で scaler が fit されます。また、Layer 2 で使うのは `y_pred_calib` のみ、`y_pred_test` は最終評価まで**開封しない**（実装上は変数として保持しますが、Skill の出力ファイルとしては test 系を暗号化してもよい）。

---

## 13.4 Layer 2：PyMC 校正曲線スキル

Layer 1 は**点予測**しか出しません。実験判断（合否・出荷）には**予測不確実性**が要ります。Layer 2 は Layer 1 の **calibration split の予測** `y_pred_calib` を「観測」として、真値 `y_calib` との関係をベイズ校正します（第10章 §10.6 の再掲）。**test set は最後まで触りません**。

```python
import numpy as np, pymc as pm

coords = {"sample": np.arange(len(y_calib))}

with pm.Model(coords=coords) as calib:
    x_pred = pm.Data("x_pred", y_pred_calib, dims="sample")   # Layer 1 の calibration 予測
    alpha  = pm.Normal("alpha", 0.0, 1.0)
    beta   = pm.Normal("beta",  1.0, 0.5)                     # 理想は 1
    sigma  = pm.HalfNormal("sigma", 1.0)
    mu     = pm.Deterministic("mu", alpha + beta * x_pred, dims="sample")
    y_obs  = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_calib, dims="sample")
    idata = pm.sample(
        draws=1000, tune=2000, chains=8,
        random_seed=list(range(42, 50)),
        target_accept=0.90,
        nuts_sampler="numpyro",
    )
    pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=42)
```

> [!TIP]
> **新しい予測に差し替える**：Skill をデプロイして新しい `y_pred_new` に対する事後分布を得るときは、`pm.set_data({"x_pred": y_pred_new}, coords={"sample": np.arange(len(y_pred_new))})` で入れ替え、`pm.sample_posterior_predictive` を呼び出します（第10章 §10.6 の out-of-sample パターン）。

### Layer 2 の診断（第12章 `check_diagnostics()` を適用）

```python
diag = check_diagnostics(idata)     # 第12章 §12.2
assert diag["diagnostics_pass"], f"L2 診断不合格: {diag['checks']}"
```

### Layer 2 の provenance（Ch9-10 の集約 + upstream 接続）

```yaml
layer_2_calibration:
  skill_id:      pymc-calibration-linear
  skill_version: 1.0.0
  inputs:
    y_pred:
      upstream_layer:  layer_1
      artifact_name:   y_pred_calib
      sha256:          <Layer 1 の output_artifacts.y_pred_calib.sha256 と一致>
    y_true:
      sha256:          <hash of y_calib>
      split:           calibration
  input_sha256:    <hash of (y_pred_calib || y_calib)>
  uncertainty_scheme: bayesian_inference
  prior_specification:
    alpha: {dist: Normal, mu: 0, sigma: 1, rationale: "校正切片は 0 中心弱情報"}
    beta:  {dist: Normal, mu: 1, sigma: 0.5, rationale: "理想は 1、0.5 で誤校正の余地"}
    sigma: {dist: HalfNormal, sigma: 1, rationale: "残差は測定単位で 1 スケール想定"}
  sampler_config:  {draws: 1000, tune: 2000, chains: 8, target_accept: 0.90, nuts_sampler: numpyro, random_seed: [42..49]}
  backend_config:  {pymc_version: "5.x", numpyro_version: "0.x", arviz_version: "0.17.x", jax_x64: true}
  posterior_artifact: {netcdf_path: ..., sha256: ...}
  diagnostics_summary: <check_diagnostics() の返り値 + runtime/versions>
  backend_reproducibility: {reference_backend: numpyro, tolerance: {posterior_mean_diff_sd_units: 0.1, cri_endpoint_diff_sd_units: 0.2}, cross_backend_tested: true}
```

> [!TIP]
> **inverse calibration も同じモデル**：Skill 実行時に「新しい観測 x を得たとき、真の y の CrI は？」を求めるには、第10章 §10.6 の inverse 用モデルを追加で組みます。本章ではスコープ簡素化のため forward だけを示します。

---

## 13.5 Layer 3：階層モデル capstone

合成階層データで、Ch11 の階層モデル + Ch12 の診断を統合します。

### モデル設計（Ch11 §11.8 の 3 階層版）

`lab_eff` と `inst_eff` の両方に **sum-to-zero 制約**を入れ、`mu_grand` と交絡しないようにします。`lot_eff` は装置内でランダム効果として非中心化のみ（ロット単位の平均は `inst_eff` に吸収される想定で、ロット内の変動だけを推定）。

```python
import pymc as pm, numpy as np

n_lab, n_inst, n_lot = 3, 8, 60      # 装置総数 8（lab をまたぐ）、ロット総数 60
lab_of_inst = np.array([...])         # 装置 -> lab の写像 (len n_inst)
inst_of_obs = np.array([...])         # 観測 -> 装置 (len N)
lot_of_obs  = np.array([...])         # 観測 -> ロット (len N)
lab_of_obs  = lab_of_inst[inst_of_obs]  # 観測 -> lab（読みやすさのため事前計算）
y_obs       = np.array([...])         # 観測値 (len N)

coords = {
    "lab":    np.arange(n_lab),
    "inst":   np.arange(n_inst),
    "lot":    np.arange(n_lot),
    "sample": np.arange(len(y_obs)),
}

with pm.Model(coords=coords) as capstone:
    mu_grand  = pm.Normal("mu_grand", 0.0, 1.0)

    # 実験室効果（sum-to-zero, non-centered scale）
    lab_raw   = pm.Normal("lab_raw", 0.0, 1.0, dims="lab")
    sigma_lab = pm.HalfNormal("sigma_lab", 1.0)
    lab_eff   = pm.Deterministic(
        "lab_eff", sigma_lab * (lab_raw - lab_raw.mean()), dims="lab"
    )

    # 装置効果（sum-to-zero + non-centered、mu_grand との交絡を防ぐ）
    inst_raw  = pm.Normal("inst_raw", 0.0, 1.0, dims="inst")
    sigma_inst= pm.HalfNormal("sigma_inst", 0.5)
    inst_eff  = pm.Deterministic(
        "inst_eff", sigma_inst * (inst_raw - inst_raw.mean()), dims="inst"
    )

    # ロット効果（装置内のランダム変動、非中心化）
    lot_raw   = pm.Normal("lot_raw", 0.0, 1.0, dims="lot")
    sigma_lot = pm.HalfNormal("sigma_lot", 0.5)
    lot_eff   = pm.Deterministic("lot_eff", sigma_lot * lot_raw, dims="lot")

    # 観測ノイズ
    sigma_y   = pm.HalfNormal("sigma_y", 0.5)

    mu = pm.Deterministic(
        "mu",
        mu_grand + lab_eff[lab_of_obs] + inst_eff[inst_of_obs] + lot_eff[lot_of_obs],
        dims="sample",
    )
    y  = pm.Normal("y", mu=mu, sigma=sigma_y, observed=y_obs, dims="sample")

    idata = pm.sample(
        draws=1000, tune=3000, chains=8,
        target_accept=0.95, nuts_sampler="numpyro",
        random_seed=list(range(100, 108)),
    )
    pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=100)
```

> [!IMPORTANT]
> **`sigma_lab` と少数群の落とし穴**：`lab_raw - lab_raw.mean()` は zero-sum non-centered parameterization ですが、`G` 個の効果の marginal sd は厳密には `sigma_lab` ではなく `sigma_lab * sqrt((G-1)/G)` です。`G = 3` では ~18% の差になり、真値回収の検証時に注意が必要です。**そもそも `n_lab = 3` は少数群**（Ch11 §11.4）で `sigma_lab` は prior に強く依存します。真値 sd との一致を「近ければ良し」で緩め、必ず sensitivity 分析（`sigma_lab` の HalfNormal(0.5)/HalfNormal(2.0) の切替）を回してください。

### 診断と識別性チェック

```python
diag = check_diagnostics(idata)
assert diag["diagnostics_pass"], diag["checks"]

# 真値との比較（合成データなので可能）
import arviz as az
posterior_sigma_lab = az.summary(idata, var_names=["sigma_lab", "sigma_inst", "sigma_lot", "sigma_y"])
# 事後平均が生成時の真値 sd（sigma_lab_true=0.4, sigma_inst_true=0.3, ...）と近いか確認
```

### Layer 3 の provenance（Ch11-12 の集約）

```yaml
layer_3_hierarchical:
  skill_id:      pymc-hierarchical-3level-synthetic
  skill_version: 1.0.0
  input_sha256:  <hash of synthetic dataset arrays>
  data_source:   synthetic          # 合成データであることを明示
  synthetic_dataset:
    generator_script_sha256: <付録 A の生成スクリプト hash>
    generator_version:       "1.0.0"
    generation_seed:         2024
    true_parameters:
      mu_grand_true:  0.0
      sigma_lab_true: 0.4
      sigma_inst_true:0.3
      sigma_lot_true: 0.2
      sigma_y_true:   0.15
  hierarchical_structure:
    model_formula: "y ~ mu_grand + lab_eff[lab_of_obs] + inst_eff[inst_of_obs] + lot_eff[lot_of_obs]"
    structure_type: nested            # lab -> inst -> lot
    index_mapping:  {lab_of_inst: [...], inst_of_obs: [...], lot_of_obs: [...]}
    zero_sum_constraints: [lab_eff, inst_eff]   # 両方に sum-to-zero
    parameterization:
      lab_eff:  zero-sum-non-centered
      inst_eff: zero-sum-non-centered
      lot_eff:  non-centered
  identifiability_check:
    grand_mean_lab_confounded: false   # lab_eff / inst_eff の sum-to-zero で解消
    posterior_correlations_max: <値>
  shrinkage_summary:
    inst_shrinkage_factor: <各装置の 1 - |α_partial - μ|/|α_no_pool - μ|>
    lot_shrinkage_factor:  <同上>
  applicability_domain:
    y_range: [<min>, <max>]
    inst_count: 8
    lot_count:  60
    obs_per_lot_median: 4
  # 以降 sampler_config, backend_config, posterior_artifact, diagnostics_summary,
  # backend_reproducibility は Layer 2 と同型
```

> [!IMPORTANT]
> **合成データを実データと誤認しない**：`data_source: synthetic` を必ず provenance に明記します。合成データで得た `sigma_inst=0.3` は「装置差はこれくらい」の**実測ではありません**。実データで再学習するまで、Layer 3 の Skill を装置判定に使ってはいけません（第14章の失敗パターンで再訪）。

---

## 13.6 3 層の統合実行と provenance チェイン

3 層を一つの capstone Skill として繋ぎます。各 `run_*` 関数の返り値スキーマを揃えると provenance チェインの検証が容易になります。

### 各層 Skill の返り値スキーマ（推奨）

```python
def run_sklearn_regression(X_train, y_train, X_calib, X_test) -> dict:
    """
    Returns:
      {
        "y_pred_calib": np.ndarray,
        "y_pred_test":  np.ndarray,
        "cv_metrics":   {"cv_rmse_mean": float, "cv_rmse_std": float},
        "provenance":   {..., "output_artifacts": {"y_pred_calib": {"sha256": str, ...}, "y_pred_test": {...}}},
      }
    """

def run_pymc_calibration(y_pred_calib, y_calib, upstream_prov) -> dict:
    """
    Args:
      upstream_prov: Layer 1 の provenance dict（upstream 検証用）
    Returns:
      {"idata": az.InferenceData, "provenance": {..., "inputs": {"y_pred": {"sha256": ..., "upstream_layer": "layer_1", ...}}}}
    """

def run_hierarchical_bayes(synthetic_dataset) -> dict:
    """
    Returns:
      {"idata": az.InferenceData, "provenance": {..., "data_source": "synthetic", "synthetic_dataset": {...}}}
    """
```

### capstone 実行

```python
def run_capstone(X_train, y_train, X_calib, y_calib, X_test, y_test, synthetic_dataset):
    # Layer 1
    l1 = run_sklearn_regression(X_train, y_train, X_calib, X_test)
    # Layer 2（calibration split で fit、test は触らない）
    l2 = run_pymc_calibration(
        y_pred_calib=l1["y_pred_calib"], y_calib=y_calib,
        upstream_prov=l1["provenance"],
    )
    # Layer 3（合成階層データ、上流なし）
    l3 = run_hierarchical_bayes(synthetic_dataset)
    return {"layer_1": l1, "layer_2": l2, "layer_3": l3}
```

### provenance チェインの検証（改竄検知）

Layer 2 の `inputs.y_pred.sha256` が Layer 1 の `output_artifacts.y_pred_calib.sha256` と**バイト単位で一致**することを確認します：

```python
def verify_provenance_chain(result):
    p1 = result["layer_1"]["provenance"]
    p2 = result["layer_2"]["provenance"]
    p3 = result["layer_3"]["provenance"]

    # Layer 2 の y_pred 入力ハッシュ == Layer 1 の y_pred_calib 出力ハッシュ
    assert p2["inputs"]["y_pred"]["sha256"] == p1["output_artifacts"]["y_pred_calib"]["sha256"], \
        "Layer 1 → Layer 2 の provenance チェインが破損"
    assert p2["inputs"]["y_pred"]["upstream_layer"] == "layer_1"

    # Layer 2 の y_true ハッシュが calibration split のハッシュと一致
    assert p2["inputs"]["y_true"]["sha256"] == p1["data_split"]["calib_sha256"]

    # Layer 3 は上流を持たない
    assert p3["data_source"] == "synthetic"
    assert "upstream_layer" not in p3.get("inputs", {})

    return True
```

**Layer 3 だけ独立している**理由：Layer 1-2 は同じ実データで繋がりますが、Layer 3 は合成データを使うので実データ上流には接続しません。ただし Layer 3 も `synthetic_dataset.generator_script_sha256` によって**合成データ生成の再現性**は担保されます。実データで階層構造を扱える段階になったら Layer 3 も upstream を持つチェインに変わります（付録 C 参照）。

---

## 13.7 Capstone の合否判定

第12章の **14 項目チェックリスト**を、Layer 2 と Layer 3 それぞれに適用します。Layer 1（sklearn）は診断項目が異なるため、別の 4 項目セットで確認します。

### Layer 1（sklearn）のチェック（6 項目）

1. ☐ `data_split` が fit 前に 3 分割（train / calibration / test）で凍結され、それぞれの SHA-256 が記録済み
2. ☐ CV は Pipeline を通しており、fold 内で scaler が fit されている（漏洩なし）
3. ☐ Layer 2 に渡す予測は **calibration split の予測** `y_pred_calib`、test 予測 `y_pred_test` は最終評価まで開封しない
4. ☐ `cv_scheme` が group を要する場合、`group_key` を指定した GroupKFold を使っている
5. ☐ `applicability_domain` の特徴量範囲・y 範囲が記録済み
6. ☐ `output_artifacts.y_pred_calib.sha256` / `y_pred_test.sha256` / `model_sha256` が記録済み

### Layer 2・Layer 3（PyMC）のチェック（各 14 項目）

第12章 §12.9 の 14 項目をそのまま適用。Layer 2 は非階層なので項目 13（`hierarchical_structure` 等）を「該当なし ✅」とマークします。Layer 3 は 14 項目**すべて**必要です。

### Capstone 全体の合否

- **Layer 1 の 6 項目**すべて ✅
- **Layer 2 の 14 項目**すべて ✅（項目 13 = 階層 provenance は該当なしで OK）
- **Layer 3 の 14 項目**すべて ✅
- **provenance チェイン**：`layer_2.inputs.y_pred.sha256 == layer_1.output_artifacts.y_pred_calib.sha256` かつ `layer_2.inputs.y_true.sha256 == layer_1.data_split.calib_sha256`

以上をすべて満たしたときのみ `capstone_certification_pass: true` を最終 provenance に記録します。

---

## 13.8 章末ワーク

1. **MatBench または RRUFF から Layer 1 を組む**：Ridge の代わりに RandomForest / GradientBoosting でも試し、CV RMSE を比較して provenance に記録
2. **Layer 2 の校正曲線に inverse を追加**：第10章 §10.6 の inverse モデルを組み込み、新しい観測 x に対する y の CrI を返す関数を実装
3. **合成階層データを増やす**：`sigma_lab_true` を大きくして生成し、Layer 3 が sigma を大きく推定するか確認（sensitivity テスト）
4. **Layer 3 で ロット数 G < 5 の場合を再実行**：Ch11 §11.4 の警告どおり `sigma_lot` が prior 依存になるかを観察
5. **provenance チェインの改竄検知**：Layer 1 の `y_pred_calib` を人為的にいじって Layer 2 を走らせ、`layer_2.inputs.y_pred.sha256 != layer_1.output_artifacts.y_pred_calib.sha256` で `verify_provenance_chain` の assert が failing することを確認
6. **14 項目チェックリストを Python 関数化**：`certify_capstone(prov: dict) -> dict` を実装し、失敗項目を列挙する形にする

---

## 13.9 本章のまとめ

- **Ch4-12 の Skill を 3 層に統合**：sklearn 回帰 → PyMC 校正 → 階層モデル
- 実データは Layer 1-2 で、階層構造は Layer 3 の合成データで扱う（実データの階層構造データは付録 C）
- Layer 2 は Layer 1 の `y_pred_calib` を入力に取り、その **SHA-256 を `inputs.y_pred.sha256` に記録**。**provenance チェインをハッシュで検証**でき、Layer 1 出力の改竄を検知できる
- Layer 3 は合成データを使うため上流を持たない。`data_source: synthetic` を必ず記録
- **合否は機械的に決める**：Layer 1（6 項目）+ Layer 2（14 項目）+ Layer 3（14 項目）+ provenance チェインのハッシュ整合。すべて ✅ で `capstone_certification_pass: true`

vol-02 の**組み立て**はこれで完成しました。第14章では、この capstone を壊す「失敗パターン」を体系的に扱います。

---

## 参考資料

### 本書内の該当章

- 第4章：anti-leakage 契約、data_split 凍結（Layer 1 の前提）
- 第5-6章：sklearn Pipeline と回帰 Skill 化（Layer 1 の素材）
- 第7-8章：CV 設計と interpretability（Layer 1 の検証）
- 第9-10章：Bayesian 校正曲線（Layer 2）
- 第11章：階層モデル、非中心化、sum-to-zero（Layer 3）
- 第12章：診断 14 項目チェックリスト（合否判定）
- 第14章：本章の capstone を壊す失敗パターン
- 第15章：capstone を組織で共有する運用パターン
- 付録 A：合成階層データ生成スクリプト、拡張 provenance スキーマ全体
- 付録 C：実データで階層構造を扱う候補（Materials Project ラウンドロビン、NIST SRD、NOMAD ほか）

### 外部参考

<a id="ref-13-1">[13-1]</a> Dunn, A. et al. (2020). Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm. *npj Computational Materials*, 6(1). — MatBench の公式論文（Layer 1 データ）
<a id="ref-13-2">[13-2]</a> Lafuente, B., Downs, R. T., Yang, H., & Stone, N. (2015). The power of databases: the RRUFF project. *Highlights in Mineralogical Crystallography*, 1–29. — RRUFF プロジェクト（Layer 1 スペクトルデータ）
<a id="ref-13-3">[13-3]</a> Gelman, A., Hill, J., & Vehtari, A. (2020). *Regression and Other Stories*. Cambridge University Press. — 階層モデルの多層 workflow
<a id="ref-13-4">[13-4]</a> McElreath, R. (2020). *Statistical Rethinking* (2nd ed.). CRC Press, Chapter 13. — 3 階層モデル（村×人×時間）の教科書的事例
<a id="ref-13-5">[13-5]</a> Betancourt, M. (2020). Towards a Principled Bayesian Workflow. [https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html) — 合成データで真値回収を確認する workflow（Layer 3 の設計思想）
