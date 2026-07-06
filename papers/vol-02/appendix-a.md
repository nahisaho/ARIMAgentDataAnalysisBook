# 付録A　統計/ML Skill テンプレート集

> **本付録の目的**：第4〜13章で作った Skill の**再利用可能な雛形**を、`SKILL.md` + `provenance schema` + `minimal example` の 3 点セットで提供します。vol-01 の provenance を **統計/ML 拡張**した完全スキーマも本付録末尾に置きます。

> [!IMPORTANT]
> テンプレートは**そのままコピーして使うためのもの**ではなく、**「何を書き、何を検証すべきか」を思い出すためのチェックリスト**です。実際のデータ・目的に合わせて調整してください。

---

## A.1 本付録の使い方

| 目的 | 参照するテンプレート | 該当章 |
|---|---|---|
| 表形式データで回帰／分類 Skill を作る | A.2（教師あり） | 第6章 |
| PCA・クラスタリング・異常検知 Skill を作る | A.3（教師なし） | 第7章 |
| ベイズ校正・回帰 Skill を作る | A.4（Bayesian） | 第11章 |
| 階層モデル（反復測定・ロット差） Skill を作る | A.5（Hierarchical） | 第12章 |
| capstone（sklearn → PyMC 連結） Skill を作る | A.6（Multi-layer） | 第14章 |
| provenance schema の全項目を確認する | A.7（完全スキーマ） | 第4・10・11章 |

**テンプレの共通要素**（6 要素）は第5章 §5.2 に準拠：①目的 / ②入力条件 / ③出力形式 / ④成功条件 / ⑤禁止事項（severity 付き）/ ⑥再現性条件。

### A.1.1 Skill / リポジトリ配置規約（vol-01 A.2.1 の継承 + vol-02 追加要素）

Skill ディレクトリ構造は **vol-01 付録A §A.2.1 と同一**です（`skills/<name>/SKILL.md`, `references/`, `scripts/`, `tests/`, `examples/`）。配置先も同じ 3 系統：

- **リポジトリ配布**：`.github/skills/<name>/`（Copilot CLI 自動検出、実運用の推奨）
- **個人利用**：`~/.copilot/skills/<name>/`
- **下書き**：`skills/<name>/`（本書の説明・演習用。完成後にどちらかへ移す）

vol-02 で新規に登場するパスは **Skill ディレクトリの内側** と **リポジトリルート** の 2 箇所に分かれます。混乱を避けるため、以下を「どこに置くか」の一次リファレンスとして参照してください。

**リポジトリルート（Skill 横断で共有される資産）**

```
<vol-02 repo root>/
├── data/
│   └── synthetic-hierarchy/          ← 本書提供の合成階層データ（第3章 §3.9）
│       ├── main.csv
│       └── repeated/                 ← 反復測定サブセット（第12章）
├── scripts/
│   └── generate_synthetic_hierarchy.py   ← 合成データ生成（snake_case で統一）
└── .github/skills/                   ← 各 Skill はここに配置（vol-01 A.2.1）
    └── <skill-name>/
        └── ...
```

**Skill ディレクトリの内側（vol-02 で追加される 2 ディレクトリ）**

vol-01 の `SKILL.md` / `references/` / `scripts/` / `tests/` / `examples/` に加え、統計/ML 章で扱う「学習成果物」と「診断図」の置き場を明示します。

```
.github/skills/<skill-name>/
├── SKILL.md                          ← vol-01 A.2.2（同一）
├── references/                       ← vol-01 A.2.3〜（同一）
├── scripts/                          ← vol-01 A.2（同一）
├── tests/                            ← vol-01 A.2（同一）
├── examples/                         ← vol-01 A.2（同一）
├── artifacts/                        ← ★vol-02 追加：学習・推論の永続化物
│   ├── model_v<SemVer>.pkl                 sklearn モデル（第5-8章）
│   ├── posterior_v<SemVer>.nc              PyMC InferenceData（第10-12章）
│   └── permimp.csv                          permutation importance（第9章）
└── figures/                          ← ★vol-02 追加：診断・可視化の PNG/PDF
    ├── pca_scree_v<SemVer>.png
    ├── ppc_bpv_v<SemVer>.png
    └── trace_v<SemVer>.png
```

**規約サマリ**

| パス | 場所 | 例 | Git 追跡 |
|---|---|---|---|
| `data/synthetic-hierarchy/` | **リポジトリルート** | 本書提供の合成階層データ | 追跡（大容量なら LFS） |
| `scripts/generate_synthetic_hierarchy.py` | **リポジトリルート**（Skill 横断で使う共通スクリプト）| 合成データ生成 | 追跡 |
| `.github/skills/<name>/scripts/` | Skill 内 | Skill 固有のスクリプト | 追跡 |
| `.github/skills/<name>/artifacts/` | Skill 内 | 学習済みモデル、posterior | **通常 gitignore**（provenance の `output_artifacts.model_sha256` で hash 検証。大きい成果物は別ストレージ + URI 参照）|
| `.github/skills/<name>/figures/` | Skill 内 | 診断図 | 通常 gitignore（PR 添付や docs/ に選抜） |

**命名規約**

- **Skill 名**：ケバブケース（`sklearn-supervised-matbench-gap`）— vol-01 A.2.1 と同じ
- **スクリプト名**：スネークケース（`run_analysis.py`, `generate_synthetic_hierarchy.py`）— vol-01 A.2 と同じ
- **成果物ファイル名**：`<name>_v<SemVer>.<ext>`（例: `model_v1.0.0.pkl`, `posterior_v1.0.0.nc`）
- **provenance からの参照**：常に **Skill ディレクトリからの相対パス**（`artifacts/model_v1.0.0.pkl`）。絶対パスや `~/...` は書かない

> **本付録以降のパス表記の読み方**
>
> - 頭に `data/` があるパス（例: `data/synthetic-hierarchy/`）→ **リポジトリルート**からの相対
> - 頭に `scripts/generate_synthetic_hierarchy.py` → 例外的に **リポジトリルート**（合成データ生成は横断的資産のため）
> - 頭に `artifacts/` / `figures/` / `references/` / `tests/` があるパス → **`.github/skills/<name>/` 内**の相対

---

## A.2 教師あり学習 Skill テンプレート（第6章）

### A.2.1 `SKILL.md` 雛形

```markdown
---
skill_id: sklearn_supervised_<domain>_v1
skill_version: 1.0.0
task_type: regression | classification
domain: <matbench_expt_gap | matbench_steels | ...>
maintainer: <ORCID or team>
created: YYYY-MM-DD
---

## ① 目的
- 何を予測するか（例：組成 → バンドギャップ eV）
- どう使うか（意思決定文脈：スクリーニング / 校正 / etc.）

## ② 入力条件（data contract 参照：Ch4）
- 特徴量：`X: DataFrame (n, p)` — 列名・単位・型
- 目的変数：`y: Series (n,)` — 単位・型
- グループ列：`group_key: Series | None` — 装置/ロット/試料 ID
- split_unit: {sample | lot | instrument | patient | none}
- 前処理：単位統一済み、欠損なし、外れ値検査済み

## ③ 出力形式
- `posterior/point_estimate`: 予測ベクトル
- `cv_scores.json`: fold ごと MAE/RMSE/R²
- `feature_importance.json`: SHAP または permutation importance
- `provenance.yaml`: A.7 スキーマ準拠
- `figures/`: 学習曲線、residual plot、feature importance

## ④ 成功条件（3 点セット）
- 予測誤差：CV RMSE ≤ <target> — 根拠：<baseline / task standard>
- 安定性：CV fold 間 RMSE の CV of Variance ≤ 20%
- 予測分布：外挿域（applicability_domain 外）で警告

## ⑤ 禁止事項（severity 付き）
- **fatal**：Pipeline 外での fit（データリーク） / test 複数回参照 / 目的変数の特徴量への混入
- **error**：group_key ありで KFold 使用（GroupKFold 必須）
- **warning**：p >> n で正則化パラメータ未指定

## ⑥ 再現性条件
- random_seed 固定、package_versions ピン、CV 分割 hash 記録
- provenance の `cv_scheme.group_key` と `data_split` を必須化
```

### A.2.2 最小コード骨格

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, GroupKFold, KFold
from sklearn.linear_model import Ridge

pipe = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])

# group_key の有無で CV を切り替え（無いのに GroupKFold を使うと落ちる）
if group_key is not None:
    cv = GroupKFold(n_splits=5)
    cv_kwargs = {"cv": cv, "groups": group_key}
else:
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_kwargs = {"cv": cv}

res = cross_validate(pipe, X, y, **cv_kwargs,
                     scoring=("neg_root_mean_squared_error", "r2"),
                     return_train_score=True, return_indices=True)
```

### A.2.3 チェックリスト（PR 前）
- [ ] `check_split_contract(train_idx, test_idx, group_key)` パス
- [ ] SHAP または permutation importance を出力
- [ ] `applicability_domain` を provenance に記録
- [ ] 過学習の目安：train R² − CV R² ≤ 0.1

---

## A.3 教師なし学習 Skill テンプレート（第7章）

### A.3.1 `SKILL.md` 雛形（PCA / clustering / anomaly を単一 template で扱う）

```markdown
---
skill_id: sklearn_unsupervised_<domain>_v1
task_type: dimensionality_reduction | clustering | anomaly_detection
---

## ④ 成功条件（task_type 別）

### dimensionality_reduction (PCA)
- 個別 loading 安定率：individual_agree ≥ 0.9
- 部分空間安定率：subspace_agree ≥ 0.9（縮退時）
- 累積寄与率：≥ 0.8 で採用成分数決定

### clustering
- 内部指標：silhouette ≥ 0.5（k-means）
                 or DBCV / cluster_persistence（HDBSCAN）
- 安定性：bootstrap ARI ≥ 0.7
- 外部評価：既知ラベルがあれば ARI / NMI

### anomaly_detection
- 感度・特異度：既知の異常例で TPR ≥ 0.9, FPR ≤ 0.1
- 閾値の provenance 記録
```

### A.3.2 安定性測定関数（第7章 §7.2）

```python
import numpy as np
from sklearn.base import clone
from sklearn.utils import _safe_indexing

def bootstrap_stability(estimator, X, measure, n_boot=50, seed=0):
    """PCA loading / cluster assignment の bootstrap 安定性。

    - estimator は毎回 clone して独立にフィット（同一インスタンスの使い回しは NG）
    - X は pandas DataFrame でも安全にサンプリング（`.iloc[idx]` 相当）
    - `measure(base, est_i, X_i)` は subspace 一致率 / ARI 等を返す user 関数
    """
    rng = np.random.default_rng(seed)
    base = clone(estimator).fit(X)
    scores = []
    n = len(X)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        X_i = _safe_indexing(X, idx)          # DataFrame でも列選択にならない
        est_i = clone(estimator).fit(X_i)     # 毎回 clone
        scores.append(measure(base, est_i, X_i))
    return float(np.mean(scores))
```

---

## A.4 Bayesian（PyMC）Skill テンプレート（第11章）

### A.4.1 `SKILL.md` 雛形

```markdown
---
skill_id: pymc_bayesian_<domain>_v1
task_type: bayesian_calibration | bayesian_regression | ...
backend: pymc-nuts | numpyro-nuts-cpu | numpyro-nuts-gpu
---

## ② 入力条件
- X, y のスケール範囲（prior 設計に必要）
- applicability_domain: prior predictive で検証済み範囲

## ③ 出力形式
- posterior_artifact: `.nc`（InferenceData NetCDF）
- diagnostics_summary: R-hat, ESS, divergences, BFMI, treedepth
- prior_specification: prior 定義（式 + 数値）
- prior_predictive_range: 95% 区間
- posterior_predictive_check: BPV, PIT calibration

## ④ 成功条件
- 診断ゲート：diagnostics_pass == true（14 項目チェックリスト）
- Prior sensitivity：main / weaker / stronger の 3 分岐でロバスト
- PPC：観測データが posterior predictive 95% 区間内

## ⑤ 禁止事項
- **fatal**：diagnostics_pass == false での運用
- **fatal**：backend 変更後の tolerance 未検証（Ch10 §11.9）
- **error**：prior_predictive_check 未実施
```

### A.4.2 骨格コード（線形回帰のベイズ版）

```python
import pymc as pm
with pm.Model() as m:
    # Skill 再利用のため入力は pm.Data で登録（Ch10 §11.6）
    x_data = pm.Data("x_data", x_obs, mutable=True)
    y_data = pm.Data("y_data", y_obs, mutable=True)

    beta = pm.Normal("beta", 0.0, 5.0)
    alpha = pm.Normal("alpha", 0.0, 5.0)
    sigma = pm.HalfNormal("sigma", 1.0)
    mu = alpha + beta * x_data
    pm.Normal("y", mu=mu, sigma=sigma, observed=y_data)

    # (1) prior predictive を fit の前に実施（Ch10 §11.3）
    idata = pm.sample_prior_predictive(random_seed=42)

    # (2) 事後サンプリング
    idata.extend(pm.sample(draws=2000, tune=2000, chains=4, cores=4,
                           nuts_sampler="numpyro", random_seed=42,
                           target_accept=0.95))

    # (3) 事後予測（学習データ）
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=42))

# 予測時は `pm.set_data({"x_data": x_new}, predictions=True)` 後に
# `pm.sample_posterior_predictive(idata, predictions=True)` を使う（Ch10 §11.6）
```

### A.4.3 診断関数（第13章 §13.9）

```python
def check_diagnostics(idata, thr=None):
    """14 項目チェックリスト（診断 10 + provenance 4）"""
    thr = thr or {"rhat": 1.01, "ess_bulk": 400, "ess_tail": 400,
                  "divergences": 0, "bfmi": 0.3, "treedepth": None}
    rhat = az.rhat(idata)
    ess = az.ess(idata, method="bulk")
    # ...（Ch12 の実装参照）
    return {"diagnostics_pass": bool(...), "rhat_max": ..., ...}
```

---

## A.5 階層モデル Skill テンプレート（第12章・Capstone Layer 3）

### A.5.1 `SKILL.md` 雛形

```markdown
---
skill_id: pymc_hierarchical_<domain>_v1
task_type: hierarchical_regression | variance_decomposition
hierarchy: [lab, instrument, lot, sample]
backend: numpyro-nuts-cpu
---

## ② 入力条件（追加）
- group_id 列：`lab_id`, `inst_id`, `lot_id` の対応表
- 各レベルの group 数：n_lab, n_inst, n_lot
- **applicability_domain の階層版**：inst_count_min, lot_count_min

## ③ 出力形式（追加）
- 各分散成分の事後：sigma_lab, sigma_inst, sigma_lot, sigma_y
- 各レベルの random effect：lab_eff, inst_eff, lot_eff
- 予測分解：total = grand_mean + lab_eff + inst_eff + lot_eff

## ④ 成功条件（追加）
- **非中心化必須**：centered だと divergences 頻発（第12章 §12.3）
- sum-to-zero 制約：lab_eff.mean() ≈ 0, inst_eff.mean() ≈ 0
- 小 group 数警告：inst_count < 5 のとき strong prior + sensitivity 必須
```

### A.5.2 非中心化 + sum-to-zero テンプレート

```python
coords = {"lab": lab_names, "inst": inst_names, "lot": lot_names,
          "obs": np.arange(n_obs)}
with pm.Model(coords=coords) as m_h:
    mu_grand = pm.Normal("mu_grand", 0.0, 5.0)
    sigma_lab = pm.HalfNormal("sigma_lab", 1.0)
    sigma_inst = pm.HalfNormal("sigma_inst", 1.0)
    sigma_lot = pm.HalfNormal("sigma_lot", 1.0)
    sigma_y = pm.HalfNormal("sigma_y", 1.0)

    # 全レベルとも非中心化（raw を Normal(0,1) で置き、sigma を Deterministic で掛ける）
    # sum-to-zero は raw から平均を引いて Deterministic 化する（Ch11 §12.3 / Ch13）
    lab_raw = pm.Normal("lab_raw", 0.0, 1.0, dims="lab")
    inst_raw = pm.Normal("inst_raw", 0.0, 1.0, dims="inst")
    lot_raw = pm.Normal("lot_raw", 0.0, 1.0, dims="lot")

    lab_eff = pm.Deterministic(
        "lab_eff", sigma_lab * (lab_raw - lab_raw.mean()), dims="lab")
    inst_eff = pm.Deterministic(
        "inst_eff", sigma_inst * (inst_raw - inst_raw.mean()), dims="inst")
    lot_eff = pm.Deterministic(
        "lot_eff", sigma_lot * lot_raw, dims="lot")  # lot は sum-to-zero 不要

    mu = (mu_grand + lab_eff[lab_idx] + inst_eff[inst_idx]
          + lot_eff[lot_idx])
    pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y, dims="obs")

    idata = pm.sample(draws=2000, tune=2000, chains=4,
                      nuts_sampler="numpyro", target_accept=0.95,
                      random_seed=0)
```

> **`pm.ZeroSumNormal` は？**：sum-to-zero は満たすが「非中心化」の raw パラメータ化とは別概念。Ch11/Ch13 の非中心化必須条件を満たすには上記のように raw + Deterministic の形にする。
                      random_seed=0)
```

---

## A.6 Multi-layer Capstone Skill テンプレート（第14章）

Layer 1（sklearn）→ Layer 2（PyMC 校正）→ Layer 3（階層モデル）を **provenance chain で連結**します。

### A.6.1 layer 間の受け渡し契約

```yaml
layer_1:
  skill_id: sklearn_regression_matbench_v1
  outputs:
    y_pred_raw: DataFrame(n, 1)
    y_pred_sha256: <hash>
layer_2:
  skill_id: pymc_calibration_v1
  inputs:
    y_pred_raw: layer_1.outputs.y_pred_raw
    y_pred_sha256: layer_1.outputs.y_pred_sha256   # verify chain
    upstream_layer: layer_1
  outputs:
    y_pred_calib: DataFrame(n, 3)   # mean, lower95, upper95
    y_pred_calib_sha256: <hash>
layer_3:
  # 合成階層データを使うため、実データ上流には接続しない（Ch13 §14.5）
  skill_id: pymc_hierarchical_v1
  data_source: synthetic
  synthetic_dataset:
    generator_script: scripts/generate_synthetic_hierarchy.py   # リポジトリルート（§A.1.1）
    generator_script_sha256: <hash>
    seed: 0
    true_params:
      sigma_lab: 0.5
      sigma_inst: 0.3
      sigma_lot: 0.2
  # inputs.upstream_layer は無し（Ch13 の verify で明示的にチェック）
  outputs:
    variance_decomposition: {sigma_lab, sigma_inst, sigma_lot}
```

> **Layer 3 に upstream を書かない**：Ch13 §14.5 で「Layer 3 は合成データを使うため実データ上流には接続しない」と定めているため、`inputs.upstream_layer` を書くと検証（`assert "upstream_layer" not in p3["inputs"]`）で落ちる。Layer 3 の再現性は `synthetic_dataset.generator_script_sha256` で担保する。

### A.6.2 chain 検証（第14章 §14.6）

```python
def verify_provenance_chain(prov: dict) -> bool:
    for layer_id, layer in prov["layers"].items():
        for input_key, expected_hash in layer.get("input_hashes", {}).items():
            upstream = layer["upstream"]
            actual = prov["layers"][upstream]["output_hashes"][input_key]
            if actual != expected_hash:
                raise AssertionError(
                    f"chain break at {layer_id}.{input_key}: "
                    f"{expected_hash[:8]} != {actual[:8]}")
    return True
```

---

## A.7 統計/ML 拡張 provenance 完全スキーマ

vol-01 の provenance を**統計/ML 拡張**したスキーマ完全版です。**必須項目**（★）と**推奨項目**（○）を明示。

```yaml
# ---------- vol-01 継承（★★★） ----------
run_id: <UUIDv7>                       # ★
run_timestamp: <ISO 8601 UTC>          # ★
operator: <ORCID or team_id>           # ★
skill_id: <namespace_snake_case>       # ★
skill_version: <SemVer>                # ★
code_sha256: <64 hex>                  # ★（Skill dir tree hash）
git_commit_sha: <40 hex>               # ★
execution_context: development | production  # ★ Ch14 §15.8

# ---------- 入力・出力（★） ----------
inputs:                                # ★
  X_sha256: <64 hex>
  y_sha256: <64 hex>
  data_contract_hash: <64 hex>
  feature_columns: [f1, f2, ...]       # ★ Ch5, Ch7
  target_column: y                     # ★
output_artifacts:                      # ★
  model_path: artifacts/model_v1.0.0.pkl   # .github/skills/<name>/artifacts/ 配下（§A.1.1）
  model_sha256: <64 hex>

package_versions:                      # ★
  python: 3.11.7
  sklearn: 1.9.0
  pymc: 5.28.5
  arviz: 0.23.4
  numpyro: 0.21.0
  jax: 0.10.2
random_seed: <int>                     # ★（top-level）

# ---------- 統計/ML 拡張：CV / データ分割（★） ----------
data_split:
  strategy: KFold | GroupKFold | StratifiedGroupKFold | TimeSeriesSplit
  n_splits: 5
  shuffle: true
  random_state: 0
  group_key: inst_id | lot_id | sample_id | null
  split_manifest_hash: <64 hex>        # ★ Ch7
  split_unit: {sample|lot|instrument|patient|none}
  # 3-way split の場合（Ch13 capstone）
  train_sha256: <64 hex>
  calib_sha256: <64 hex>
  test_sha256: <64 hex>

cv_scheme:                             # ★ Ch7
  outer: {type: GroupKFold, n_splits: 5, group_key: inst_id}
  inner: {type: GroupKFold, n_splits: 3, group_key: lot_id}
  purge: null | {gap: 5}

# ---------- モデル設定（★） ----------
model_config:
  estimator: Ridge | RandomForestRegressor | pm.Model
  hyperparameters: {alpha: 1.0, ...}
  pipeline_steps: [scaler, model]      # sklearn の場合
  applicability_domain:
    x_range: [[-3, 3], ...]
    y_range: [0, 10]
    inst_count_min: 3                  # 階層モデル用
    lot_count_min: 5

metric_definition:                     # ★ Ch5, Ch7
  primary: neg_root_mean_squared_error
  secondary: [r2, mae]
  cv_score:                            # ★ 学習/CV スコアの双方を必ず記録
    neg_rmse_mean: -0.42
    neg_rmse_std: 0.05
    r2_mean: 0.81
  training_score:
    neg_rmse: -0.35
    r2: 0.87

hyperparam_search:                     # ○ nested CV の場合
  n_trials: 20
  search_space: {alpha: [0.01, 100]}
  selection_metric: neg_rmse
  test_set_access_count: 0
  selection_metric_changed: false      # ★ p-hacking 防止

# ---------- Bayesian 追加項目（★ Ch10-12） ----------
sampler_config:
  draws: 2000
  tune: 2000
  chains: 4
  cores: 4
  target_accept: 0.95
  random_seeds: [0, 1, 2, 3]

backend_config:
  backend: numpyro-nuts-cpu
  jax_enable_x64: true
  jax_platform: cpu | gpu
  hardware: <CPU/GPU model>

backend_reproducibility:               # ★ Ch10 §11.9 / Ch14 §15.7
  cross_backend_tested: true
  tolerance_rhat_delta: 0.01
  tolerance_ess_ratio: 0.9
  reference_backend: pymc-nuts

prior_specification:
  beta: {dist: Normal, mu: 0, sigma: 5}
  sigma: {dist: HalfNormal, sigma: 1}
  sigma_group:
    dist: HalfNormal
    sigma: 1
    sensitivity_alternatives:          # ★ Ch11 (n_group < 5 のとき必須)
      - {sigma: 0.3}
      - {sigma: 3.0}

prior_predictive_range:                # ★ Ch10 §11.3
  y_95ci: [-2.5, 2.5]

uncertainty_scheme:                    # ★ Ch10, Ch12
  type: posterior_predictive_95ci | conformal_prediction | bootstrap
  coverage_target: 0.95
  measured_coverage: 0.94

diagnostics_summary:                   # ★ Ch12 §13.9（項目名は check_diagnostics() の出力に統一）
  diagnostics_pass: true
  max_rhat: 1.005
  min_ess_bulk: 850
  min_ess_tail: 720
  n_divergences: 0
  min_bfmi: 0.5
  max_tree_depth: 8
  reached_max_treedepth: false
  thresholds:                          # 使用した閾値も記録
    rhat: 1.01
    ess_bulk: 400
    ess_tail: 400
    divergences: 0
    bfmi: 0.3

posterior_predictive_check:            # ★ Ch12 §13.5
  bpv_pass: true
  pit_calibration_ks_p: 0.42
  rank_uniformity_pass: true

posterior_artifact:                    # ★ Ch10
  path: artifacts/posterior_v1.0.0.nc
  sha256: <64 hex>
  format: netcdf4

# ---------- 階層モデル追加項目（Ch11） ----------
hierarchical_structure:
  levels: [lab, instrument, lot, sample]
  n_groups: {lab: 3, inst: 9, lot: 45, sample: 180}
  parameterization: non_centered       # ★ Ch11 §12.3
  sum_to_zero: [lab_eff, inst_eff]
  applicability_domain:
    inst_count: 9
  identifiability_check:               # ★ Ch11
    passed: true
    method: prior_vs_posterior_ratio
  shrinkage_summary:                   # ○ Ch11
    lab_shrinkage: 0.72
    inst_shrinkage: 0.55

# ---------- Multi-layer chain（Ch13） ----------
upstream_run_ids: [<UUIDv7>, ...]      # ○（実データ上流を持つ Layer のみ）
input_hashes: {y_pred_raw: <sha256>, ...}
output_hashes: {y_pred_calib: <sha256>, ...}

# ---------- 監査・失敗管理（Ch14-15） ----------
data_source: real | synthetic          # ★ Ch14 §15.8
synthetic_dataset:                     # data_source=synthetic のとき必須
  generator_script: scripts/generate_synthetic_hierarchy.py   # リポジトリルート（§A.1.1）
  generator_script_sha256: <64 hex>
  seed: 0
  true_params: {sigma_lab: 0.5, ...}   # 合成データなので既知
synthetic_production_override:         # data_source=synthetic かつ execution_context=production のときのみ必須
  approved_by: <ORCID>
  justification: <text>

known_issues_snapshot_hash: <64 hex>   # ★ Ch14 §15.9
acknowledged_known_issue_ids: []

certification_pass: true               # ★ Ch13 §14.7

# ---------- 決定紐付け（Ch15 §16.5.4） ----------
decision:                              # ○（production 実行）
  type: pilot_go_nogo | lot_release | ...
  value: go | hold
  rationale_link: <URI>
  reviewer: <ORCID>

# ---------- ログ完全性（Ch15 §16.5.5） ----------
integrity:
  log_sha256: <64 hex>
  prev_log_sha256: <64 hex>
  signature: <PKI signature>
access_policy_id: <ID>
retention_policy_id: <ID>
```

### A.7.1 task_type 別 必須項目マトリクス

| フィールド | supervised | unsupervised | Bayesian | hierarchical | capstone |
|---|:-:|:-:|:-:|:-:|:-:|
| `inputs.feature_columns`, `target_column` | ★ | ★（target 除く） | ★ | ★ | ★ |
| `data_split.split_manifest_hash` | ★ | ○ | ★ | ★ | ★（train/calib/test 全て）|
| `cv_scheme` | ★ | ○ | ○ | ○ | ★ |
| `model_config.applicability_domain` | ★ | ★ | ★ | ★（inst/lot_count_min）| ★ |
| `metric_definition.cv_score` | ★ | ★（内部指標）| ○ | ○ | ★ |
| `sampler_config`, `backend_config` | — | — | ★ | ★ | ★（Layer 2/3）|
| `backend_reproducibility` | — | — | ★ | ★ | ★ |
| `prior_specification`, `prior_predictive_range` | — | — | ★ | ★ | ★（Layer 2/3）|
| `diagnostics_summary` | — | — | ★ | ★ | ★（Layer 2/3）|
| `posterior_predictive_check` | — | — | ★ | ★ | ★（Layer 2/3）|
| `hierarchical_structure` | — | — | — | ★ | ★（Layer 3）|
| `data_source`, `synthetic_dataset` | — | — | ○ | ○ | ★（Layer 3 は synthetic）|
| `upstream_run_ids`, `input_hashes` | — | — | — | — | ★（Layer 2、Layer 3 は除外）|
| `known_issues_snapshot_hash` | ★ | ★ | ★ | ★ | ★ |

---

## A.8 テンプレートを新規 Skill に適用する手順

1. **対象 task_type を A.2〜A.6 から選ぶ**（複数該当なら Multi-layer）
2. **`SKILL.md` 雛形をコピー**、①〜⑥ をドメイン固有に書き換え
3. **④成功条件を数値化**（第5章 §5.4 の 3 点セット準拠）
4. **⑤禁止事項に severity を付与**（第5章 §5.5、fatal/error/warning）
5. **provenance schema を A.7 から抽出**（task_type に該当する項目のみ）
6. **CI ゲート**：`check_split_contract`, `check_diagnostics`, `verify_provenance_chain` を追加
7. **pytest**：第15章 §15.10 のテストを実装

---

## A.9 まとめ

- Skill 雛形は **6 要素 × task_type 別**（教師あり / 教師なし / Bayesian / 階層 / Multi-layer）
- provenance は **vol-01 継承 + 統計/ML 拡張** の 2 層構造
- **A.7 完全スキーマ**は必須★・推奨○を明示。task_type に応じて必要項目を抽出
- テンプレは「コピペ元」ではなく「チェックリスト」として使う

**次の付録**：付録B（cheatsheet + PyMC↔Stan 対応表）、付録C（トラブルシューティング + 実データ候補）。

---

## 参考資料

### 本書内の該当章

- 第5章 §5.2：Skill 設計原則（6 要素）
- 第5章 §5.4：成功条件 3 点セット
- 第5章 §5.5：禁止事項 severity
- 第5-8章：sklearn Skill 実装
- 第8章：CV 設計・データリーク検知
- 第10-12章：Bayesian Skill 実装・診断
- 第12章：階層モデル・非中心化
- 第14章：Multi-layer capstone・provenance chain
- 第15章 §15.9：known_issues schema
- 第16章 §16.5：監査ログ 4 レイヤ

### 外部参考

- Kapoor, S. & Narayanan, A. *Leakage and the Reproducibility Crisis in ML-Based Science*. Patterns (2023). — Leakage taxonomy
- Gelman, A., et al. *Bayesian Workflow*. arXiv:2011.01808 (2020). — Bayesian Skill 設計の指針
- Vehtari, A., et al. *Rank-Normalization, Folding, and Localization: An Improved R̂*. Bayesian Analysis (2021). — 診断指標
