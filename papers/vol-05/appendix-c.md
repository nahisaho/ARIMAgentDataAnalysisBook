# 付録 C: BO 演習データ候補と benchmark 関数

> [!NOTE]
> 本付録は、vol-05 本文（第 5〜16 章）で確立した **21 フィールド provenance 契約** と **Ch14 §14.2.1 canonical 6-var search space** に整合する演習用データセットを整理する。vol-04 付録 C（因果推論用合成データ）と対を成し、BO / Active Learning Skill を **開発・検証・受け入れテスト** する目的で以下の 3 系列を提供する。
>
> - **系列 1（ARIM 風合成 BO データ）**: 真の目的関数 $f^\star(x)$ と装置差バイアス $\delta_{\text{facility}}(x)$ を持つ半合成データセット。Ch11 hierarchical GP と Ch15 §15.2 失敗モードのテストに使う
> - **系列 2（Olympus / Bayesian benchmark 関数）**: 材料 BO コミュニティで標準化されている外部ベンチマーク（Branin, Hartmann-6, Ackley, Rosenbrock, Schwefel, Levy）と Olympus dataset
> - **系列 3（Materials Project ラップ）**: 実物性 API を "測定コスト付き" にラップし、Ch5 §5.5 `stop_condition` の budget scope を実データで演習する
>
> **canonical 依存**:
>
> - Ch5 §5.2 21 フィールド provenance（グループ A/B/C/D）
> - Ch10 §10.6 `hard_constraints_ceramic_v0_2`
> - Ch14 §14.2.1 canonical search space（`x_temperature` / `x_pressure` / `x_hold_time` / `x_mass_frac_A` / `x_mass_frac_B` / `x_atmosphere`）
> - vol-04 付録 C §C.12 `causal_dag_ceramic_v1`（search space の SoT）

> [!IMPORTANT]
> **canonical schema 準拠ルール**（vol-05 全体で不変、本付録も継承）:
>
> - `x_atmosphere`: `{Ar=0, N2=1, Air=2}` の ordinal encoding、hamming kernel（Ch6 §6.4 / Ch14 §14.2.2）
> - `x_mass_frac_A + x_mass_frac_B <= 0.85`（Ch14 §14.2.1 linear constraint、原料 C が最低 15% 必要）
> - すべての観測データは Ch5 §5.2 21 フィールド provenance に対応する **event 単位** で export し、`event_hash` は RFC 8785 JCS + SHA-256（canonical anchor: `vol-05:ch10:event_canonical_serialization`）
> - `authorization_id`: `elauth_YYYYMMDD_HHMMSS_iter<n>`（Ch16 §16.1.1 H-R2-1）
> - `status` enum: **4 値** `pending | approved | denied | revoked`（Ch16 §16.1.1 H7）
> - **付録 C ローカル拡張**: 履歴データ import event に限り `authorization_applicable: false` + `experiment_launch_authorization: null` + `historical_import_provenance` を許容（本付録 C.6.4 定義、canonical 4-value status enum に該当しないため）。Ch16 監査ツールは `authorization_applicable == false` を優先評価する

---

## C.1 概観 — ARIM BO 演習データの位置づけ

### C.1.1 なぜ三段階を用意するのか

BO / AL Skill は、以下の 3 段階の検証を経ないと運用に耐えない:

1. **contract test**: Skill の 21 フィールド contract が **既知の真値** に対して壊れないかを検証する。**合成データ**が必要（真の $f^\star$ / 真の Pareto 前線 / 真の feasibility 集合が既知）
2. **benchmark test**: BO の性能を **他ライブラリ・他手法と比較可能** な形で測る。**標準ベンチマーク関数**が必要（BoTorch / Ax / Emukit の他ユーザと同じ土俵）
3. **realism test**: 実データの分布・外れ値・欠測パターンに Skill が耐えるかを見る。**実物性データを "測定コスト付き"** にラップして budget を回す

これら 3 段階を混ぜると、以下の失敗が起きる:

| 混同パターン | 失敗症状 | 対策 |
|---|---|---|
| 合成データだけで開発 | 実データの外れ値で surrogate が壊れる | 系列 3 の Materials Project ラップで stress test |
| benchmark 関数だけで開発 | Ch11 hierarchical GP / Ch10 hard_constraints が未検証のまま | 系列 1 の合成データで装置差 + 制約を明示的にテスト |
| 実データだけで開発 | 真の最適解を知らないので regret が計算できず、Skill の性能を証明できない | 系列 1 / 2 の合成データで regret curve を作り GA test に含める |

### C.1.2 各データセットが対応する章

| データセット ID | 種別 | 主用途 | 対応章 | 真値の既知性 |
|:---|:---|:---|:---|:---|
| **C-BO-01** | 合成（ARIM 風、単目的） | Ch6 GP surrogate / Ch7 acquisition | Ch6, Ch7, Ch14 §14.2 | ground truth $f^\star$ 既知 |
| **C-BO-02** | 合成（ARIM 風、多目的 2 目的） | Ch9 qLogEHVI / Pareto | Ch9, Ch14 §14.3 | Pareto 前線 semi-analytical |
| **C-BO-03** | 合成（ARIM 風、制約付き） | Ch10 cEI / hard_constraints | Ch10, Ch14 §14.3 | feasibility 集合既知 |
| **C-BO-04** | 合成（ARIM 風、装置差 3 拠点） | Ch11 hierarchical GP / MTGP | Ch11 | 装置差 $\delta_j(x)$ 既知 |
| **C-BO-05** | 合成（ARIM 風、batch） | Ch12 qLogNEI + pending | Ch12 | 同上 + pending fantasize |
| **C-AL-01** | 合成（分類、learning mode） | Ch13 uncertainty sampling | Ch13 | true boundary 既知 |
| **C-BM-01** | Bayesian benchmark（Branin） | Ch7 / Ch14 sanity check | 全章 sanity | 全大域解既知 |
| **C-BM-02** | Bayesian benchmark（Hartmann-6） | vol-05 6-var との次元一致 | Ch14 §14.2 | 6D 大域解既知 |
| **C-BM-03** | Bayesian benchmark（Ackley 30D） | 高次元 stress test | Ch15 §15.1（外挿） | 大域解既知 |
| **C-BM-04** | Bayesian benchmark（Rosenbrock / Schwefel / Levy） | 混合形状 stress | Ch15 | 大域解既知 |
| **C-OLY-01** | Olympus dataset（Photo_Wf3） | 光触媒 BO 事例 | Ch9 | dataset 実測、"真値" 相当 |
| **C-OLY-02** | Olympus dataset（HPLC） | クロマト最適化 | Ch9 / Ch10 | 同上 |
| **C-OLY-03** | Olympus dataset（SnAr） | 反応最適化 | Ch10 制約 | 同上 |
| **C-MP-01** | Materials Project ラップ | 実 API + 疑似コスト | Ch5 §5.5 stop_condition | Materials Project 実測 |
| **C-ARIM-01** | ARIM 実データ持ち込み | 実運用受け入れ | Ch16 §16.9 | 施設持込データ |

### C.1.3 v0.2 schema 準拠マトリクス

各データセットが Ch5 21 フィールド contract のどれをテストするか:

| ID | search_space_bounds | hard_constraints | surrogate_treatment_of_facility_variance | pending_experiments | hallucination_flag |
|:---|:---:|:---:|:---:|:---:|:---:|
| C-BO-01 | ✅ | (linear only) | not_applicable | – | ✅ |
| C-BO-02 | ✅ | (linear only) | not_applicable | – | ✅ |
| C-BO-03 | ✅ | ✅ | not_applicable | – | ✅ |
| C-BO-04 | ✅ | ✅ | hierarchical_shrinkage | – | ✅ |
| C-BO-05 | ✅ | ✅ | not_applicable | ✅ | ✅ |
| C-AL-01 | ✅ | – | – | – | ✅ (proxy) |
| C-BM-\* | ✅ (synth bounds) | – | not_applicable | – | ✅ |
| C-OLY-\* | ✅ | (varies) | not_applicable | – | ✅ |
| C-MP-01 | ✅ | (linear) | not_applicable | – | ✅ |
| C-ARIM-01 | ✅ | ✅ | (varies) | (optional) | ✅ |

---

## C.2 ARIM 風合成 BO データ生成スクリプト

### C.2.1 目的関数と装置差モデル

Ch14 §14.2.1 の 6 変数 search space に対して、**真の目的関数** $f^\star(x)$ と **装置差バイアス** $\delta_{\text{facility}}(x)$ を明示的に定義する。観測 $y$ は次式で生成する:

$$
y_{i,j} = f^\star(x_i) + \delta_{j}(x_i) + \varepsilon_{i,j}, \quad \varepsilon_{i,j} \sim \mathcal{N}(0, \sigma_j^2)
$$

ここで $j \in \{A, B, C\}$ は装置 index（Ch11 hierarchical GP のテストで 3 装置を使う）。真の関数は焼結収率風の物理妥当性を模した polynomial + Gaussian bump:

$$
f^\star(x) = A_0 - a_1(T - T^\star)^2 - a_2(P - P^\star)^2 - a_3(\tau - \tau^\star)^2 + B_0 \exp\!\left(-\|x_{\text{comp}} - x_{\text{comp}}^\star\|^2 / \ell^2\right) + \phi(\text{atm})
$$

- $T = x_{\text{temperature}} / 100$（無次元化）、$P = x_{\text{pressure}}$、$\tau = x_{\text{hold\_time}} / 60$
- $x_{\text{comp}} = (x_{\text{mass\_frac\_A}}, x_{\text{mass\_frac\_B}})$
- $\phi(\text{Ar}) = 0.05, \phi(\text{N2}) = 0.0, \phi(\text{Air}) = -0.10$（雰囲気効果）

**装置差** $\delta_j$ は多項式バイアス + additive shift:

- 装置 A（reference）: $\delta_A(x) = 0$、$\sigma_A = 0.02$
- 装置 B（bias）: $\delta_B(x) = -0.03 + 0.005(T - 5)$、$\sigma_B = 0.03$
- 装置 C（noisy）: $\delta_C(x) = +0.02$、$\sigma_C = 0.06$

### C.2.2 canonical 6-var search space の再掲

```yaml
search_space_bounds:
  version: "v0.2"
  derived_from: "vol-04:appendix-c:causal_dag_ceramic_v1"
  variables:
    x_temperature:   {type: continuous,  low: 200.0, high: 800.0, unit: degC}
    x_pressure:      {type: continuous,  low: 0.1,   high: 5.0,   unit: MPa}
    x_hold_time:     {type: continuous,  low: 30.0,  high: 180.0, unit: min}
    x_mass_frac_A:   {type: continuous,  low: 0.1,   high: 0.6,   unit: "1"}
    x_mass_frac_B:   {type: continuous,  low: 0.1,   high: 0.5,   unit: "1"}
    x_atmosphere:
      type: categorical
      choices: [Ar, N2, Air]
      ordinal_encoding: {Ar: 0, N2: 1, Air: 2}
      kernel: hamming
  linear_constraints:
    - name: mass_frac_sum
      expr: "x_mass_frac_A + x_mass_frac_B"
      operator: "<="
      threshold: 0.85
      source: causal_dag_derivation
  hard_constraints_ref: "vol-05:ch10:hard_constraints_ceramic_v0_2"
```

> [!TIP]
> 付録 A §A.2 の Skill テンプレート `arim.bo.single_objective_gp_ei.v0.2` の `search_space_bounds` セクションと完全に一致する。合成データを Skill contract test に流し込むときは、ここの YAML を `search_space_bounds` として渡すだけでよい。

### C.2.3 生成コード（`arim_synth_bo.py`）

```python
"""
ARIM 風合成 BO データ生成スクリプト (C-BO-01 〜 C-BO-05)。

真の目的関数 f*(x) と装置差 δ_j(x) を持ち、Ch11 hierarchical GP、
Ch10 hard_constraints、Ch12 batch BO のテストに使う。

依存: numpy, pandas
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

# ---- canonical 6-var search space (Ch14 §14.2.1) ----
BOUNDS = {
    "x_temperature": (200.0, 800.0),
    "x_pressure":    (0.1,   5.0),
    "x_hold_time":   (30.0,  180.0),
    "x_mass_frac_A": (0.1,   0.6),
    "x_mass_frac_B": (0.1,   0.5),
}
ATMOSPHERE_CHOICES = ["Ar", "N2", "Air"]
ATMOSPHERE_ORDINAL = {"Ar": 0, "N2": 1, "Air": 2}

# ---- 真の最適点（合成の解として設計上定義）----
X_STAR = {
    "T_star_scaled": 6.5,       # x_temperature=650 degC
    "P_star":        2.5,
    "tau_star_scaled": 1.75,    # x_hold_time=105 min
    "compA_star":    0.42,
    "compB_star":    0.28,
    "atm_star":      "Ar",
}

# ---- 装置差モデル ----
@dataclass(frozen=True)
class FacilityModel:
    facility_id: str
    bias_fn_name: str
    sigma: float


FACILITIES = {
    "A": FacilityModel("A", "zero",         0.02),   # reference
    "B": FacilityModel("B", "linear_temp", 0.03),   # T 依存 bias
    "C": FacilityModel("C", "constant",    0.06),   # noisy constant shift
}


def _true_objective(x: pd.DataFrame) -> np.ndarray:
    """f*(x): 収率風の polynomial + Gaussian bump + atmosphere effect."""
    T = x["x_temperature"].to_numpy() / 100.0
    P = x["x_pressure"].to_numpy()
    tau = x["x_hold_time"].to_numpy() / 60.0
    fA = x["x_mass_frac_A"].to_numpy()
    fB = x["x_mass_frac_B"].to_numpy()
    atm = x["x_atmosphere"].to_numpy()

    quad = (
        -0.010 * (T - X_STAR["T_star_scaled"]) ** 2
        - 0.020 * (P - X_STAR["P_star"]) ** 2
        - 0.030 * (tau - X_STAR["tau_star_scaled"]) ** 2
    )
    dist_comp2 = (fA - X_STAR["compA_star"]) ** 2 + (fB - X_STAR["compB_star"]) ** 2
    bump = 0.30 * np.exp(-dist_comp2 / 0.02)
    atm_effect = np.array(
        [{"Ar": 0.05, "N2": 0.0, "Air": -0.10}[a] for a in atm]
    )
    return 0.75 + quad + bump + atm_effect


def _facility_bias(x: pd.DataFrame, fac: FacilityModel) -> np.ndarray:
    T = x["x_temperature"].to_numpy() / 100.0
    if fac.bias_fn_name == "zero":
        return np.zeros(len(x))
    if fac.bias_fn_name == "linear_temp":
        return -0.03 + 0.005 * (T - 5.0)
    if fac.bias_fn_name == "constant":
        return np.full(len(x), 0.02)
    raise ValueError(f"unknown bias_fn: {fac.bias_fn_name}")


def sample_search_space(
    n: int, seed: int, respect_linear_constraint: bool = True
) -> pd.DataFrame:
    """search space からのランダムサンプル。x_mass_frac_A + x_mass_frac_B <= 0.85 を rejection sampling で満たす。

    Note: seed reproducibility は respect_linear_constraint の値に依存する
    （reject 数が異なると RNG stream の消費位置が変わる）。厳密再現が必要な場合は
    respect_linear_constraint を固定するか、rejection を経由しないサンプラを別途用意すること。
    """
    rng = np.random.default_rng(seed)
    rows = []
    max_rejections = 10000
    rejections = 0
    while len(rows) < n:
        row = {k: rng.uniform(lo, hi) for k, (lo, hi) in BOUNDS.items()}
        if respect_linear_constraint:
            if row["x_mass_frac_A"] + row["x_mass_frac_B"] > 0.85:
                rejections += 1
                if rejections > max_rejections:
                    raise RuntimeError(
                        "sample_search_space: 10000 回の rejection でも constraint 満たす候補を得られず。"
                        "search space bounds もしくは A+B<=0.85 制約を確認してください。"
                    )
                continue
        row["x_atmosphere"] = rng.choice(ATMOSPHERE_CHOICES)
        rows.append(row)
    return pd.DataFrame(rows)


def generate_dataset(
    dataset_id: str, n_per_facility: int, seed: int
) -> pd.DataFrame:
    """C-BO-01 〜 C-BO-04 の統一 generator。dataset_id で mode を切替。"""
    rng = np.random.default_rng(seed)
    frames = []
    facilities = ["A"] if dataset_id in ("C-BO-01", "C-BO-02", "C-BO-03", "C-BO-05") else list(FACILITIES.keys())
    for fac_id in facilities:
        fac = FACILITIES[fac_id]
        x = sample_search_space(n_per_facility, seed=rng.integers(0, 10**9))
        f_true = _true_objective(x)
        bias = _facility_bias(x, fac)
        noise = rng.normal(0.0, fac.sigma, size=len(x))
        y = f_true + bias + noise
        x = x.copy()
        x["facility_id"] = fac_id
        x["y_true"] = f_true          # ground truth（訓練には使わない、regret 計算用）
        x["y_observed"] = y
        frames.append(x)
    df = pd.concat(frames, ignore_index=True)
    df["dataset_id"] = dataset_id
    # ordinal encoding for x_atmosphere（Skill 入力用の追加列）
    df["x_atmosphere_ord"] = df["x_atmosphere"].map(ATMOSPHERE_ORDINAL)
    return df


if __name__ == "__main__":
    # 契約テスト: _true_objective(pd.DataFrame([X_optimum])) は 1.10 ± 1e-12 を返す
    #   （生成ロジックが将来変更された場合の early detection 用）
    for did, n, seed in [
        ("C-BO-01", 200, 20260704),
        ("C-BO-02", 200, 20260705),
        ("C-BO-03", 200, 20260706),
        ("C-BO-04",  80, 20260707),  # 3 facilities × 80 = 240 行
        ("C-BO-05", 200, 20260708),
    ]:
        df = generate_dataset(did, n_per_facility=n, seed=seed)
        # split into Skill-safe and groundtruth files with stable join key
        df = df.reset_index(drop=True)
        df["row_uid"] = [f"{did}_row_{i:06d}" for i in range(len(df))]
        df_skill_safe = df.drop(columns=["y_true"])
        out = f"{did.lower()}.parquet"
        df_skill_safe.to_parquet(out, index=False)
        # groundtruth は row_uid で必ず結合（positional 依存を回避）
        df[["row_uid", "y_true"]].to_parquet(
            f"{did.lower()}.groundtruth.parquet", index=False)
        print(f"[{did}] N={len(df)} facilities={df['facility_id'].unique().tolist()} -> {out}")
```

> [!IMPORTANT]
> Skill に渡す artifact は `{did}.parquet` (y_true 列を除いた版) を作成し、
> ground truth 参照は `{did}.groundtruth.parquet` に分離することを推奨（Ch15 §15.2.4 対策）。
> 両ファイルは `row_uid` 列で結合できるので、positional 順序に依存せず regret 計算が可能。
> 単一ファイル運用時は Skill 側で `df.drop(columns=["y_true"])` を厳格化すること。

> [!WARNING]
> `y_true` 列は **regret 計算専用**。Skill には絶対渡さない。Skill が `y_true` を surrogate 学習に使ってしまうと、Ch15 §15.2.4 "外挿を自信ありと報告" の逆パターン（**真値を覗き見して自信を偽造**）が起きる。データローダ側で `y_true` は Skill 契約入力から明示的に除外すること。

### C.2.4 期待 optimum と convergence 特性

C-BO-01 の設計上の真の最適点と、そこでの $f^\star$ の値:

```yaml
c_bo_01_ground_truth:
  x_optimum:
    x_temperature: 650.0     # T_scaled = 6.5
    x_pressure: 2.5
    x_hold_time: 105.0       # tau_scaled = 1.75
    x_mass_frac_A: 0.42
    x_mass_frac_B: 0.28
    x_atmosphere: Ar
  f_star_at_optimum: 1.10    # 0.75 + 0.30 (bump) + 0.05 (atm=Ar)
  expected_convergence:
    iterations_to_regret_below_0_05: 25   # qLogEI, 8 init points
    iterations_to_regret_below_0_02: 45
    final_regret_at_iter_50:        0.015
  hallucination_expected:
    at_iter_below: 5           # 初期 5 iter は search_space 端で外挿 flag が立つ想定
    at_iter_above: 40          # convergence 後は再 flag しない想定
```

### C.2.5 装置差の統計的性質（Ch11 hierarchical GP テスト用）

C-BO-04 は 3 装置 × 各 80 点 = 240 点で、Ch11 §11.x hierarchical / MTGP のパラメータ回復テストに使う:

```yaml
c_bo_04_facility_variance_ground_truth:
  reference_facility: A
  facility_bias_functions:
    A: {form: "zero",         params: {},                     sigma_noise: 0.02}
    B: {form: "linear_temp",  params: {intercept: -0.03, slope: 0.005, anchor_T_scaled: 5.0}, sigma_noise: 0.03}  # bias = intercept + slope*(T_scaled - anchor_T_scaled)
    C: {form: "constant",     params: {shift: +0.02},          sigma_noise: 0.06}
  expected_hierarchical_gp_recovery:
    # NOTE: 以下の閾値は illustrative reference。CI での strict acceptance criteria として
    # 用いる場合は N >= 20 seeds を fit した mean ± 2σ で校正すること。
    task_covariance_rank:           2   # A,B は相関、C は独立寄り
    inter_task_correlation_AB_min:  0.5
    inter_task_correlation_AC_max:  0.3
    per_task_noise_ratio_C_over_A_min: 2.5
  surrogate_treatment_of_facility_variance: hierarchical_shrinkage   # Ch5 §5.2 canonical enum
```

> [!TIP]
> Ch11 hierarchical GP Skill を実装するときは、**このデータで `inter_task_correlation_AB` が 0.5 以上に推定されるか** を acceptance test に加える。それが取れなければ kernel 設計か prior が壊れている。

### C.2.6 provenance に埋める synthetic 由来の宣言

Ch5 §5.2 21 フィールド provenance のうち、合成データを使ったことを **明示的に宣言** するフィールド:

```yaml
observed_data_source_declaration:
  dataset_id: "C-BO-01"
  origin: "synthetic"                      # enum canonical: {synthetic | benchmark_function | public_api | facility_real | emulated}
  generator_ref: "vol-05:appendix-c:arim_synth_bo.py"
  generator_git_sha: "sha256:<git head>"   # 実運用では埋める
  seed_root: 20260704
  ground_truth_available: true
  ground_truth_leak_risk: "isolated_column_y_true"
  disclosure_in_publications: "must_cite_appendix_c_2"
```

---

## C.3 Olympus benchmark 連携

### C.3.1 Olympus とは何か

Olympus（Häse et al., 2021, Aspuru-Guzik 研）は、**実際の実験ラン** をもとに構築された材料化学 BO ベンチマークライブラリ。ARIM が扱う「装置あり・コストあり・制約あり」の設定に近い dataset を提供する。BoTorch 単独では合成関数しかない不足を埋める:

- **論文実験の再現性**: 各 dataset に元論文がある
- **surrogate ベース**: 実測点を interpolate した surrogate を "疑似ブラックボックス" として使うので、任意の $x$ でクエリできる
- **material domain 準拠**: 化学合成、光触媒、HPLC 条件など、ARIM 施設で扱う類似ドメイン

### C.3.2 インストールと最小コード

> [!NOTE]
> Olympus API は release により差異あり。以下は 2024 版準拠。実際の invoke 前に `help(Emulator)` で署名確認を推奨。

```bash
pip install olymp>=0.2  # or: pip install git+https://github.com/aspuru-guzik-group/olympus.git
```

```python
"""
Olympus dataset を BO Skill に流し込む最小例。
"""
from olympus.datasets import Dataset
from olympus import Emulator, BayesNeuralNet
import numpy as np


def load_olympus_dataset(name: str = "photo_wf3"):
    """dataset をロードし、param_space / value_space を返す。"""
    ds = Dataset(kind=name)
    return {
        "param_space": ds.param_space,   # 連続 / 離散変数の仕様
        "value_space": ds.value_space,   # 目的値の仕様
        "data": ds.data,                 # 実測 (X, y) DataFrame
    }


def olympus_as_blackbox(name: str):
    """Emulator を "疑似実験装置" として返す。BO Skill から関数として呼ばれる。"""
    emu = Emulator(dataset=name, model=BayesNeuralNet())
    emu.train()  # training は Emulator コンストラクタで自動実行される場合もあり (Olympus 版に依存)

    def blackbox(x: dict) -> float:
        arr = np.array([[x[p] for p in emu.param_space.param_names]])
        y = emu.run(features=arr, num_samples=1)
        return float(np.mean(y))

    return blackbox


if __name__ == "__main__":
    ds = load_olympus_dataset("photo_wf3")
    print(f"Photo_Wf3 params: {[p.name for p in ds['param_space']]}")
    bb = olympus_as_blackbox("photo_wf3")
    sample_x = {p.name: (p.low + p.high) / 2 for p in ds["param_space"]}
    print(f"blackbox at midpoint: {bb(sample_x):.4f}")
```

### C.3.3 ARIM workflow への埋め込み（Ch5 provenance 付与）

Olympus 由来のクエリは **`olympus_dataset_id`** を Ch5 §5.2 provenance に追加宣言する。合成でも実測でもない **"emulated"** 由来として扱う:

```yaml
observed_data_source_declaration:
  dataset_id: "C-OLY-01"
  origin: "emulated"                       # enum canonical: {synthetic | benchmark_function | public_api | facility_real | emulated}
  emulator: "olympus"
  olympus_dataset_id: "photo_wf3"
  olympus_emulator_model: "BayesNeuralNet"
  olympus_version: ">=0.2"
  emulator_training_seed: 20260704
  ground_truth_available: false            # emulator の予測は真値ではない
  citation_required: "Häse et al., 2021, Matter"
```

> [!IMPORTANT]
> Olympus emulator の予測は **surrogate の surrogate** であり、真の物理を保証しない。**ground_truth_available: false** を明示的に立て、regret 計算には使わない（あくまで BO の "動作" 検証に使う）。真の regret を要求する検証は C-BO-01〜05 で行う。

### C.3.4 dataset 表と ARIM 章対応

| Olympus dataset | ドメイン | 変数数 | 目的 | ARIM 対応章 |
|:---|:---|:---:|:---|:---|
| `photo_wf3` | 光触媒 quantum yield | 4 | 単目的 | Ch7 |
| `hplc` | HPLC 分離条件 | 6 | 多目的（分離度・時間） | Ch9 |
| `snar` | SnAr 反応最適化 | 4 | 単目的（収率） | Ch7 |
| `benzylation` | ベンジル化反応 | 4 | 単目的 | Ch7 |
| `alkox` | アルコキシ化 | 6 | 制約付き単目的 | Ch10 |
| `crossed_barrel` | 材料強度実験 | 4 | 単目的 | Ch14 sanity |
| `perovskites` | ペロブスカイト stability | 3 | 単目的 | Ch11（拡張） |

---

## C.4 Bayesian benchmark 関数

### C.4.1 Branin-Hoo（2D）

Branin-Hoo は BO 論文で **必ず** 出てくる 2D テスト関数。全大域最適点が 3 つ、局所解なし、大域最適で $f = 0.397887$。ARIM 6 変数と次元は違うが、**BO Skill の疎通確認** に使う:

$$
f_{\text{Branin}}(x_1, x_2) = a(x_2 - bx_1^2 + cx_1 - r)^2 + s(1 - t)\cos(x_1) + s,
$$

with $a=1$, $b=5.1/(4\pi^2)$, $c=5/\pi$, $r=6$, $s=10$, $t=1/(8\pi)$, $x_1 \in [-5, 10]$, $x_2 \in [0, 15]$。

### C.4.2 Hartmann-6（6D）

**vol-05 canonical search space と同じ 6D** なので、Ch14 §14.2 の Skill を Hartmann-6 で sanity check できる:

$$
f_{\text{Hartmann6}}(x) = -\sum_{i=1}^{4} \alpha_i \exp\!\left(-\sum_{j=1}^{6} A_{ij}(x_j - P_{ij})^2\right)
$$

$\alpha = (1.0, 1.2, 3.0, 3.2)$、$A, P$ は既知定数。$x_j \in [0, 1]$、大域最適 $f^\star \approx -3.32237$、$x^\star \approx (0.20169, 0.15001, 0.47687, 0.27533, 0.31165, 0.6573)$。

### C.4.3 Ackley（30D）

**高次元 stress test**。Ackley 関数は多数の局所解を持ち、BO が高次元で破綻するかどうかを見る。ARIM 実運用では 6 変数しか使わないが、**Ch15 §15.1 "外挿誤用"** のテストとして「BO を敢えて 30D で回すと何が起きるか」を検証する用途。

### C.4.4 Rosenbrock / Schwefel / Levy（各 briefly）

| 関数 | 特徴 | 用途 |
|:---|:---|:---|
| **Rosenbrock** | 曲がった谷、局所解ほぼ無、勾配 anisotropic | ARD length scale の回復テスト |
| **Schwefel** | 大量の deceptive 局所解、大域解が端に偏る | acquisition の探索性テスト |
| **Levy** | 中程度の局所解、smooth | 標準 BO sanity |

### C.4.5 Python 実装（botorch.test_functions 活用）

```python
"""
Bayesian benchmark 関数の統一 wrapper。
BoTorch の test_functions を使いつつ、ARIM Skill 契約用の
YAML metadata を付与する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import torch

try:
    from botorch.test_functions import (
        Branin, Hartmann, Ackley, Rosenbrock, Levy
    )
    _BOTORCH_AVAILABLE = True
except ImportError:
    _BOTORCH_AVAILABLE = False


@dataclass
class BenchmarkSpec:
    id: str
    dim: int
    bounds: list                                 # [(low, high), ...] length = dim
    global_minimum: float
    global_minimizer: list                       # length = dim
    negate_for_maximization: bool = True         # BoTorch は minimize 前提
    provenance: dict = field(default_factory=dict)


BENCHMARKS = {
    "C-BM-01": BenchmarkSpec(
        id="C-BM-01",
        dim=2,
        bounds=[(-5.0, 10.0), (0.0, 15.0)],
        global_minimum=0.397887,
        global_minimizer=[-np.pi, 12.275],
        provenance={"function": "Branin-Hoo", "reference": "Dixon & Szego 1978"},
    ),
    "C-BM-02": BenchmarkSpec(
        id="C-BM-02",
        dim=6,
        bounds=[(0.0, 1.0)] * 6,
        global_minimum=-3.32237,
        global_minimizer=[0.20169, 0.15001, 0.47687, 0.27533, 0.31165, 0.6573],
        provenance={"function": "Hartmann-6", "matches_arim_dim": True},
    ),
    "C-BM-03": BenchmarkSpec(
        id="C-BM-03",
        dim=30,
        bounds=[(-32.768, 32.768)] * 30,
        global_minimum=0.0,
        global_minimizer=[0.0] * 30,
        provenance={"function": "Ackley-30D", "stress_test": True},
    ),
    "C-BM-04a": BenchmarkSpec(
        id="C-BM-04a", dim=6, bounds=[(-5.0, 10.0)] * 6,
        global_minimum=0.0, global_minimizer=[1.0] * 6,
        provenance={"function": "Rosenbrock-6D"},
    ),
    "C-BM-04b": BenchmarkSpec(
        id="C-BM-04b", dim=6, bounds=[(-500.0, 500.0)] * 6,
        global_minimum=0.0, global_minimizer=[420.9687] * 6,
        provenance={"function": "Schwefel-6D"},
    ),
    "C-BM-04c": BenchmarkSpec(
        id="C-BM-04c", dim=6, bounds=[(-10.0, 10.0)] * 6,
        global_minimum=0.0, global_minimizer=[1.0] * 6,
        provenance={"function": "Levy-6D"},
    ),
}


def _pure_python_ackley(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1]
    a, b, c = 20.0, 0.2, 2 * np.pi
    s1 = np.sum(x ** 2, axis=-1) / d
    s2 = np.sum(np.cos(c * x), axis=-1) / d
    return -a * np.exp(-b * np.sqrt(s1)) - np.exp(s2) + a + np.e


def make_callable(spec: BenchmarkSpec) -> Callable[[np.ndarray], np.ndarray]:
    """BenchmarkSpec から Skill 用の関数を返す。BoTorch があれば使い、なければ純 numpy fallback。"""
    if _BOTORCH_AVAILABLE:
        func_map = {
            "C-BM-01": Branin(negate=spec.negate_for_maximization),
            "C-BM-02": Hartmann(dim=6, negate=spec.negate_for_maximization),
            "C-BM-03": Ackley(dim=30, negate=spec.negate_for_maximization),
            "C-BM-04a": Rosenbrock(dim=6, negate=spec.negate_for_maximization),
            "C-BM-04c": Levy(dim=6, negate=spec.negate_for_maximization),
        }
        if spec.id in func_map:
            f = func_map[spec.id]

            def wrapped(x: np.ndarray) -> np.ndarray:
                t = torch.as_tensor(x, dtype=torch.double)
                return f(t).detach().cpu().numpy()

            return wrapped
    # fallback: Ackley だけ純 numpy 実装を提供
    if spec.id == "C-BM-03":
        if spec.negate_for_maximization:
            return lambda x: -_pure_python_ackley(x)
        return _pure_python_ackley
    raise NotImplementedError(f"install botorch for {spec.id}")


if __name__ == "__main__":
    for bid, spec in BENCHMARKS.items():
        try:
            f = make_callable(spec)
            x0 = np.array([spec.global_minimizer])
            expected = (-1.0 if spec.negate_for_maximization else 1.0) * spec.global_minimum
            print(f"[{bid}] dim={spec.dim} f(x*)={float(f(x0).ravel()[0]):+.4f} "
                  f"(expected: {expected:+.4f})")
        except NotImplementedError as e:
            print(f"[{bid}] skipped: {e}")
```

### C.4.6 参考ベンチマーク結果表

各関数 × acquisition {qLogEI, qUCB} × 50 iter × 8 initial LHS × 20 seed（平均 ± 標準誤差）:

| Function | Dim | qLogEI regret@50 | qUCB regret@50 | qLogEI conv iter | Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| Branin | 2 | 0.001 ± 0.0002 | 0.003 ± 0.0008 | ~15 | 全 acq が容易に解く |
| Hartmann-6 | 6 | 0.020 ± 0.005 | 0.045 ± 0.010 | ~35 | ARIM と同次元、qLogEI 優位 |
| Ackley-30 | 30 | 5.2 ± 0.3 | 6.1 ± 0.4 | 収束せず | 高次元 stress、GP 適用外の警告 |
| Rosenbrock-6 | 6 | 1.5 ± 0.4 | 2.1 ± 0.5 | ~40 | anisotropic、ARD 必須 |
| Schwefel-6 | 6 | 250 ± 60 | 320 ± 80 | 収束せず | deceptive、探索優位 acq が有利 |
| Levy-6 | 6 | 0.05 ± 0.01 | 0.09 ± 0.02 | ~30 | 標準的挙動 |

> [!NOTE]
> C.4.6 の regret 値は **illustrative reference** であり strict acceptance threshold ではない。
> 再現には N >= 20 seeds × pinned BoTorch version で mean ± 2σ を計算すること。
> 特に Hartmann-6 で `regret@50 <= 0.05` が達成できなければ、Ch14 §14.2 の canonical BO Skill 実装に問題がある可能性が高い。

---

## C.5 Materials Project "測定コスト付き" ラップ

### C.5.1 Materials Project API の概観

[Materials Project](https://materialsproject.org)（LBNL, DOE 支援）は DFT 計算に基づく 150,000+ 材料の物性データベース。`pymatgen` の `MPRester` で REST API 経由でクエリできる。BO 演習では **候補材料の物性を "実測相当" として扱い、クエリに疑似コストを付ける** ことで budget scope を回す。

### C.5.2 なぜ "測定コスト付き" にラップするのか

Materials Project API 自体は **無料・無制限** に近い（API key 登録は必要）。しかし BO 演習では:

- 実験装置予約や試薬コストを **模擬** する必要がある
- Ch5 §5.5 `stop_condition.budget` を実データで回す訓練が必要
- Ch14 §14.2 の **iteration ごとに budget を消費する** ループを検証したい

ため、**形式的なコストモデル** をラップ層で付与する。実際の API 呼び出しコストではなく、"仮想実験時間 + 仮想費用" として管理する:

### C.5.3 コストモデル

```yaml
mp_costwrap_cost_model:
  version: "v0.2"
  cost_per_query:
    time_hours: 2.0                   # 1 材料の物性照会 = 2 時間の仮想装置予約
    money_jpy: 3000                   # 1 材料 = 3000 円の仮想試薬 + 消耗品
  discount_rules:
    batch_of_5_or_more:
      time_multiplier: 0.75           # batch 割引（Ch12 §12.x の batch 効果）
      money_multiplier: 0.80
  budget_stop_condition_hookup:
    budget_scope_variable: "money_jpy"
    default_total_budget_jpy: 300000
    default_total_budget_hours: 200
```

### C.5.4 ラップ層コード（`mp_costwrap.py`）

```python
"""
Materials Project を "測定コスト付き" にラップするミニ層。
Ch5 §5.5 stop_condition.budget と結合する。

依存: pymatgen (>=2023.x), python-dotenv (任意)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import os


@dataclass
class BudgetState:
    total_money_jpy: float = 300000.0
    total_hours: float = 200.0
    spent_money_jpy: float = 0.0
    spent_hours: float = 0.0
    query_log: list = field(default_factory=list)

    def can_afford(self, cost_money: float, cost_hours: float) -> bool:
        return (
            self.spent_money_jpy + cost_money <= self.total_money_jpy
            and self.spent_hours + cost_hours <= self.total_hours
        )

    def charge(self, cost_money: float, cost_hours: float, query: dict) -> None:
        self.spent_money_jpy += cost_money
        self.spent_hours += cost_hours
        self.query_log.append(
            {"query": query, "money_jpy": cost_money, "hours": cost_hours}
        )


class MPCostWrapper:
    """Materials Project クエリに時間 + 金額の疑似コストを付与する層。"""

    UNIT_MONEY_JPY = 3000.0
    UNIT_HOURS = 2.0

    def __init__(self, api_key: str | None = None, budget: BudgetState | None = None):
        try:
            from pymatgen.ext.matproj import MPRester
        except ImportError as e:
            raise ImportError("pip install pymatgen") from e
        key = api_key or os.environ.get("MP_API_KEY")
        if not key:
            raise ValueError("Materials Project API key not provided")
        self._mp = MPRester(key)
        self.budget = budget or BudgetState()

    def _batch_cost(self, n: int) -> tuple[float, float]:
        money = n * self.UNIT_MONEY_JPY
        hours = n * self.UNIT_HOURS
        # NOTE: batch discount 係数は C.5.3 mp_costwrap_cost_model YAML の
        # {money_multiplier: 0.80, time_multiplier: 0.75} と同期。将来的には YAML から load 推奨。
        if n >= 5:
            money *= 0.80
            hours *= 0.75
        return money, hours

    def query_material(self, mp_ids: list[str], fields: list[str]) -> dict[str, Any]:
        """budget 内なら Materials Project にクエリ、超過なら fatal。"""
        money, hours = self._batch_cost(len(mp_ids))
        if not self.budget.can_afford(money, hours):
            raise RuntimeError(
                f"budget_exceeded: money={self.budget.spent_money_jpy}+"
                f"{money} > {self.budget.total_money_jpy}"
            )
        results = self._mp.materials.summary.search(
            material_ids=mp_ids, fields=fields
        )
        self.budget.charge(money, hours, {"mp_ids": mp_ids, "fields": fields})
        return {"results": results, "budget_state": self.budget}


if __name__ == "__main__":
    # 演習例：TiO2 系材料の band_gap を BO 目的にする
    wrap = MPCostWrapper(budget=BudgetState(total_money_jpy=50000, total_hours=40))
    try:
        r = wrap.query_material(
            mp_ids=["mp-2657", "mp-390"],
            fields=["material_id", "formula_pretty", "band_gap"],
        )
        print(f"spent: {wrap.budget.spent_money_jpy} JPY, "
              f"{wrap.budget.spent_hours} h, "
              f"{len(wrap.budget.query_log)} queries")
    except RuntimeError as e:
        print(f"budget guard triggered (expected in demo): {e}")
```

### C.5.5 ARIM Ch10 hard_constraints へのマッピング

Materials Project のクエリ結果を Ch10 §10.6 `hard_constraints` 相当に翻訳する:

```yaml
mp_costwrap_hard_constraints_mapping:
  version: "v0.2"
  constraints:
    - name: "band_gap_upper"
      variable: "band_gap"
      operator: "<="
      threshold: 6.0
      unit: "eV"
      rationale: "光触媒応用として現実的な上限"
      enforcement: "pre_authorization_filter"
      violation_action: "skip_candidate"
    - name: "band_gap_lower"
      variable: "band_gap"
      operator: ">="
      threshold: 1.0
      unit: "eV"
      rationale: "半導体特性が必要"
      enforcement: "pre_authorization_filter"
      violation_action: "skip_candidate"
    - name: "formation_energy_stability"
      variable: "formation_energy_per_atom"
      operator: "<="
      threshold: 0.0
      unit: "eV/atom"
      rationale: "合成可能性（安定相）"
      enforcement: "pre_authorization_filter"
      violation_action: "skip_candidate"
```

### C.5.6 責任範囲: Materials Project のライセンスと再配布

> [!WARNING]
> Materials Project データは **CC-BY-4.0** ライセンス。演習で取得したデータを:
>
> - **再配布する場合**: 出典（"Data from Materials Project"）と DOI 引用が必須
> - **論文発表する場合**: [Materials Project 引用ページ](https://materialsproject.org/about) の推奨引用を追加
> - **本書の演習コード出力を GitHub 等で公開する場合**: raw dump は含めず、`mp_ids` と `fields` のクエリ仕様を含めた再現用スクリプトのみ公開する
>
> ARIM 施設運用側は、演習用の "研修環境" API key と、実プロジェクト用の API key を分けて管理し、コスト境界（本節の budget model）を **CI で強制** することが望ましい。

---

## C.6 ARIM 実データ持ち込みガイド

### C.6.1 匿名化の必要性

ARIM 施設で得られた **実験データ** を演習・書籍・GitHub で使う場合、以下 3 系統の情報保護が必要:

1. **施設情報**: 装置固有 ID、施設名、担当者名、シフト表など内部運用情報
2. **発案者情報**: 実験計画立案者、共同研究者、研究テーマ名など人の帰属情報
3. **特許出願前データ**: 未公開の材料組成、プロセス条件、性能数値（**匿名化しても identifiable な場合は持込禁止**）

### C.6.2 匿名化手順（YAML チェックリスト）

```yaml
arim_data_anonymization_checklist:
  version: "v0.2"
  applied_at: null                                # YYYY-MM-DD で埋める
  reviewer_role: "arim_data_steward"              # 担当ロール
  approver_role: "arim_pi_or_delegated"           # 承認ロール
  checks:
    facility_information:
      - {id: "AN-F-01", item: "facility_id を facility_A / B / C など匿名 label に置換", required: true}
      - {id: "AN-F-02", item: "装置メーカー・型番を除去またはカテゴリラベル化", required: true}
      - {id: "AN-F-03", item: "シフト時刻・予約 slot を YYYY-MM-DD の日付粒度に丸め", required: true}
      - {id: "AN-F-04", item: "施設内部 URL / パス情報を除去", required: true}
    personnel_information:
      - {id: "AN-P-01", item: "実験実行者名を staff_XX の匿名 ID に置換", required: true}
      - {id: "AN-P-02", item: "承認者・PI 氏名を role 名（pi, coordinator）に置換", required: true}
      - {id: "AN-P-03", item: "共著者・共同研究先の組織名を除去", required: true}
      - {id: "AN-P-04", item: "email / 内線 / チャット ID を除去", required: true}
    material_identifiability:
      - {id: "AN-M-01", item: "材料組成の商用名 → 化学式に置換", required: true}
      - {id: "AN-M-02", item: "特許出願前の組成範囲は範囲を broaden するか除外", required: true}
      - {id: "AN-M-03", item: "サプライヤ・ロット番号を除去", required: true}
      - {id: "AN-M-04", item: "内部プロジェクト code name を汎用 code に置換", required: true}
    provenance_scrubbing:
      - {id: "AN-V-01", item: "authorization_id 中の staff ID / date の staff 部分を匿名化", required: true}
      - {id: "AN-V-02", item: "event_hash の再計算を行い chain 整合性を維持", required: true}
      - {id: "AN-V-03", item: "sharing_authorization_id の宛先を role 名に置換", required: true}
      - {id: "AN-V-04", item: "dag_of_record の内部リビジョン URI を汎用 URI に置換", required: true}
    optional_hardening:
      - {id: "AN-O-01", item: "連続変数の絶対値を dimensionless / normalized 値に変換", required: false}
      - {id: "AN-O-02", item: "date を epoch 相対の day offset に変換", required: false}
      - {id: "AN-O-03", item: "y の絶対値を rescaling して非公開閾値を隠す", required: false}
```

> [!IMPORTANT]
> `required: true` の項目は **全部 pass しないと持込禁止**。`optional_hardening` は特に公開範囲が広い（GitHub public, arXiv preprint）ときに追加適用する。

### C.6.3 search space 定義のレビュー手順

匿名化しても、**search space 定義そのもの** が identifiable な場合がある（例: ある材料系でのみ意味を持つ boundary）。以下 5 観点でレビュー:

```yaml
search_space_review_procedure:
  version: "v0.2"
  reviewers: ["arim_data_steward", "domain_scientist_delegate"]
  observations:
    - id: "SS-R-01"
      focus: "変数選択の妥当性"
      questions:
        - "6 変数以外に真に影響する変数が省略されていないか（vol-04 因果 DAG 参照）"
        - "定数として固定した変数の値が過度に identifiable でないか"
    - id: "SS-R-02"
      focus: "単位の一貫性"
      questions:
        - "全変数の unit が明示され SI 系で書かれているか"
        - "x_temperature = degC vs K の混在がないか"
    - id: "SS-R-03"
      focus: "範囲の妥当性"
      questions:
        - "bounds が装置耐性・化学的妥当性・因果 DAG 由来のどれに基づくか明示されているか"
        - "bounds を過度に狭く取っていて BO の探索が意味を成さなくなっていないか"
    - id: "SS-R-04"
      focus: "制約の完全性"
      questions:
        - "linear / nonlinear / categorical whitelist の全種類が declare されているか"
        - "Ch10 hard_constraints_ref が付与されているか"
    - id: "SS-R-05"
      focus: "変数間相関"
      questions:
        - "vol-04 因果 DAG で示された相関構造が反映されているか"
        - "冗長変数（mediator など）が入っていないか"
  sign_off:
    required_approvals: 2
    approval_ids_template: "approval:sr-YYYYMMDD-<seq>"
```

### C.6.4 データ持ち込み時の記録テンプレート（Ch5 §5.2 21 フィールド）

ARIM 実データを演習に組み込むときの provenance テンプレート:

```yaml
arim_realdata_ingestion_record:
  version: "v0.2"
  dataset_id: "C-ARIM-01"                        # 演習側の ID
  origin: "facility_real"                              # enum canonical: {synthetic | benchmark_function | public_api | facility_real | emulated}
  origin_facility_ref: "vol-05:appendix-c:facility_A_anonymized"
  ingested_at: "2026-07-07T09:33:20+09:00"
  anonymization_checklist_ref: "arim_data_anonymization_checklist:v0.2"
  anonymization_completed_at: "2026-07-05T10:00:00+09:00"
  anonymization_approval_id: "approval:anon-20260705-01"
  search_space_review_id: "approval:sr-20260706-01"
  # ---- Ch5 §5.2 21 フィールド互換の抜粋 ----
  iteration_index: 0                             # 持ち込み時は 0
  search_space_bounds:
    version: "v0.2"
    derived_from: "vol-04:appendix-c:causal_dag_ceramic_v1"
    dag_of_record_sha256: "sha256:aaaa..."       # 実データ用に更新
    variables:
      x_temperature: {type: continuous, low: 200.0, high: 800.0, unit: degC}
      x_pressure:    {type: continuous, low: 0.1,   high: 5.0,   unit: MPa}
      x_hold_time:   {type: continuous, low: 30.0,  high: 180.0, unit: min}
      x_mass_frac_A: {type: continuous, low: 0.1,   high: 0.6,   unit: "1"}
      x_mass_frac_B: {type: continuous, low: 0.1,   high: 0.5,   unit: "1"}
      x_atmosphere:  {type: categorical, choices: [Ar, N2, Air]}
  bo_library_stack: "botorch_direct"
  surrogate_model_family: "single_task_gp"
  retraining_policy: "every_iteration"
  observed_data_size: 42                          # 匿名化済み観測点数
  observed_data_hash: "sha256:bbbb..."            # 参照実データの hash
  facility_id_encoding: "facility_A_anonymized"
  surrogate_treatment_of_facility_variance: "categorical_input"
  hallucinated_recommendation_detection:
    declared: true
    operational_ref: "vol-05:ch07:hrd_surrogate_v0_1"
  # 履歴データ import はオペレーショナル契約として本付録 C.6.4 に定義するローカル例外パターン
  # （canonical Ch5 §5.3 の experiment_launch_authorization は "新規実験実施" 前提のため、
  #  過去の測定データを ARIM 資産として取り込む import event には適用不能）。
  authorization_applicable: false                    # historical-data import event なので elauth 発行対象外
  experiment_launch_authorization: null              # canonical 4-value status enum (pending|approved|denied|revoked) には該当しない
                                                     # 監査時は authorization_applicable: false で識別する（本付録 C.6.4 import 例外パターン）
  historical_import_provenance:                      # ingestion source と anonymization 承認を pin する
    ingested_from: "arim_facility_A_2025_archive"
    anonymization_performed_by: "^human:staff_A0007#anonymization-2026-07-07"
    anonymization_approval_id: "approval:anon-20260705-01"
    ingestion_boundary_date: "2026-07-01"            # この日以降の実験は elauth 経由で登録する
  parent_authorization_id: "vol-04:L3:l3_auth_20260501_120000_iter0"  # vol-04 L3 intervention authorization; vol-05 canonical elauth_ とは別 layer
```

### C.6.5 監査可能性: dataset_hash と import event の記録

Ch5 §5.2 の `event_hash` チェーンに、**dataset import** を明示的な event として append する:

```yaml
dataset_import_event:
  event_id: "import_C-ARIM-01_20260707T093320"
  event_type: "dataset_import"
  timestamp: "2026-07-07T09:33:20+09:00"
  actor: "^human:arim_data_steward"                  # import は human-driven action として記録（本付録 C.6.4 例外パターン）
  human_approver: "^human:staff_A0007"
  dataset_id: "C-ARIM-01"
  dataset_hash: "sha256:bbbb..."
  anonymization_checklist_hash: "sha256:cccc..."
  search_space_review_hash: "sha256:dddd..."
  previous_event_hash: "sha256:eeee..."            # チェーンの直前 event
  # RFC 8785 JCS + SHA-256 で以下 payload から計算
  event_hash: "sha256:ffff..."
```

> [!TIP]
> `dataset_import_event` は **Ch16 §16.1.1 authorization_events_stream** の event 型として back-registration する。監査時に「この BO ラン は正しく anonymized data を使ったか」を追跡可能にするため。

---

## C.7 dataset マトリクスと選択ガイド

### C.7.1 演習別 dataset マップ

| 演習の目的 | 推奨 dataset | 補助 dataset |
|:---|:---|:---|
| Ch6 GP surrogate の fit と length_scale 診断 | C-BO-01 | C-BM-02 (Hartmann-6) |
| Ch7 acquisition の比較（qLogEI vs qUCB） | C-BM-01 (Branin) | C-BO-01 |
| Ch7 hallucination flag のテスト | C-BO-01（bounds 外を強制注入） | C-BM-03 (Ackley 30D) |
| Ch9 多目的 Pareto | C-BO-02 | C-OLY-02 (HPLC) |
| Ch10 hard_constraints | C-BO-03 | C-MP-01（band_gap 制約） |
| Ch11 hierarchical GP | C-BO-04（3 施設） | – |
| Ch12 batch BO | C-BO-05 | C-BO-01 (batch_size=1 と比較) |
| Ch13 Active Learning | C-AL-01 | – |
| Ch14 §14.2 basic loop の sanity | C-BM-02 (Hartmann-6) | C-BO-01 |
| Ch14 §14.3 integrated case | C-BO-05 + C-BO-04 | – |
| Ch15 §15.2 Agentic 失敗モードの injection | C-BO-01（監査 dry-run） | – |
| Ch16 §16.9 施設運用受け入れ | C-ARIM-01 | – |

### C.7.2 trade-off 表（computational cost / realism / reproducibility）

| Dataset | 計算コスト | 実物性 realism | 再現性 | 真値既知 | 導入コスト |
|:---|:---:|:---:|:---:|:---:|:---:|
| C-BO-01 〜 05 | 極小（純 numpy） | 中（合成だが物理妥当） | ◎ | ◎ | 極小 |
| C-BM-01 〜 04 | 小（BoTorch API） | 低（純数学関数） | ◎ | ◎ | 小 |
| C-OLY-01 〜 03 | 中（emulator 訓練込み） | 中〜高（実論文由来） | ○ | △（emulator） | 中 |
| C-MP-01 | 中（API round-trip） | 高（実物性 DFT） | △（API 版依存） | ✗（真値なし） | 中（API key 要） |
| C-ARIM-01 | 中〜高（施設連携） | ◎（実験実測） | △（匿名化前後） | ✗ | 大（匿名化・レビュー） |

> [!NOTE]
> **選択の原則**: 開発初期は C-BO-01 と C-BM-02 で **contract test を機械的に回す**。中期に C-OLY / C-MP で **実データ耐性** を見る。最終段階で C-ARIM-01 の **持ち込みレビューを通してから** 施設運用に投入する。

---

## C.8 章末チェックリスト

### C.8.1 演習データ準備前の pre-flight check

```yaml
pre_flight_checklist:
  version: "v0.2"
  items:
    - {id: "PF-01", item: "search_space_bounds が Ch14 §14.2.1 canonical と一致するか"}
    - {id: "PF-02", item: "x_atmosphere の ordinal encoding {Ar:0, N2:1, Air:2} を使っているか"}
    - {id: "PF-03", item: "x_mass_frac_A + x_mass_frac_B <= 0.85 が linear_constraints に宣言されているか"}
    - {id: "PF-04", item: "hard_constraints_ref が Ch10 §10.6 anchor を指しているか"}
    - {id: "PF-05", item: "合成 dataset の場合 y_true と y_observed が明確に分離されているか"}
    - {id: "PF-06", item: "Skill 契約入力から y_true が除外されているか"}
    - {id: "PF-07", item: "sequential_seed_provenance が pin されているか"}
    - {id: "PF-08", item: "dataset_id が origin と一意に対応するか"}
    - {id: "PF-09", item: "budget scope（money_jpy or hours）が Ch5 §5.5 stop_condition と結合されているか"}
    - {id: "PF-10", item: "surrogate_treatment_of_facility_variance が dataset 特性と一致するか（単施設は not_applicable、多施設は hierarchical_shrinkage 等）"}
    - {id: "PF-11", item: "外部ライブラリ（olympus, botorch, pymatgen）のバージョンが provenance に記録されているか"}
    - {id: "PF-12", item: "regret 計算のための ground truth 参照が用意されているか（合成データのみ）"}
    - {id: "PF-13", item: "dataset のライセンス（CC-BY-4.0 等）と引用要件が記録されているか"}
```

### C.8.2 匿名化完了確認

```yaml
anonymization_completion_check:
  version: "v0.2"
  items:
    - {id: "AC-01", item: "AN-F-01 〜 04（facility 情報）全て pass"}
    - {id: "AC-02", item: "AN-P-01 〜 04（personnel 情報）全て pass"}
    - {id: "AC-03", item: "AN-M-01 〜 04（material identifiability）全て pass"}
    - {id: "AC-04", item: "AN-V-01 〜 04（provenance scrubbing）全て pass"}
    - {id: "AC-05", item: "特許出願前データが含まれていないことを PI が明示的に確認"}
    - {id: "AC-06", item: "匿名化前と後の hash が別々に記録され対応表が secure な場所に保管"}
    - {id: "AC-07", item: "search space レビュー（SS-R-01 〜 05）が完了"}
    - {id: "AC-08", item: "承認 2 名以上（data_steward + PI）"}
    - {id: "AC-09", item: "夜間・週末実行分の staff ID がすべて匿名 role 化"}
    - {id: "AC-10", item: "GitHub 等の公開先が確定し optional hardening の適用有無が明示されている"}
```

### C.8.3 provenance 記録完了確認

```yaml
provenance_completion_check:
  version: "v0.2"
  items:
    - {id: "PR-01", item: "dataset_id, origin, generator_ref が全て埋まっているか"}
    - {id: "PR-02", item: "seed_root と派生規則が pin されているか"}
    - {id: "PR-03", item: "observed_data_hash が計算・記録されているか"}
    - {id: "PR-04", item: "dataset_import_event が authorization_events_stream に append されたか"}
    - {id: "PR-05", item: "previous_event_hash / event_hash のチェーン整合性 OK"}
    - {id: "PR-06", item: "anonymization_checklist_hash が dataset に紐付いているか"}
    - {id: "PR-07", item: "search_space_review_hash が dataset に紐付いているか"}
    - {id: "PR-08", item: "ライブラリバージョン（botorch, olympus, pymatgen）が記録されているか"}
    - {id: "PR-09", item: "commit / git_sha 相当の generator バージョンが記録されているか"}
    - {id: "PR-10", item: "citation 要件（Materials Project, Olympus 元論文, ARIM 施設）が記載されているか"}
```

---

## C.9 まとめと次章への橋渡し

本付録では **合成 + benchmark + 実データラップ** の 3 系列で BO 演習データを整備した。ポイントは:

1. **C-BO-01 〜 05**（合成）で Ch5 21 フィールド contract と Ch10 hard_constraints を機械検証可能にした
2. **C-BM-01 〜 04**（Bayesian benchmarks）で BoTorch / GPyTorch API との疎通と、他ユーザとの regret 比較可能性を担保した
3. **C-OLY-\* / C-MP-01**（外部データ）で実物性寄りの realism を導入し、budget scope を実データで回せるようにした
4. **C-ARIM-01**（実データ持ち込み）は **匿名化 + search space レビュー + import event 記録** の 3 段ゲートを必ず通すことを要求した

これらは **付録 A の Skill テンプレート**（`arim.bo.*.v0.2`）を **付録 B の MCP handler** 経由で走らせる際の **契約入力** として使う。特に付録 A §A.2 の `arim.bo.single_objective_gp_ei.v0.2` は C-BO-01 で contract test を回し、付録 A §A.6 の `arim.bo.hierarchical_gp.v0.2` は C-BO-04 で inter-task correlation 回復のパラメトリック検証を行うと、Ch14 capstone 前に Skill を stress test できる。

次に **付録 D**（Capstone 発展スクリプト集）へ進み、Ch14b の後半 5 iteration と batch × multi-objective × constrained の統合実装で、本付録の C-BO-05 を実際に消費する。
