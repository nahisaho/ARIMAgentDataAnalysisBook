# 第11章 生成候補の surrogate ランキング — 離散候補集合の絞り込み

## 11.1 本章の目的：離散候補集合の T stage を Skill 化する

本章では、Ch5-7 の生成 Skill、Ch8 の物理制約フィルタ、Ch9 の DFT proxy、Ch10 の分布妥当性判定を通過した **離散候補集合** を、**多目的 Pareto ランキング** で top-k に絞り込む Skill `arim.gen.candidate_ranking.v0.1` を pin します。本章は Ch2 で導入した 5 stage パイプライン（F → S1 → S2 → H → T）の **T stage（top-k 選抜）** を担当します。

### 11.1.1 動機 A：連続 BO と離散候補ランキングは別種の問題

vol-05 では **連続 search space** に対する Bayesian Optimization（BO）を扱いました。本章の対象は **既に生成された離散候補集合**（数十〜数千個の材料組成 / 結晶構造 / 分子候補）です。両者の違いを canonical に区別します：

| 項目 | 連続 BO（vol-05） | 離散候補ランキング（本章） |
|---|---|---|
| **search space** | 連続（例：組成比 $x_1 + x_2 + x_3 = 1$、$x_i \in [0, 1]$） | 離散有限集合（生成モデル出力の $N$ 個候補） |
| **acquisition の役割** | 次の 1 点を search space から探索的に選ぶ | 全候補にスコア付けし top-k を選抜 |
| **surrogate 学習ループ** | acquisition → 実測 → surrogate 更新 → acquisition の反復 | 実測データで surrogate を再訓練後、次の生成回に反映（**ループの外側**） |
| **多目的の扱い** | multi-objective acquisition（EHVI / ParEGO 等） | Pareto 非劣選抜 + crowding distance で top-k 選抜 |
| **典型サイズ** | 数回〜数十回の逐次実験 | 1 回の生成で数百〜数千候補、top-k で 5〜50 選抜 |

**canonical 立場（v0.2 reframe）**：本章は生成候補を **BO の "初期点" として扱わない**。既に生成された離散有限集合に対する **surrogate ランキング + Pareto 選抜** に責務を絞ります。BO の逐次的な search-space 探索は vol-05 の責務、本章は「1 バッチの生成候補集合から top-k を選抜」に専念します。

### 11.1.2 動機 B：多目的の Pareto 選抜が canonical

Ch8 の合成可能性スコア、Ch9 の物性予測スコア、Ch10 の OOD score は **互いにトレードオフ関係にある独立軸** です。単一 scalar への合成（線形和・重み付き和）は Ch2 §2.4 canonical で禁止（`hallucinatory_composition_detection` 内部の composite_score は Ch10 内部の非劣選抜のためのものであり、Ch11 の 3 軸間で使ってはいけない）。本章は **3 軸 Pareto 非劣選抜** を canonical とします：

- **軸 1（合成可能性）**：$1 - s_{\text{synth}}$（Ch8 `synthesizability_proxy_score`、低いほど良い）
- **軸 2（物性目標達成度）**：$s_{\text{prop\_dev}}$（Ch11 内部で Ch9 `dft_proxy_predicted_properties` と `property_target_spec` から derive、低いほど良い、§11.4.0）
- **軸 3（分布内らしさ）**：$1 - s_{\text{in\_dist}}$（Ch10 `ood_score` = "in-distribution らしさ" から one_minus 変換、低いほど良い）

3 軸すべてを最小化する多目的最適化として Pareto フロントを構成し、top-k を選抜します。

### 11.1.3 動機 C：positivity_violation は Pareto 軸ではない（Human 承認境界）

Ch10 §10.7.2 で確立した canonical：`positivity_violation` は **Pareto ranking の軸に含めない**。理由：

1. **因果解釈の意図的差別化を Human 判断領域に残す**：positivity 外の候補は「外挿だが物性的に興味深い」ケースがあり、自動除外は探索空間を過度に縮小する
2. **Ch3 §3.5 B3 の Human 承認境界と一致**：条件変数 support 外候補の採否は Human が意識的に判断すべき境界

**canonical 実装**：`positivity_violation=True` の候補は **Pareto ranking には含める** が、result 態 metadata に **警告フラグを並記** し、Ch12 の Phase 5（Human 承認 + Safety governance）で明示的な意思確認を要求します。

---

## 11.2 vol-05 未読者向け BO 最小知識（3 ページ）

vol-05 を未読の読者向けに、本章で最低限必要な BO 概念を 3 ページで整理します。深堀りは vol-05 全体を参照。

### 11.2.1 GP surrogate（1 ページ）

**Gaussian Process surrogate**：入力 $x$ に対する未知関数 $f(x)$ を、平均関数 $\mu(x)$ と共分散関数 $k(x, x')$ を持つ確率過程としてモデル化します。$n$ 個の観測 $\{(x_i, y_i)\}_{i=1}^n$ が与えられたとき、任意の新規点 $x_*$ での予測分布は：

$$
p(y_* | x_*, X, y) = \mathcal{N}(\mu_*(x_*), \sigma_*^2(x_*))
$$

- $\mu_*(x_*)$：予測平均（`predicted_property_mean`）
- $\sigma_*^2(x_*)$：予測分散（`predicted_property_variance`、vol-03 第9章の epistemic uncertainty）

**本章での位置付け**：Ch9 の DFT proxy Skill が返す `predicted_property_mean` と `predicted_property_variance` を **既に学習済み surrogate の予測** として受け取ります。本章では surrogate の再学習はしません（それは Ch12 capstone Phase 6 の "生成 → surrogate ランキング → 実験 → 追加学習" ループの責務）。

**pin 契約**：surrogate の kernel type / hyperparameter / training data fingerprint は Ch9 result 態に pin されているものと仮定。Ch11 は **Ch9 の invocation_id + skill_version format のみを検証** し（§11.4.4 `INVOCATION_ID_PATTERN` / `mix_upstream_skill_versions` 検査）、pinned surrogate の内容照合（fingerprint hash 一致等）は Ch12 capstone Phase 4 orchestrator の責務とします（§11.5.4 candidate bundler 契約）。

### 11.2.2 Acquisition function（1 ページ）

**Acquisition function** $\alpha(x)$：surrogate の予測 $(\mu_*, \sigma_*)$ から「次に評価すべき点の優先度」を返すスカラー関数。canonical 3 択：

| Acquisition | 定義 | 特性 |
|---|---|---|
| **EI（Expected Improvement）** | $\alpha(x) = \mathbb{E}[\max(0, y_{\text{best}} - y(x))]$ | 現最良値 $y_{\text{best}}$ を下回る improvement の期待値。exploitation 寄り |
| **UCB（Upper Confidence Bound）** | $\alpha(x) = \mu_*(x) + \beta \sigma_*(x)$ | 平均 + 不確かさ×係数。$\beta$ で explore/exploit を制御 |
| **LCB（Lower Confidence Bound）** | $\alpha(x) = \mu_*(x) - \beta \sigma_*(x)$ | 最小化問題での lower bound。物性値の "最小化目標" に |

**本章での位置付け**：連続 BO では acquisition から次の 1 点を選びますが、**本章の離散候補ランキングでは acquisition を軸として使わず**、直接 predicted mean / uncertainty から Pareto 軸を構成します。理由：離散候補は既に有限集合であり、acquisition のような "探索的な次点選択" は不要。**vol-05 の EI / UCB / LCB は本章 canonical では使わない**（Ch11 §11.8.3 の失敗パターンで詳述）。

### 11.2.3 Pareto 非劣選抜（1 ページ）

**Pareto 非劣**：多目的最適化で、候補 $x_a$ が候補 $x_b$ を **Pareto dominate** するとは、以下を満たすこと（すべての軸で $x_a$ が $x_b$ 以下、少なくとも 1 軸で厳密に小さい）：

$$
\forall i: f_i(x_a) \leq f_i(x_b) \quad \land \quad \exists j: f_j(x_a) < f_j(x_b)
$$

**Pareto フロント**：他のどの候補にも dominate されない候補集合（**非劣集合**）。本章は 3 軸（synth / property / OOD）の Pareto フロントを構成し、**フロント上の top-k を選抜** します。

**crowding distance（NSGA-II 由来）**：Pareto フロント上に $k$ より多くの候補がある場合、多様性維持のため crowding distance（各軸で隣接候補との距離和）が大きい順に top-k を選抜します。以下 canonical：

$$
\text{crowding}(x_i) = \sum_{j=1}^{M} \frac{f_j(x_{i+1}) - f_j(x_{i-1})}{f_j^{\max} - f_j^{\min}}
$$

ここで $M$ は目的関数数（本章は $M = 3$）、$x_{i+1}, x_{i-1}$ は軸 $j$ で $x_i$ の両隣の候補。

**canonical 実装**：Ch11 は **NSGA-II 型の "Pareto rank + crowding distance" 2 段階選抜** を採用（§11.4）。理由：determinantal point process（DPP）等の代替は kernel の pin が困難で reproducibility を損なう。

---

## 11.3 canonical schema：config 態と result 態

Ch4 §4.5 の 3 態 schema を本章で **完全 pin** します。

### 11.3.1 config 態（Skill 起動時）

```yaml
candidate_ranking:
  mode: "config"                                        # Ch4 §4.5 canonical 態識別子（literal）
  ranking_method: "pareto_nsga2"                        # canonical 単一 method、literal pin（本章 canonical では他方法禁止）
  # 目的軸定義（3 軸固定、canonical）
  objectives:
    - axis_name: "synth_cost"                           # canonical 軸名（literal）
      source_skill: "arim.gen.synthesizability_proxy.v0.1"
      source_field: "synthesizability_proxy_score"
      transform: "one_minus"                            # 1 - score, minimization に統一
      required_upstream_stage: "S1"                     # Ch2 §2.7 canonical stage enum
    - axis_name: "property_cost"                        # canonical 軸名（literal）
      source_skill: "arim.gen.dft_proxy.v0.1"
      source_field: "dft_proxy_predicted_properties"    # Ch9 raw dict (property_id → {mean, std_raw, std_calibrated})
      transform: "target_deviation"                     # Ch11 §11.4.0 で property_target_spec を使い [0, 1] に derive
      required_upstream_stage: "S1"                     # Ch2 §2.7 canonical、Ch9 は S1 併走
    - axis_name: "ood_cost"                             # canonical 軸名（literal）
      source_skill: "arim.gen.ood_detection.v0.1"
      source_field: "ood_score"                         # = 1 - composite_score (Ch10 §10.6 canonical, "in-distribution らしさ")
      transform: "one_minus"                            # 1 - ood_score, minimization に統一（"低い分布内らしさ" ≒ "高いコスト"）
      required_upstream_stage: "H"                      # Ch2 §2.7 canonical
  # 物性目標仕様（Ch9 raw dict から property_cost を derive、canonical）
  property_target_spec:
    targets:                                             # 1 目的物性ごとの target 定義（canonical: 順序 pin、literal 名 pin）
      - property_id: "band_gap_eV"
        target_value: 1.5                               # float, canonical unit で pin
        tolerance: 0.3                                  # float > 0, deviation を [0, 1] に正規化する basis
        direction: "match"                              # canonical 3 択: match | minimize | maximize
    aggregation: "max_normalized_deviation"             # canonical 単一 method、literal pin (§11.4.0)
  # top-k 選抜設定
  selection:
    top_k: 10                                           # int, positive, ≤ n_candidates
    crowding_distance_enabled: true                     # NSGA-II 型 diversification（false は同順位内 lexicographic sort に fallback）
    tie_breaking: "crowding_distance"                   # canonical 2 択: crowding_distance | lexicographic
  # Ch4 §4.3.8 canonical: ランキング Skill の top-k 上限宣言（selection.top_k と一致必須）
  top_k_returned: 10                                    # int, 1 ≤ x ≤ 20, selection.top_k と equality 制約
  # Ch4 §4.3.7 canonical: 3-tier ranking method family（Ch11 は canonical single method）
  screening_model_family:
    family: "pareto_nsga2"                              # canonical fixed for Ch11（他値は disallowed_operations.expand_ranking_method_beyond_canonical）
    id: "arim.ranking.pareto_nsga2"
    version: "v0.1.0"
  # positivity_violation の扱い（Ch10 §10.7.2 canonical）
  positivity_handling:
    include_in_pareto: false                            # canonical false 固定（Pareto 軸に含めない）
    metadata_field: "positivity_violation"              # Ch10 result 態の field 名
    human_review_required: true                         # positivity_violation=True の候補は Human 承認境界フラグ立て
  # NF family 縮退の受け取り（Ch10 §10.5.2）
  nf_family_handling:
    accept_family_degenerate_to_2axis: true             # Ch10 provenance に family_degenerate_to_2axis: true があれば warn log のみ、Pareto 軸数は変えない
```

### 11.3.2 result 態（Skill 実行後）

```yaml
candidate_ranking:
  mode: "result"                                        # Ch4 §4.5 canonical 態識別子（literal）
  # 全候補に対する Pareto rank + top-k フラグ
  ranked_candidates:
    - candidate_id: "cand_0007"
      pareto_rank: 1                                    # 1 が非劣集合（frontier）、2 以降はより悪い集合
      crowding_distance: 0.87                           # 0 以上の実数、大きいほど多様性寄与大
      axis_values:
        synth_cost: 0.12
        property_cost: 0.08
        ood_cost: 0.15
      selected_in_top_k: true                           # top-k 選抜の結果フラグ
      positivity_violation: false                       # Ch10 result 態からの pass-through
      positivity_review_required: false                 # positivity_handling.human_review_required かつ violation の場合 true
  # aggregate metadata
  n_candidates_input: 250
  n_pareto_front: 42                                    # rank=1 の候補数
  n_selected_top_k: 10
  n_positivity_review_required: 3                      # top-k のうち Human 承認境界要のもの
  # provenance（実行 pin）
  upstream_invocation_ids_format_valid: true           # Ch8/Ch9/Ch10 の invocation_id を canonical format 検査済み（内容照合は Ch12 責務）
  ranking_method_used: "pareto_nsga2"
  config_ref:
    skill: "arim.gen.candidate_ranking.v0.1"
    skill_version: "v0.1.0"
    invocation_id: "inv_20260707_150000"
```

### 11.3.3 delegated_to 態

Ch4 §4.5 canonical 3 態目。本章 canonical に許可される委譲パターンは **1 つのみ**：

- **完全委譲**：外部 Skill（例：将来の `arim.gen.foundation_ranker.v0.2`）が T stage を独立実装。この場合のみ delegated_to 態を使う。

```yaml
candidate_ranking_delegated_to:
  skill: "arim.gen.foundation_ranker.v0.2"
  skill_version: "v0.2.0"
  invocation_stage: "T"                                 # Ch2 §2.7 canonical stage enum
  delegation_reason: "foundation_model_native_ranking"  # canonical 理由 list（下記）
  downstream_expectation:
    required_output_keys:                               # 代行 Skill の result 態が返す field 集合
      - "ranked_candidates"
      - "n_pareto_front"
      - "n_selected_top_k"
    pareto_axes_expected: 3                             # 3 軸維持を要求
```

**canonical delegation_reason（2 択のみ、literal pin）**：
- `foundation_model_native_ranking`：Foundation Model が候補生成と同時に ranking を副産物として持つ
- `external_multi_objective_engine`：pymoo / platypus 等の専用エンジンを外部 Skill で使う

---

## 11.4 実装：4 レイヤの canonical 契約

本節は Ch11 T stage の実装を **5 つの独立関数** に分解し、canonical 契約を pin します（axes 集約 / property_cost 導出 / Pareto rank / crowding distance / apply_ranking）。

### 11.4.0 物性コスト導出：`_derive_property_cost`

Ch9 `dft_proxy_predicted_properties` は raw 予測 dict（`property_id → {mean, std_raw, std_calibrated, ...}`）を返す契約です（Ch9 §9.6）。Pareto の "低いほど良い" 軸に載せるには、**目標物性 spec に対する正規化 deviation** を Ch11 側で導出する必要があります。この責務境界は Ch9 責務外（Ch9 は予測に閉じ、目標仕様との比較は下流責務）。

```python
import math
from typing import Any

def _derive_property_cost(
    predicted_properties: dict[str, Any],
    target_spec: dict[str, Any],
) -> float:
    """Ch9 raw dict + target_spec から [0, 1] の cost を導出。

    canonical 契約：
    - direction ∈ {match, minimize, maximize}
      - match: cost = min(1.0, |mean - target| / tolerance)
      - minimize: cost = min(1.0, max(0.0, (mean - target) / tolerance)) （target 以下なら 0）
      - maximize: cost = min(1.0, max(0.0, (target - mean) / tolerance)) （target 以上なら 0）
    - 複数 target がある場合は canonical aggregation で 1 値化（"max_normalized_deviation" のみ）
    - 全 target の property_id が predicted_properties に存在すること必須（欠落は fail-fast）
    - mean 非有限は fail-fast（Ch9 nan/inf 検査の下流保証）
    """
    targets = target_spec.get("targets", [])
    if not targets:
        raise ValueError("property_target_spec.targets is empty; ≥ 1 target required")
    aggregation = target_spec.get("aggregation")
    if aggregation != "max_normalized_deviation":
        raise ValueError(
            f"property_target_spec.aggregation must be 'max_normalized_deviation' "
            f"(got {aggregation!r}); canonical single method (§11.4.0)"
        )
    per_target_costs: list[float] = []
    for tgt in targets:
        pid = tgt.get("property_id")
        if not (isinstance(pid, str) and pid):
            raise ValueError(
                f"property_target_spec.targets[*].property_id must be non-empty str "
                f"(got {pid!r})"
            )
        target_value = float(tgt["target_value"])
        tolerance = float(tgt["tolerance"])
        direction = tgt["direction"]
        if not math.isfinite(target_value):
            raise ValueError(
                f"property_target_spec.targets[{pid!r}].target_value must be finite "
                f"(got {target_value!r})"
            )
        if not (math.isfinite(tolerance) and tolerance > 0.0):
            raise ValueError(
                f"property_target_spec.targets[{pid!r}].tolerance must be finite and > 0 "
                f"(got {tolerance!r})"
            )
        if direction not in ("match", "minimize", "maximize"):
            raise ValueError(
                f"property_target_spec.targets[{pid!r}].direction must be "
                f"'match' | 'minimize' | 'maximize' (got {direction!r})"
            )
        pred = predicted_properties.get(pid)
        if pred is None:
            raise ValueError(
                f"predicted_properties lacks required property_id {pid!r}; "
                "target spec requires it (§11.4.0)"
            )
        # ---- Ch9 calibration contract enforcement (§11.4.0, Ch9 §9.8.1 mode 2) ----
        cal_status = pred.get("calibration_status")
        if cal_status not in ("calibrated", "uncalibrated"):
            raise ValueError(
                f"predicted_properties[{pid!r}].calibration_status must be "
                f"'calibrated' | 'uncalibrated' (got {cal_status!r}); "
                "Ch9 §9.6 output_schema required field"
            )
        for k in ("std_raw", "std_calibrated"):
            v = pred.get(k)
            if not (isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) >= 0.0):
                raise ValueError(
                    f"predicted_properties[{pid!r}].{k} must be finite non-negative "
                    f"(got {v!r}); Ch9 §9.6 output_schema required"
                )
        mean = float(pred["mean"])
        if not math.isfinite(mean):
            raise ValueError(
                f"predicted_properties[{pid!r}].mean is non-finite ({mean!r}); "
                "Ch9 §9.6 fail-fast contract should have prevented this"
            )
        if direction == "match":
            raw = abs(mean - target_value) / tolerance
        elif direction == "minimize":
            raw = max(0.0, (mean - target_value) / tolerance)
        else:  # maximize
            raw = max(0.0, (target_value - mean) / tolerance)
        per_target_costs.append(min(1.0, raw))
    cost = max(per_target_costs)
    # Ch9 §9.8.1 mode 2 canonical: uncalibrated 予測は penalty で保守化。
    # 全 target のうち 1 つでも uncalibrated なら 1.0 に clamp（Pareto 最悪化）。
    any_uncalibrated = any(
        predicted_properties[tgt["property_id"]].get("calibration_status") == "uncalibrated"
        for tgt in targets
    )
    if any_uncalibrated:
        cost = 1.0
    if not (0.0 <= cost <= 1.0):
        raise RuntimeError(
            f"derived property_cost {cost!r} out of [0, 1]; internal consistency violation"
        )
    return cost
```

### 11.4.1 入力集約と upstream 契約検証：`_aggregate_candidate_axes`

```python
import hashlib
import math
import re
from typing import Any, Iterable
import numpy as np

SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
INVOCATION_ID_PATTERN = re.compile(r"\Ainv_[0-9]{8}_[0-9]{6}\Z")
SKILL_VERSION_PATTERN = re.compile(r"\Av[0-9]+\.[0-9]+\.[0-9]+\Z")
CANONICAL_STAGES = frozenset({"F", "S1", "S2", "H", "T"})
CANONICAL_AXIS_NAMES = ("synth_cost", "property_cost", "ood_cost")
CANONICAL_OBJECTIVES = (
    {
        "axis_name": "synth_cost",
        "source_skill": "arim.gen.synthesizability_proxy.v0.1",
        "source_field": "synthesizability_proxy_score",
        "transform": "one_minus",
        "required_upstream_stage": "S1",
    },
    {
        "axis_name": "property_cost",
        "source_skill": "arim.gen.dft_proxy.v0.1",
        "source_field": "dft_proxy_predicted_properties",
        "transform": "target_deviation",
        "required_upstream_stage": "S1",
    },
    {
        "axis_name": "ood_cost",
        "source_skill": "arim.gen.ood_detection.v0.1",
        "source_field": "ood_score",
        "transform": "one_minus",
        "required_upstream_stage": "H",
    },
)

def _sha256(data: bytes) -> str:
    """canonical hash 形式 'sha256:<64-hex>' を返す。"""
    return "sha256:" + hashlib.sha256(data).hexdigest()

def _canonical_array_bytes(arr: np.ndarray) -> bytes:
    """dtype/shape header + C-contiguous payload の canonical serialization
    （byte 一致だが dtype/shape が異なる snapshot を verify で見逃さないため、§11.5.1）。"""
    header = f"dtype={arr.dtype.str}|shape={arr.shape}|order=C".encode("utf-8")
    payload = np.ascontiguousarray(arr).tobytes()
    return header + b"|" + payload

def _aggregate_candidate_axes(
    candidates: list[dict[str, Any]],
    objectives_config: list[dict[str, Any]],
    property_target_spec: dict[str, Any],
) -> np.ndarray:
    """全候補について 3 軸の cost 値を集約し (n_candidates, 3) ndarray として返す。

    canonical 契約：
    - candidates は Ch8/Ch9/Ch10 の result 態を全て持つ（欠落は fail-fast）
    - 3 軸すべてに finite 値が必須（NaN/inf は fail-fast、Ch10 §10.8.4 と同旨）
    - objectives_config は canonical 3 軸順・literal 軸名で pin されている必要あり
    - property_cost 軸は _derive_property_cost で property_target_spec を使い derive
    """
    # objectives_config の canonical 軸名検査
    axis_names = tuple(obj["axis_name"] for obj in objectives_config)
    if axis_names != CANONICAL_AXIS_NAMES:
        raise ValueError(
            f"objectives axis_names must be canonical {CANONICAL_AXIS_NAMES} "
            f"in this order (got {axis_names}); "
            "disallowed_operations.reorder_or_rename_pareto_axes"
        )

    n = len(candidates)
    if n == 0:
        raise ValueError("candidates list is empty; T stage requires ≥ 1 candidate")

    cost_matrix = np.full((n, 3), np.nan, dtype=np.float64)
    for i, cand in enumerate(candidates):
        for j, obj in enumerate(objectives_config):
            # upstream stage 契約検査（Ch9 は S1、Ch10 は H）
            trace = cand.get("upstream_stage_trace", [])
            if obj["required_upstream_stage"] not in trace:
                raise ValueError(
                    f"candidate {cand.get('candidate_id')!r} lacks required upstream stage "
                    f"{obj['required_upstream_stage']!r} for axis {obj['axis_name']!r} "
                    f"(upstream_stage_trace={trace}); "
                    "disallowed_operations.rank_without_full_upstream_trace"
                )
            # source_field 抽出 + transform 適用
            transform = obj["transform"]
            if transform == "target_deviation":
                # property_cost 軸専用：Ch9 raw dict + target_spec から derive
                pred_dict = cand.get(obj["source_field"])
                if not isinstance(pred_dict, dict) or not pred_dict:
                    raise ValueError(
                        f"candidate {cand.get('candidate_id')!r} lacks required dict field "
                        f"{obj['source_field']!r} for axis {obj['axis_name']!r}"
                    )
                value = _derive_property_cost(pred_dict, property_target_spec)
            else:
                raw_value = cand.get(obj["source_field"])
                if raw_value is None:
                    raise ValueError(
                        f"candidate {cand.get('candidate_id')!r} lacks field "
                        f"{obj['source_field']!r} required for axis {obj['axis_name']!r}"
                    )
                if transform == "one_minus":
                    value = 1.0 - float(raw_value)
                elif transform == "identity":
                    value = float(raw_value)
                else:
                    raise ValueError(
                        f"unknown transform {transform!r} for axis {obj['axis_name']!r}; "
                        "canonical: 'one_minus' | 'identity' | 'target_deviation'"
                    )
            if not math.isfinite(value):
                raise ValueError(
                    f"candidate {cand.get('candidate_id')!r} axis {obj['axis_name']!r} "
                    f"produced non-finite value {value!r}; "
                    "disallowed_operations.propagate_non_finite_axis_value"
                )
            # canonical range [0, 1] 検査（Ch8/Ch9/Ch10 全て [0, 1] 契約）
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"candidate {cand.get('candidate_id')!r} axis {obj['axis_name']!r} "
                    f"value {value!r} out of canonical [0, 1] range; "
                    "check upstream Skill contract"
                )
            cost_matrix[i, j] = value

    # 最終 sanity（NaN 残留チェック）
    if not np.all(np.isfinite(cost_matrix)):
        raise ValueError(
            f"cost_matrix contains non-finite entries after aggregation; "
            f"n_nan={int(np.sum(~np.isfinite(cost_matrix)))}"
        )
    assert cost_matrix.shape == (n, 3), f"cost_matrix shape {cost_matrix.shape} != ({n}, 3)"
    return cost_matrix
```

### 11.4.2 Pareto rank 計算：`_compute_pareto_ranks`

```python
def _compute_pareto_ranks(cost_matrix: np.ndarray) -> np.ndarray:
    """NSGA-II 型の非劣ソートで Pareto rank を計算し (n_candidates,) int ndarray を返す。

    canonical 契約：
    - rank=1 は Pareto frontier（他のどの候補にも dominate されない）
    - rank=k は "rank<k をすべて除いた後の frontier"
    - 同点候補（identical rows）は同じ rank に配置
    - strict dominance（全軸で ≤ かつ 1 軸で <）で判定、tie は non-dominated 扱い
    """
    if cost_matrix.ndim != 2 or cost_matrix.shape[1] != 3:
        raise ValueError(
            f"cost_matrix must be (n, 3), got shape {cost_matrix.shape}"
        )
    n = cost_matrix.shape[0]
    ranks = np.zeros(n, dtype=np.int64)
    remaining = np.arange(n, dtype=np.int64)
    current_rank = 1
    while remaining.size > 0:
        sub = cost_matrix[remaining]
        # 各候補 i について、"i を strict dominate する候補が sub 内に存在しない" なら i は frontier
        # dominate: 全軸で ≤ かつ 少なくとも 1 軸で <
        dominated = np.zeros(sub.shape[0], dtype=bool)
        for i in range(sub.shape[0]):
            # candidate i vs all others
            leq = np.all(sub <= sub[i], axis=1)             # (n_sub,) other ≤ i in all axes
            strict = np.any(sub < sub[i], axis=1)           # (n_sub,) other < i in ≥1 axis
            # other strictly dominates i iff leq[other] ∧ strict[other] ∧ other ≠ i
            dom_by_others = leq & strict
            dom_by_others[i] = False
            dominated[i] = bool(np.any(dom_by_others))
        frontier_mask = ~dominated
        frontier_indices = remaining[frontier_mask]
        if frontier_indices.size == 0:
            # 論理的には発生しない（少なくとも 1 候補は非劣）が防衛的に fail-fast
            raise RuntimeError(
                f"empty Pareto frontier at rank {current_rank}, remaining={remaining.size}; "
                "internal consistency violation"
            )
        ranks[frontier_indices] = current_rank
        remaining = remaining[~frontier_mask]
        current_rank += 1

    # sanity: 全候補が rank 割当済み
    if np.any(ranks == 0):
        raise RuntimeError(
            f"pareto rank assignment incomplete: {int(np.sum(ranks == 0))} candidates un-ranked"
        )
    return ranks
```

### 11.4.3 crowding distance：`_compute_crowding_distance`

```python
def _compute_crowding_distance(
    cost_matrix: np.ndarray,
    ranks: np.ndarray,
    cid_order: np.ndarray,
) -> np.ndarray:
    """NSGA-II crowding distance を計算し (n_candidates,) float ndarray を返す。

    canonical 契約：
    - 同 rank 内で軸ごとにソートし、両端候補は +inf（境界候補は最優先で残す）
    - 中央候補は正規化された隣接距離の和
    - 正規化：軸の (max - min) が 0 の場合は当該軸の寄与を 0（degenerate 軸の除外）
    - 出力は非負の float、+inf を許容
    - `cid_order` は candidate_id 昇順に対応する rank 配列（§11.4.4 で構築）。
      軸値が tied の場合の順序を canonical に pin し、bundler 入力順に依存しない
      境界 +inf 割当・隣接距離計算を保証する。
    """
    if cost_matrix.shape[0] != ranks.shape[0]:
        raise ValueError(
            f"shape mismatch: cost_matrix {cost_matrix.shape} vs ranks {ranks.shape}"
        )
    if cid_order.shape[0] != ranks.shape[0]:
        raise ValueError(
            f"shape mismatch: cid_order {cid_order.shape} vs ranks {ranks.shape}"
        )
    n, m = cost_matrix.shape
    crowding = np.zeros(n, dtype=np.float64)
    unique_ranks = np.unique(ranks)
    for r in unique_ranks:
        indices = np.where(ranks == r)[0]
        if indices.size <= 2:
            # 2 候補以下は全員境界扱い（+inf）
            crowding[indices] = np.inf
            continue
        for j in range(m):
            values = cost_matrix[indices, j]
            # 軸値 tied の場合の secondary key として cid_order を pin（bundler 順に依存しない）
            order = np.lexsort((cid_order[indices], values))
            sorted_indices = indices[order]
            v_min = float(values[order[0]])
            v_max = float(values[order[-1]])
            span = v_max - v_min
            if span <= 0.0:
                # degenerate 軸：全員同値。当該軸は境界 +inf を割り当てず、寄与 0 のまま。
                continue
            # 両端候補は +inf（tied の場合は cid_order で canonical 選抜）
            crowding[sorted_indices[0]] = np.inf
            crowding[sorted_indices[-1]] = np.inf
            for k in range(1, indices.size - 1):
                prev_v = float(values[order[k - 1]])
                next_v = float(values[order[k + 1]])
                crowding[sorted_indices[k]] += (next_v - prev_v) / span
    return crowding
```

### 11.4.4 top-k 選抜と result 態組立：`apply_candidate_ranking`

```python
def apply_candidate_ranking(
    ranking_config: dict[str, Any],
    candidates: list[dict[str, Any]],
    invocation_id: str,
) -> dict[str, Any]:
    """本 Skill のトップレベル entry point。

    canonical 契約：
    - config 態の literal 検査（axis_names, top_k > 0, positivity_handling.include_in_pareto=False）
    - upstream Skill (Ch8/Ch9/Ch10) の invocation_id を canonical format で全候補 verify
      （複数 invocation 由来の混在は許容。ただし per-candidate で Ch8/Ch9/Ch10 invocation_id が
      全て pin 必須、§11.5.1 canonical）
    - upstream_stage_trace が {F, S1, H} を全て包含することを全候補で literal 検査
    - Ch10 provenance の family_degenerate_to_2axis を検査し warn log を残す
    - result 態は Ch4 §4.5 canonical shape で返す
    - reject は行わない（top-k 外の候補も ranked_candidates 全件返す、pass-through 契約）
    """
    if ranking_config.get("mode") != "config":
        raise ValueError(
            f"ranking_config.mode must be 'config' (got {ranking_config.get('mode')!r})"
        )

    # ranking_method canonical literal 検査
    if ranking_config.get("ranking_method") != "pareto_nsga2":
        raise ValueError(
            f"ranking_method must be 'pareto_nsga2' (got {ranking_config.get('ranking_method')!r}); "
            "disallowed_operations.expand_ranking_method_beyond_canonical"
        )

    # objectives canonical shape 検査（3 軸、canonical 順・literal 名 + source_skill/source_field/transform/stage 完全一致）
    objectives = ranking_config.get("objectives", [])
    if len(objectives) != 3:
        raise ValueError(
            f"objectives must have exactly 3 axes (got {len(objectives)}); "
            "canonical: synth_cost, property_cost, ood_cost"
        )
    for actual, canonical in zip(objectives, CANONICAL_OBJECTIVES):
        for key in ("axis_name", "source_skill", "source_field", "transform",
                    "required_upstream_stage"):
            if actual.get(key) != canonical[key]:
                raise ValueError(
                    f"objective {canonical['axis_name']!r}.{key} must be "
                    f"{canonical[key]!r} (got {actual.get(key)!r}); "
                    "canonical fixed axis contract, disallowed_operations.reorder_or_rename_pareto_axes"
                )
        if actual["required_upstream_stage"] not in CANONICAL_STAGES:
            raise ValueError(
                f"required_upstream_stage {actual['required_upstream_stage']!r} "
                f"not in canonical enum {sorted(CANONICAL_STAGES)}"
            )

    # positivity_handling canonical literal 検査
    pos = ranking_config.get("positivity_handling", {})
    if pos.get("include_in_pareto") is not False:
        raise ValueError(
            f"positivity_handling.include_in_pareto must be False literal "
            f"(got {pos.get('include_in_pareto')!r}); "
            "Ch10 §10.7.2 canonical, Human 承認境界を Pareto 軸に含めることは禁止 "
            "(disallowed_operations.include_positivity_in_pareto_axes)"
        )
    if not isinstance(pos.get("human_review_required"), bool):
        raise ValueError(
            f"positivity_handling.human_review_required must be bool "
            f"(got {type(pos.get('human_review_required')).__name__})"
        )

    # selection canonical 検査
    sel = ranking_config.get("selection", {})
    top_k = sel.get("top_k")
    if not (isinstance(top_k, int) and top_k > 0):
        raise ValueError(f"selection.top_k must be positive int (got {top_k!r})")
    # Ch4 §4.3.8 canonical: top_k_returned 上限 20
    if top_k > 20:
        raise ValueError(
            f"selection.top_k={top_k} exceeds Ch4 §4.3.8 canonical upper bound (20); "
            "disallowed_operations.exceed_top_k_returned_cap"
        )
    # Ch4 §4.3.8 canonical: top_k_returned field と selection.top_k の equality 制約
    top_k_returned = ranking_config.get("top_k_returned")
    if not (isinstance(top_k_returned, int) and 1 <= top_k_returned <= 20):
        raise ValueError(
            f"top_k_returned must be int in [1, 20] (got {top_k_returned!r}); "
            "Ch4 §4.3.8 canonical required for ranking Skills"
        )
    if top_k_returned != top_k:
        raise ValueError(
            f"top_k_returned={top_k_returned} must equal selection.top_k={top_k}; "
            "canonical equality contract prevents cap drift"
        )
    # Ch4 §4.3.7 canonical: screening_model_family literal 検査
    smf = ranking_config.get("screening_model_family", {})
    if smf.get("family") != "pareto_nsga2":
        raise ValueError(
            f"screening_model_family.family must be 'pareto_nsga2' for Ch11 "
            f"(got {smf.get('family')!r}); "
            "disallowed_operations.expand_ranking_method_beyond_canonical"
        )
    for k in ("id", "version"):
        v = smf.get(k)
        if not (isinstance(v, str) and v):
            raise ValueError(
                f"screening_model_family.{k} must be non-empty str (got {v!r}); "
                "Ch4 §4.3.7 3-tier canonical required"
            )
    n = len(candidates)
    if top_k > n:
        raise ValueError(
            f"top_k={top_k} exceeds n_candidates={n}; "
            "cannot select more than available candidates"
        )
    tie_breaking = sel.get("tie_breaking")
    if tie_breaking not in ("crowding_distance", "lexicographic"):
        raise ValueError(
            f"selection.tie_breaking must be 'crowding_distance' | 'lexicographic' "
            f"(got {tie_breaking!r})"
        )
    # crowding_distance_enabled と tie_breaking の consistency 検査（§11.6 canonical）
    crowding_enabled = sel.get("crowding_distance_enabled")
    if not isinstance(crowding_enabled, bool):
        raise ValueError(
            f"selection.crowding_distance_enabled must be bool "
            f"(got {type(crowding_enabled).__name__})"
        )
    if tie_breaking == "crowding_distance" and not crowding_enabled:
        raise ValueError(
            "inconsistent selection config: tie_breaking='crowding_distance' but "
            "crowding_distance_enabled=False. When crowding is disabled, tie_breaking "
            "must be 'lexicographic' (canonical, §11.4.4)"
        )

    # ---- candidate_id 非空 + 一意性検査 (§11.4.4 canonical) ----
    seen_ids: set[str] = set()
    for cand in candidates:
        cid = cand.get("candidate_id")
        if not (isinstance(cid, str) and cid):
            raise ValueError(
                f"candidate_id must be non-empty str (got {cid!r}); "
                "Ch11 provenance and Ch12 Human review require unique id per candidate"
            )
        if cid in seen_ids:
            raise ValueError(
                f"duplicate candidate_id {cid!r} in bundled input; "
                "candidate_id must be globally unique within the ranking invocation"
            )
        seen_ids.add(cid)

    # ---- upstream_stage_trace {F, S1, H} 包含検査 (§11.5.1 canonical) ----
    REQUIRED_STAGES = {"F", "S1", "H"}
    for cand in candidates:
        trace = cand.get("upstream_stage_trace")
        if not isinstance(trace, list) or not REQUIRED_STAGES.issubset(set(trace)):
            raise ValueError(
                f"candidate {cand.get('candidate_id')!r} upstream_stage_trace must be a list "
                f"containing at least {sorted(REQUIRED_STAGES)} (got {trace!r}); "
                "disallowed_operations.rank_without_full_upstream_trace"
            )

    # ---- upstream invocation_id verification: 全候補で同一 Ch8/9/10 invocation を期待しない
    # （複数 invocation 由来の混在は許容するが、per-candidate で invocation_id を pin 済みとする）----
    UPSTREAM_SKILLS = (
        "arim.gen.synthesizability_proxy.v0.1",
        "arim.gen.dft_proxy.v0.1",
        "arim.gen.ood_detection.v0.1",
    )
    for cand in candidates:
        for skill_name in UPSTREAM_SKILLS:
            inv = cand.get("upstream_invocation_ids", {}).get(skill_name)
            if not (isinstance(inv, str) and INVOCATION_ID_PATTERN.match(inv)):
                raise ValueError(
                    f"candidate {cand.get('candidate_id')!r} lacks canonical upstream_invocation_ids["
                    f"{skill_name!r}] (got {inv!r}); "
                    "canonical: 'inv_YYYYMMDD_HHMMSS' format required for full provenance chain"
                )

    # Ch11 自身の invocation_id も canonical format 検査
    if not (isinstance(invocation_id, str) and INVOCATION_ID_PATTERN.match(invocation_id)):
        raise ValueError(
            f"Ch11 invocation_id must match canonical 'inv_YYYYMMDD_HHMMSS' pattern "
            f"(got {invocation_id!r})"
        )

    # ---- upstream skill_version drift check (§11.5.2 canonical) ----
    # 全候補で per-skill の skill_version が一致する必要あり。混在は fail-fast。
    reference_versions: dict[str, str] = {}
    for cand in candidates:
        versions = cand.get("upstream_skill_versions", {})
        if not isinstance(versions, dict):
            raise ValueError(
                f"candidate {cand.get('candidate_id')!r} lacks upstream_skill_versions dict "
                f"(got {type(versions).__name__})"
            )
        for skill_name in UPSTREAM_SKILLS:
            ver = versions.get(skill_name)
            if not (isinstance(ver, str) and SKILL_VERSION_PATTERN.match(ver)):
                raise ValueError(
                    f"candidate {cand.get('candidate_id')!r} upstream_skill_versions["
                    f"{skill_name!r}] must match canonical 'vX.Y.Z' semver (got {ver!r})"
                )
            if skill_name not in reference_versions:
                reference_versions[skill_name] = ver
            elif reference_versions[skill_name] != ver:
                raise ValueError(
                    f"upstream skill_version mismatch for {skill_name!r}: "
                    f"candidate {cand.get('candidate_id')!r} has {ver!r} but reference is "
                    f"{reference_versions[skill_name]!r}; "
                    "disallowed_operations.mix_upstream_skill_versions"
                )

    # ---- NF family 縮退 warn log ----
    n_nf_degenerate = 0
    for cand in candidates:
        ood_prov = cand.get("ood_provenance", {})
        if ood_prov.get("family_degenerate_to_2axis") is True:
            n_nf_degenerate += 1
    nf_handling = ranking_config.get("nf_family_handling", {})
    if n_nf_degenerate > 0:
        if nf_handling.get("accept_family_degenerate_to_2axis") is not True:
            raise ValueError(
                f"{n_nf_degenerate} candidates have Ch10 family_degenerate_to_2axis=True, "
                "but nf_family_handling.accept_family_degenerate_to_2axis is not True; "
                "explicit acceptance required (§11.4)"
            )
        # warn log は本 canonical 実装では print で代替（本番は logging 推奨）
        print(
            f"[WARN] {n_nf_degenerate}/{n} candidates are from Ch10 NF-degenerate 2-axis family; "
            "Pareto axis count remains 3 but axes 1 (property) and 3 (OOD) may share information"
        )

    # ---- 実算出 ----
    property_target_spec = ranking_config.get("property_target_spec")
    if not isinstance(property_target_spec, dict) or not property_target_spec.get("targets"):
        raise ValueError(
            "ranking_config.property_target_spec must be dict with non-empty targets "
            "(canonical, §11.3.1). property_cost 軸の導出に必要"
        )
    cost_matrix = _aggregate_candidate_axes(candidates, objectives, property_target_spec)
    ranks = _compute_pareto_ranks(cost_matrix)

    # candidate_id 昇順を canonical tie-breaker として先に構築（crowding/top-k 両方で共有）
    cid_order = np.argsort(
        np.argsort([str(c["candidate_id"]) for c in candidates], kind="stable"),
        kind="stable",
    )  # candidate_id 昇順に対応する rank 配列

    if sel.get("crowding_distance_enabled") is True and tie_breaking == "crowding_distance":
        crowding = _compute_crowding_distance(cost_matrix, ranks, cid_order)
    else:
        # lexicographic tie-break: crowding は全 0 にし、後段で lex sort
        crowding = np.zeros(n, dtype=np.float64)

    # ---- top-k 選抜: (rank asc, crowding desc, candidate_id asc) で sort ----
    # 最終 tie-breaker として candidate_id 昇順を pin することで、
    # bundler の入力順に依存しない deterministic な top-k を保証する（§11.4.4 canonical）。
    if tie_breaking == "crowding_distance":
        # 優先度 (低→高): candidate_id_rank, -crowding, ranks
        sort_key = np.lexsort((cid_order, -crowding, ranks))
    else:  # lexicographic
        # 優先度 (低→高): candidate_id_rank, ood, property, synth, ranks
        sort_key = np.lexsort(
            (cid_order, cost_matrix[:, 2], cost_matrix[:, 1], cost_matrix[:, 0], ranks)
        )
    selected_top_k_set = frozenset(int(i) for i in sort_key[:top_k].tolist())

    # ---- result 態組立 ----
    ranked = []
    n_pos_review = 0
    for i, cand in enumerate(candidates):
        pos_raw = cand.get("positivity_violation")
        if not isinstance(pos_raw, bool):
            raise ValueError(
                f"candidate {cand.get('candidate_id')!r} positivity_violation must be "
                f"exactly bool (got {type(pos_raw).__name__} {pos_raw!r}); "
                "Ch10 §10.7.2 pass-through canonical requires strict bool, no default/coercion"
            )
        pos_violation = pos_raw
        pos_review = bool(pos_violation and pos.get("human_review_required"))
        if pos_review and (i in selected_top_k_set):
            n_pos_review += 1
        ranked.append({
            "candidate_id": cand["candidate_id"],
            "pareto_rank": int(ranks[i]),
            "crowding_distance": float(crowding[i]),
            "axis_values": {
                "synth_cost": float(cost_matrix[i, 0]),
                "property_cost": float(cost_matrix[i, 1]),
                "ood_cost": float(cost_matrix[i, 2]),
            },
            "selected_in_top_k": (i in selected_top_k_set),
            "positivity_violation": pos_violation,
            "positivity_review_required": pos_review,
            # Ch12 content-verification 用の per-candidate provenance pass-through
            "upstream_invocation_ids": cand["upstream_invocation_ids"],
            "upstream_skill_versions": cand["upstream_skill_versions"],
            "upstream_stage_trace": cand["upstream_stage_trace"],
        })

    n_pareto_front = int(np.sum(ranks == 1))
    result = {
        "candidate_ranking": {
            "mode": "result",
            "ranked_candidates": ranked,
            "n_candidates_input": n,
            "n_pareto_front": n_pareto_front,
            "n_selected_top_k": top_k,
            "n_positivity_review_required": n_pos_review,
            "upstream_invocation_ids_format_valid": True,
            "ranking_method_used": "pareto_nsga2",
            "config_ref": {
                "skill": "arim.gen.candidate_ranking.v0.1",
                "skill_version": "v0.1.0",
                "invocation_id": invocation_id,
            },
        }
    }
    return result
```

---

## 11.5 参照 snapshot と versioning

Ch11 T stage は **参照分布 snapshot を pin しない** 数少ない Skill です。理由：Ch11 の役割は **上流 Skill の result 態を集約して Pareto 選抜する** ことに閉じており、独自の統計基盤（μ / Σ / dist）を持ちません。したがって Ch10 §10.5.1 のような snapshot pinning は不要です。

代わりに Ch11 は **上流の invocation_id を pin** します（§11.4.4 の `upstream_invocation_ids` 検査）。

### 11.5.1 upstream invocation_id pinning の canonical

各候補は **Ch8 / Ch9 / Ch10 それぞれの invocation_id** を持ちます。Ch11 は候補ごとに 3 つの invocation_id が **全て canonical format（`inv_YYYYMMDD_HHMMSS`）** で存在することを literal 検査します。invocation_id が欠落・不正形式の候補は fail-fast（`disallowed_operations.rank_without_full_upstream_trace`）。

**canonical 契約**：
- 同一候補集合内で Ch8/Ch9/Ch10 の invocation_id は **異なっていて良い**（例：Ch8 は 250 候補一括、Ch9 は 250 候補一括、Ch10 は 250 候補一括で別 invocation）
- ただし候補ごとに 3 つの invocation_id が **全て pin されている** ことが必須
- **Ch11 result 態は自身の `invocation_id` を持ち、Ch12 capstone がこれを Pareto の根拠として pin する**

### 11.5.2 upstream Skill version drift の検査

Ch11 は候補の provenance から Ch8/Ch9/Ch10 の `skill_version` を抽出し、**全候補で同一 skill_version** を要求します。異なる skill_version の候補混在は fail-fast（`disallowed_operations.mix_upstream_skill_versions`）。

**理由**：Ch8 v0.1.0 と Ch8 v0.2.0 は synthesizability_proxy_score の意味が微妙に異なる可能性があり、これを混ぜて Pareto を組むと axis basis が食い違います（Ch10 §10.5.1 dataset_fingerprint 混在禁止と同じ思想）。

### 11.5.3 Pareto 前線の再現性

Ch11 result 態は **NumPy の deterministic な非劣ソート + crowding distance 計算** のみで完全再現可能です。乱数を使わない canonical 実装のため、同じ入力・同じ config に対して bit-exact に同じ result を返します。Ch12 capstone Phase 5 の Human 承認レビューで **同じ候補集合から同じ top-k が導出されることを検査可能** です。

### 11.5.4 candidate bundler：Ch11 入力構築の canonical

Ch8 / Ch9 / Ch10 の result 態はそれぞれ **独立した Skill 出力** であり、Ch11 が期待する **per-candidate bundled record**（`upstream_invocation_ids: dict`, `upstream_skill_versions: dict`, `upstream_stage_trace: [F, S1, H]`, `synthesizability_proxy_score`, `dft_proxy_predicted_properties`, `ood_score`, `positivity_violation` を単一 candidate オブジェクトに merge した shape）は **自動生成されません**。この bundling は **Ch11 の外側（Ch12 capstone Phase 4 orchestrator）の責務** です。

**canonical bundler 契約**：

1. **stage_trace の合成**：Ch8 result（stage_trace に `F`, `S1` を含む S1 Skill）と Ch9 result（同じく `F`, `S1`）と Ch10 result（`F`, `H`）を候補 ID 単位で JOIN し、`upstream_stage_trace = ["F", "S1", "H"]` を合成する。Ch11 は `{F, S1, H}` の subset 存在を検査するのみで、trace の出所生成は行わない。
2. **invocation_id dict の構築**：Ch8/Ch9/Ch10 それぞれの result 態 top-level `invocation_id`（`inv_YYYYMMDD_HHMMSS` 形式）を、
   ```python
   upstream_invocation_ids = {
       "arim.gen.synthesizability_proxy.v0.1": ch8_result["invocation_id"],
       "arim.gen.dft_proxy.v0.1":               ch9_result["invocation_id"],
       "arim.gen.ood_detection.v0.1":           ch10_result["invocation_id"],
   }
   ```
   の形で per-candidate に付与する（同一 invocation から派生した候補は同じ値を共有）。
3. **skill_version dict の構築**：各 Skill result 態の `provenance.skill_version`（または YAML `metadata.skill_version` に対応する top-level field）から
   ```python
   upstream_skill_versions = {
       "arim.gen.synthesizability_proxy.v0.1": ch8_result["skill_version"],
       "arim.gen.dft_proxy.v0.1":               ch9_result["skill_version"],
       "arim.gen.ood_detection.v0.1":           ch10_result["skill_version"],
   }
   ```
   を構築する。
4. **score field の直リフト**：Ch8 の `synthesizability_proxy_score`、Ch9 の `dft_proxy_predicted_properties` dict、Ch10 の `ood_score` / `positivity_violation` を候補オブジェクトの top-level にそのまま lift する（rename/変換禁止）。

**responsibility 境界**：
- **bundler の位置**：Ch12 capstone Phase 4「Surrogate 選抜」の直前、Ch10 result 態受領後に実行される orchestrator step（Ch12 §12.5 で具体的 pseudocode を提示予定）
- **Ch11 の契約**：bundled record を **受け取るのみ**。Ch8/9/10 raw result 態から自動 join は行わない（Ch11 は単一 Skill の責務境界を厳守）
- **fail-fast の意義**：bundler がバグって invocation_id や skill_version の合成に失敗した場合、Ch11 の literal 検査で必ず fail-fast する（`disallowed_operations.rank_without_full_upstream_trace` / `mix_upstream_skill_versions`）

---

## 11.6 Skill YAML 完全 shape（arim.gen.candidate_ranking.v0.1）

以下は Ch4 §4.7 の canonical shape に準拠した Skill YAML の完全形。

```yaml
skill_id: "arim.gen.candidate_ranking.v0.1"
skill_version: "v0.1.0"
stage: "T"                                                  # Ch2 §2.7 canonical stage enum

# --- Ch4 §4.3 canonical provenance fields (top-level, Skill YAML SoT) ---
top_k_returned: 10                                          # Ch4 §4.3.8, int 1-20
screening_model_family:                                     # Ch4 §4.3.7 3-tier canonical
  family: "pareto_nsga2"
  id: "arim.ranking.pareto_nsga2"
  version: "v0.1.0"
inverse_design_authorization: "semi_autonomous"             # Ch4 §4.3.9 canonical (T stage)
safety_screening_passed: true                               # Ch4 §4.6 canonical
dual_use_review_completed: true                             # Ch4 §4.6 canonical
# 上記 top-level 値と candidate_ranking 内 config 値の equality を実装で強制:
# apply_candidate_ranking で top_k_returned / screening_model_family を検査 (§11.4.4)。
# Ch12 orchestrator は起動時に top-level ↔ candidate_ranking.* の equality を pre-check する。

description: >
  Ch8 synthesizability + Ch9 DFT proxy + Ch10 OOD detection の 3 軸を統合し、
  NSGA-II 型 Pareto 非劣選抜 + crowding distance による top-k 選抜を行う T stage Skill。
  離散候補集合の surrogate ランキング（連続 BO とは別枠、§11.1.1 canonical）。

input_schema:
  required_keys:
    - "candidate_ranking"                                   # config 態のトップレベル key
    - "candidates"                                          # list[dict], 各 dict に upstream result 態が pin 済み
  candidate_ranking_required_fields:
    - "mode"                                                # literal "config"
    - "ranking_method"                                      # literal "pareto_nsga2"
    - "objectives"                                          # list[dict], canonical 3 軸 (§11.6 canonical_objectives)
    - "property_target_spec"                                # dict, property_cost 軸 derive 用 (§11.3.1, §11.4.0)
    - "selection"                                           # {top_k, crowding_distance_enabled, tie_breaking}
    - "top_k_returned"                                      # int, 1 ≤ x ≤ 20, selection.top_k と equality 制約 (Ch4 §4.3.8)
    - "screening_model_family"                              # {family: "pareto_nsga2", id, version} (Ch4 §4.3.7)
    - "positivity_handling"                                 # {include_in_pareto, metadata_field, human_review_required}
    - "nf_family_handling"                                  # {accept_family_degenerate_to_2axis, ...}
  candidate_required_fields:
    - "candidate_id"
    - "upstream_stage_trace"                                # list[str], 少なくとも {"F", "S1", "H"} を包含
    - "upstream_invocation_ids"                             # dict[str, str], Ch8/Ch9/Ch10 invocation_id を含む
    - "upstream_skill_versions"                             # dict[str, str], §11.5.2 の drift 検査用（"vX.Y.Z" format）
    - "synthesizability_proxy_score"                        # float [0, 1], Ch8 result 態
    - "dft_proxy_predicted_properties"                      # dict[str, {mean, std_raw, std_calibrated, ...}], Ch9 result 態 raw
    - "ood_score"                                           # float [0, 1], Ch10 result 態 ("in-distribution らしさ")
    - "positivity_violation"                                # bool, Ch10 result 態
    - "ood_provenance"                                      # dict, family_degenerate_to_2axis 等

output_schema:
  required_keys:
    - "candidate_ranking"                                   # result 態のトップレベル key
  result_required_fields:
    - "mode"                                                # literal "result"
    - "ranked_candidates"                                   # list[dict], 全候補分（top-k 外含む pass-through）
    - "n_candidates_input"
    - "n_pareto_front"
    - "n_selected_top_k"
    - "n_positivity_review_required"
    - "upstream_invocation_ids_format_valid"                # literal True（format 検査のみ、内容照合は Ch12 責務）
    - "ranking_method_used"                                 # literal "pareto_nsga2"
    - "config_ref"                                          # {skill, skill_version, invocation_id}
  ranked_candidate_required_fields:
    - "candidate_id"
    - "pareto_rank"                                         # int ≥ 1
    - "crowding_distance"                                   # float ≥ 0 or +inf
    - "axis_values"                                         # {synth_cost, property_cost, ood_cost}, 各 [0, 1] float
    - "selected_in_top_k"                                   # bool
    - "positivity_violation"                                # bool（Ch10 pass-through）
    - "positivity_review_required"                          # bool
    - "upstream_invocation_ids"                             # dict (Ch12 content verify 用 provenance pass-through)
    - "upstream_skill_versions"                             # dict (Ch12 content verify 用 provenance pass-through)
    - "upstream_stage_trace"                                # list[str] ⊇ {F, S1, H}
  validation_rule: >
    n_selected_top_k == sum(selected_in_top_k for c in ranked_candidates) を検査。
    n_pareto_front == sum(pareto_rank == 1 for c in ranked_candidates) を検査。
    n_positivity_review_required == sum(positivity_review_required and selected_in_top_k for c in ranked_candidates) を検査。
    n_candidates_input == len(ranked_candidates) を検査（pass-through 契約：全候補 rank 割当）。

# --- 目的軸定義（canonical 3 軸、順序・literal 名 pin）---
canonical_objectives:
  - axis_name: "synth_cost"
    source_skill: "arim.gen.synthesizability_proxy.v0.1"
    source_field: "synthesizability_proxy_score"
    transform: "one_minus"                                  # 1 - score, minimization に統一
    required_upstream_stage: "S1"
  - axis_name: "property_cost"
    source_skill: "arim.gen.dft_proxy.v0.1"
    source_field: "dft_proxy_predicted_properties"          # Ch9 raw dict (property_id → {mean, std_raw, std_calibrated})
    transform: "target_deviation"                           # Ch11 §11.4.0 で property_target_spec を使い [0, 1] に derive
    required_upstream_stage: "S1"
  - axis_name: "ood_cost"
    source_skill: "arim.gen.ood_detection.v0.1"
    source_field: "ood_score"                               # Ch10 §10.6 canonical: = 1 - composite_score ("in-distribution らしさ")
    transform: "one_minus"                                  # 1 - ood_score, minimization に統一（低分布内らしさ ≒ 高コスト）
    required_upstream_stage: "H"

# --- 物性目標仕様（property_cost 軸の derive 用、canonical）---
property_target_spec_schema:
  targets_required_fields:
    - "property_id"                                         # str, Ch9 dft_proxy_predicted_properties の key と一致必須
    - "target_value"                                        # float, canonical unit
    - "tolerance"                                           # float > 0, deviation を [0, 1] に正規化する basis
    - "direction"                                           # canonical 3 択: match | minimize | maximize
  aggregation_canonical: "max_normalized_deviation"         # canonical 単一 method（§11.4.0）

# --- 選抜設定 canonical default ---
selection_defaults:
  top_k: 10
  crowding_distance_enabled: true
  tie_breaking: "crowding_distance"                         # canonical 2 択: crowding_distance | lexicographic

# --- positivity 扱い canonical（Ch10 §10.7.2）---
positivity_handling:
  include_in_pareto: false                                  # literal False 固定（変更禁止）
  metadata_field: "positivity_violation"
  human_review_required: true                               # 変更可、ただし false 設定は Ch3 §3.5 B3 逸脱警告

# --- NF family 縮退の受け取り契約 ---
nf_family_handling:
  accept_family_degenerate_to_2axis: true                   # Ch10 provenance の family_degenerate_to_2axis を受け入れる
  behavior: "warn_log_only_axis_count_unchanged"            # Pareto 軸数は 3 のまま維持、警告 log のみ

# --- ステージング契約 ---
handoff_from:                                               # 上流 3 Skill を全て要求
  - "arim.gen.synthesizability_proxy.v0.1"
  - "arim.gen.dft_proxy.v0.1"
  - "arim.gen.ood_detection.v0.1"
handoff_to:
  - "arim.gen.experimental_authorization.v0.1"              # Ch12 capstone Phase 5 の Human 承認 Skill（想定）
  - "arim.gen.safety_governance.v0.1"                       # Ch15 の Safety governance 3-layer（想定）
outputs_disallowed_natural_language:                       # Ch4 §4.7 canonical: 直接 list
  - "この候補は自動的に合成試行の対象になります"           # T stage は launch authorization を持たない (Ch3 §3.5 B4)
  - "top-k 外の候補は破棄されました"                       # pass-through 契約の虚偽陳述
  - "Pareto rank 1 の候補は必ず最良です"                   # 多目的最適化の理論的誤解
  - "positivity_violation の候補は自動除外されました"     # §11.1.3 canonical の虚偽陳述
  - "crowding distance は重要度を表します"                 # crowding は多様性維持指標、重要度ではない
  - "top-k は連続 BO の次点候補です"                       # §11.1.1 canonical 逸脱
  - "acquisition function で選抜しました"                  # §11.2.2 canonical: 本 Skill は acquisition を使わない
  - "NF family の 3 軸目は独立です"                        # Ch10 §10.5.2 縮退契約の虚偽陳述

# --- disallowed operations ---
disallowed_operations:
  - expand_ranking_method_beyond_canonical                  # pareto_nsga2 のみ
  - reorder_or_rename_pareto_axes                           # axis_names の順序・literal 名は canonical 固定
  - include_positivity_in_pareto_axes                       # §11.1.3 canonical、Ch10 §10.7.2 canonical と一致
  - propagate_non_finite_axis_value                         # NaN/inf axis 伝達禁止
  - rank_without_full_upstream_trace                        # F/S1/H stage trace 全欠落禁止
  - mix_upstream_skill_versions                             # §11.5.2 canonical
  - auto_launch_synthesis_from_ranking                      # T stage は launch 権限を持たない (Ch3 §3.5 B4)
  - reject_top_k_outside_candidates                         # top-k 外候補は pass-through 必須、削除禁止
  - use_acquisition_function_instead_of_pareto              # §11.2.2 canonical: acquisition は使わない
  - collapse_axes_by_weighted_sum                           # Ch2 §2.4 canonical: 3 軸を scalar 合成禁止
  - propagate_flag_without_run_id                           # Ch10 の pattern と同旨
  - alter_top_k_ordering_by_positivity                      # positivity は metadata only、順位変更禁止
  - exceed_top_k_returned_cap                               # Ch4 §4.3.8 canonical: top_k ≤ 20
  - use_crowding_tie_break_without_crowding_enabled          # §11.4.4 consistency 契約
  - rank_uncalibrated_property_without_penalty              # §11.4.0 Ch9 §9.8.1 mode 2 canonical
  - accept_duplicate_or_empty_candidate_id                  # §11.4.4 一意性契約

provenance:
  human_boundary_reference: "Ch3 §3.5 B4"                   # 合成試行 launch は Ch11 の権限外
  aggregation_rule_reference: "Ch2 §2.4"                    # 3 軸 scalar 合成禁止の origin
  pareto_reference: "vol-05 第X章 (Pareto 選抜) / §11.2.3"
  causal_boundary_reference: "vol-04 第9章 (positivity) / Ch10 §10.7.2"
  invocation_id_format: "inv_YYYYMMDD_HHMMSS"
```

---

## 11.7 責務マトリクス：Ch8 / Ch9 / Ch10 / Ch11

本節は Ch8 (F/S1 stage)、Ch9 (S1 stage 併走)、Ch10 (H stage)、Ch11 (T stage) の境界を canonical に整理します。

### 11.7.1 pipeline stage / 出力 / reject 権限

| 項目 | Ch8 physics_filter (F) | Ch8 synthesizability (S1) | Ch9 dft_proxy (S1) | Ch10 ood_detection (H) | Ch11 candidate_ranking (T) |
|---|---|---|---|---|---|
| **pipeline stage** | F（フィルタ） | S1（surrogate 並走） | S1（surrogate 並走） | H（幻覚検知） | T（top-k 選抜） |
| **主出力** | physics_filter_passed, n_rule_violations | synthesizability_proxy_score | dft_proxy_predicted_properties (raw dict; Ch11 が target_deviation を導出) | z_M / z_R / v / composite_score / ood_score / ood_flag / positivity_violation | ranked_candidates + top-k フラグ |
| **reject 権限** | あり（rule violation で剥落） | なし（scoring のみ） | なし（predict のみ） | **なし**（reject 禁止契約 §10.7.1） | **なし**（top-k 外も pass-through） |
| **参照 snapshot** | rule set version | 学習済み合成可能性モデル + fingerprint | surrogate model + fingerprint | 4 core + family-cond + optional support snapshot | **なし**（上流 invocation_id のみ pin） |
| **入力軸** | 生成候補（Ch5-7） | Ch8 F 通過後 | Ch8 F 通過後 | Ch8 F 通過後（Ch9 と並走） | Ch8/Ch9/Ch10 全 result 態 |
| **Human 承認境界** | B1 (rule 更新) | B2 (model 更新) | B2 (model 更新) | B3 (snapshot 更新) | **B4 (合成試行 launch)** |

### 11.7.2 top-k 選抜と Human 承認境界の canonical

Ch11 result 態の `selected_in_top_k=True` フラグは **合成試行の launch 権限ではありません**。Ch3 §3.5 B4（合成試行 launch）は **Human 承認境界** であり、Ch11 は候補を **絞り込む** のみ、実際の launch は Ch12 capstone Phase 5 で Human が明示的に承認します。

**canonical 決定表（top-k フラグ × positivity_violation）**：

| top-k フラグ | positivity_violation | Ch12 Phase 5 の Human 承認要件 |
|---|---|---|
| False | False | Ch12 で除外（合成試行対象外） |
| False | True | Ch12 で除外（同上、positivity 判断は不要） |
| True | False | **合成試行候補**：Human が Ch8/Ch9/Ch10 の scoring と Pareto 位置を確認して承認 or 却下 |
| True | True | **合成試行候補（要因果境界レビュー）**：Human が positivity_violation の意味を明示的に評価。「因果解釈できないが物性的に興味深い」候補として承認する場合は追加の実験 protocol pinning を要求 |

**disallowed_operations**：
- `auto_launch_synthesis_from_ranking`：top-k フラグを合成試行 launch のトリガーとする実装は禁止
- `alter_top_k_ordering_by_positivity`：positivity_violation を top-k 順位付けに使うのは禁止（Pareto rank + crowding distance のみで決定）

---

## 11.8 失敗パターン 7 種

### 11.8.1 失敗 1：Pareto 軸を scalar に合成

**症状**：`ranking_score = 0.4 * synth + 0.4 * property + 0.2 * ood` のような線形合成で単一 scalar を作り、それでソート。

**根本原因**：多目的最適化を単目的に "簡略化" しようとして、trade-off 情報を失う。

**予防契約**：`disallowed_operations.collapse_axes_by_weighted_sum`。Ch2 §2.4 canonical と一致（`hallucinatory_composition_detection` の内部 composite_score は Ch10 レイヤ内部の合成のみで許容、Ch11 の 3 軸間には適用不可）。

### 11.8.2 失敗 2：positivity_violation で top-k を自動除外

**症状**：`positivity_violation=True` の候補を top-k から自動除外（`selected_in_top_k=False` に強制上書き）。

**根本原因**：因果解釈可能性を "必須" と誤解し、Human 判断領域を Skill 内部で先取り。

**予防契約**：`disallowed_operations.include_positivity_in_pareto_axes` + `alter_top_k_ordering_by_positivity`。Ch10 §10.7.2 canonical と一致。§11.7.2 の "top-k=True, positivity=True" ケースは Human 承認境界に残す。

### 11.8.3 失敗 3：acquisition function を軸に使う

**症状**：EI / UCB / LCB を Pareto 軸に混ぜる、または acquisition value で top-k を選抜。

**根本原因**：vol-05 の連続 BO パターンを離散候補ランキングにそのまま持ち込む。

**予防契約**：`disallowed_operations.use_acquisition_function_instead_of_pareto`。§11.2.2 canonical: acquisition は本 Skill では使わない。Ch12 capstone Phase 6 の "追加学習ループ" で、次バッチ生成の探索性を高めるために間接的に使う余地はあるが、それは本 Skill の外側。

### 11.8.4 失敗 4：upstream skill_version の混在

**症状**：Ch8 v0.1.0 と Ch8 v0.2.0 の synthesizability_proxy_score を混ぜた候補集合で Pareto を組む。

**根本原因**：候補集合の管理が分散し、per-candidate の skill_version トラッキングが漏れる。

**問題点**：異なる model version の score は basis が微妙に違い、Pareto rank が意味を失う。

**予防契約**：`disallowed_operations.mix_upstream_skill_versions`。全候補で Ch8/Ch9/Ch10 の skill_version が同一であることを起動時に検査（§11.5.2）。

### 11.8.5 失敗 5：top-k フラグを launch 権限と誤認

**症状**：`selected_in_top_k=True` を受けて Ch12 の合成試行 launch を自動起動（Human 承認を skip）。

**根本原因**：Ch11 の "top-k 絞り込み" と Ch12 の "launch authorization" の責務混同。

**予防契約**：`disallowed_operations.auto_launch_synthesis_from_ranking`。§11.7.2 canonical: launch は Ch3 §3.5 B4 の Human 承認境界。Ch11 result 態には `synthesis_launch_authorized` 相当のフィールドを持たない（意図的欠落）。

### 11.8.6 失敗 6：top-k 外候補の削除（pass-through 違反）

**症状**：`ranked_candidates` に top-k 選抜候補のみ含める（`n_candidates_input=250, len(ranked_candidates)=10`）。

**根本原因**：出力サイズを "整理" するために top-k 外を drop。

**問題点**：Ch12 capstone や事後分析で Pareto 全体の分布（rank 2, 3 の候補）を確認できなくなり、次バッチ生成の diversity 改善根拠を失う。

**予防契約**：`disallowed_operations.reject_top_k_outside_candidates`。output_schema の validation_rule で `n_candidates_input == len(ranked_candidates)` を literal 検査（§11.6）。

### 11.8.7 失敗 7：Pareto 軸の順序・名前変更

**症状**：`objectives` に `[ood_cost, synth_cost, property_cost]` の順で並べる、または `axis_name` を `"cost_A", "cost_B", "cost_C"` に匿名化。

**根本原因**：可搬性を高めるために axis metadata を "汎用化"、または上流 Skill の名前と切り離す試み。

**問題点**：Ch12 capstone Phase 5 の Human レビューで軸の物理的意味が復元できず、承認判断の根拠が失われる。

**予防契約**：`disallowed_operations.reorder_or_rename_pareto_axes`。`_aggregate_candidate_axes` 冒頭で `axis_names == CANONICAL_AXIS_NAMES` を tuple 比較で literal 検査（§11.4.1）。

---

## 11.9 チェックリスト

- [ ] **§11.1 の 3 動機**（連続 BO vs 離散候補、Pareto の canonical、positivity 非軸）を説明でき、vol-05 の BO との責務境界を pipeline stage で示せる
- [ ] **§11.2 vol-05 未読者 3 ページ最小知識**（GP surrogate / acquisition / Pareto）を再現でき、本章での位置付け（acquisition は使わない、Pareto は使う）を明言できる
- [ ] **§11.3 canonical schema**（config 態 / result 態 / delegated_to）の全 field を Ch4 §4.5 と一致させ、objectives の canonical 3 軸順・literal 名を書ける
- [ ] **§11.4 実装 4 レイヤ**（axes 集約 / Pareto rank / crowding distance / apply_ranking）の canonical 契約を書ける
- [ ] **§11.5 上流 invocation_id pinning の canonical** と snapshot を持たない設計理由を説明できる
- [ ] **§11.6 Skill YAML 完全 shape**（12 disallowed_operations、canonical_objectives の 3 軸固定、positivity_handling.include_in_pareto=false 固定）を再現できる
- [ ] **§11.7 責務マトリクス**：Ch8/Ch9/Ch10/Ch11 の境界と Human 承認境界 B4（合成試行 launch）を pipeline stage 観点で分離できる
- [ ] **§11.7.2 top-k × positivity 決定表** 4 象限で Ch12 Phase 5 の Human 承認要件を説明でき、**top-k フラグが launch 権限ではない** ことを canonical に論じられる
- [ ] **§11.8 失敗パターン 7 種**をチェックリスト化でき、それぞれに canonical 予防契約（`disallowed_operations` フィールド名）を対応させられる

---

## 11.10 参考文献

- Deb, K. et al. (2002). *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*. IEEE Trans. Evol. Comput. 6(2):182. [NSGA-II 型 Pareto rank + crowding distance の canonical reference]
- Miettinen, K. (1999). *Nonlinear Multiobjective Optimization*. Springer. [多目的最適化と Pareto 支配の教科書的整理]
- Coello, C. A. C. et al. (2007). *Evolutionary Algorithms for Solving Multi-Objective Problems*, 2nd ed. Springer. [Pareto 選抜の algorithm survey]
- Frazier, P. I. (2018). *A Tutorial on Bayesian Optimization*. arXiv:1807.02811. [BO の入門的整理、§11.2.1-11.2.2 の origin]
- Shahriari, B. et al. (2016). *Taking the Human Out of the Loop: A Review of Bayesian Optimization*. Proc. IEEE 104(1):148. [BO の全体像 review]
- vol-05 全体：Bayesian Optimization Skill と active learning、階層 BO capstone（本章 §11.2 の詳細版）
- vol-03 第8-9章：uncertainty quantification（Ch9 surrogate 予測の epistemic uncertainty の origin）
- vol-04 第9章：refutation テストと positivity 条件（本章 §11.1.3、§11.7.2 の因果境界の origin）
- vol-06 第2章 §2.4：`hallucinatory_composition_detection` 合成規則 canonical（本章 §11.8.1 の 3 軸 scalar 合成禁止の origin）
- vol-06 第3章 §3.5 B4：合成試行 launch の Human 承認境界（本章 §11.7 の T stage 権限境界の origin）
- vol-06 第4章 §4.5, §4.7：3 態 schema と Skill YAML canonical shape（本章 §11.3, §11.6 の完全 pin）
- vol-06 第8章 §8.5, §8.8：F/S1 stage 責務分離と handoff（本章 §11.7 責務マトリクスの upstream）
- vol-06 第9章 §9.6：DFT proxy Skill YAML の provenance pin pattern（本章 §11.5 invocation_id pinning の parallel）
- vol-06 第10章 §10.5-§10.7：4 snapshot + family-cond dist pinning と responsibility matrix（本章 §11.5 で snapshot を持たない設計理由の対比、§11.7 responsibility 継承）

---

**次章予告**（第12章）：**総合ハンズオン（Advanced Capstone）— Foundation Model → 候補生成 → 物理フィルタ → 分布的妥当性 → Surrogate 選抜 → Human 承認**。本章 §11.6 で pin した `candidate_ranking` の result 態を、Ch12 capstone Phase 4 の入力とし、**Phase 5: Human 承認 + Safety governance** で `selected_in_top_k=True` かつ `positivity_review_required=True` の候補について明示的な意思確認を行います。§11.7.2 の 4 象限決定表は Phase 5 の canonical judgment tree の直接根拠となります。
