# 第9章 DFT proxy と surrogate による物性予測を Skill 化する

> **本章の使い方**
> - **Part III の第 2 章**です。Ch8 で構築した F ステージ（`arim.gen.physics_filter.v0.1`）と S1 ステージ内の合成可能性 proxy（`arim.gen.synthesizability_proxy.v0.1`）に**並走**する **S1 ステージ第 2 Skill** `arim.gen.dft_proxy.v0.1` を実装します。
> - **本章の scope**：**DFT proxy（MEGNet / M3GNet / MACE 等の pre-trained universal potentials）** による物性予測、**予測不確かさ管理**（Deep Ensemble / MC Dropout / calibration；vol-03 Ch8-9 canonical の再利用）、**vol-03 Foundation Model 特徴を surrogate 入力に混ぜる 2 経路の使い分け**、そして **Skill 契約としての "proxy であって measurement ではない" 宣言**を Ch4 §4.3 canonical に沿って落とし込みます。
> - **Ch8 との責務分離**：Ch8 `synthesizability_proxy` は「合成前例に近いか」の proxy score を返します。**本章 `dft_proxy` は「予測物性値（formation energy / band gap / elastic modulus 等）と不確かさ」を返します**。両 Skill は S1 ステージで **同時に走り**、いずれも reject せず、Ch11 T ステージのランキング入力になります（§9.7 責務マトリクス）。
> - **YAML block は全て `yaml.safe_load` 準拠**：本書 CI で機械検証されます。
> - **教育目的の DFT proxy**：MEGNet / M3GNet / MACE は既存 pre-trained weight（Materials Project や MPtrj で学習された universal potential）を **fetch → 推論** する Route A（Ch4 §4.9 PRE ステージ `arim.gen.fm_fetch.v0.1` 経由）を扱います。本番運用の precision/recall と外挿域挙動は §9.5 で議論します。
>
> **本章の到達目標**
> - **DFT proxy と DFT measurement の categorical 差**（proxy は surrogate、真値ではない）を Skill YAML `dft_proxy_role` field で literal に固定できる
> - **canonical 3 surrogate 手法**（MEGNet / M3GNet / MACE）の対象物性と入出力契約を書き分けられる
> - **予測不確かさの 3 経路**（Deep Ensemble / MC Dropout / Bayesian last-layer）を vol-03 Ch8-9 の canonical と共通化して Skill 出力 schema に埋め込める
> - **`arim.gen.dft_proxy.v0.1` の完全 Skill YAML** を Ch4 §4.3 の 14 provenance fields + Safety 3-layer + `outputs_disallowed_natural_language` + pipeline_position の全項目を満たして書ける
> - **Ch8 synthesizability_proxy と Ch9 dft_proxy の責務マトリクス**（§9.7）を Ch11 T ステージ入力視点で整合させられる
> - **vol-03 Foundation Model 特徴を surrogate に足す 2 経路**（Route α: FM features + MLP、Route β: FM を fine-tune）を、本 Skill の入力 shape と provenance field で pin できる
>
> **本章で扱わないこと**
> - 物理制約フィルタ（Ch8、`arim.gen.physics_filter.v0.1` F ステージ）
> - 合成可能性 proxy（Ch8、`arim.gen.synthesizability_proxy.v0.1` S1 併走 Skill）
> - 分布的妥当性判定（Ch10、`arim.gen.ood_detection.v0.1` H ステージ完全計算）
> - 離散候補ランキングと top-k 選抜（Ch11、`arim.gen.candidate_ranking.v0.1` T ステージ）
> - GP surrogate による active learning（vol-05 canonical、Ch11 で流用）
> - MEGNet / M3GNet / MACE 内部の GNN アーキテクチャ詳細（vol-03 Ch11 材料 FM 参照）
> - Agentic 失敗パターン全般（Ch14）

---

## 9.1 なぜ DFT proxy を Skill 化するのか — 3 つの動機

### 9.1.1 動機 1：DFT 直接計算は agentic pipeline に載らない

第一原理計算（VASP / Quantum ESPRESSO / GPAW 等の DFT）は 1 組成あたり CPU 数時間〜数日を要します。Ch5-7 の生成器は 1 回の推論で数百〜数千候補を返す設計であり、**agentic loop の 1 iteration 内で全候補を DFT 計算することは物理的に不可能**です。

DFT proxy（pre-trained universal potentials）は **GNN による物性予測を 1 候補あたり ms オーダで**返します。これにより S1 ステージが agentic loop の 1 サイクル内に収まり、Ch11 T ステージへの候補ランキング入力を実時間で作れます。**Skill 化の目的は「DFT で置き換える」ことではなく、「DFT が届く前段で候補を絞る proxy 契約層」を提供すること**です。

### 9.1.2 動機 2：合成可能性 proxy だけでは物性推定が欠落する

Ch8 `synthesizability_proxy` は「合成前例に近いか」の soft score を返しますが、**「その候補が目標物性を満たすか」は判定できません**。逆設計（inverse design）の目的は「合成可能かつ目標物性を満たす候補を top-k で提案する」ことなので、**合成可能性軸と物性予測軸は直交に必要**です。

Ch11 の Pareto ランキング（vol-05 §5.5 の非劣ソート）は、**S1 ステージから返される 2 種類のスコア**（`synthesizability_proxy_score` と `dft_proxy_predicted_properties`）を **同時に受け取ってこそ多目的最適化として機能**します。本章の Skill は Ch11 の非劣ソート入力を produce する契約層です。

### 9.1.3 動機 3：予測不確かさなしの proxy は agentic 誤用を招く

vol-03 Ch8-9 で議論した通り、**予測不確かさを返さない model は agentic pipeline に載せてはいけません**——エージェントが「予測値が良ければ推奨」と判断する時、**外挿域の予測値が学習分布内と同じ confidence で使われる**のが agentic 誤用の第一パターン（Ch14 §14.2 予定）です。

DFT proxy は「学習分布」が Materials Project や MPtrj であり、**生成候補は必ずしもその分布内ではない**（Ch5-7 の生成分布は独自の学習データからのため）。したがって、**予測値と不確かさをペアで返し、Ch10 H ステージ OOD スコアと合わせて Ch11 が下降補正できる**契約が必須です。

> [!IMPORTANT]
> **本章 Skill は「予測物性値 + 予測不確かさ + provenance」の三点セットを返す契約**：この 3 つのいずれか 1 つでも欠落した出力は、Ch4 §4.7 canonical 禁止フレーズ違反（"性能が良い" 等の自然言語集約）を招く直接原因になります。§9.6 Skill YAML で 3 点セットを literal に強制します。

---

## 9.2 DFT proxy Skill の canonical 契約 — 3 レイヤ構造

`arim.gen.dft_proxy.v0.1` は **3 レイヤの内部段**で構成されます。Ch8 の Filter A/B/C とは責務が異なるため、あえて別名で呼びます。

- **Predict レイヤ（P-layer）**：pre-trained universal potential（MEGNet / M3GNet / MACE 等）による物性予測
- **Uncertainty レイヤ（U-layer）**：Predict レイヤの各予測値に不確かさ（Deep Ensemble / MC Dropout / Bayesian last-layer）を紐付け
- **Calibration レイヤ（C-layer）**：validation set（学習データ hold-out）で得た **coverage 校正曲線**を使って confidence interval を calibrate

### 9.2.1 3 レイヤの reject/soft-score 契約

Ch8 と同様、**本 Skill は reject しません**。全候補に予測値 + 不確かさ + calibration flag を付与して S2/T ステージ（Ch11）に流します。「予測が悪い候補を drop する」判断は **Ch11 の T ステージが所有**します（vol-05 の Pareto 非劣ソートで自然に淘汰されるため、Skill 層で reject する必要がない）。

> [!NOTE]
> **なぜ hard reject しないか**：DFT proxy の unreliable 予測（U-layer の予測分散が大きい / calibration が破綻）は、**Ch11 が uncertainty penalty として組み込む**設計です。Ch9 Skill 層で reject するとエージェントが「reject 閾値を勝手に緩める」失敗（Ch14 予定）を招くため、**判定は上流に委譲**する canonical を維持します。

### 9.2.2 Ch10 H ステージ OOD detection との協調

生成候補が学習分布外（OOD）である場合、DFT proxy の予測不確かさは **統計的に不十分**（epistemic uncertainty を過小評価）です。この場合、**Ch10 H ステージ OOD スコア（`ood_score`）を Ch11 が Pareto の第 3 軸として組み込む**ことで補正します。本 Skill は `ood_score` を計算しませんが、`upstream_ood_score` field で H ステージからの値を **保持し** Ch11 に流す義務があります（§9.6 Skill YAML）。

---

## 9.3 canonical 3 surrogate 手法の対象物性と契約

### 9.3.1 3 手法の対象物性と入出力

| method_id | 対象物性 | 入力 | 出力 | 参照 |
|---|---|---|---|---|
| `megnet_v0.4` | formation energy / bandgap / elastic modulus | pymatgen `Structure` | scalar values | Materials Project |
| `m3gnet_v0.2` | formation energy / forces / stress | pymatgen `Structure` | scalar + tensor | MPtrj |
| `mace_mp0_v1` | formation energy / forces / stress / phonon | pymatgen `Structure` | scalar + tensor | MPtrj + MACE-MP-0 |

> [!NOTE]
> **手法選択の canonical**：3 手法は **物性依存で使い分ける** canonical です。formation energy と bandgap は MEGNet が長年の実績を持ち、forces / stress を要する場合は M3GNet / MACE を優先します。**Skill 呼び出しごとに 1 method** の契約を維持（§9.6 YAML `method_id` 単一値）、複数手法の重ね合わせは Ch11 のランキング Skill に委譲します。

### 9.3.2 入力 composition の扱い

Ch8 §8.3.2 で議論した canonicalize 済み整数比 dict と Structure の使い分けを本章でも継承します：

| method_id | 入力形式 | 理由 |
|---|---|---|
| `megnet_v0.4` (全 property) | pymatgen `Structure` 必須 | formation energy を含めて全物性が原子座標・格子に依存 |
| `m3gnet_v0.2` | pymatgen `Structure` 必須 | universal potential、座標・格子に鋭敏 |
| `mace_mp0_v1` | pymatgen `Structure` 必須 | MACE の GNN receptor は座標依存 |

> [!IMPORTANT]
> **composition-only 候補は DFT proxy を通せない canonical**：DFT proxy 予測物性はすべて structure-conditioned（MEGNet formation energy も原子座標・格子に依存する）です。Ch5-7 生成器のうち組成のみを出力する family（VAE / Flow / AR）の `generated_x` を fictitious cell 上で走らせると、Ch11 Pareto を汚染します。canonical 対応方針は 2 択：
> 1. Ch6 の結晶 Diffusion refinement もしくは explicit relaxer（例: `arim.gen.structure_relaxer.v0.1`）で **real relaxed `Structure` に refine** してから DFT proxy に流す
> 2. Ch8 `synthesizability_proxy` の proxy score のみで S1 を完了させ、DFT proxy を skip（Ch11 Pareto は 2 軸縮退で agent 側が承認取り付け）
>
> 本 Skill は `structure` field を **required_keys** として declare し、composition-only は fail-fast します（§9.6 YAML / `disallowed_operations.invoke_dft_proxy_on_composition_only_candidate`）。

### 9.3.3 canonical 予測物性 list（本書での固定 set）

本章では以下 canonical 5 物性を pin します（§9.6 Skill YAML で literal 固定）：

| property_id | 単位 | 対応 method | 意味 |
|---|---|---|---|
| `formation_energy_per_atom` | eV/atom | megnet, m3gnet, mace | 生成熱の proxy（thermodynamic stability） |
| `bandgap` | eV | megnet | 電子構造 proxy |
| `log10_bulk_modulus_gpa` | log10(GPa) | megnet | 弾性物性 proxy（MEGNet MP-2018 pretrained は Voigt-Reuss-Hill 平均 K の log10 を native 出力）。GPa 変換は Ch11 下流で `10**x` する canonical。 |
| `log10_shear_modulus_gpa` | log10(GPa) | megnet | 弾性物性 proxy（同上、G_VRH の log10）。 |
| `energy_above_hull` | meV/atom | derived from formation_energy + MP hull reference | 準安定度 proxy（**非負値 canonical**——負値は surrogate 誤差 or hull snapshot 不整合の signature として 0 に clamp し、生値は audit 用 `signed_hull_offset_mev_per_atom` に保持。Ch8 `synthesizability_proxy` の `hull_distance` method と **同一 MP hull snapshot** を参照する canonical——Ch11 Pareto 突合の整合性のため） |

これら以外の物性を Ch11 で使う場合は **本章の拡張として `predicted_properties_extended` field に格納**し、canonical 5 物性とは別 dict で管理します（§9.6 で分離）。

---

## 9.4 実装：Predict / Uncertainty / Calibration 3 レイヤ

### 9.4.1 canonical import と共通定数

**コードスニペット 1 — Ch9 canonical import**

```python
from typing import Callable
from dataclasses import dataclass
import hashlib
import json
import math
import re
import numpy as np
from pymatgen.core import Structure, Composition
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
# 以下はサードパーティ optional（実行時 fetch は arim.gen.fm_fetch.v0.1 経由）
# from megnet.models import MEGNetModel
# from m3gnet.models import Relaxer, M3GNet
# from mace.calculators import MACECalculator

# Ch9 canonical constants
CANONICAL_METHODS = ("megnet_v0.4", "m3gnet_v0.2", "mace_mp0_v1")
CANONICAL_PROPERTIES = (
    "formation_energy_per_atom",
    "bandgap",
    "log10_bulk_modulus_gpa",
    "log10_shear_modulus_gpa",
    "energy_above_hull",
)
CANONICAL_UNCERTAINTY_METHODS = ("deep_ensemble", "mc_dropout", "bayesian_last_layer")

# Uncertainty 推定に十分な標本数の canonical 下限
# 1 だと std=0.0（fake certainty）、0 だと mean=NaN。どちらも Ch11 Pareto / Safety 契約を
# 破壊するので fail-fast。canonical default は §9.4 コンスタント参照。
DEEP_ENSEMBLE_MIN_MEMBERS = 2
MC_DROPOUT_MIN_SAMPLES = 2
BAYESIAN_LAST_LAYER_MIN_SAMPLES = 2

# 各 method 対応可能 property の canonical mapping
# formation_energy_per_atom, bandgap, log10_bulk_modulus_gpa, log10_shear_modulus_gpa は method が直接予測する native property。
# energy_above_hull は formation_energy_per_atom + MP hull reference から本 Skill 内 derive する canonical
# derived property であり、下記 map には native property のみ列挙する。
#
# 注意（Ch9 canonical）：M3GNet / MACE の predict_energy() は universal potential の "total potential energy"
# を返す設計であり、そのままでは MP-compatible formation_energy_per_atom にならない（elemental reference
# 差分と atomic reference 補正が要る）。本 Skill では **elemental reference correction handle** を
# predict_config["_elemental_reference_correction"] として要求する canonical で扱う（§9.4.2 参照）。
# 未提供時は fail-fast（formation_energy_per_atom を requested_properties に含めた場合）。
METHOD_PROPERTY_MAP: dict[str, tuple[str, ...]] = {
    "megnet_v0.4": (
        "formation_energy_per_atom", "bandgap", "log10_bulk_modulus_gpa", "log10_shear_modulus_gpa",
    ),
    "m3gnet_v0.2": ("formation_energy_per_atom",),  # 要 elemental reference correction（下 note）
    "mace_mp0_v1": ("formation_energy_per_atom",),  # 要 elemental reference correction（下 note）
}

# universal potential 型 method は elemental reference correction が必須
METHODS_REQUIRING_ELEMENTAL_REFERENCE = frozenset({"m3gnet_v0.2", "mace_mp0_v1"})

# derived properties: 各 derived property の必要 native properties を列挙。
# energy_above_hull は formation_energy_per_atom があれば MP hull と照合して計算できる。
DERIVED_PROPERTIES: dict[str, tuple[str, ...]] = {
    "energy_above_hull": ("formation_energy_per_atom",),
}

def _method_supports(method_id: str, prop: str) -> bool:
    """native or derivable かの判定 canonical。"""
    native = METHOD_PROPERTY_MAP[method_id]
    if prop in native:
        return True
    if prop in DERIVED_PROPERTIES:
        return all(dep in native for dep in DERIVED_PROPERTIES[prop])
    return False

# Uncertainty method canonical parameters
DEEP_ENSEMBLE_MEMBERS = 5              # vol-03 Ch8 canonical
MC_DROPOUT_SAMPLES = 30                # vol-03 Ch8 canonical
CALIBRATION_COVERAGE_TARGET = 0.9      # 90% CI target
```

### 9.4.2 Predict レイヤ実装

**コードスニペット 2 — Predict layer canonical**

```python
def _predict_layer(
    candidate: dict,
    method_id: str,
    requested_native: tuple[str, ...],
    predict_config: dict,
) -> dict[str, float]:
    """
    Predict layer: 1 候補 → requested_native subset の scalar 物性 dict。
    requested_native は method_id の native supported set の subset（呼び出し側で保証）。
    """
    if method_id not in CANONICAL_METHODS:
        raise ValueError(f"unknown method_id: {method_id}, expected one of {CANONICAL_METHODS}")
    if not (set(requested_native) <= set(METHOD_PROPERTY_MAP[method_id])):
        raise ValueError(
            f"requested_native {requested_native} exceeds native support of {method_id}"
        )

    # DFT proxy 全体で composition-only は fail-fast（apply_dft_proxy 側で事前 check 済みだが、
    # _predict_layer 単独呼び出しでも防御）。実物 Structure を要求。
    if "structure" not in candidate or candidate["structure"] is None:
        raise ValueError(
            f"candidate {candidate.get('candidate_id')} lacks real Structure; "
            "DFT proxy properties are structure-conditioned. Refine composition-only candidates "
            "before invoking dft_proxy (Ch6 diffusion / explicit relaxer)."
        )
    struct: Structure = candidate["structure"]

    # requested_native の subset のみ計算する（未 requested property の leak を防ぐ canonical）
    properties: dict[str, float] = {}
    if method_id == "megnet_v0.4":
        # MEGNet は property ごとに別モデル handle を要求（single-handle predict_structure(prop=...) は
        # 標準 API ではない）。予測物性ごとに事前 fit された handle を dict で受け取る canonical。
        model_handles: dict = predict_config.get("_model_handles_by_property", {})
        for prop in requested_native:
            if prop not in model_handles:
                raise ValueError(
                    f"megnet_v0.4 requires predict_config['_model_handles_by_property']['{prop}'] "
                    f"(property-specific pretrained MEGNet handle); missing for candidate "
                    f"{candidate.get('candidate_id')}"
                )
            val = float(model_handles[prop].predict_structure(struct))
            if not math.isfinite(val):
                raise ValueError(
                    f"megnet_v0.4 returned non-finite {prop}={val} for candidate "
                    f"{candidate.get('candidate_id')}; refuse to propagate NaN/inf downstream"
                )
            properties[prop] = val
    elif method_id in ("m3gnet_v0.2", "mace_mp0_v1"):
        model = predict_config["_model_handle"]
        if "formation_energy_per_atom" in requested_native:
            # universal potential total energy を MP-compatible formation energy per atom に変換
            # elemental reference correction は predict_config で提供必須（§9.4.1 note 参照）
            corr = predict_config.get("_elemental_reference_correction")
            if corr is None:
                raise ValueError(
                    f"method_id {method_id} requires predict_config['_elemental_reference_correction'] "
                    "(callable: (Structure, total_energy) -> formation_energy_per_atom) to produce "
                    "MP-compatible formation_energy_per_atom; total_energy alone is not comparable to MP hull"
                )
            e_total = float(model.predict_energy(struct))
            if not math.isfinite(e_total):
                raise ValueError(
                    f"{method_id} returned non-finite total energy for candidate "
                    f"{candidate.get('candidate_id')}"
                )
            fe = float(corr(struct, e_total))
            if not math.isfinite(fe):
                raise ValueError(
                    f"elemental reference correction returned non-finite formation_energy_per_atom "
                    f"({fe}) for candidate {candidate.get('candidate_id')}"
                )
            properties["formation_energy_per_atom"] = fe
    return properties
```

### 9.4.3 Uncertainty レイヤ実装

**コードスニペット 3 — Uncertainty layer canonical**

```python
def _uncertainty_layer(
    candidate: dict,
    method_id: str,
    requested_native: tuple[str, ...],
    predict_config: dict,
    uncertainty_method: str,
    uncertainty_config: dict,
) -> dict[str, dict[str, float]]:
    """
    各 requested native 物性に不確かさ（mean と std）を紐付ける。
    Deep Ensemble: 独立 5 モデルの predict を stack → mean / std
    MC Dropout: dropout on 30 samples の predict を stack → mean / std
    Bayesian last-layer: last-layer posterior sampling → mean / std
    vol-03 Ch8-9 canonical と同じ 3 経路を再利用（実装本体は vol-03 Ch8 §8.3 参照）。
    """
    if uncertainty_method not in CANONICAL_UNCERTAINTY_METHODS:
        raise ValueError(
            f"unknown uncertainty_method: {uncertainty_method}, "
            f"expected one of {CANONICAL_UNCERTAINTY_METHODS}"
        )
    if not (set(requested_native) <= set(METHOD_PROPERTY_MAP[method_id])):
        raise ValueError(
            f"requested_native {requested_native} exceeds native support of {method_id}"
        )

    predictions_per_property: dict[str, list[float]] = {prop: [] for prop in requested_native}

    if uncertainty_method == "deep_ensemble":
        # predict_config["_ensemble_handles"] は 5 個の independent model（M3GNet/MACE）。
        # MEGNet は property 別 handle canonical のため、_ensemble_handles_by_property[prop] = list[handle]
        # の shape で供給する（各 property について独立に n_members の property-specific ensemble）。
        # 両方式を method_id で分岐する canonical。
        if method_id == "megnet_v0.4":
            ens_by_prop: dict[str, list] = predict_config.get("_ensemble_handles_by_property", {})
            n_members = uncertainty_config.get("n_members", DEEP_ENSEMBLE_MEMBERS)
            for prop in requested_native:
                if prop not in ens_by_prop:
                    raise ValueError(
                        f"megnet_v0.4 deep_ensemble requires "
                        f"predict_config['_ensemble_handles_by_property']['{prop}']; missing"
                    )
                if len(ens_by_prop[prop]) != n_members:
                    raise ValueError(
                        f"megnet_v0.4 deep_ensemble for {prop!r} requires exactly {n_members} "
                        f"handles (got {len(ens_by_prop[prop])})"
                    )
            for i in range(n_members):
                # ensemble member i: 各 property の i 番目 handle を束ねた {_model_handles_by_property}
                member_handles = {prop: ens_by_prop[prop][i] for prop in requested_native}
                member_config = {**predict_config, "_model_handles_by_property": member_handles}
                preds = _predict_layer(candidate, method_id, requested_native, member_config)
                for prop in requested_native:
                    predictions_per_property[prop].append(preds[prop])
        else:
            n_members = uncertainty_config.get("n_members", DEEP_ENSEMBLE_MEMBERS)
            if len(predict_config["_ensemble_handles"]) != n_members:
                raise ValueError(
                    "deep_ensemble requires exactly n_members loaded model handles "
                    f"(got {len(predict_config['_ensemble_handles'])}, expected {n_members})"
                )
            for handle in predict_config["_ensemble_handles"]:
                member_config = {**predict_config, "_model_handle": handle}
                preds = _predict_layer(candidate, method_id, requested_native, member_config)
                for prop in requested_native:
                    predictions_per_property[prop].append(preds[prop])
    elif uncertainty_method == "mc_dropout":
        n_samples = uncertainty_config.get("n_samples", MC_DROPOUT_SAMPLES)
        # predict_config["_model_handle"] は dropout on mode の model
        for _ in range(n_samples):
            preds = _predict_layer(candidate, method_id, requested_native, predict_config)
            for prop in requested_native:
                predictions_per_property[prop].append(preds[prop])
    elif uncertainty_method == "bayesian_last_layer":
        # last-layer weights posterior sampling; 実装は vol-03 §8.3 参照
        n_samples = uncertainty_config.get("n_samples", MC_DROPOUT_SAMPLES)
        for _ in range(n_samples):
            preds = _predict_layer(candidate, method_id, requested_native, predict_config)
            for prop in requested_native:
                predictions_per_property[prop].append(preds[prop])

    # requested_native subset について mean / std を組み立てる
    # 事前 fail-fast で n >= 2 が保証されているため std は ddof=1 で計算する canonical
    # 加えて、mc_dropout / bayesian_last_layer で全サンプルが同一値（std==0）となる場合は
    # 「decorrelated stochastic sampler」契約違反（決定的 handle を渡した / dropout が inactive /
    # posterior sampler が seed 固定）。fake certainty は Ch11 Pareto と Safety CI 契約を破壊する
    # ため fail-fast する。deep_ensemble は各 member が独立学習された「independent point estimate」
    # のため std==0 も理論上ありうるが、実装エラーの signature でもあるので同様に fail-fast。
    STD_EPS = 1e-12
    result: dict[str, dict[str, float]] = {}
    for prop in requested_native:
        arr = np.array(predictions_per_property[prop])
        mean_v = float(arr.mean())
        std_v = float(arr.std(ddof=1))
        if not (math.isfinite(mean_v) and math.isfinite(std_v) and std_v >= 0.0):
            raise ValueError(
                f"uncertainty layer produced non-finite or negative moments for {prop}: "
                f"mean={mean_v}, std={std_v} (candidate {candidate.get('candidate_id')})"
            )
        if std_v <= STD_EPS:
            raise ValueError(
                f"uncertainty layer produced zero-variance samples for {prop} "
                f"(std={std_v}, uncertainty_method={uncertainty_method}); "
                "this indicates a non-stochastic handle (dropout inactive / "
                "posterior sampler seed-locked / duplicated ensemble members). "
                "fake certainty violates Ch11 Pareto and Safety CI contracts."
            )
        result[prop] = {"mean": mean_v, "std": std_v}
    return result
```

### 9.4.4 Calibration レイヤ実装

**コードスニペット 4 — Calibration layer canonical**

```python
@dataclass(frozen=True)
class CalibrationTable:
    """validation set で作った coverage 校正表。property_id → (nominal_coverage → actual_coverage) の非減少 monotone table。
    fit_calibration_from_validation() で作り、Skill YAML calibration_provenance に fingerprint を pin する。
    """
    property_id: str
    coverage_target: float                    # 例：0.9
    scale_factor: float                       # std × scale で actual coverage を coverage_target に合わせる
    validation_set_fingerprint: str           # 学習時 hold-out set の hash
    n_validation_samples: int


def _calibration_layer(
    uncertainty_output: dict[str, dict[str, float]],
    calibration_tables: dict[str, CalibrationTable],
    coverage_target: float = CALIBRATION_COVERAGE_TARGET,
    calibration_provenance: dict | None = None,
) -> dict[str, dict[str, float]]:
    """
    各物性の std に calibration_tables[property_id].scale_factor を掛けて calibrated interval を組む。
    coverage_target 未 fit の物性は raw std を返し `calibration_status: uncalibrated` を明示する。

    calibration_provenance（YAML の calibration_provenance 由来）が渡された場合、
    table の property_id / scale_factor / fingerprint / n_validation_samples を pin と突合し、
    差異があれば fail-fast（stale/mis-swapped table による false trust interval を防ぐ）。
    """
    calibrated: dict[str, dict[str, float]] = {}
    for prop, moments in uncertainty_output.items():
        mean = moments["mean"]
        std_raw = moments["std"]
        if prop not in calibration_tables:
            calibrated[prop] = {
                "mean": mean,
                "std_raw": std_raw,
                "std_calibrated": std_raw,   # fallback として raw を expose
                "coverage_target": coverage_target,
                "calibration_status": "uncalibrated",
                "calibration_scale_factor": 1.0,
            }
            continue
        table = calibration_tables[prop]
        # 基本 sanity
        if table.property_id != prop:
            raise ValueError(
                f"calibration table property_id {table.property_id!r} != map key {prop!r} "
                "(mis-assigned table; refuse to apply)"
            )
        if table.coverage_target != coverage_target:
            raise ValueError(
                f"calibration table {prop} target {table.coverage_target} != requested {coverage_target}"
            )
        if not (math.isfinite(table.scale_factor) and table.scale_factor > 0.0):
            raise ValueError(
                f"calibration table {prop} scale_factor must be finite positive "
                f"(got {table.scale_factor})"
            )
        if not isinstance(table.n_validation_samples, int) or table.n_validation_samples < 1:
            raise ValueError(
                f"calibration table {prop} n_validation_samples must be positive int "
                f"(got {table.n_validation_samples!r})"
            )
        if not isinstance(table.validation_set_fingerprint, str) \
                or not table.validation_set_fingerprint:
            raise ValueError(
                f"calibration table {prop} validation_set_fingerprint must be non-empty str"
            )
        # calibration_provenance pin と突合（stale/swap 検知）
        # calibrated property は必ず provenance に pin されている必要がある（unpinned = bypass 経路）
        if calibration_provenance is not None:
            per_prop = calibration_provenance.get("per_property_scale_factors", {})
            if prop not in per_prop:
                raise ValueError(
                    f"calibration table {prop} is not pinned in calibration_provenance."
                    "per_property_scale_factors; refuse to emit 'calibrated' status "
                    "(unpinned tables bypass stale/swap detection)"
                )
            pinned = per_prop[prop]
            if pinned.get("scale_factor") != table.scale_factor:
                raise ValueError(
                    f"calibration table {prop} scale_factor {table.scale_factor} "
                    f"!= pinned {pinned.get('scale_factor')} (stale or swapped table)"
                )
            if pinned.get("validation_set_fingerprint") != table.validation_set_fingerprint:
                raise ValueError(
                    f"calibration table {prop} validation_set_fingerprint mismatch pinned value "
                    "(stale or swapped validation set)"
                )
            if pinned.get("n_validation_samples") != table.n_validation_samples:
                raise ValueError(
                    f"calibration table {prop} n_validation_samples "
                    f"{table.n_validation_samples} != pinned {pinned.get('n_validation_samples')}"
                )
        std_cal = std_raw * table.scale_factor
        calibrated[prop] = {
            "mean": mean,
            "std_raw": std_raw,
            "std_calibrated": std_cal,
            "coverage_target": coverage_target,
            "calibration_status": "calibrated",
            "calibration_scale_factor": table.scale_factor,
            "validation_set_fingerprint": table.validation_set_fingerprint,
            "n_validation_samples": table.n_validation_samples,
        }
    return calibrated
```

### 9.4.5 3 レイヤ統合：`apply_dft_proxy` canonical 実装

**コードスニペット 5 — 3 レイヤ統合と F/H upstream 検証**

```python
def apply_dft_proxy(
    candidates_after_fh: list[dict],
    dft_proxy_config: dict,
) -> dict:
    """
    Ch9 S1 併走 Skill canonical: DFT proxy による物性予測 + 不確かさ + calibration。
    入力候補は必ず F ステージ（Ch8 apply_physics_filter の `survivors`）と
    H ステージ（Ch10 arim.gen.ood_detection.v0.1）を pass 済みであること。
    低予測 / 高不確かさ候補も drop せず全て S2/T ステージに流す（Ch11 が top-k 選抜を所有）。
    """
    # F/H pass provenance の enforcement（Ch8 §8.7 synthesizability_proxy と対称）
    for cand in candidates_after_fh:
        if not cand.get("physics_filter_run_id"):
            raise ValueError(
                f"candidate {cand.get('candidate_id')} lacks physics_filter_run_id; "
                "S1 requires concrete F stage execution id (flag alone insufficient)"
            )
        if not cand.get("physics_filter_passed", False):
            raise ValueError(
                f"candidate {cand.get('candidate_id')} lacks physics_filter_passed=True"
            )
        if not cand.get("ood_detection_run_id"):
            raise ValueError(
                f"candidate {cand.get('candidate_id')} lacks ood_detection_run_id; "
                "S1 requires concrete H stage execution id (flag alone insufficient)"
            )
        if not cand.get("ood_detection_passed", False):
            raise ValueError(
                f"candidate {cand.get('candidate_id')} lacks ood_detection_passed=True"
            )
        trace = set(cand.get("upstream_stage_trace", []))
        if not {"F", "H"} <= trace:
            raise ValueError(
                f"candidate {cand.get('candidate_id')} upstream_stage_trace missing F/H: got {trace}"
            )
        # ood_score は Ch10 H stage で必ず finite numeric として confer される（Ch11 Pareto 第 3 軸）。
        # ood_detection_passed=True の状態で ood_score が欠落 / bool / non-numeric / NaN / inf なら fail-fast。
        # そのまま下流に流すと Ch11 が OOD 軸を欠く or NaN 汚染 Pareto を作ってしまう。
        _os = cand.get("ood_score", None)
        if isinstance(_os, bool) or _os is None or not isinstance(_os, (int, float)) \
                or not math.isfinite(float(_os)):
            raise ValueError(
                f"candidate {cand.get('candidate_id')} lacks finite numeric ood_score "
                f"(got {_os!r}); Ch10 H stage must confer finite ood_score for Ch11 Pareto third axis"
            )
        # DFT proxy 全 property は本質的に structure-conditioned（MEGNet formation energy も
        # 原子座標・格子に依存する）。composition-only 候補の fictitious cell 上で走らせた予測は
        # Ch11 Pareto を汚染するので、DFT proxy 段階では fail-fast。
        # composition-only 候補は Ch8 synthesizability_proxy（proxy score）のみ走らせ、
        # DFT proxy を通す前に Ch6 diffusion refinement もしくは explicit relaxer で
        # real relaxed Structure に refine する契約（§9.7 責務マトリクス）。
        if ("structure" not in cand) or (cand.get("structure") is None):
            raise ValueError(
                f"candidate {cand.get('candidate_id')} is composition-only (no real Structure); "
                "DFT proxy properties are all structure-conditioned. Refine to real relaxed "
                "Structure via Ch6 diffusion or explicit relaxer before invoking dft_proxy, "
                "or route composition-only candidates through Ch8 synthesizability_proxy only."
            )

    # method_id enforcement（canonical set 内）
    method_id = dft_proxy_config["method_id"]
    if method_id not in CANONICAL_METHODS:
        raise ValueError(f"unknown method_id: {method_id}, expected {CANONICAL_METHODS}")

    # requested_properties が method_id で native supported または derivable であることを検証
    requested_properties: tuple[str, ...] = tuple(dft_proxy_config["requested_properties"])
    if len(requested_properties) == 0:
        raise ValueError(
            "requested_properties must contain at least one canonical property; "
            "empty set leaves Ch11 without the performance axis (Ch11 Pareto 契約違反)"
        )
    for p in requested_properties:
        if p not in CANONICAL_PROPERTIES:
            raise ValueError(f"{p} is not in CANONICAL_PROPERTIES {CANONICAL_PROPERTIES}")
        if not _method_supports(method_id, p):
            raise ValueError(
                f"method_id {method_id} does not support (native or derived) property {p}; "
                f"native set is {METHOD_PROPERTY_MAP[method_id]}, "
                f"derived requires deps {DERIVED_PROPERTIES.get(p)}"
            )

    # requested_native = native properties actually computed (derivations の deps を含む)
    requested_native_set: set[str] = set()
    for p in requested_properties:
        if p in METHOD_PROPERTY_MAP[method_id]:
            requested_native_set.add(p)
        elif p in DERIVED_PROPERTIES:
            requested_native_set.update(DERIVED_PROPERTIES[p])
    # canonical order（決定論的順序）で tuple 化
    requested_native: tuple[str, ...] = tuple(
        prop for prop in CANONICAL_PROPERTIES if prop in requested_native_set
    )

    uncertainty_method = dft_proxy_config["uncertainty_method"]
    coverage_target = dft_proxy_config.get("coverage_target", CALIBRATION_COVERAGE_TARGET)
    # YAML allowed_range [0.5, 0.99]。bool は int 派生なので isinstance check で除外し、
    # 有限性 + 範囲を fail-fast（範囲外の coverage は Ch11 CI 契約と非整合）
    if isinstance(coverage_target, bool) or not isinstance(coverage_target, (int, float)) \
            or not math.isfinite(coverage_target) or not (0.5 <= float(coverage_target) <= 0.99):
        raise ValueError(
            f"coverage_target must be a finite float in [0.5, 0.99] (got {coverage_target!r}); "
            "out-of-range or non-finite coverage breaks Ch11 CI/uncertainty semantics"
        )
    coverage_target = float(coverage_target)

    predict_config = dft_proxy_config["predict_config"]
    uncertainty_config = dft_proxy_config.get("uncertainty_config", {})
    calibration_tables: dict[str, CalibrationTable] = dft_proxy_config.get("calibration_tables", {})
    calibration_provenance: dict = dft_proxy_config.get("calibration_provenance", {})  # YAML pin と突合
    hull_reference: dict = dft_proxy_config.get("hull_reference", {})  # {phase_diagrams: {...}} + fingerprint
    # YAML pin：swap 検知のため runtime hull_reference と突合する source of truth
    # 実装上は proxy_reference_provenance.hull_reference（Ch8 と共有 snapshot）を渡す
    hull_reference_pin: dict = dft_proxy_config.get("proxy_reference_provenance", {}).get("hull_reference", {})
    elemental_reference_pin: dict = dft_proxy_config.get("elemental_reference_correction_provenance", {})

    # Uncertainty サンプル数の下限 enforcement（0/1 は NaN or fake certainty を生む → Ch11/Safety 契約違反）
    if uncertainty_method == "deep_ensemble":
        if method_id == "megnet_v0.4":
            # MEGNet: property-specific ensembles（_ensemble_handles_by_property[prop] = list[handle]）
            ens_by_prop = predict_config.get("_ensemble_handles_by_property", {})
            for prop in requested_native:
                n = len(ens_by_prop.get(prop, []))
                if n < DEEP_ENSEMBLE_MIN_MEMBERS:
                    raise ValueError(
                        f"megnet_v0.4 deep_ensemble requires "
                        f">= {DEEP_ENSEMBLE_MIN_MEMBERS} handles for property {prop!r} (got {n}); "
                        "fewer members produce zero or NaN std, violating Ch11 Pareto contract"
                    )
        else:
            n_members = len(predict_config.get("_ensemble_handles", []))
            if n_members < DEEP_ENSEMBLE_MIN_MEMBERS:
                raise ValueError(
                    f"deep_ensemble requires >= {DEEP_ENSEMBLE_MIN_MEMBERS} ensemble handles "
                    f"(got {n_members}); fewer members produce zero or NaN std, violating Ch11 Pareto contract"
                )
    elif uncertainty_method == "mc_dropout":
        n_samples = int(uncertainty_config.get("n_samples", MC_DROPOUT_SAMPLES))
        if n_samples < MC_DROPOUT_MIN_SAMPLES:
            raise ValueError(
                f"mc_dropout requires uncertainty_config.n_samples >= {MC_DROPOUT_MIN_SAMPLES} "
                f"(got {n_samples})"
            )
    elif uncertainty_method == "bayesian_last_layer":
        n_samples = int(uncertainty_config.get("n_samples", MC_DROPOUT_SAMPLES))
        if n_samples < BAYESIAN_LAST_LAYER_MIN_SAMPLES:
            raise ValueError(
                f"bayesian_last_layer requires uncertainty_config.n_samples >= "
                f"{BAYESIAN_LAST_LAYER_MIN_SAMPLES} (got {n_samples})"
            )
    else:
        raise ValueError(
            f"unknown uncertainty_method: {uncertainty_method}, expected {CANONICAL_UNCERTAINTY_METHODS}"
        )

    # universal potential 型 method で formation_energy_per_atom を要求するなら
    # elemental reference correction handle が必須（M3GNet/MACE の total energy は MP-compatible ではない）
    if method_id in METHODS_REQUIRING_ELEMENTAL_REFERENCE and \
            "formation_energy_per_atom" in requested_native:
        if predict_config.get("_elemental_reference_correction") is None:
            raise ValueError(
                f"method_id {method_id} requires predict_config['_elemental_reference_correction'] "
                "to produce MP-compatible formation_energy_per_atom "
                "(and thus energy_above_hull); provide callable or use megnet_v0.4"
            )
        runtime_prov = predict_config.get("_elemental_reference_provenance")
        if runtime_prov is None:
            raise ValueError(
                f"method_id {method_id} requires predict_config['_elemental_reference_provenance'] "
                "(dict describing reference source/version) for audit log; "
                "unaudited elemental reference is disallowed"
            )
        # YAML pin と runtime dict を突合（stale/swap 検知）：correction_scheme / source / reference_sha256
        for req in ("correction_scheme", "source", "reference_sha256"):
            v = elemental_reference_pin.get(req)
            if v is None or (isinstance(v, str) and not v):
                raise ValueError(
                    f"elemental_reference_correction_provenance is missing required field {req!r}; "
                    "provenance pin must be non-empty for stale/swap detection"
                )
        _rsha = elemental_reference_pin["reference_sha256"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", _rsha):
            raise ValueError(
                f"elemental_reference reference_sha256 must match 'sha256:<64-hex>' (got {_rsha!r})"
            )
        for field in ("correction_scheme", "source", "reference_sha256"):
            if runtime_prov.get(field) != elemental_reference_pin.get(field):
                raise ValueError(
                    f"elemental reference provenance field {field!r} mismatch "
                    f"(runtime={runtime_prov.get(field)!r}, "
                    f"pinned={elemental_reference_pin.get(field)!r}); refuse to apply "
                    "(stale/swap correction produces MP-incompatible formation_energy_per_atom)"
                )

    # energy_above_hull を要求する場合、hull_reference の swap 検知を強制する
    if "energy_above_hull" in tuple(requested_properties):
        # pin dict の必須 field が非空でなければ、runtime も欠落と偶然一致（None == None）して bypass できる。
        for req in ("source", "api_version", "hull_snapshot_sha256", "energy_basis",
                    "phase_diagrams_sha256_by_chemsys"):
            v = hull_reference_pin.get(req)
            if v is None or (isinstance(v, str) and not v) or (isinstance(v, dict) and not v):
                raise ValueError(
                    f"proxy_reference_provenance.hull_reference is missing required field {req!r}; "
                    "provenance pin must be non-empty for stale/swap detection"
                )
        # sha256 は 'sha256:' prefix + 64 hex 桁
        _sha = hull_reference_pin["hull_snapshot_sha256"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", _sha):
            raise ValueError(
                f"hull_snapshot_sha256 must match 'sha256:<64-hex>' (got {_sha!r})"
            )
        # canonical energy_basis：MEGNet の native formation_energy_per_atom と同一 basis のみ許可
        if hull_reference_pin["energy_basis"] != "formation_energy_per_atom_MP_compatible":
            raise ValueError(
                f"hull_reference.energy_basis must be 'formation_energy_per_atom_MP_compatible' "
                f"(got {hull_reference_pin['energy_basis']!r}); "
                "different basis (raw entry energy 等) breaks e_above_hull = fe - e_hull semantics"
            )
        # phase_diagrams payload の per-chemsys sha256 pin：全 hash が 'sha256:<64-hex>'
        _pd_pins: dict = hull_reference_pin["phase_diagrams_sha256_by_chemsys"]
        for _cs, _h in _pd_pins.items():
            if not (isinstance(_h, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", _h)):
                raise ValueError(
                    f"phase_diagrams_sha256_by_chemsys[{_cs!r}] must be 'sha256:<64-hex>' (got {_h!r})"
                )
        for field in ("source", "api_version", "hull_snapshot_sha256", "energy_basis"):
            if hull_reference.get(field) != hull_reference_pin.get(field):
                raise ValueError(
                    f"hull_reference field {field!r} mismatch "
                    f"(runtime={hull_reference.get(field)!r}, "
                    f"pinned={hull_reference_pin.get(field)!r}); refuse to apply "
                    "(runtime hull swap breaks Ch8/Ch9/Ch11 突合契約; "
                    "disallowed_operations.swap_hull_reference_snapshot_at_inference)"
                )
        # runtime phase_diagrams payload が pin と一致することを field-by-field で検証
        # （metadata が一致していても phase_diagrams payload だけ swap されるのを検知）
        rt_pd = hull_reference.get("phase_diagrams", {})
        if not isinstance(rt_pd, dict) or not rt_pd:
            raise ValueError(
                "hull_reference.phase_diagrams must be a non-empty dict "
                "(chemsys → PhaseDiagram); runtime payload missing"
            )
        # (1) chemsys key set は完全一致
        if set(rt_pd.keys()) != set(_pd_pins.keys()):
            raise ValueError(
                f"hull_reference.phase_diagrams chemsys key set mismatch "
                f"(runtime={sorted(rt_pd.keys())}, pinned={sorted(_pd_pins.keys())}); "
                "phase_diagrams payload swap detected"
            )
        # (2) 各 chemsys の serialize hash が pin と一致
        #     PhaseDiagram の entries を canonical に serialize → sha256 でハッシュ
        for _cs, _pd in rt_pd.items():
            _canonical = _serialize_phase_diagram_for_hash(_pd)   # utility: deterministic JSON of entries
            _rt_hash = "sha256:" + hashlib.sha256(_canonical.encode("utf-8")).hexdigest()
            if _rt_hash != _pd_pins[_cs]:
                raise ValueError(
                    f"phase_diagrams[{_cs!r}] payload hash mismatch "
                    f"(runtime={_rt_hash}, pinned={_pd_pins[_cs]}); "
                    "phase_diagrams payload swap detected "
                    "(disallowed_operations.swap_hull_reference_snapshot_at_inference)"
                )
        # (3) energy_basis == "formation_energy_per_atom_MP_compatible" の実質検証：
        #     PDEntry ベースの e_above_hull 計算は pd が formation-energy basis で構築されていることを要請する。
        #     元素ごとに **最も安定な単元素 entry（elemental reference）** の formation energy が ~0 で
        #     あることを確認する。metastable 単元素 allotrope は formation energy > 0 が正当に存在しうる
        #     ので、min を取って ground-state のみ検査する canonical。負値 (< -tol) は basis 不整合の
        #     signature（total-energy basis で構築された pd）として fail-fast。
        BASIS_TOL_EV_PER_ATOM = 1e-3
        for _cs, _pd in rt_pd.items():
            _min_by_el: dict[str, float] = {}
            for _e in _pd.all_entries:
                _sym_amts = _e.composition.get_el_amt_dict()
                if len(_sym_amts) == 1:
                    _sym = next(iter(_sym_amts))
                    _epa = float(_e.energy_per_atom)
                    if (_sym not in _min_by_el) or (_epa < _min_by_el[_sym]):
                        _min_by_el[_sym] = _epa
            for _sym, _min_epa in _min_by_el.items():
                # ground-state 単元素 entry の formation energy は 0 でなければならない。
                # 正の tol 超え → 別 basis（total-energy 等）で構築された pd の signature。
                # 負の tol 超え → 元素基準がさらに下にある = basis 不整合。
                if abs(_min_epa) > BASIS_TOL_EV_PER_ATOM:
                    raise ValueError(
                        f"hull_reference.phase_diagrams[{_cs!r}] ground-state elemental entry "
                        f"for {_sym} has energy_per_atom={_min_epa} eV/atom "
                        f"(|·| > {BASIS_TOL_EV_PER_ATOM}); pd must be built in formation-energy "
                        f"basis (ground-state elemental refs ≈ 0) for "
                        f"energy_basis == 'formation_energy_per_atom_MP_compatible'. "
                        "Rebuild pd from PDEntry(comp, formation_energy_per_atom * n_atoms) "
                        "or use MP-compatible corrected total energies with proper elemental refs."
                    )

    per_candidate_log: list[dict] = []
    scored: list[dict] = []
    for cand in candidates_after_fh:
        # P → U → C の 3 layer 順序 canonical（requested_native subset のみ計算）
        # single_shot は audit 用（ensemble mean vs single-shot drift 検知）。
        # deep_ensemble: MEGNet は property 別 ensemble の 0 番目 handle を束ねる、それ以外は _ensemble_handles[0]。
        if uncertainty_method == "deep_ensemble":
            if method_id == "megnet_v0.4":
                ens_by_prop = predict_config["_ensemble_handles_by_property"]
                single_handles = {prop: ens_by_prop[prop][0] for prop in requested_native}
                single_config = {**predict_config, "_model_handles_by_property": single_handles}
            else:
                first_handle = predict_config["_ensemble_handles"][0]
                single_config = {**predict_config, "_model_handle": first_handle}
        else:
            single_config = predict_config
        _single = _predict_layer(cand, method_id, requested_native, single_config)
        u_out = _uncertainty_layer(
            cand, method_id, requested_native, predict_config, uncertainty_method, uncertainty_config,
        )
        c_out = _calibration_layer(u_out, calibration_tables, coverage_target, calibration_provenance)

        # requested properties について canonical dict を組み立てる（native + derived）
        canonical_pred: dict[str, dict[str, float]] = {}
        for p in requested_properties:
            if p in c_out:
                canonical_pred[p] = c_out[p]
            elif p in DERIVED_PROPERTIES:
                # 現状 canonical derived は energy_above_hull のみ。formation_energy_per_atom + MP hull から derive
                if p == "energy_above_hull":
                    fe = c_out["formation_energy_per_atom"]
                    # PhaseDiagram + PDEntry で e_above_hull を直接得る（basis mismatch 構造的排除）
                    signed_delta_ev = _compute_e_above_hull_ev_per_atom(cand, fe["mean"], hull_reference)
                    signed_delta_mev = 1000.0 * signed_delta_ev
                    # 定義上 energy_above_hull は非負。負値は surrogate 誤差 or hull snapshot 不整合の
                    # signature であり、Ch11 Pareto で「hull より安定」と誤って高評価されるのを防ぐ。
                    # canonical value は clamp、生値は audit 用に別 key で保持。
                    e_above_hull_mev = max(0.0, signed_delta_mev)
                    canonical_pred[p] = {
                        "mean": e_above_hull_mev,
                        "std_raw": 1000.0 * fe["std_raw"],
                        "std_calibrated": 1000.0 * fe["std_calibrated"],
                        "coverage_target": fe["coverage_target"],
                        "calibration_status": fe["calibration_status"],
                        "calibration_scale_factor": fe["calibration_scale_factor"],
                        "derived_from": "formation_energy_per_atom",
                        "hull_reference_source": hull_reference.get("source"),
                        "hull_snapshot_sha256": hull_reference.get("hull_snapshot_sha256"),
                        "signed_hull_offset_mev_per_atom": signed_delta_mev,   # audit（< 0 は surrogate/hull 不整合の signature）
                        "clamped_negative_to_zero": signed_delta_mev < 0.0,
                    }

        cand["dft_proxy_predicted_properties"] = canonical_pred
        cand["dft_proxy_predicted_properties_extended"] = {}   # canonical set 外は本 Skill では produce しない
        # H ステージ ood_score は本 Skill が算出しないが、Ch11 に流す義務あり
        cand["upstream_ood_score"] = float(cand["ood_score"])  # 事前 fail-fast 済み、numeric 保証

        scored.append(cand)
        _neg_hull = bool(canonical_pred.get("energy_above_hull", {}).get("clamped_negative_to_zero", False))
        per_candidate_log.append({
            "candidate_id": cand["candidate_id"],
            "method_id": method_id,
            "uncertainty_method": uncertainty_method,
            "coverage_target": coverage_target,
            "requested_properties": list(requested_properties),
            "requested_native": list(requested_native),
            "predicted_properties_canonical": canonical_pred,
            "single_shot_predict": _single,                    # audit 用：single-shot 値と ensemble mean の drift 検知
            "signed_hull_offset_negative": _neg_hull,          # audit: surrogate/hull 不整合 signature
        })

    n_neg_hull = sum(1 for r in per_candidate_log if r.get("signed_hull_offset_negative"))
    return {
        "scored_candidates": scored,
        "dft_proxy_provenance": {
            "method_id": method_id,
            "uncertainty_method": uncertainty_method,
            "coverage_target": coverage_target,
            "requested_properties_canonical": list(requested_properties),
            "requested_native": list(requested_native),
            "hull_reference_source": hull_reference.get("source"),
            "n_scored": len(scored),
            "n_signed_hull_offset_negative": n_neg_hull,       # ≥ 1 で Ch14 §14.x で監視（surrogate 誤差 or snapshot drift）
            "per_candidate_log": per_candidate_log,
        },
    }


def _serialize_phase_diagram_for_hash(pd) -> str:
    """
    PhaseDiagram を deterministic JSON 文字列に serialize してハッシュ入力とする。
    pinned `phase_diagrams_sha256_by_chemsys[chemsys]` と runtime を突合するための canonical form。
    entries は (reduced_formula, energy_per_atom) を reduced_formula 昇順で並べたリストに正規化する。
    浮動小数はハッシュ安定性のため 12 桁固定丸め。
    """
    entries = []
    for e in pd.all_entries:
        red = e.composition.reduced_formula
        # energy_per_atom は pymatgen entry.energy_per_atom（correction 込み）
        entries.append((red, round(float(e.energy_per_atom), 12)))
    entries.sort(key=lambda x: (x[0], x[1]))
    return json.dumps(entries, separators=(",", ":"), ensure_ascii=False)


def _compute_e_above_hull_ev_per_atom(
    candidate: dict, formation_energy_per_atom: float, hull_reference: dict
) -> float:
    """
    候補の composition + predicted formation_energy_per_atom から e_above_hull [eV/atom] を
    直接返す。pymatgen `PhaseDiagram.get_decomp_and_e_above_hull()` を用いて basis mismatch を
    構造的に排除する canonical。

    契約：`hull_reference["phase_diagrams"][chemsys]` の PhaseDiagram は
    `formation_energy_per_atom_MP_compatible` basis で構築されていること
    （= 元素基準の entries の energy_per_atom が 0 に近い）。apply_dft_proxy 冒頭で
    energy_basis literal と PhaseDiagram の elemental entry 基準を検証済み。

    PDEntry(composition, formation_energy_per_atom * n_atoms) を temporary entry として渡し、
    PhaseDiagram の hull 上の分解経路に対する超過分を得る。逆設計で生じた novel composition でも
    convex hull interpolation で扱える（formula lookup 不可の場合と同じ機構）。

    fail-fast 条件：
      - phase_diagrams 欠落 / chemsys 未収録
      - get_decomp_and_e_above_hull() が非有限（surrogate / PD 破綻）
    """
    if "structure" in candidate and candidate["structure"] is not None:
        comp = candidate["structure"].composition
    elif "generated_x" in candidate:
        comp = Composition({el: float(v) for el, v in candidate["generated_x"].items()})
    else:
        raise ValueError(
            f"candidate {candidate.get('candidate_id')} lacks structure / generated_x "
            "(cannot derive Composition for hull lookup)"
        )
    chemsys = "-".join(sorted({str(el) for el in comp.elements}))

    if "phase_diagrams" not in hull_reference:
        raise ValueError(
            "hull_reference lacks phase_diagrams (dict[chemical_system → PhaseDiagram]) "
            "(derived property energy_above_hull cannot be computed; PhaseDiagram-based snapshot required)"
        )
    pd_map = hull_reference["phase_diagrams"]
    if chemsys not in pd_map:
        raise ValueError(
            f"hull_reference missing PhaseDiagram for chemical_system={chemsys} "
            f"(candidate {candidate.get('candidate_id')}); expand snapshot to cover this system "
            "or drop energy_above_hull from requested_properties"
        )
    phase_diagram: PhaseDiagram = pd_map[chemsys]
    # PDEntry は total energy [eV] を要求。formation-energy basis の pd 上では
    # entry energy = formation_energy_per_atom * n_atoms が canonical。
    n_atoms = int(comp.num_atoms)
    entry_energy_ev = float(formation_energy_per_atom) * n_atoms
    if not math.isfinite(entry_energy_ev):
        raise ValueError(
            f"predicted formation_energy_per_atom={formation_energy_per_atom} is non-finite for "
            f"candidate {candidate.get('candidate_id')}"
        )
    temp_entry = PDEntry(composition=comp, energy=entry_energy_ev)
    # get_decomp_and_e_above_hull returns (decomp, e_above_hull) with e_above_hull in [eV/atom]
    _decomp, e_above_hull = phase_diagram.get_decomp_and_e_above_hull(
        temp_entry, allow_negative=True
    )
    e_above_hull = float(e_above_hull)
    if not math.isfinite(e_above_hull):
        raise ValueError(
            f"PhaseDiagram returned non-finite e_above_hull for {chemsys} at composition "
            f"{comp.reduced_formula} (surrogate / PD破綻; fail-fast to avoid Ch11 poisoning)"
        )
    return e_above_hull   # 単位：eV/atom（signed；下流で meV/atom へ 1000×、非負 clamp は apply_dft_proxy 側）
```

---

## 9.5 vol-03 Foundation Model 特徴の合成 — 2 経路と使い分け

vol-03 Ch11 で登場した材料 Foundation Model（材料 FM）を DFT proxy に組み合わせる方法は 2 経路あります。本節では **Skill 契約としての pin 方法**を扱います。

### 9.5.1 Route α：FM 特徴 + MLP head

材料 FM の embedding を fixed feature として抽出し、そこから MLP で物性回帰します。**利点**：FM 重みを凍結するため学習コストが低い、canonical fingerprint（vol-03 §11.4）が確立している。**欠点**：物性依存の adaptation が弱い。

### 9.5.2 Route β：FM を fine-tune

材料 FM の GNN backbone を含めて fine-tune します。**利点**：ドメイン特化性能が高い。**欠点**：fine-tune 前後で fingerprint drift が起き、vol-03 canonical `foundation_model_fingerprint` と衝突する場合がある（vol-03 §11.4 で「fine-tune は fingerprint reset を伴う」と canonical に pin されている）。

### 9.5.3 Skill YAML での経路 pin

`arim.gen.dft_proxy.v0.1` の Skill YAML では、`foundation_model_integration` field で経路を明示します：

- `"none"`：MEGNet / M3GNet / MACE のみ（本章 default）
- `"route_alpha"`：MEGNet + FM feature concat（FM fingerprint を YAML provenance に pin）
- `"route_beta"`：MEGNet を FM で fine-tune（fine-tune 後の fingerprint を再登録）

Route α / β を選ぶ場合、**Ch3 `arim.gen.candidate_rescore.v0.1`（FM 再スコア Skill）と本 Skill の順序を pipeline manifest で pin**する義務があります（§9.6 YAML `pipeline_position.depends_on`）。

> [!WARNING]
> **Route β の canonical drift 注意**：fine-tune 後の FM fingerprint は vol-03 `candidate_rescore` のそれとは異なります。両 Skill を同一 pipeline で走らせる場合、**同じ FM を 2 度 fine-tune することはできず**、どちらかが凍結モデルとして参照する契約が必要です（§9.7 責務マトリクスで整理）。

---

## 9.6 `arim.gen.dft_proxy.v0.1` — 完全 Skill YAML

Ch4 §4.3 の 14 provenance fields + Safety 3-layer + `outputs_disallowed_natural_language` + `pipeline_position` の canonical を継承した完全 Skill YAML です。

```yaml
# =============================================================
# arim.gen.dft_proxy.v0.1 — DFT proxy 物性予測 Skill（Ch9 canonical、S1 併走）
# 継承: Ch4 §4.2 template ⑧、Ch8 §8.7 synthesizability_proxy と一貫（S1 併走 Skill 対称形）
# 責務: MEGNet/M3GNet/MACE による予測物性 + calibrated 不確かさ付与
#       reject はせず、S1 → S2 → T の下流に予測 + 不確かさ付き候補を流す
# =============================================================
skill:
  id: "arim.gen.dft_proxy.v0.1"
  version: "v0.1"
  description: "Ch9 canonical DFT proxy 3 手法（megnet_v0.4 / m3gnet_v0.2 / mace_mp0_v1）による予測物性 + calibrated 不確かさ付与。reject はせず、Ch11 ランキング入力を提供。"

pipeline_position:
  stage: "S1"
  upstream_stages: ["H"]                         # immediate: H → S1 canonical（Ch4 §4.9）
  downstream_stages: ["S2"]                      # immediate: S1 → S2 canonical
  runs_in_parallel_with:
    - "arim.gen.synthesizability_proxy.v0.1"     # Ch8 canonical S1 Skill
    - "arim.gen.candidate_rescore.v0.1"          # vol-03 FM 再スコア（optional）

input_schema:
  candidates_in:
    type: "list_of_dict"
    required_keys:
      - "candidate_id"
      - "structure"                              # DFT proxy は structure-conditioned のため実物 Structure 必須。
                                                 # composition-only 候補は Ch6 diffusion / explicit relaxer で refine してから通す。
      # F → H → S1 canonical 前提条件（Ch8 §8.7 と対称）
      - "physics_filter_run_id"
      - "physics_filter_passed"
      - "ood_detection_run_id"
      - "ood_detection_passed"
      - "ood_score"                              # Ch10 H stage confer（finite numeric 必須、Ch11 Pareto 第 3 軸）
      - "upstream_stage_trace"                   # list[str]、["F", "H"] を含むこと必須
    optional_keys:
      - "chemical_system"                        # audit hint（Composition から derive 可、任意）
      - "reduced_formula"                        # audit hint（Composition から derive 可、任意）
    condition: "required"
    validation_rule: |
      すべての candidate_in について:
        assert physics_filter_run_id, "F stage run_id missing (flag alone insufficient)"
        assert physics_filter_passed is True, "F stage did not pass"
        assert ood_detection_run_id, "H stage run_id missing (flag alone insufficient)"
        assert ood_detection_passed is True, "H stage did not pass"
        assert isinstance(ood_score, (int, float)) and not isinstance(ood_score, bool) \
            and math.isfinite(float(ood_score)), \
            "ood_score must be finite numeric (Ch11 Pareto third axis); bool/None/NaN/inf → fail-fast"
        assert set(upstream_stage_trace) >= {"F", "H"}, "F/H trace missing"
        assert structure is not None, \
            "DFT proxy requires real relaxed Structure; composition-only candidates must be refined first"
      違反時は fail-fast（silent proceed 禁止）
  hyperparameters:
    type: "dict"
    schema:
      method_id:
        type: "enum"
        values: ["megnet_v0.4", "m3gnet_v0.2", "mace_mp0_v1"]
      requested_properties:
        type: "list_of_enum"
        values: ["formation_energy_per_atom", "bandgap", "log10_bulk_modulus_gpa", "log10_shear_modulus_gpa", "energy_above_hull"]
        min_items: 1
        description: "canonical 5 properties の subset（**空 list 禁止**——Ch11 Pareto の performance 軸を欠くため）。native support または derivable な property のみ許可"
      uncertainty_method:
        type: "enum"
        values: ["deep_ensemble", "mc_dropout", "bayesian_last_layer"]
      uncertainty_config:
        type: "dict"
        description: |
          deep_ensemble:
            - method_id == "megnet_v0.4": predict_config._ensemble_handles_by_property[prop] の len が n_members
              （各 requested_native property について独立に property-specific ensemble）
            - method_id in {m3gnet_v0.2, mace_mp0_v1}: predict_config._ensemble_handles の len が n_members
            default 5, min 2
          mc_dropout: n_samples（default 30, min 2）
          bayesian_last_layer: n_samples（default 30, min 2）
          いずれも min 未満は fail-fast（n=0 → NaN, n=1 → std=0.0 fake certainty）
      coverage_target:
        type: "float"
        allowed_range: [0.5, 0.99]
        default: 0.9
      hull_reference:
        type: "dict"
        description: |
          energy_above_hull を requested_properties に含む場合必須。canonical shape（全フィールド required、空文字禁止）:
            source: "materials_project"                                    # str, non-empty
            api_version: "<materials_project_api_version>"                 # str, non-empty
            hull_snapshot_sha256: "sha256:<64 hex>"                        # str, regex "^sha256:[0-9a-f]{64}$"
            energy_basis: "formation_energy_per_atom_MP_compatible"        # literal (basis 互換性の宣言)
            phase_diagrams: {chemical_system → pymatgen.analysis.phase_diagram.PhaseDiagram}
          proxy_reference_provenance.hull_reference は上記に加え **phase_diagrams_sha256_by_chemsys**
          （dict: chemsys(str) → "sha256:<64 hex>"）を pin として必ず記載する。runtime `phase_diagrams`
          の chemsys key 集合完全一致 + 各 PhaseDiagram の canonical serialize hash と一致必須
          （metadata が一致していても payload swap があれば hard_reject）。
          Ch8 synthesizability_proxy.hull_distance と **同一 snapshot** を使うこと canonical
          （swap_hull_reference_snapshot_at_inference で hard_reject）。
          PhaseDiagram-based のため、逆設計で生じた novel reduced_formula も
          convex hull interpolation で扱える（formula-map lookup 不可）。
          runtime dict は proxy_reference_provenance.hull_reference と **field-by-field 一致必須**
          （bypass_via_pin_omission で hard_reject）。
        condition: "required if 'energy_above_hull' in requested_properties"
    validation_rule: |
      # native or derivable の互換性
      METHOD_PROPERTY_MAP = {
        "megnet_v0.4": {"formation_energy_per_atom", "bandgap", "log10_bulk_modulus_gpa", "log10_shear_modulus_gpa"},
        "m3gnet_v0.2": {"formation_energy_per_atom"},
        "mace_mp0_v1": {"formation_energy_per_atom"},
      }
      DERIVED_PROPERTIES = {"energy_above_hull": {"formation_energy_per_atom"}}
      for p in requested_properties:
          if p in METHOD_PROPERTY_MAP[method_id]:
              continue
          if p in DERIVED_PROPERTIES and DERIVED_PROPERTIES[p] <= METHOD_PROPERTY_MAP[method_id]:
              continue
          raise AssertionError(f"{method_id} does not support (native or derived) {p}")
      # deep_ensemble は n_members = 5 canonical、変更する場合 uncertainty_config.n_members を明示
      違反時は fail-fast
      # hull_reference validation（energy_above_hull を requested_properties に含む場合）:
      required_hull_fields = {
          "source": (str, "non-empty"),
          "api_version": (str, "non-empty"),
          "hull_snapshot_sha256": (str, r"^sha256:[0-9a-f]{64}$"),
          "energy_basis": (str, "formation_energy_per_atom_MP_compatible"),
          "phase_diagrams": (dict, "non-empty"),
      }
      for k, (t, constraint) in required_hull_fields.items():
          assert k in hull_reference and isinstance(hull_reference[k], t)
          # 空文字 / 空 dict / regex 不一致 / literal 不一致 で fail-fast
      # proxy_reference_provenance.hull_reference と field-by-field 一致確認
      # （bypass_via_pin_omission / swap_hull_reference_snapshot_at_inference で hard_reject）

output_schema:
  scored_candidates:
    type: "list_of_dict"
    required_keys:
      - "candidate_id"
      - "dft_proxy_predicted_properties"              # dict: property_id → {mean, std_raw, std_calibrated, calibration_status, ...}
      - "dft_proxy_predicted_properties_extended"     # canonical set 外の物性（本 Skill では常に空 dict、拡張用 slot）
      - "upstream_ood_score"                          # H ステージ pass-through（**numeric 必須**、input で validated 済み、Ch11 Pareto 第 3 軸）
    description: "全入力候補（reject なし）に予測物性 + calibrated 不確かさを付与"
  dft_proxy_provenance:
    type: "dict"
    required_keys:
      - "method_id"
      - "uncertainty_method"
      - "coverage_target"
      - "requested_properties_canonical"
      - "requested_native"
      - "hull_reference_source"                       # null 可（energy_above_hull を含まない場合）
      - "n_scored"
      - "n_signed_hull_offset_negative"               # audit: energy_above_hull で clamp が発火した候補数（surrogate/hull 不整合 signature）
      - "per_candidate_log"

# --- 生成器 field は null（本 Skill は screening/prediction 側）---
generative_model_family: null
latent_dim: null
training_data_provenance: null
condition_variables_declared: []
condition_extensibility:
  allow_agent_to_add_conditions: false
  additional_conditions_require: "human_approval"
generation_temperature: null
generation_temperature_range_allowed: null
guidance_scale: null
guidance_scale_range_allowed: null
sampling_config: null

# --- screening_model_family（Ch4 §4.3.7 canonical 3-tier）---
screening_model_family:
  family: "GNN_universal_potential"
  id: "as declared in screening_config.method_id"
  version: "v0.1.0"

screening_config:                                # Ch8 と対称、method_id は呼び出しごとに 1 つに固定
  method_id: "megnet_v0.4"                       # 呼び出しごとに上書き
  uncertainty_method: "deep_ensemble"
  coverage_target: 0.9
  # --- predict_config 要求 shape（uncertainty_method + method_id で分岐、fail-fast 対象）---
  # deep_ensemble:
  #   - method_id == megnet_v0.4:
  #       predict_config._ensemble_handles_by_property: dict[property_id, list[MEGNetModel]]
  #       各 property について len == n_members の property-specific ensemble
  #       （_single 用に各 property の [0] 番目 handle を借りるため _model_handles_by_property は不要）
  #   - method_id in {m3gnet_v0.2, mace_mp0_v1}:
  #       predict_config._ensemble_handles: list[Model]（len == DEEP_ENSEMBLE_MEMBERS）
  #       （_single 用に _ensemble_handles[0] を借りるため _model_handle は不要）
  # mc_dropout / bayesian_last_layer:
  #   - method_id == megnet_v0.4:
  #       predict_config._model_handles_by_property: dict[property_id, MEGNetModel]（dropout on mode / posterior sampler 付き）
  #   - method_id in {m3gnet_v0.2, mace_mp0_v1}:
  #       predict_config._model_handle: Model
  # universal potential (m3gnet/mace) + formation_energy_per_atom:
  #     predict_config._elemental_reference_correction: Callable[[Structure, float], float]
  #     predict_config._elemental_reference_provenance: dict（source / scheme / sha256）

# --- proxy_indicators（Ch9 canonical: 予測物性の意味論）---
proxy_indicators:
  method_id: "as declared in screening_config"
  raw_output_range: "method-dependent (eV/atom, eV, GPa, meV/atom)"
  raw_output_unit: "as per property_id"
  normalized_score_range: "N/A (本 Skill は raw 予測値と calibrated std を返し、正規化は Ch11 の Pareto 側で扱う)"
  normalization_formula: null
  high_score_semantics: "N/A (物性ごとに desired direction が異なるため、Ch11 側で objective 定義)"

# --- proxy 参照データ provenance（Ch9 canonical: pre-trained weight source と hull ref）---
proxy_reference_provenance:
  megnet_v0.4:
    source: "materialsvirtuallab/megnet"
    weight_release: "v0.4"
    fetched_at: "2026-06-15T00:00:00Z"
    weight_sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  m3gnet_v0.2:
    source: "materialsvirtuallab/m3gnet"
    weight_release: "v0.2"
    fetched_at: "2026-06-15T00:00:00Z"
    weight_sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  mace_mp0_v1:
    source: "ACEsuit/mace-mp"
    weight_release: "mp0_v1"
    fetched_at: "2026-06-15T00:00:00Z"
    weight_sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  hull_reference:                                # Ch8 §8.7 synthesizability_proxy.proxy_reference_provenance.hull_distance と同一 snapshot 必須
    source: "materials_project"
    api_version: "2024.9"
    fetched_at: "2026-06-15T00:00:00Z"
    hull_snapshot_sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    energy_basis: "formation_energy_per_atom_MP_compatible"   # 予測 formation_energy_per_atom と同一 basis 必須（MP-compatible elemental reference で構築された PhaseDiagram の formation energy [eV/atom]）
    phase_diagrams_sha256_by_chemsys:            # 各 chemsys の PhaseDiagram payload hash（_serialize_phase_diagram_for_hash の canonical serialize に対する sha256）
      "Fe-O": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
      "Li-Fe-O": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
      # ... runtime `hull_reference.phase_diagrams` の全 chemsys を列挙必須（key 集合完全一致 canonical）
    same_as_synthesizability_proxy_hull: true    # Ch11 が両 Skill 出力を突合する canonical 契約

# --- Ch9 独自：DFT proxy 契約（3 fields）---
dft_proxy_role: "surrogate_not_measurement"      # literal canonical、変更禁止
foundation_model_integration:
  mode: "none"                                   # none | route_alpha | route_beta（§9.5 canonical 3 択）
  foundation_model_fingerprint: null             # Route α/β 使用時のみ pin（vol-03 §11.4 と共有）
  fine_tune_reset_flag: false                    # Route β 時のみ true（fingerprint drift 記録）
calibration_provenance:
  coverage_target: 0.9
  per_property_scale_factors:                    # nested per-property records（stale/swap 検知契約）
    formation_energy_per_atom:
      scale_factor: 1.15
      validation_set_fingerprint: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
      n_validation_samples: 5000
    bandgap:
      scale_factor: 1.30
      validation_set_fingerprint: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
      n_validation_samples: 5000
    log10_bulk_modulus_gpa:
      scale_factor: 1.20
      validation_set_fingerprint: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
      n_validation_samples: 5000
    log10_shear_modulus_gpa:
      scale_factor: 1.20
      validation_set_fingerprint: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
      n_validation_samples: 5000

# --- Ch9 独自: universal potential 用 elemental reference correction ---
# M3GNet/MACE は total potential energy を返すため、MP-compatible formation_energy_per_atom
# への変換に elemental reference correction が必須（§9.4.2 note 参照）。
# MEGNet は元から formation energy を native 予測するので本 block は N/A。
elemental_reference_correction_provenance:
  applies_to_methods: ["m3gnet_v0.2", "mace_mp0_v1"]
  correction_scheme: "MP_compatible_elemental_reference_v2020_09"
  source: "materials_project_elemental_reference_energies"
  fetched_at: "2026-06-15T00:00:00Z"
  reference_sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  note: |
    universal potential の total energy を MP hull と比較可能な formation_energy_per_atom
    に変換するための elemental reference。predict_config.["_elemental_reference_correction"]
    として callable、["_elemental_reference_provenance"] として本 provenance dict を渡す。
    未提供時は fail-fast（YAML input validation と apply_dft_proxy コード両方で）。

foundation_model_reference: null                 # 本 Skill は独立で完結、vol-03 candidate_rescore と 1:1 なし

# --- filter_rules_applied は本 Skill 内 3 レイヤ順序 canonical ---
filter_rules_applied:
  - "P_layer_predict"
  - "U_layer_uncertainty"
  - "C_layer_calibration"

# --- Ch11 委譲 ---
top_k_returned: null

# --- Ch2 §2.5 canonical: 合成 GO/NO-GO は Human 判断 ---
synthesis_decision_owner: "human"                # canonical 固定

# --- Ch4 §4.4 canonical ---
inverse_design_authorization: "autonomous"       # S1 ステージは自律実行
synthesis_launch_authorization: null

hallucinatory_composition_detection_delegated_to:
  skill: "arim.gen.ood_detection.v0.1"
  skill_version: "v0.1"
  invocation_stage: "H"

safety_screening_passed: false
dual_use_review_completed: false
handoff_to:
  layer_1:
    stage: "SF"
    skill: "arim.gen.safety_screening.v0.1"
    required_field: "safety_screening_passed"
    relation: "upstream"
  layer_2:
    stage: "D"
    reviewer_role: "organizational_dual_use_committee"
    required_field: "dual_use_review_id"
  layer_3:
    stage: "P"
    reviewer_role: "facility_policy_officer"
    required_field: "facility_policy_review_id"

outputs_disallowed_natural_language:
  disallowed_phrases:                            # Ch4 §4.7 canonical 継承 + 本 Skill 独自
    - "この物性は良好です"
    - "合成後の性能を保証します"
    - "DFT 計算相当の精度です"
    - "予測誤差は無視できます"
    - "top 候補として推奨します"
    - "この材料は目標性能を満たすと予測されます"
  allowed_phrases:                               # canonical field 名の直接引用のみ許可
    - "formation_energy_per_atom (mean): X eV/atom"
    - "std_calibrated at coverage_target 0.9: X eV/atom"
    - "calibration_status: calibrated"
    - "upstream_ood_score: X"
    - "dft_proxy_role: surrogate_not_measurement"

# --- Ch9 独自：DFT proxy 操作逸脱 ---
disallowed_operations:
  - name: "invoke_without_verified_f_h_pass_provenance"
    description: "F/H stage pass 未検証の候補で本 Skill 呼び出し禁止（input_schema.validation_rule 参照）。physics_filter_run_id / physics_filter_passed / ood_detection_run_id / ood_detection_passed / upstream_stage_trace ⊇ {F,H} のいずれかが欠落したら fail-fast"
    enforcement: "hard_reject_and_audit_log"
  - name: "reject_candidate_at_dft_proxy"
    description: "本 Skill は soft score のみ。予測物性・不確かさに閾値をかけて reject 禁止（Ch11 T ステージが Pareto 選抜で担当）"
    enforcement: "hard_reject"
  - name: "return_dft_measurement_claim"
    description: "dft_proxy_role: surrogate_not_measurement canonical に反し、予測値を DFT 計算値または実測値として表現することを禁止（Ch4 §4.7 と対称）"
    enforcement: "hard_reject"
  - name: "swap_calibration_table_at_inference"
    description: "訓練時に pin した calibration_tables（per_property_scale_factors）の推論時変更禁止"
    enforcement: "hard_reject"
  - name: "expand_requested_properties_beyond_method_support"
    description: "METHOD_PROPERTY_MAP + DERIVED_PROPERTIES に反する requested_properties の推論時追加禁止（agent が Ch11 top-k を有利にするため properties を勝手に拡張する失敗モード）"
    enforcement: "hard_reject_and_audit_log"
  - name: "modify_foundation_model_route_at_inference"
    description: "foundation_model_integration.mode の推論時変更禁止。Route α → β の切替は fingerprint drift を伴うため訓練 pipeline から re-fit 必須"
    enforcement: "hard_reject"
  - name: "swap_hull_reference_snapshot_at_inference"
    description: "hull_reference snapshot（proxy_reference_provenance.hull_reference）を Ch8 synthesizability_proxy と異なる snapshot に切替禁止。両 Skill は同一 snapshot でなければ Ch11 Pareto 突合が不整合"
    enforcement: "hard_reject_and_audit_log"
  - name: "invoke_universal_potential_without_elemental_reference_correction"
    description: "m3gnet_v0.2 / mace_mp0_v1 で formation_energy_per_atom を requested_properties に含める場合、predict_config._elemental_reference_correction と ._elemental_reference_provenance が必須。未提供時は fail-fast（total energy を per-atom で割っただけでは MP-compatible formation energy とならず、energy_above_hull derive も意味論的に破綻するため）"
    enforcement: "hard_reject_and_audit_log"
  - name: "invoke_dft_proxy_on_composition_only_candidate"
    description: "structure が無い composition-only 候補（Ch5/Ch7 VAE/Flow/AR 直後の generated_x のみ）に DFT proxy を適用することを全 method / 全 property で禁止。DFT proxy 予測物性はすべて structure-conditioned（MEGNet formation energy も原子座標・格子に依存）で、fictitious cell 上の予測は Ch11 Pareto を汚染する。composition-only 候補は Ch8 synthesizability_proxy のみ通す or Ch6 diffusion refinement / explicit relaxer で real relaxed Structure に refine してから通す"
    enforcement: "hard_reject_and_audit_log"
  - name: "apply_stale_or_swapped_calibration_table"
    description: "calibration_tables に渡した CalibrationTable の property_id / scale_factor / validation_set_fingerprint / n_validation_samples が YAML pin calibration_provenance.per_property_scale_factors と一致しない場合、fail-fast。stale table や swap による false trust interval を防ぐ"
    enforcement: "hard_reject_and_audit_log"
  - name: "propagate_non_finite_prediction_or_calibration"
    description: "predict_layer / uncertainty_layer / calibration_layer が NaN・inf・負の std / 非正の scale_factor を返した場合、fail-fast で下流に流さない（Ch11 Pareto を汚染しないため）"
    enforcement: "hard_reject_and_audit_log"
  - name: "return_natural_language_property_recommendation"
    description: "outputs_disallowed_natural_language 継承。自然言語での「性能保証」型 output 禁止"
    enforcement: "hard_reject"

provenance:
  input_sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  skill_version: "v0.1"
  run_datetime_utc: "2026-07-08T00:00:00Z"
  package_versions:
    pymatgen: "2024.10.3"
    megnet: "1.3.2"
    m3gnet: "0.2.4"
    mace-torch: "0.3.6"
    numpy: "2.0.2"
  random_seed: 20260708
  event_hash: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
```

---

## 9.7 Ch8 synthesizability_proxy と Ch9 dft_proxy の責務マトリクス

S1 ステージで並走する 2 Skill の責務境界を Ch11 T ステージ入力視点で pin します。

| 論点 | Ch8 `synthesizability_proxy` | Ch9 `dft_proxy` | Ch10 `ood_detection`（参考）|
|---|---|---|---|
| **pipeline stage** | S1（併走） | S1（併走） | H（両 S1 の upstream） |
| **返す量** | 合成前例 proxy score（[0, 1] float） | 予測物性 dict + calibrated 不確かさ | ood_score + physical_violation |
| **reject 契約** | reject 禁止（Ch11 委譲） | reject 禁止（Ch11 委譲） | H 単体で hard reject（violation ≥ threshold） |
| **入力 provenance 要件** | F/H pass 済み provenance 必須 | F/H pass 済み provenance 必須 | F pass 済み provenance 必須 |
| **主要外部依存** | RDKit, pymatgen, Materials Project hull, ICSD | pymatgen, MEGNet/M3GNet/MACE weights | 学習分布統計 + F ステージ violation 数 |
| **Ch11 での使途** | Pareto 第 1 軸（合成軸） | Pareto 第 2 軸（性能軸） | Pareto 第 3 軸（信頼軸） |
| **Foundation Model 統合** | 非対応（proxy は heuristic 主体） | Route α / β を YAML pin（§9.5） | 非対応（統計軸のみ） |
| **予測不確かさ** | 単一 score（不確かさなし） | mean + calibrated std 必須 | ood_score（uncertainty proxy） |
| **canonical 予測物性数** | N/A | 5 物性（§9.3.3 pin） | 3 スコア（Ch10 で pin） |
| **同時使用の必然性** | 必須（合成軸なしでは top-k が物性のみ支配） | 必須（性能軸なしでは top-k が合成のみ支配） | 必須（信頼軸なしで外挿域が top-k に入る） |

> [!IMPORTANT]
> **3 軸の独立性が Ch11 Pareto 非劣ソートの前提**：Ch8 score・Ch9 予測・Ch10 ood_score は **同じ upstream provenance（F/H run_id）を共有**するが、判定基準は互いに独立です。1 つでも欠落するとエージェントが「利用可能軸だけで top-k を組む」失敗（Ch14 予定）を招きます。Ch11 は 3 軸すべてを required 入力として declare する契約を持ちます。

---

## 9.8 失敗モードと Skill 契約による回避

### 9.8.1 6 大失敗モード

| # | 失敗モード | 発生原因 | 回避手段 |
|---|---|---|---|
| 1 | 予測値のみ返し不確かさ欠落 | uncertainty_method 未指定 or 実装 skip | YAML `output_schema.scored_candidates` に `dft_proxy_predicted_properties` の 3 field（mean/std_raw/std_calibrated）必須化（§9.6） |
| 2 | calibration 前の raw std を confidence として提示 | calibration_layer skip または `calibration_status: uncalibrated` を無視 | Ch11 が `calibration_status` を required 入力で受け取り、`uncalibrated` は penalty 付与（§9.7 責務マトリクス） |
| 3 | requested_properties を推論時に拡張 | agent が top-k 有利化目的で properties を追加 | `disallowed_operations.expand_requested_properties_beyond_method_support`（§9.6） |
| 4 | Route β fine-tune 後の fingerprint drift 未記録 | `fine_tune_reset_flag` を false のまま fine-tune 実施 | vol-03 §11.4 canonical `foundation_model_fingerprint` reset 契約と YAML `fine_tune_reset_flag` の同時 pin（§9.5） |
| 5 | DFT proxy の予測を "DFT 計算値" と表現 | 自然言語出力で `dft_proxy_role: surrogate_not_measurement` を無視 | `outputs_disallowed_natural_language` + `disallowed_operations.return_dft_measurement_claim`（§9.6） |
| 6 | Ch8 との責務混同：Ch9 で "合成可能性" を判定 | Ch8 skip して Ch9 predicted_properties から合成可能性を推定 | §9.7 責務マトリクスで軸独立性を declare、Ch11 が 3 軸 required 入力で受け取る契約 |

### 9.8.2 コード / YAML / 責務境界の 3 層対応

各失敗モードは **コード（fail-fast）・YAML（宣言）・責務境界（Ch11 契約）** の 3 層で守ります。1 層でも欠けると agentic 実装で bypass されるため、§9.6 YAML では `disallowed_operations` に 7 entries、`outputs_disallowed_natural_language` に 6 phrases、code の `apply_dft_proxy` に 3 種の fail-fast（F/H provenance / method_id / requested_properties compatibility）を pin します。

---

## 9.9 章末チェックリスト

以下 8 項目で自己検証してください。**7 項目以上「はい」で第 10 章へ進めます**。

- [ ] **§9.1 の 3 動機**（DFT 直接計算は agentic loop に載らない / Ch8 は物性推定を持たない / 不確かさなし proxy は agentic 誤用を招く）を、Ch11 Pareto 3 軸の必然性と関連付けて説明できる
- [ ] **§9.2 の 3 レイヤ**（Predict / Uncertainty / Calibration）の順序と、いずれも reject しない契約の理由を説明できる
- [ ] **§9.3.3 の canonical 5 物性**（formation_energy_per_atom / bandgap / log10_bulk_modulus_gpa / log10_shear_modulus_gpa / energy_above_hull）と、METHOD_PROPERTY_MAP による method_id 別 supported set の切り分けを書ける
- [ ] **§9.4 の `apply_dft_proxy` コード**で、F/H upstream provenance の fail-fast enforcement と method_id / requested_properties の互換性検証を実装できる
- [ ] **§9.4.4 の calibration** の意味（`std_raw × scale_factor = std_calibrated`、`coverage_target` に合わせる）と、`uncalibrated` fallback の Ch11 側の扱いを説明できる
- [ ] **§9.5 の Foundation Model 2 経路**（Route α / β）を Skill YAML `foundation_model_integration` で pin でき、Route β 時の fingerprint reset 契約を説明できる
- [ ] **§9.6 の Skill YAML 全項目**（pipeline_position / provenance 14 fields + Ch9 独自 3 fields / input/output schema / outputs_disallowed_natural_language / disallowed_operations）を書ける
- [ ] **§9.7 責務マトリクス**で、Ch8 / Ch9 / Ch10 の 3 Skill が Ch11 Pareto 3 軸に対応する独立性を説明できる

> [!TIP]
> 「8 項目のうち 4 つ以下」なら、§9.2（3 レイヤ契約）と §9.7（責務マトリクス）を再読してください。この 2 節が Ch10 H ステージと Ch11 T ステージ実装の前提条件です。

---

## 9.10 参考資料

- **本書内**
  - Ch2 §2.7 canonical pipeline（G → SF → F → H → S1 → S2 → T → D → P → A）
  - Ch4 §4.3 canonical 14 provenance fields、§4.6 Safety 3-layer、§4.7 outputs_disallowed_natural_language、§4.9 pipeline stage anchor
  - Ch8 §8.6 `arim.gen.physics_filter.v0.1` YAML、§8.7 `arim.gen.synthesizability_proxy.v0.1` YAML、§8.8 責務マトリクス
  - Ch10 予定：`arim.gen.ood_detection.v0.1` の完全計算（本章 `upstream_ood_score` の produce 元）
  - Ch11 予定：`arim.gen.candidate_ranking.v0.1`（本章 S1 出力 + Ch8 S1 出力 + Ch10 H 出力を Pareto 3 軸で選抜）
  - Ch14 予定：Agentic 失敗パターン全体 taxonomy（§9.8 の 6 モードは Ch14 §14.2 の一部）
  - 付録 A：DFT proxy Skill テンプレート（本章 §9.6 YAML の骨格）
- **vol-01〜05**
  - vol-03 Ch8-9：予測不確かさ canonical（Deep Ensemble / MC Dropout / Bayesian last-layer）
  - vol-03 Ch11：材料 Foundation Model canonical（`foundation_model_fingerprint`、Route α / β 概念）
  - vol-03 §11.4：FM fingerprint と fine-tune reset 契約
  - vol-05 §5.5：Pareto 非劣ソート canonical（Ch11 で本章 3 軸を統合）
- **外部（教育目的の参照、本書での literal 検証 scope 外）**
  - Materials Project pre-computed hull energies
  - MEGNet / M3GNet / MACE-MP-0 の pretrained weights と学習データ分布
  - MPtrj 学習セット構成
