# 第14章 総合ハンズオン（Advanced Capstone）— Foundation Model → 深層特徴 → PyMC 階層モデル

> [!NOTE]
> **本章の到達目標**
> - **ARIM 風合成階層データ**（装置間・ロット間・研究室間）に対して、**Foundation Model の深層特徴** → **PyMC 階層モデル**の統合 Skill を組み立てられる
> - **エージェントが階層構造を認識して fine-tune 戦略を切り替える**判断を実装できる（装置差が大きい層では fine-tune、小さい層は frozen feature）
> - **「深層特徴の不確かさ」と「階層のプーリング」の共存**を PyMC で表現できる
> - **3 つの Human-in-the-loop 承認ゲート**（fine-tune 起動・不確かさ閾値超え・階層プーリング構造変更）を Skill 契約に統合できる
> - vol-02 第14章 capstone（PyMC 階層）と本書 Ch4-12 の全 Skill を **1 本の統合契約**にまとめられる
>
> **本章で扱わないこと**
> - **深層モデルの新規学習**（本章は Ch11 FM + Ch7 fine-tune + Ch12 SSL の成果を統合するのみ）
> - **PyMC の詳細**（vol-02 第11-13章 + Ch11 階層モデルを参照）
> - **失敗パターンの網羅** → **第15章**（本章は成功シナリオを組み立てる）
> - **組織展開** → **第16章**

---

## 14.1 この章で作る統合 Skill

**1 つの capstone Skill（`hierarchical_deep_bayes`）**と、その内部で orchestrate される **4 つのサブ Skill**を作ります。

| Skill / 成果物 | 役割 | 依存する章 |
|---|---|---|
| **`capstone_hierarchy_detection`** | データから装置 / ロット / 研究室の階層構造を検出し、層ごとに fine-tune 戦略を決定 | Ch4, Ch7 |
| **`capstone_deep_feature_extraction`** | FM で深層特徴を抽出（fine-tune or frozen を層ごとに切替）+ 不確かさ伝搬 | Ch7, Ch8, Ch9, Ch11, Ch12 |
| **`capstone_hierarchical_bayes`** | PyMC 階層モデルで深層特徴 → 材料物性を推定、posterior で不確かさを表現 | vol-02 Ch10-12, Ch11 |
| **`capstone_three_gate_orchestrator`** | 3 つの Human-in-the-loop 承認ゲートを統合管理（fine-tune 起動 / 不確かさ閾値超え / 階層構造変更） | Ch4 §5.7, Ch10, Ch11 |
| **`hierarchical_deep_bayes`**（capstone） | 上記 4 つを 1 本の契約に統合。ARIM 風合成階層データを入力、posterior + 監査レポートを出力 | 全章 |

**継承 DNA**（Ch4-12 の provenance 拡張ブロックを全て継承）：
- Layer 1-3 (Ch4)
- `layer_4_pretrained_weights` (Ch7)
- `bayesian_inference_config` (Ch9)
- `layer_attribution` + `layer_human_review` (Ch10)
- `foundation_model_provenance` + `fm_query_provenance` (Ch11)
- `ssl_pretrain_provenance` + `ssl_representation_eval_provenance` (Ch12)
- **本章で新設**: `capstone_integrated_provenance`

---

## 14.2 なぜ統合が難しいか — 深層 × 階層 × 不確かさ の三重奏

以下の 3 つは、それぞれ独立には確立された技術ですが、**統合したときに整合させるのが難しい**：

1. **深層 FM は "1 サンプル = 1 ベクトル"** を返す。階層構造（装置 / ロット / 研究室）の情報は特徴に埋まっていない
2. **PyMC 階層モデルは "低次元共変量 + 階層ラベル"** で書かれる想定。数百次元の深層特徴を直接投入すると事前分布設計が破綻
3. **不確かさは 2 種類ある**：深層側の不確かさ（Ch8-9）と、Bayesian posterior の不確かさ（vol-02 Ch10-12）。**両者を同じスケールで語れない**

```mermaid
flowchart LR
    A["ARIM 風合成階層データ<br/>(装置 x ロット x 研究室)"] --> B["capstone_hierarchy_detection<br/>(Ch4 + Ch7)"]
    B --> C{"層ごとの<br/>domain_gap 判定"}
    C -->|"gap 大"| D["fine-tune with Ch7<br/>(training_job_approval)"]
    C -->|"gap 小"| E["frozen feature"]
    D --> F["capstone_deep_feature_extraction<br/>(Ch11 FM + Ch8/9 不確かさ)"]
    E --> F
    F --> G["深層特徴 + per-sample uncertainty"]
    G --> H["capstone_hierarchical_bayes<br/>(PyMC 階層モデル)"]
    H --> I["posterior + trace diagnostics"]
    I --> J["capstone_three_gate_orchestrator<br/>(3 gates: fine-tune / uncertainty / pooling)"]
    J -->|"gate 全通過"| K["最終 report + Ch10 deep_report_template"]
    J -->|"gate NG"| L["Human 判断"]
```

> [!IMPORTANT]
> **本章の主題は "3 つを繋げる契約"** です。個別技術の詳細は各章に譲り、**接続面の整合性**（特徴の次元、不確かさのスケール、階層ラベルの引き渡し、監査ログの統合）に紙面を割きます。

---

## 14.3 ARIM 風合成階層データの構造

vol-02 第14章の合成階層データを **画像 × 階層**に拡張します：

| 階層 | 例 | 個数 |
|---|---|---|
| 研究室（lab） | LabA / LabB / LabC | 3 |
| 装置（instrument）※ lab に nested | LabA-SEM1, LabA-SEM2, LabB-SEM1, ... | 各 lab に 2〜3 台 |
| ロット（lot）※ instrument に nested | 撮影セッション | 各装置に 5〜10 lot |
| 画像（sample）※ lot に nested | 個別 SEM 画像 | 各 lot に 50〜200 枚 |

**合成の設計原則**：

- 装置ごとに **明るさ / コントラスト / ノイズ特性**が異なる（実装置固有性を模擬）
- ロットごとに **サンプル準備の微差**による drift を持たせる
- **材料物性（target）は lab 依存 + 装置依存の階層構造**を持つ
- 一部の装置は **他装置と domain_gap が大きい**（fine-tune 判断が分かれる）

### 合成データ生成契約（要旨）

```python
# 生成関数の骨子（実装は付録 A / vol-02 データ生成の拡張）
def generate_arim_hierarchy_dataset(
    n_labs: int = 3,
    n_instruments_per_lab: tuple = (2, 3, 2),
    n_lots_per_instrument: int = 8,
    n_samples_per_lot: int = 100,
    material_property_hierarchy: dict = None,  # 階層構造の真値
    instrument_domain_shift: dict = None,      # 装置ごとの明るさ/ノイズ
    seed: int = 42,
) -> "ARIMHierarchyDataset":
    ...
```

> [!NOTE]
> 合成データは **本章の学習を隔離する装置**です。実 ARIM データは装置固有性が強すぎ、階層構造の "正解" が観測不能なため、本章の練習には合成データを使い、実データは読者の現場に持ち込む前提です。

---

## 14.4 `capstone_hierarchy_detection` — 階層検出と fine-tune 戦略

エージェントが **層ごとに fine-tune するか frozen で行くか**を判断する Skill。

### アルゴリズム

各階層（lab / instrument / lot）について：

1. サンプルを層ラベルで分割
2. 層内 vs 層間で **Ch7 `domain_gap_gate`** を実行
3. 層間 domain_gap が大きい階層 → **fine-tune 候補**
4. 層間 domain_gap が小さい階層 → **frozen feature で共有**

### 契約 YAML

```yaml
# capstone_hierarchy_detection.yaml
skill: "capstone_hierarchy_detection"
version: "1.0.0"

requires:
  hierarchy_labels_provided: true                   # lab / instrument / lot / sample の完全ラベル
  ch7_domain_gap_gate_available: true               # 層ごとに呼び出すため
  minimum_samples_per_group: 30                     # 層内比較の統計的最小

decision_logic:
  for_each_hierarchy_level:
    - level: "lab"
    - level: "instrument"
    - level: "lot"
  per_level_action:
    domain_gap_low_all_pairs: "frozen_feature_shared_across_this_level"
    domain_gap_high_any_pair: "fine_tune_candidate_this_level"
    domain_gap_mixed: "defer_to_human_this_level"
  overall_strategy:
    fine_tune_hierarchy_levels: "list (subset of lab/instrument/lot)"
    frozen_hierarchy_levels: "complement"

pairwise_scaling_policy:                              # n^2 対策（多装置 / 多 lot 時）
  max_pairwise_comparisons_per_level: 300             # 例：25 装置なら 300 で打ち切り
  when_exceeding_max:
    strategy: "reference_group_plus_clustering"       # (a) reference (最大 n の group) との 1-vs-all
    clustering_method: "kmeans_on_group_centroids"    #     (b) 残りを cluster 化して代表と比較
    cluster_k_max: 20
  fdr_correction: "benjamini_hochberg"                # 多重比較補正
  skipped_pairs_recorded_in_provenance: true

canonical_top_level_flags:                            # 下流 (orchestrator) が参照する派生フラグ
  fine_tune_recommended: "bool == (len(overall_strategy.fine_tune_hierarchy_levels) > 0)"
  fine_tune_deferred_to_human: "bool == any(level.recommendation == 'defer_to_human')"
  derivation_deterministic: true

acceptance:
  ch7_score_computed_for_all_evaluated_level_pairs: true   # scaling で pruned した pair は skipped_pairs に記録
  fdr_correction_applied_if_multiple_pairs: true
  canonical_flags_derived_and_recorded: true
  provenance_ref_recorded_per_level: true

agent_authorization:
  L1: "read_hierarchy_detection_report"
  L2: "propose_strategy_but_not_execute"
  L3:
    can_recommend_fine_tune_scope: true
    cannot_launch_fine_tune_without_gate1: "forbidden_all_levels"
    cannot_merge_hierarchy_levels_silently: "forbidden_all_levels"
  never_allowed:
    - "override_ch7_domain_gap_result"
    - "collapse_lab_and_instrument_levels"
    - "invent_hierarchy_labels"

provenance:
  capstone_hierarchy_detection_provenance:
    hierarchy_levels_detected: "list"
    per_level_domain_gap:                            # 各層で Ch7 結果を保存
      - level: "str"
        pairwise_gaps: "dict of {(a,b): score}"
        pairwise_ch7_actions: "dict of {(a,b): action}"
        recommendation: "frozen | fine_tune_candidate | defer_to_human"
        ch7_provenance_refs: "list"
    overall_strategy:
      fine_tune_levels: "list"
      frozen_levels: "list"
    strategy_timestamp: "iso8601"
```

> [!WARNING]
> **`lab` レベルで fine-tune 候補と判定されたら、実装レベルでは `instrument` 別 fine-tune を強く推奨**します。研究室間で装置校正が別なら、混ぜて fine-tune すると装置差が押し潰されて posterior に不適切な影響が出ます。

---

## 14.5 `capstone_deep_feature_extraction` — FM 特徴 + 不確かさ伝搬

Ch11 の FM を使って深層特徴を抽出し、**per-sample uncertainty** を並列に出す Skill。

### 実装骨子

```python
# capstone_deep_feature_extraction.py
import torch
import numpy as np


def capstone_deep_feature_extraction(
    fm_model,                                       # Ch11 fm_fetch_and_verify で取得済み
    dataset,                                        # 階層ラベル付き（train/test split id を持つ）
    fine_tune_scope: dict,                          # Ch4 hierarchy detection の結果
    feature_probe: dict,                            # {"layer_id": "backbone.layer4", "pooling": "gap", "expected_shape": [D]}
    uncertainty_method: str = "mc_dropout",         # "mc_dropout" | "bnn" | "deep_ensemble"
    uncertainty_config: dict = None,                # Ch9 config (n_mc_samples, temperature, etc.)
    reduction_config: dict = None,                  # {"method":"pca","target_dim":16,"fit_split":"train",...}
    reduction_transformer=None,                     # train split で事前 fit 済み（提供必須／None なら本関数内で fit）
    train_split_hash: str = None,
    device: str = "cuda",
) -> dict:
    """
    fm_model から深層特徴を抽出。
    - feature_probe.layer_id / pooling で「どの層を特徴とみなすか」を明示
    - fine_tune_scope で指定された層は fine-tune 済み head、それ以外は frozen backbone
    - per-sample uncertainty を Ch8/Ch9 の named metrics として出力
    - PCA/PLS/AE 次元削減は本関数内で必ず適用（train split only で fit）
    """
    # ---- (a) FM を eval モードにするが、MC-Dropout 用に dropout 層のみ再有効化
    fm_model.eval()
    if uncertainty_method == "mc_dropout":
        _enable_dropout_layers_only(fm_model)       # Ch9 パターン: BN/LN は eval, Dropout のみ train

    # ---- (b) 指定 layer から特徴を取り出す hook を登録
    probe_hook = _register_feature_probe(
        fm_model, layer_id=feature_probe["layer_id"], pooling=feature_probe["pooling"]
    )

    features_by_sample = []
    monitors_by_sample = []                          # Ch8 named metrics [n_samples, n_metrics]
    hierarchy_labels = []
    split_ids = []

    for batch in dataset:
        images = batch["image"].to(device)
        labels = batch["hierarchy"]
        split_ids.append(batch["split_id"])          # train / val / test

        # fine-tune スコープに応じて分岐
        if _should_use_finetuned(labels, fine_tune_scope):
            head = _select_finetuned_head(labels, fine_tune_scope)
            _ = head(fm_model.backbone(images))      # head 側の中間層も probe 対象になり得る
        else:
            with torch.no_grad():
                _ = fm_model.backbone(images)        # frozen path

        feat = probe_hook.pop()                      # [B, D] — 契約 shape と一致確認済み

        # per-sample uncertainty（Ch8/Ch9 named metrics）
        if uncertainty_method == "mc_dropout":
            monitors = _mc_dropout_monitors(fm_model, images, uncertainty_config)
        elif uncertainty_method == "bnn":
            monitors = _bnn_monitors(fm_model, images, uncertainty_config)
        elif uncertainty_method == "deep_ensemble":
            monitors = _deep_ensemble_monitors(fm_model, images, uncertainty_config)
        else:
            raise ValueError(f"unknown uncertainty_method: {uncertainty_method}")
        # monitors: dict of ["predictive_entropy_normalized", "mutual_information",
        #                    "max_softmax_uncertainty"] → [B]

        features_by_sample.append(feat.cpu().numpy())
        monitors_by_sample.append(monitors)
        hierarchy_labels.append(labels)

    features = np.concatenate(features_by_sample, axis=0)      # [n_samples, D]
    monitors_stacked = _stack_named_monitors(monitors_by_sample)  # [n_samples, n_metrics]

    # ---- (c) 次元削減（train split only で fit → 全 split に transform）
    if reduction_transformer is None:
        reduction_transformer, reduction_meta = _fit_reduction(
            features=features, split_ids=split_ids, config=reduction_config
        )
    else:
        reduction_meta = _validate_reduction_config(reduction_transformer, reduction_config)
    reduced_features = reduction_transformer.transform(features)  # [n_samples, target_dim]

    # ---- (d) Ch8 combined_gate と互換な per-sample stop/warn 判定 + 派生スカラー
    per_sample_gate = _apply_ch8_combined_gate(
        monitors=monitors_stacked, thresholds=uncertainty_config["ch8_thresholds"]
    )                                                # {"state":[n], "triggered_metrics":[n]}
    sigma_deep = _calibrate_sigma_deep(              # target scale へ較正
        monitors=monitors_stacked,
        calibration=uncertainty_config["sigma_deep_calibration"],
    )                                                # [n_samples] スカラー、単位は y と同じ

    high_uncertainty_mask = per_sample_gate["state"] == "stop"

    return {
        "features_raw": features,                    # [n_samples, D]  (D = probe.expected_shape[-1])
        "reduced_features": reduced_features,        # [n_samples, target_dim]
        "uncertainty_monitors": monitors_stacked,    # [n_samples, n_metrics] — Ch8 named metrics
        "sigma_deep": sigma_deep,                    # [n_samples] — PyMC 用スカラー（較正済み）
        "per_sample_gate_state": per_sample_gate["state"],       # "pass"|"warn"|"stop"
        "per_sample_triggered_metrics": per_sample_gate["triggered_metrics"],
        "high_uncertainty_sample_ids": _collect_ids(hierarchy_labels, high_uncertainty_mask),
        "hierarchy_labels": hierarchy_labels,
        "split_ids": split_ids,
        "uncertainty_method": uncertainty_method,
        "fine_tune_scope_applied": fine_tune_scope,
        "feature_probe": feature_probe,
        "reduction_meta": reduction_meta,            # method/target_dim/transformer_hash/seed/fit_split
        "attribution_provenance_ref": _optional_ch10_attribution_ref(high_uncertainty_mask),
        "instrument_history_ref": _optional_instrument_history_ref(hierarchy_labels, high_uncertainty_mask),
    }
```

### 特徴の次元削減（PyMC 投入前）

深層特徴は通常 512〜2048 次元。PyMC 階層モデルに直接投入すると事前分布設計が破綻するため、**PCA / partial least squares / Autoencoder**で 8〜32 次元に圧縮します。

```yaml
# feature_reduction_config.yaml (capstone 内部)
skill_step: "feature_dim_reduction"
method: "pca | pls | autoencoder"
target_dim: 16                                       # PyMC 事前分布と釣り合う次元
require_variance_explained_min: 0.80                 # PCA では 80% 以上
fit_only_on_train_split: true                        # test 情報漏洩防止
random_seed: 20260704                                # 再現性
train_split_hash: "sha256_of_train_indices"          # split lock
standardizer_hash: "sha256"                          # 前段の zero-mean/unit-var 変換
library_versions:                                    # 数値再現性
  sklearn: "1.5.x"
  torch: "2.4.x"
  numpy: "1.26.x"
ae_config_if_autoencoder:                            # AE のときのみ
  architecture: "encoder_layers | latent_dim | activation"
  train_epochs: "int"
  optimizer_config: "dict"
  early_stopping: "dict"
provenance_recorded:
  - "reduction_transformer_hash"
  - "random_seed"
  - "train_split_hash"
  - "standardizer_hash"
  - "library_versions"
```

### 契約 YAML

```yaml
# capstone_deep_feature_extraction.yaml
skill: "capstone_deep_feature_extraction"
version: "1.0.0"

requires:
  fm_model_provenance_ref: true                     # Ch11 fm_fetch_and_verify の出力
  fine_tune_scope_from_hierarchy_detection: true    # capstone_hierarchy_detection の出力
  uncertainty_method_in_ch9_registry: true          # MC-Dropout / BNN / Ensemble のいずれか
  feature_reduction_fit_only_on_train_split: true   # 漏洩防止

uncertainty_gate:                                    # Ch8 uncertainty_stop_gate と完全互換
  compatibility_mode: "ch8_named_metrics_preserved"
  monitored_metrics:                                 # Ch8 の named metrics を保持
    - "predictive_entropy_normalized"
    - "mutual_information"
    - "max_softmax_uncertainty"
  per_metric_thresholds:                             # Ch8 と同じ辞書構造で受ける
    predictive_entropy_normalized: { warn: 0.6, stop: 0.85 }
    mutual_information:            { warn: 0.5, stop: 0.8  }
    max_softmax_uncertainty:       { warn: 0.6, stop: 0.85 }
  calibration_required:
    ece_absolute_max: 0.05
    ece_relative_max_vs_baseline: 0.5
    temperature_recorded: true
  combined_gate_states: ["pass", "warn", "stop"]
  stop_precedence: true
  triggered_metric_names_recorded_per_sample: true
  action_on_stop: "route_to_human_gate2"             # Gate 2 に escalate

feature_probe_contract:
  layer_id_required: true                            # 例: "backbone.encoder.layer4"
  pooling_required: true                             # "gap"|"cls_token"|"mean_over_seq"
  expected_shape_recorded: true                      # [D] を Ch11 manifest と突合
  probe_hash_recorded: true                          # hook 実装の sha256

feature_reduction_applied_in_this_skill: true         # skeleton 内で必ず transform を通す

acceptance:
  reduced_features_dim_matches_reduction_target: true
  uncertainty_monitors_shape_is_n_samples_by_n_metrics: true
  sigma_deep_shape_is_n_samples: true
  sigma_deep_calibration_recorded: true
  fine_tune_scope_applied_matches_input_scope: true
  no_feature_leakage_train_to_test: true
  ch8_triggered_metric_names_present_when_state_ne_pass: true

agent_authorization:
  L1: "read_features_only"
  L2: "extract_with_signed_scope"
  L3:
    can_propose_uncertainty_method: true
    cannot_change_fine_tune_scope_after_gate1: "forbidden_all_levels"
    cannot_reuse_features_across_projects_without_approval: "forbidden_all_levels"
  never_allowed:
    - "extract_without_hierarchy_scope"
    - "silently_swap_uncertainty_method"
    - "fit_reduction_on_test_data"
    - "drop_uncertainty_field"
    - "collapse_named_metrics_to_single_scalar_without_calibration"
    - "run_full_eval_disabling_mc_dropout_layers"

provenance:
  capstone_deep_feature_extraction_provenance:
    fm_model_provenance_ref: "id"
    feature_probe:
      layer_id: "str"
      pooling: "str"
      expected_shape: "list"
      probe_hash: "sha256"
    fine_tune_scope: "dict"
    fine_tune_scope_provenance_ref: "id"
    uncertainty_method: "mc_dropout | bnn | deep_ensemble"
    uncertainty_method_provenance_ref: "id"          # Ch9 の該当実行 ID
    uncertainty_config:
      n_mc_samples: "int (if mc_dropout)"
      enable_dropout_only: true                      # BN/LN は eval のまま
      posterior_samples: "int (if bnn)"
      ensemble_members: "int (if deep_ensemble)"
      temperature: "float"
    calibration:
      ece_absolute: "float"
      ece_relative_vs_baseline: "float"
      temperature: "float"
    feature_reduction:
      method: "pca | pls | autoencoder"
      target_dim: "int"
      variance_explained: "float (if pca)"
      transformer_hash: "sha256"
      random_seed: "int"
      train_split_hash: "sha256"
      standardizer_hash: "sha256"
      fit_split: "train"
      library_versions: "dict"
      ae_config: "dict (if autoencoder)"
    per_sample_uncertainty_stats:
      per_metric_mean_p95_max: "dict"
      warn_ratio: "float"
      stop_ratio: "float"
      triggered_metric_names_histogram: "dict"
    sigma_deep_calibration:
      calibration_method: "isotonic | scaling_by_holdout_residual_std | temperature"
      calibration_provenance_ref: "id"
      target_scale_units: "same_as_y"
    features_shape_raw: "list [n_samples, D]"
    reduced_features_shape: "list [n_samples, target_dim]"
    uncertainty_monitors_shape: "list [n_samples, n_metrics]"
    sigma_deep_shape: "list [n_samples]"
    high_uncertainty_sample_ids: "list of str"        # Gate 2 payload で必須
    attribution_provenance_ref: "id (Ch10)"           # Gate 2 payload で必須
    instrument_history_ref: "id"                      # Gate 2 payload で必須
    extraction_timestamp: "iso8601"
```

> [!IMPORTANT]
> **`triggered_metric_state == "stop"` のサンプルが 5% 以上ある場合**、下流の PyMC 階層モデルはそのサンプルを **`known_high_uncertainty` フラグ付きで投入**し（除外は禁止）、posterior summary で **別セクションとして分離レポート**します（§14.6 で詳述、追加の重み付けは行わない — 尤度の分散項で自然に扱う）。

---

## 14.6 `capstone_hierarchical_bayes` — 深層特徴 × 階層プーリング

vol-02 第12章の階層モデルを **深層特徴入力に対応**させます。

### モデル設計

- 各サンプル $i$（研究室 $\ell$, 装置 $j$, ロット $k$, sample $s$）に対して、**縮約された深層特徴 $\mathbf{z}_i \in \mathbb{R}^{16}$** と観測 uncertainty $\sigma_{\mathrm{deep}, i}$ が入る
- 材料物性 $y_i$ を階層 GLM で表現：

$$
y_i \sim \mathrm{Normal}\bigl(\mu_i,\ \sigma_i\bigr),\quad
\sigma_i = \sqrt{\sigma_{\mathrm{obs}}^2 + w \cdot \sigma_{\mathrm{deep}, i}^2}
$$

$$
\mu_i = \alpha_{\ell(i)} + \gamma_{\ell(i), j(i)} + \delta_{\ell(i), j(i), k(i)} + \mathbf{z}_i^\top \boldsymbol{\beta}
$$

- $\alpha_\ell$: 研究室効果（partial pooling with Normal(μ_α, σ_α)）
- $\gamma_{\ell, j}$: 装置効果（研究室内 partial pooling）
- $\delta_{\ell, j, k}$: ロット効果（装置内 partial pooling）
- $\boldsymbol{\beta}$: 深層特徴の係数
- $w$: **深層 uncertainty をどの程度観測ノイズに加算するかの重み**（弱情報事前で識別性を確保、感度分析必須）
- **PyMC 実装では `pm.Normal(mu=mu_i, sigma=σ_i)`（`sigma` パラメータ）**を使う。`sigma_obs^2 + w·σ_deep²` は分散、`σ_i` はその平方根

> [!WARNING]
> **`sigma_deep` は較正済み・y と同じ単位のスカラー**（§14.5 の `sigma_deep_calibration` で担保）。Ch8/Ch9 の生の normalized risk score をそのまま渡すと `w` が自明な吸収項になり識別性が崩壊する。calibration provenance の確認は Bayes 実行前の acceptance。

### 契約 YAML

```yaml
# capstone_hierarchical_bayes.yaml
skill: "capstone_hierarchical_bayes"
version: "1.0.0"

requires:
  features_from_capstone_deep_feature_extraction: true
  hierarchy_labels_complete: true                    # lab/instrument/lot/sample 全て
  reduction_target_dim_recorded: true                # 事前分布設計の透明性
  uncertainty_propagation_configured: true           # deep uncertainty をノイズに加算するか
  sigma_deep_calibration_provenance_present: true    # 較正なしでは w が非識別
  sigma_deep_units_match_y_scale: true

model_family: "hierarchical_glm_with_deep_features"

pooling_strategy:
  lab_level: "partial"
  instrument_level: "partial_within_lab"
  lot_level: "partial_within_instrument"
  sample_level: "no_pooling_use_deep_uncertainty_only"

priors:
  alpha_lab: "Normal(mu_alpha, sigma_alpha), sigma_alpha ~ HalfNormal(1.0)"
  gamma_instrument: "Normal(0, sigma_gamma), sigma_gamma ~ HalfNormal(1.0)"
  delta_lot: "Normal(0, sigma_delta), sigma_delta ~ HalfNormal(0.5)"
  beta_deep: "Normal(0, 1.0)"                         # 特徴を標準化前提
  sigma_obs: "HalfNormal(1.0)"
  deep_uncertainty_weight_w: "HalfNormal(0.5)"        # 弱情報、target scale 較正済み前提

likelihood:
  parameterization: "sigma (standard deviation, not variance)"
  formula_variance: "obs_variance_i = sigma_obs^2 + w * sigma_deep_i^2"
  formula_sigma:    "sigma_i = sqrt(obs_variance_i)"
  pymc_call: "pm.Normal('y', mu=mu_i, sigma=sigma_i, observed=y_obs)"

deep_uncertainty_integration:
  scalar_source: "sigma_deep from extraction.sigma_deep (calibrated, [n_samples])"
  formula: "observation_variance = sigma_obs^2 + w * sigma_deep_i^2"
  known_high_uncertainty_flag_handling:
    definition: "per_sample_gate_state == 'stop' upstream (Ch8-compatible)"
    treatment: "flag_only_no_additional_weighting"    # 尤度の分散増で自然に downweight される
    exclusion_forbidden: true
    posterior_summary:
      report_flagged_samples_in_separate_section: true
      also_report_ppc_metrics_conditioned_on_flag: true
  identifiability_safeguards:
    weak_informative_prior_on_w: "HalfNormal(0.5)"
    require_w_posterior_ci_upper_finite: true
    sensitivity_analysis_required:
      variants: ["fixed w=0", "fixed w=1", "prior HalfNormal(0.1)", "prior HalfNormal(2.0)"]
      report_field: "w_sensitivity_analysis"
    posterior_correlation_w_vs_sigma_obs_max: 0.7     # 高相関なら非識別警告

sampler_config:
  backend: "pymc"
  algorithm: "nuts"
  chains: 4
  target_accept: 0.9
  draws: 2000
  tune: 1000
  cores: 4
  random_seed_per_chain: "list of int"

diagnostics_required:                                  # 基準（違反時は必ず Gate 3 へ）
  r_hat_max: 1.01
  ess_min_ratio_of_draws: 0.4
  divergences_max: 0
  bfmi_min: 0.3
  action_on_diagnostic_fail: "route_to_human_gate3"

# 診断不合格は「不合格として記録」する。緩和承認は Gate 3 が発行する
# documented_exception のみ許可され、`diagnostics_all_pass` は書き換えない。
diagnostic_exception_policy:
  who_can_grant: "gate3_only (statistician AND pi, reviewers_min=2)"
  granted_exception_fields:
    exception_id: "sha256"
    violated_metric: "r_hat | ess | divergences | bfmi"
    violation_value: "float or int"
    remediation_plan: "str (required)"
    re_run_reference_provenance_id: "id or null"
  effect_on_report:
    diagnostics_all_pass: false                        # 書き換え禁止
    report_must_include_exception_block: true

posterior_predictive_checks:                           # PPC は必須
  required: true
  checks:
    - "residuals_by_group: lab | instrument | lot"
    - "posterior_predictive_coverage_50_80_95"
    - "calibration_by_sigma_deep_bin"
    - "group_level_shrinkage_diagnostics: alpha_lab_prior_vs_posterior"
    - "ppc_conditioned_on_known_high_uncertainty_flag"
  fail_threshold:
    coverage_95_min: 0.90
    coverage_95_max: 0.985
    calibration_slope_min: 0.8
    calibration_slope_max: 1.2
  action_on_ppc_fail: "route_to_human_gate3"

acceptance:
  posterior_summary_computed: true
  diagnostics_all_pass_or_gate3_exception_granted: true
  known_high_uncertainty_samples_reported_separately: true
  known_high_uncertainty_samples_never_excluded: true
  sigma_parameterization_verified_as_stddev: true
  w_sensitivity_analysis_reported: true
  ppc_reported: true

agent_authorization:
  L1: "read_posterior_summary"
  L2: "run_sampler_with_signed_config"
  L3:
    can_propose_pooling_change: true
    cannot_change_pooling_structure_without_gate3: "forbidden_all_levels"
    cannot_hide_divergences: "forbidden_all_levels"
    cannot_downgrade_diagnostic_thresholds_silently: "forbidden_all_levels"
  never_allowed:
    - "collapse_hierarchy_to_avoid_divergences"
    - "flatten_priors_to_hide_convergence_issues"
    - "drop_known_high_uncertainty_samples"
    - "apply_additional_downweight_on_top_of_variance_augmentation"
    - "pass_variance_as_sigma_to_pm_normal"

provenance:
  capstone_hierarchical_bayes_provenance:
    features_provenance_ref: "id"
    model_family: "str"
    pooling_strategy: "dict"
    priors: "dict"
    sampler_config: "dict"
    posterior_artifact_uri: "str (arviz netcdf or similar)"
    posterior_artifact_sha256: "str"
    diagnostics:
      r_hat: "dict"
      ess: "dict"
      divergences: "int"
      bfmi: "float"
      all_pass: "bool"                                # false のときは exception_block 必須
      exception_block:                                # 記録専用 (all_pass=true には格上げしない)
        exception_id: "sha256 or null"
        violated_metric: "str or null"
        violation_value: "float or int or null"
        remediation_plan: "str or null"
        gate3_provenance_ref: "id or null"
    posterior_predictive_checks:
      residuals_by_group: "dict (lab|instrument|lot)"
      coverage_50_80_95: "dict"
      calibration_by_sigma_deep_bin: "dict"
      shrinkage_diagnostics: "dict"
      ppc_conditioned_on_known_high_uncertainty: "dict"
      ppc_all_pass: "bool"
    deep_uncertainty_weight_w_posterior: "dict"
    w_sensitivity_analysis:
      variants_run: "list"
      per_variant_summary: "dict"
      identifiability_verdict: "identified | weakly_identified | non_identified"
      posterior_corr_w_vs_sigma_obs: "float"
    known_high_uncertainty_samples_count: "int"
    known_high_uncertainty_samples_excluded: false    # 常に false（除外禁止）
    inference_timestamp: "iso8601"
```

> [!WARNING]
> **`collapse_hierarchy_to_avoid_divergences: never_allowed`** は本章で最も重要な never_allowed の 1 つです。divergences が出たら階層を collapse（no pooling / complete pooling に降格）するのは統計的に禁忌で、Gate 3（Human 承認）を通す必要があります。

---

## 14.7 `capstone_three_gate_orchestrator` — 3 つの Human-in-the-loop ゲート

**本章の核心**は、以下 3 つの Gate が **順番に** Human 承認を要求する点です：

| Gate | 発火タイミング | Human に見せる情報 |
|---|---|---|
| **Gate 1: Fine-tune 起動承認** | `capstone_hierarchy_detection` が fine_tune_candidate を返した直後 | Ch7 domain_gap の pairwise 表 + fine-tune スコープ提案 + GPU 予算見積 |
| **Gate 2: 不確かさ閾値超え** | `capstone_deep_feature_extraction` の stop_ratio が閾値超過 | High-uncertainty sample のリスト + Ch10 attribution マップ + 該当装置の recent history |
| **Gate 3: 階層プーリング構造変更** | `capstone_hierarchical_bayes` の diagnostics 不合格または pooling 変更提案 | R̂ / ESS / divergences レポート + pooling structure diff + trace plot |

### Gate 発火ロジック

```python
# capstone_three_gate_orchestrator.py
def capstone_three_gate_orchestrator(
    hierarchy_result: dict,
    extraction_result: dict = None,
    bayes_result: dict = None,
    human_approvals: dict = None,      # {"gate1": ApprovalRecord, ...} — role/expiry 付き
    prior_state: dict = None,          # 直前の orchestrator 呼び出し結果（re-run 追跡）
) -> dict:
    """
    3 つの Gate を順番に評価。
    各 Gate は {pending, approved, rejected, expired, disputed} の state machine。
    approved 以外の状態はすべて詳細を provenance に記録し、以降には進まない。
    """
    gates_status = {"gate1": None, "gate2": None, "gate3": None}

    # ---- Gate 1
    if hierarchy_result["fine_tune_recommended"]:
        g1 = _resolve_gate_state(
            gate="gate1",
            approvals=human_approvals,
            required_roles={"any_of": ["ml_lead", "pi"]},
            reviewers_min=1,
            payload={
                "pairwise_ch7_table": hierarchy_result["per_level_domain_gap"],
                "proposed_scope": hierarchy_result["overall_strategy"]["fine_tune_hierarchy_levels"],
                "gpu_budget_estimate": hierarchy_result.get("gpu_budget_estimate"),
                "fine_tune_deferred_to_human": hierarchy_result.get("fine_tune_deferred_to_human"),
            },
            prior_state=prior_state,
        )
        if g1["state"] != "approved":
            return _pause_or_terminate(gate="gate1", resolution=g1)
        gates_status["gate1"] = g1
    else:
        gates_status["gate1"] = {"state": "not_triggered"}

    # ---- Gate 2
    if extraction_result is not None:
        stop_ratio = extraction_result["per_sample_uncertainty_stats"]["stop_ratio"]
        warn_ratio = extraction_result["per_sample_uncertainty_stats"]["warn_ratio"]
        escalation = _gate2_escalation_reason(
            stop_ratio=stop_ratio,
            warn_ratio=warn_ratio,
            per_sample_gate_state=extraction_result["per_sample_gate_state"],
            hierarchy_labels=extraction_result["hierarchy_labels"],
        )
        if escalation is not None:
            g2 = _resolve_gate_state(
                gate="gate2",
                approvals=human_approvals,
                required_roles={"any_of": ["domain_expert", "pi"]},
                reviewers_min=1,
                payload={
                    "escalation_reason": escalation,       # stop/warn/cluster/consecutive
                    "stop_ratio": stop_ratio,
                    "warn_ratio": warn_ratio,
                    "high_uncertainty_sample_ids": extraction_result["high_uncertainty_sample_ids"],
                    "attribution_provenance_ref": extraction_result["attribution_provenance_ref"],
                    "instrument_history_ref": extraction_result["instrument_history_ref"],
                    "triggered_metrics_histogram": extraction_result[
                        "per_sample_uncertainty_stats"
                    ]["triggered_metric_names_histogram"],
                },
                prior_state=prior_state,
            )
            if g2["state"] != "approved":
                return _pause_or_terminate(gate="gate2", resolution=g2)
            gates_status["gate2"] = g2
        else:
            gates_status["gate2"] = {"state": "not_triggered"}

    # ---- Gate 3
    if bayes_result is not None:
        needs_gate3 = (
            (not bayes_result["diagnostics"]["all_pass"])
            or (not bayes_result["posterior_predictive_checks"]["ppc_all_pass"])
            or bayes_result.get("pooling_change_proposed", False)
        )
        if needs_gate3:
            g3 = _resolve_gate_state(
                gate="gate3",
                approvals=human_approvals,
                required_roles={"all_of": ["statistician", "pi"]},
                reviewers_min=2,
                forbid_duplicate_identity=True,
                forbid_self_sign_agent_id=True,
                payload={
                    "diagnostics": bayes_result["diagnostics"],
                    "ppc": bayes_result["posterior_predictive_checks"],
                    "current_pooling": bayes_result["pooling_strategy"],
                    "proposed_pooling": bayes_result.get("proposed_pooling"),
                    "trace_plot_uri": bayes_result.get("trace_plot_uri"),
                    "w_sensitivity_analysis": bayes_result["w_sensitivity_analysis"],
                    "remediation_plan_required": True,
                },
                prior_state=prior_state,
            )
            if g3["state"] != "approved":
                return _pause_or_terminate(gate="gate3", resolution=g3)
            gates_status["gate3"] = g3
        else:
            gates_status["gate3"] = {"state": "not_triggered"}

    return {
        "status": "all_gates_passed",
        "gates": gates_status,
        "next_action": "generate_final_report",
    }
```

### 契約 YAML

```yaml
# capstone_three_gate_orchestrator.yaml
skill: "capstone_three_gate_orchestrator"
version: "1.0.0"

requires:
  three_gates_defined: true
  each_gate_has_signed_approval_slot: true
  approval_registry_provides_role_and_identity: true
  human_approvers_min_per_gate: 1                    # Gate 3 は 2 名必須

gate_state_machine:
  states: ["pending", "approved", "rejected", "expired", "disputed", "not_triggered"]
  transitions:
    pending_to_approved: "signed_by_valid_role_within_expiry"
    pending_to_rejected: "signed_reject_with_reason"
    pending_to_expired:  "no_signature_before_expires_at"
    pending_to_disputed: "reviewers_signed_contradictory_decisions"
  on_rejected: "orchestrator_terminates_pipeline; re_run_requires_new_provenance_id"
  on_expired:  "orchestrator_terminates_pipeline; timeout_recorded"
  on_disputed: "escalate_to_higher_authority; agent_cannot_break_tie"
  re_run_after_rejection_or_expiry:
    new_run_provenance_id_required: true
    link_to_previous_rejection_or_expiry_id: "required"
    fresh_signatures_required_for_all_downstream_gates: true

approval_validation:
  role_conjunction_enforced: true                    # "AND" is a hard AND
  role_disjunction_enforced: true                    # "OR" は列挙のうち 1 名以上
  duplicate_identity_forbidden: true                 # 同一 human ID は 1 回のみ
  self_sign_by_agent_forbidden: true                 # agent の identity では絶対に署名不可
  signature_verified_against_registry_public_keys: true
  expiry_recorded_per_signature: true

gate_definitions:
  gate1_fine_tune_launch:
    trigger: "hierarchy_result.fine_tune_recommended == true"
    required_roles: { any_of: ["ml_lead", "pi"] }
    reviewers_min: 1
    approval_expiry_hours: 168                       # 7 日
    payload_must_include: ["pairwise_ch7_table", "proposed_scope", "gpu_budget_estimate", "fine_tune_deferred_to_human"]
  gate2_uncertainty_stop:
    trigger_any_of:
      - "stop_ratio >= 0.05"
      - "warn_ratio >= 0.20"
      - "consecutive_warn_batches_over_group >= 3"
      - "group_local_warn_cluster_detected"          # 単一 instrument/lot に warn が集中
    required_roles: { any_of: ["domain_expert", "pi"] }
    reviewers_min: 1
    approval_expiry_hours: 72
    payload_must_include:
      - "escalation_reason"
      - "stop_ratio"
      - "warn_ratio"
      - "high_uncertainty_sample_ids"
      - "attribution_provenance_ref"
      - "instrument_history_ref"
      - "triggered_metrics_histogram"
  gate3_pooling_or_diagnostics:
    trigger_any_of:
      - "bayes.diagnostics.all_pass == false"
      - "bayes.posterior_predictive_checks.ppc_all_pass == false"
      - "pooling_change_proposed == true"
    required_roles: { all_of: ["statistician", "pi"] }
    reviewers_min: 2
    duplicate_identity_forbidden: true
    approval_expiry_hours: 72
    payload_must_include:
      - "diagnostics"
      - "ppc"
      - "current_pooling"
      - "proposed_pooling"
      - "trace_plot_uri"
      - "w_sensitivity_analysis"
      - "remediation_plan_required"

acceptance:
  no_gate_bypassed: true
  all_approvals_signed_and_registry_verified: true
  gate_state_transitions_recorded: true
  re_run_provenance_chain_verified_if_prior_rejection_or_expiry: true

agent_authorization:
  L1: "read_gate_status"
  L2: "prepare_gate_payload_and_notify_human"
  L3:
    can_propose_scope_or_pooling_change: true
    cannot_self_approve_any_gate: "forbidden_all_levels"
    cannot_reorder_gates: "forbidden_all_levels"
    cannot_bypass_gate_by_downgrading_thresholds: "forbidden_all_levels"
    cannot_break_disputed_tie: "forbidden_all_levels"
  never_allowed:
    - "auto_approve_any_gate"
    - "skip_gate_1_when_fine_tune_recommended"
    - "skip_gate_2_when_any_escalation_reason_triggered"
    - "skip_gate_3_when_diagnostics_or_ppc_fail"
    - "self_sign_as_approver"
    - "reuse_expired_or_rejected_signature"
    - "count_same_human_id_twice_for_reviewers_min"

provenance:
  capstone_three_gate_orchestrator_provenance:
    gates_evaluated: "list"
    gate1_resolution:
      state: "one of gate_state_machine.states"
      approvers: "list of {hashed_id, role, signature_id, signed_at, expires_at}"
      rejection_reason: "str or null"
      expired_at: "iso8601 or null"
      previous_run_link: "provenance_id or null"
    gate2_resolution: "same schema as gate1_resolution"
    gate3_resolution:
      state: "one of gate_state_machine.states"
      approvers: "list (>=2 unique identities, roles include statistician AND pi)"
      exception_granted_id: "sha256 or null"          # bayes diagnostic exception を発行した場合
      remediation_plan: "str or null"
      previous_run_link: "provenance_id or null"
    approval_registry_signatures_verified: true
    gate_evaluation_timestamps: "list of iso8601"
```

> [!IMPORTANT]
> **エージェントが自分で承認者になることは全レベル forbidden**（`self_sign_as_approver`）。approval registry が Ch11 で導入した署名検証を行い、承認者 ID の hash が Human の登録済み ID と一致することを確認します。

---

## 14.8 統合 Skill `hierarchical_deep_bayes` — 全体契約

上記 4 つを **1 本の統合契約**にまとめます。

```yaml
# hierarchical_deep_bayes.yaml (capstone integrated)
skill: "hierarchical_deep_bayes"
version: "1.0.0"

sub_skills_orchestrated:
  - "capstone_hierarchy_detection"
  - "capstone_deep_feature_extraction"
  - "capstone_hierarchical_bayes"
  - "capstone_three_gate_orchestrator"

execution_order:                                     # 順序固定
  1: "capstone_hierarchy_detection"
  2: "capstone_three_gate_orchestrator (gate1 eval)"
  3: "capstone_deep_feature_extraction (if gate1 approved or not triggered)"
  4: "capstone_three_gate_orchestrator (gate2 eval)"
  5: "capstone_hierarchical_bayes (if gate2 approved or not triggered)"
  6: "capstone_three_gate_orchestrator (gate3 eval)"
  7: "generate_final_report_via_ch10_deep_report_template"

integrated_provenance_chain:                         # Ch4-12 の provenance を全て継承（省略不可）
  # 適用されない block も必ず nullhash sentinel で記録し、決定的順序を維持する
  chain_entries_ordered:                              # この順序が hash_chain の canonical order
    - key: "layer_1_data"
      source: "vol-02 datasets"
      required: true
    - key: "layer_2_augmentation"
      source: "Ch4"
      required: true
    - key: "layer_3_split"
      source: "Ch4"
      required: true
    - key: "layer_4_pretrained_weights"
      source: "Ch7 (if fine-tune applied)"
      required: false                                 # 未適用時は null_sentinel で記録
    - key: "foundation_model_provenance"
      source: "Ch11 fm_fetch_and_verify"
      required: true
    - key: "fm_query_provenance"
      source: "Ch11 fm_query"
      required: false
    - key: "ssl_pretrain_provenance"
      source: "Ch12 (if SSL encoder used)"
      required: false
    - key: "ssl_representation_eval_provenance"
      source: "Ch12 ssl_representation_eval"
      required: false
    - key: "bayesian_inference_config"
      source: "Ch9"
      required: true
    - key: "layer_attribution"
      source: "Ch10 (for gate2 payload)"
      required: true
    - key: "layer_human_review"
      source: "Ch10 (for all 3 gates)"
      required: true
    - key: "capstone_hierarchy_detection_provenance"
      source: "this chapter"
      required: true
    - key: "capstone_deep_feature_extraction_provenance"
      source: "this chapter"
      required: true
    - key: "capstone_hierarchical_bayes_provenance"
      source: "this chapter"
      required: true
    - key: "capstone_three_gate_orchestrator_provenance"
      source: "this chapter"
      required: true
  null_sentinel:                                      # required:false が欠けたときの決定的表現
    hash: "0000000000000000000000000000000000000000000000000000000000000000"
    reason_field_required: true                       # なぜ適用されないかを人間可読で記録

final_report_generation:
  template: "ch10_deep_report_template"
  ch10_section_input_mapping:                         # Ch10 各セクションへの capstone 出力マッピング
    executive_summary:
      from:
        - "hierarchy_result.overall_strategy"
        - "bayes_result.posterior_summary_by_hierarchy_level"
        - "gates.summary"
    data_and_provenance:
      from:
        - "integrated_provenance_chain (entire)"
        - "capstone_integrated_provenance.hash_chain"
    domain_gap_and_fine_tune:
      from:
        - "hierarchy_result.per_level_domain_gap"
        - "hierarchy_result.overall_strategy"
        - "hierarchy_result.pairwise_scaling_policy"
    feature_extraction_and_uncertainty:
      from:
        - "extraction_result.feature_probe"
        - "extraction_result.reduction_meta"
        - "extraction_result.per_sample_uncertainty_stats"
        - "extraction_result.calibration"
    hierarchical_bayes:
      from:
        - "bayes_result.pooling_strategy"
        - "bayes_result.priors"
        - "bayes_result.diagnostics"
        - "bayes_result.posterior_predictive_checks"
        - "bayes_result.w_sensitivity_analysis"
    layer_attribution:
      from: ["extraction_result.attribution_provenance_ref"]
    three_gate_audit_trail:
      from: ["orchestrator_result.gate1_resolution", "orchestrator_result.gate2_resolution", "orchestrator_result.gate3_resolution"]
    known_high_uncertainty_samples:
      from: ["extraction_result.high_uncertainty_sample_ids", "bayes_result.known_high_uncertainty_samples_count"]
    reproducibility:
      from: ["capstone_integrated_provenance.hash_chain", "capstone_integrated_provenance.canonical_manifest"]
  omit_any_section: "fatal"
  additional_sections_required:
    - "hierarchy_summary_diagram"
    - "fine_tune_scope_rationale"
    - "per_sample_uncertainty_distribution"
    - "posterior_summary_by_hierarchy_level"
    - "three_gate_audit_trail"
    - "known_high_uncertainty_samples_list"
    - "reproducibility_manifest_hash_chain"

acceptance:
  all_sub_skills_provenance_present: true
  all_three_gates_status_recorded: true
  posterior_diagnostics_pass_or_gate3_approved: true
  reproducibility_manifest_hashes_chain_verified: true

agent_authorization:
  L1: "read_final_report"
  L2: "execute_sub_skills_up_to_gate1_pause"
  L3:
    can_execute_with_gate_approvals: true
    cannot_reorder_execution_steps: "forbidden_all_levels"
    cannot_skip_gate_orchestrator_between_steps: "forbidden_all_levels"
    cannot_generate_report_before_all_gates_resolved: "forbidden_all_levels"
  never_allowed:
    - "execute_bayes_before_gate2_resolved"
    - "generate_report_before_gate3_resolved"
    - "reuse_sub_skill_output_across_projects_without_provenance_re_verification"
    - "collapse_provenance_chain_to_simplify_report"

provenance:
  capstone_integrated_provenance:
    sub_skill_provenance_refs:
      hierarchy_detection: "id"
      feature_extraction: "id"
      hierarchical_bayes: "id"
      three_gate_orchestrator: "id"
    canonical_manifest:                              # hash 決定性の根拠
      format: "canonical_json"                       # RFC 8785 JCS 準拠
      keys_ordered_by: "chain_entries_ordered above"
      absent_optional_blocks_encoded_as: "null_sentinel with reason field"
      per_entry_fields: ["key", "type_label", "provenance_id", "sha256", "required", "reason_if_absent"]
    hash_chain:
      chain_algorithm: "sha256_merkle_over_canonical_manifest"
      leaf_hash: "sha256(canonical_json_of_entry)"
      internal_node_hash: "sha256(left_hash || right_hash)"
      chain_root: "sha256 (merkle root)"
      chain_root_verified: true
    execution_timestamps: "list"
    final_report_uri: "str"
    final_report_sha256: "str"
```

---

## 14.9 全体フロー実装（オーケストレーション骨子）

```python
# hierarchical_deep_bayes.py
def hierarchical_deep_bayes(
    dataset,
    fm_provenance_ref: str,
    approval_registry,
) -> dict:
    """
    capstone: 全 Skill を順番に呼び出し、3 Gate を通しながら最終レポートを生成。
    """
    approvals = approval_registry.get_pending(project_id=dataset.project_id)

    # Step 1: 階層検出
    h = capstone_hierarchy_detection(dataset)

    # Step 2: Gate 1 評価
    orchestrator_state = capstone_three_gate_orchestrator(
        hierarchy_result=h, human_approvals=approvals
    )
    if orchestrator_state["status"] != "all_gates_passed":
        return orchestrator_state                    # Human 待ちで一時停止

    # Step 3: 深層特徴抽出（reduced_features と sigma_deep が出る）
    ext = capstone_deep_feature_extraction(
        fm_model=_load_fm(fm_provenance_ref),
        dataset=dataset,
        fine_tune_scope=h["overall_strategy"],
        feature_probe=_ch11_manifest_feature_probe(fm_provenance_ref),
        uncertainty_method="mc_dropout",
        uncertainty_config=_load_ch9_config(),
        reduction_config=_load_reduction_config(),
        train_split_hash=dataset.train_split_hash,
    )

    # Step 4: Gate 2 評価
    orchestrator_state = capstone_three_gate_orchestrator(
        hierarchy_result=h, extraction_result=ext, human_approvals=approvals
    )
    if orchestrator_state["status"] != "all_gates_passed":
        return orchestrator_state

    # Step 5: PyMC 階層モデル（reduced_features と sigma_deep を渡す）
    bayes = capstone_hierarchical_bayes(
        features=ext["reduced_features"],            # [n, target_dim]
        sigma_deep=ext["sigma_deep"],                # [n], y と同じ単位に較正済み
        hierarchy_labels=ext["hierarchy_labels"],
        known_high_uncertainty_mask=(
            ext["per_sample_gate_state"] == "stop"
        ),                                            # 除外はしない、フラグとしてのみ使用
    )

    # Step 6: Gate 3 評価
    orchestrator_state = capstone_three_gate_orchestrator(
        hierarchy_result=h, extraction_result=ext, bayes_result=bayes,
        human_approvals=approvals,
    )
    if orchestrator_state["status"] != "all_gates_passed":
        return orchestrator_state

    # Step 7: 最終レポート（Ch10 deep_report_template）
    report = _generate_final_report_ch10(
        hierarchy=h, extraction=ext, bayes=bayes, gates=orchestrator_state
    )
    return {
        "status": "complete",
        "report_uri": report["uri"],
        "capstone_integrated_provenance": _build_provenance_chain(h, ext, bayes, orchestrator_state),
    }
```

---

## 14.10 エージェント役割の具体シーン

capstone における エージェントの動きを、実際の対話シーンで示します：

| フェーズ | エージェントの行動 | Human の役割 |
|---|---|---|
| 階層検出 | 「lab 間で domain_gap = 0.72、instrument 間で 0.55、lot 間で 0.18 でした。lab / instrument を fine-tune 候補、lot は frozen で提案します」 | pairwise 表を確認し、Gate 1 を承認 |
| 深層抽出 | 「MC-Dropout で per-sample uncertainty を計算しました。stop_ratio = 8.3% です（閾値 5% 超過）」 | 該当サンプルの attribution マップと該当装置の recent history を確認し、Gate 2 を承認（または該当装置のサンプル除外を指示） |
| Bayes 推論 | 「R̂ 最大 1.008、ESS 最小比率 0.52、divergences 3 件、PPC coverage_95 = 0.87（下限違反）。**diagnostics_all_pass = false** です。Gate 3 のご判断をお願いします」 | trace plot と PPC を確認し、reparameterization 提案 or 再サンプリング要求。承認する場合は `documented_exception` として remediation_plan と共に記録（`diagnostics.all_pass` は false のまま）。差し戻し時は新 provenance_id で再実行 |
| 最終レポート | 「Ch10 deep_report_template で最終レポートを生成しました。hash chain が検証済みです」 | レポートに sign-off |

---

## 14.11 失敗パターンと対策

| 失敗 | 症状 / 兆候 | 対策（参照する契約フィールド） |
|---|---|---|
| Gate をエージェントが self-approve | 監査崩壊 | `self_sign_as_approver: never_allowed` + `approval_registry_signatures_verified` + `self_sign_by_agent_forbidden` |
| Divergences が出たら pooling を collapse | 階層情報を失う統計的禁忌 | `collapse_hierarchy_to_avoid_divergences: never_allowed` + Gate 3 経由必須 |
| 高不確かさサンプルを silent に drop | 分布尾の情報を失う | `drop_known_high_uncertainty_samples: never_allowed` + `known_high_uncertainty_samples_never_excluded: true` + `known_high_uncertainty_samples_reported_separately` |
| 特徴次元削減を test data で fit | 情報漏洩 | `fit_only_on_train_split: true` + `fit_reduction_on_test_data: never_allowed` + `train_split_hash` |
| 深層 uncertainty を PyMC ノイズに加算し忘れ | posterior が過信 | `uncertainty_propagation_configured: true` + `likelihood.formula_sigma` 明記 |
| PyMC で分散を sigma として渡す | 実効ノイズが 2 乗される致命バグ | `likelihood.parameterization: sigma` + `pass_variance_as_sigma_to_pm_normal: never_allowed` |
| deep_uncertainty を較正せず投入 | `w` が非識別、posterior が病理的 | `sigma_deep_calibration_provenance_present: true` + `w_sensitivity_analysis_reported: true` |
| 尤度分散増と ad hoc 重みで二重補正 | posterior 精度が過度に減衰 | `treatment: flag_only_no_additional_weighting` + `apply_additional_downweight_on_top_of_variance_augmentation: never_allowed` |
| 診断 fail を Gate 3 で `all_pass=true` に格上げ | 監査記録の改竄 | `diagnostic_exception_policy.effect_on_report.diagnostics_all_pass: false` (書き換え禁止) + `exception_block` に必ず記録 |
| PPC を省略して収束のみ確認 | R̂ 良好でも予測不適合を見逃す | `posterior_predictive_checks.required: true` + `ppc_all_pass` |
| Ch8 named metrics を単一スカラーに畳む | 較正なしで意味不明な閾値化 | `collapse_named_metrics_to_single_scalar_without_calibration: never_allowed` + `triggered_metric_names_recorded_per_sample: true` |
| MC-Dropout で `model.eval()` により dropout 無効化 | uncertainty が常に 0 | `enable_dropout_only: true` + `run_full_eval_disabling_mc_dropout_layers: never_allowed` |
| FM 特徴層を暗黙に選択 | 実装依存で再現不能 | `feature_probe.layer_id / pooling / expected_shape / probe_hash` 全て記録必須 |
| 縮約 transformer の hash のみ記録 | 再現には seed/split/version が必要 | `random_seed` + `train_split_hash` + `standardizer_hash` + `library_versions` |
| 多装置で pairwise が n² 爆発 | Ch7 コスト超過、実行時間破綻 | `pairwise_scaling_policy.max_pairwise_comparisons_per_level` + reference_group_plus_clustering + FDR |
| Fine-tune スコープを Gate 1 後にエージェントが変更 | 承認された scope からの逸脱 | `cannot_change_fine_tune_scope_after_gate1: forbidden_all_levels` |
| Gate 拒否 / expiry を無視して再実行 | 承認履歴が壊れる | `re_run_after_rejection_or_expiry.new_run_provenance_id_required: true` + `link_to_previous_rejection_or_expiry_id: required` |
| Gate 3 で同一人物を 2 回カウント | reviewers_min の空洞化 | `duplicate_identity_forbidden: true` + `count_same_human_id_twice_for_reviewers_min: never_allowed` |
| Reviewer 間の矛盾を agent が調停 | 決裁権の逸脱 | `on_disputed: escalate_to_higher_authority` + `cannot_break_disputed_tie: forbidden_all_levels` |
| Gate 2 が stop_ratio のみ監視 | warn クラスターや drift を見逃す | `gate2_uncertainty_stop.trigger_any_of` に `warn_ratio` / `consecutive` / `group_local_cluster` |
| 実行順序を並列化して整合性を破壊 | provenance chain が構築できない | `cannot_reorder_execution_steps: forbidden_all_levels` |
| Provenance chain を "簡潔化" のため省略 | 監査不能 | `collapse_provenance_chain_to_simplify_report: never_allowed` + `chain_entries_ordered` + `null_sentinel` |
| Hash chain の順序を実行時決定 | 同一 provenance が異なる root を返す | `canonical_manifest.format: canonical_json` + `keys_ordered_by: chain_entries_ordered` + `sha256_merkle_over_canonical_manifest` |
| Ch10 template のセクションを omit | 監査観点の欠落 | `ch10_section_input_mapping` + `omit_any_section: fatal` |
| Gate 2 承認前に Bayes を走らせる | uncertainty をレビューせず posterior に埋め込む | `execute_bayes_before_gate2_resolved: never_allowed` |
| Gate 3 未解決でレポート発行 | 未承認の推論結果を公開 | `generate_report_before_gate3_resolved: never_allowed` |
| 事前分布を flatten して divergences 回避 | データ駆動の見かけ、実は事前弱化 | `flatten_priors_to_hide_convergence_issues: never_allowed` |
| Sub-skill 出力を別プロジェクトで再利用 | provenance の対象が変わっているのに監査不能 | `reuse_sub_skill_output_across_projects_without_provenance_re_verification: never_allowed` |

---

## 14.12 まとめ

- 深層 × 階層 × 不確かさ の **3 者統合は接続面の契約が全て**
- **`capstone_hierarchy_detection`** で層ごとに fine-tune / frozen を決定（Ch7 domain_gap_gate を全 pair で実行）
- **`capstone_deep_feature_extraction`** で FM 特徴 + per-sample uncertainty を並列化、Ch8 stop_gate と互換
- **`capstone_hierarchical_bayes`** で PyMC 階層 GLM に深層特徴と uncertainty を投入、`collapse_hierarchy_to_avoid_divergences` は禁忌
- **`capstone_three_gate_orchestrator`** で 3 つの Human 承認ゲートを厳格化（fine-tune / uncertainty / pooling）
- **`hierarchical_deep_bayes`** 統合 Skill が Ch4-12 の provenance chain を hash で連結、監査可能性を担保
- 失敗パターン 12 件を Skill 契約で予防

## 14.13 章末チェックリスト

- [ ] `capstone_hierarchy_detection` で各階層に Ch7 domain_gap_gate が pairwise で走ったか
- [ ] fine-tune スコープが Gate 1 で Human 承認されたか
- [ ] Gate 1 承認後にエージェントが scope を書き換えていないか
- [ ] 深層特徴の per-sample uncertainty が stop_gate 互換で計算されているか
- [ ] stop_ratio >= 5% で Gate 2 が発火したか（該当時）
- [ ] 特徴次元削減が train split のみで fit されているか
- [ ] Bayes モデルの観測ノイズに deep uncertainty が加算されているか（`w * sigma_deep^2`）
- [ ] R̂ / ESS / divergences / BFMI の 4 診断が全て通ったか（不通なら Gate 3）
- [ ] Gate 3 で pooling 構造変更が承認されているか（該当時）
- [ ] `capstone_integrated_provenance.hash_chain` が Ch4-12 の provenance を全て包含しているか
- [ ] 最終レポートが Ch10 `deep_report_template` に沿っており hash chain 検証済みか
- [ ] Gate をエージェントが self-approve していないか（`self_sign_as_approver: never_allowed`）

## 14.14 ワーク

**W13-1**: ARIM 風合成階層データ（3 lab × 2〜3 instrument × 8 lot × 100 sample）を vol-02 のジェネレータを拡張して作成せよ。装置ごとの明るさ / コントラストにわざと差を持たせ、`capstone_hierarchy_detection` で lab / instrument / lot の per-level domain_gap が期待通り出ることを確認せよ。

**W13-2**: MatBERT または MoCo v3 で学習済みの encoder を使い、W13-1 のデータから `capstone_deep_feature_extraction` で深層特徴 + MC-Dropout uncertainty を抽出せよ。stop_ratio を計測し、5% を超えるサンプルの attribution マップを Ch10 の Skill で可視化せよ。

**W13-3**: W13-2 の特徴 + uncertainty を入力に、`capstone_hierarchical_bayes` で PyMC 階層 GLM を fit せよ。observation variance に deep uncertainty を加算する / しないの 2 パターンで posterior を比較し、加算した場合の lab-level effect の CI がどう変わるかを報告せよ。

**W13-4**: `capstone_three_gate_orchestrator` を実装し、Gate 1 / 2 / 3 の各パターン（trigger あり / なし / 却下）を再現するテストケースを 6 通り書け。`self_sign_as_approver` を試行した際に assert が発火することを確認せよ。

**W13-5**: 統合 Skill `hierarchical_deep_bayes` を最後まで走らせ、`capstone_integrated_provenance.hash_chain` を出力せよ。Ch4 Layer 1-3 → Ch7 Layer 4 → Ch11 fm_provenance → Ch9 uncertainty → Ch10 attribution → capstone 4 provenance が全て hash で繋がることを確認せよ。

## 14.15 参考資料

- Gelman, A., et al. (2013). Bayesian Data Analysis (3rd ed.), Chapters 5, 11 (階層モデル + partial pooling)
- McElreath, R. (2020). Statistical Rethinking (2nd ed.), Chapters 13-14 (multilevel models)
- Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? NeurIPS.
- Wilson, A. G., & Izmailov, P. (2020). Bayesian Deep Learning and a Probabilistic Perspective of Generalization. NeurIPS.
- 本書 全章（Ch4-12）— 特に Ch4（provenance 3 レイヤ）、Ch7（domain_gap_gate + fine-tune）、Ch8-9（uncertainty）、Ch10（deep_report_template + human_review）、Ch11（FM）、Ch12（SSL）
- vol-01 第6章（Human-in-the-loop 承認プロセス）
- vol-02 第11-13章（PyMC / MCMC 診断 / 階層モデル）、第14章（capstone: 合成階層データ + PyMC）
- 本書 付録A（Skill テンプレート集）
