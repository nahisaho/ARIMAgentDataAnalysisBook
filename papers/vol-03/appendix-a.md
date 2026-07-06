# 付録A　深層 × Agentic Skill テンプレート集

> **本付録の目的**：第4〜15章で作った深層 × Agentic Skill の**再利用可能な雛形**を、`SKILL.md` + `provenance schema` + `minimal example` の 3 点セットで提供します。vol-02 の provenance を **GPU / 深層 / Agentic 拡張**した完全スキーマも本付録末尾に置きます。

> [!IMPORTANT]
> テンプレートは**そのままコピーして使うためのもの**ではなく、**「何を書き、何を検証すべきか」を思い出すためのチェックリスト**です。実際のデータ・目的に合わせて調整してください。深層学習では学習コスト・GPU リソースが大きいため、テンプレの `④ 成功条件` `⑤ 禁止事項` を最初にレビューしてから走らせる運用を強く推奨します。

---

## A.1 本付録の使い方

| 目的 | 参照するテンプレート | 該当章 |
|---|---|---|
| CNN / ViT で画像分類 Skill を作る | A.2（画像分類） | 第7章 |
| 画像→物性値の回帰 Skill を作る | A.3（画像回帰） | 第7章・第8章 |
| 1D CNN / 1D Transformer でスペクトル分類 Skill を作る | A.4（スペクトル分類） | 第6章 |
| TabNet / FT-Transformer で表形式回帰 Skill を作る | A.5（表形式回帰） | 第6章・第13章 |
| Foundation Model 転移学習 Skill を作る | A.6（転移学習） | 第8章・第12章 |
| SSL（SimCLR / MAE 等）事前学習 Skill を作る | A.7（自己教師あり） | 第8章 |
| BNN / MC-Dropout / Deep Ensemble Skill を作る | A.8（不確かさ） | 第9章・第10章 |
| provenance schema の全項目を確認する | A.9（完全スキーマ） | 第4・11・13章 |

**テンプレの共通要素**（6 要素）は vol-01 §A.2.2 に準拠：①目的 / ②入力条件 / ③出力形式 / ④成功条件 / ⑤禁止事項（severity 付き）/ ⑥再現性条件。vol-03 では **⑦ Agentic 契約** を追加します（agent tier / 承認ゲート / 不確かさ停止条件）。

### A.1.1 Skill / リポジトリ配置規約（vol-01 A.2.1・vol-02 A.1.1 の継承 + vol-03 追加要素）

Skill ディレクトリ構造は **vol-01 §A.2.1・vol-02 §A.1.1 と同一**です。配置先も同じ 3 系統：

- **リポジトリ配布**：`.github/skills/<name>/`（Copilot CLI 自動検出、実運用の推奨）
- **個人利用**：`~/.copilot/skills/<name>/`
- **下書き**：`skills/<name>/`（本書の説明・演習用。完成後にどちらかへ移す）

vol-03 で新規に追加されるパスは **Skill ディレクトリの内側** の 2 ディレクトリです。

**リポジトリルート（Skill 横断で共有される資産）**

```text
<vol-03 repo root>/
├── data/
│   ├── synthetic-hierarchy/          ← vol-02 継承（第14章 capstone で再利用）
│   └── synthetic-deep/               ← ★vol-03 追加：深層向け合成データ（付録C）
│       ├── images/                   ← 画像サンプル（SEM 風、Raman 風）
│       ├── spectra/                  ← 1D スペクトル
│       └── metadata.parquet          ← ラベル・階層情報
├── scripts/
│   ├── generate_synthetic_hierarchy.py   ← vol-02 継承
│   └── generate_synthetic_deep.py        ← ★vol-03 追加
├── weights/                          ← ★vol-03 追加：ローカル weight cache
│   ├── foundation/                       ← FM の hash-verified copy（第12章）
│   └── fine_tuned/                       ← 派生モデル
└── .github/skills/                   ← 各 Skill はここに配置（vol-01 A.2.1）
    └── <skill-name>/
        └── ...
```

**Skill ディレクトリの内側（vol-03 で追加される 2 ディレクトリ）**

vol-02 の `artifacts/` / `diagnostics/` に加え、深層学習の中間物と GPU 実行環境情報の置き場を明示します。

```text
.github/skills/<skill-name>/
├── SKILL.md                          ← vol-01 §A.2.2（同一）
├── references/                       ← vol-01 §A.2.3〜（同一）
├── scripts/                          ← vol-01 §A.2（同一）
├── tests/                            ← vol-01 §A.2（同一）
├── examples/                         ← vol-01 §A.2（同一）
├── artifacts/                        ← vol-02 §A.1.1（同一、深層章では拡張）
│   ├── model_v<SemVer>.safetensors        ← 深層モデルは safetensors 推奨
│   ├── model_state_v<SemVer>.pt           ← state_dict のみ保存する場合
│   └── ...
├── diagnostics/                      ← vol-02 §A.1.1（同一）
├── checkpoints/                      ← ★vol-03 追加：学習中間 checkpoint
│   ├── epoch_<NNN>.safetensors            ← 各 epoch or best のみ保存
│   ├── checkpoint_registry.jsonl          ← append-only、Ch4 の overwrite policy
│   └── README.md                          ← 保持ポリシー・削除規則
└── wandb_run/                        ← ★vol-03 追加（optional、第16章 §16.10）
    ├── run_id.txt                         ← MLflow / W&B run_id の pin
    └── metrics_hash.txt                   ← tracking との改ざん検知
```

> [!TIP]
> `checkpoints/` は **append-only**（Ch4 `checkpoint_overwrite_policy: append_only`）が原則。「best」は上書きファイル（`best.safetensors`）ではなく**不変な checkpoint への署名付きポインタ**として `checkpoint_registry.jsonl` に記録する（例：`{"best_of_run_<id>": "epoch_042.safetensors", "signed_by": "..."}`）。ディスク節約が必要な場合でも最終 signed checkpoint と best pointer 対象の実体は残し、中間削除は `checkpoint_registry.jsonl` に append で記録する。**silent な上書きは AG-03（第15章）に該当**。

### A.1.2 テンプレの読み方

各テンプレは以下 3 セクションで構成：

1. **`SKILL.md` 雛形** — 6 + 1 要素をコピペ可能な形で提示
2. **最小コード骨格** — PyTorch / JAX / Hugging Face の実装スケルトン
3. **チェックリスト（PR 前）** — 提出前に埋めるべき項目

**書き方の共通ルール**：

- 目的行は 1 文で書く（`- 目的: ...`）
- 入力・出力条件は data contract（Ch4）を参照 (`- input_schema_ref: data-contract-<name>.yaml`)
- Agentic 契約は Ch4 §5.6-4.9 の tier / 承認 / 停止契約を Skill 側で pin
- Severity は `CRITICAL` / `HIGH` / `MEDIUM` / `LOW` の 4 段（Ch14 §15.3 準拠）

---

## A.2 画像分類 Skill テンプレート（第7章）

### A.2.1 `SKILL.md` 雛形

```yaml
skill_name: "image_classification_skill"
version: "0.1.0"
agent_tier_required: "L2"                    # 承認範囲内での fine-tune 可
purpose: >
  ARIM 画像データ（SEM / 光学顕微鏡 / TEM 等）から材料相・結晶構造・
  欠陥種別を分類する深層 Skill。ImageNet 事前学習 or 材料 FM 転移。

inputs:
  data_contract_ref: "data-contract-image-<domain>.yaml"
  minimum_samples_per_class: 30              # 少ないと augmentation 依存が過大
  class_balance_ratio_max: 20                # 不均衡 20:1 超は要 stratify + weighted loss

outputs:
  predicted_class: "int"
  class_probabilities: "list[float]"
  uncertainty_score: "float (Ch8 継承)"
  gradcam_reference: "URI (Ch10 継承)"

success_criteria:
  # 絶対値の閾値はタスク依存。baseline 相対 + per-class 最低の両面で評価
  macro_f1_baseline_relative_improvement_pct_min: 10   # 直近ベースライン比
  macro_f1_absolute_min: "task-defined (例: 0.70-0.85)"
  per_class_recall_min: 0.60                 # 全クラスで
  per_group_recall_min: 0.60                 # 施設 / 装置 / 撮像条件毎
  uncertainty_calibration_ece_max: 0.05      # Ch8

class_imbalance_handling:                    # Ch5 継承
  ratio_max_observed: "computed"
  class_weights_source: "computed_from_train | declared_in_skill"
  class_weights_hash: "sha256"
  loss_fn: "cross_entropy | weighted_ce | focal"
  sampler: "random | weighted | class_balanced"
  stratified_or_group_split: "stratified"
  minority_augmentation_policy: "policy_name or null"

forbidden:
  - action: "test_time_augmentation_beyond_declared_policy"
    severity: "HIGH"
    failure_pattern_ids: [DG-04, MX-02]
  - action: "class_rebalance_by_dropping_minority_at_inference"
    severity: "CRITICAL"
    failure_pattern_ids: [DG-06, AG-05]
  - action: "leakage_via_augmentation_over_split_boundary"
    severity: "CRITICAL"
    failure_pattern_ids: [DG-01, DG-08]
  - action: "silent_checkpoint_overwrite"
    severity: "HIGH"
    failure_pattern_ids: [AG-03]
  - action: "class_weights_recomputed_by_agent_without_approval"
    severity: "HIGH"
    failure_pattern_ids: [AG-05, DG-06]

reproducibility:
  gpu_backend_declared: true
  cudnn_deterministic: true
  random_seed_per_worker: "int"
  augmentation_policy_hash: "sha256"
  weights_sha256: "sha256"

agent_authorization:                         # Ch4 canonical
  level: "L2"                                # Ch4 canonical
  approved_hp_range:                         # Ch4 canonical
    lr: [1e-5, 1e-3]
    batch_size: [8, 128]
    epochs_max: 100
  training_job_approval:                     # Ch4 canonical (nested)
    required: true
    approval_record_id: "sha256"
    approver_role: "ml_lead | pi"
    approver_is_non_agent: true
  checkpoint_overwrite_policy: "append_only" # Ch4 canonical enum
  fm_update_gate_required_if_using_fm: true  # Ch11
  uncertainty_stop_threshold_ref: "Ch8 skill"
  gate_state_machine_ref: "Ch13 gate1_fine_tune_launch"
```

### A.2.2 最小コード骨格（PyTorch + timm）

```python
# scripts/train_image_classifier.py
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import timm
from torch.utils.data import DataLoader

def train(cfg: dict, data: dict, approval_record: dict) -> dict:
    """
    cfg: SKILL.md の approved_hp_range 内であることを事前検証済み
    approval_record: training_job_approval provenance (Ch4)
    """
    # 1. 決定論設定
    os.environ["PYTHONHASHSEED"] = str(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.deterministic = cfg["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = False

    # 2. モデル構築（timm から）
    model = timm.create_model(
        cfg["model_name"],
        pretrained=True,
        num_classes=cfg["num_classes"],
    )

    # 3. checkpoint dir（append-only）
    ckpt_dir = Path(cfg["output_dir"]) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    registry_path = ckpt_dir / "checkpoint_registry.jsonl"

    # 4. 学習ループ（省略）
    for epoch in range(cfg["epochs"]):
        # ... train + val
        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.safetensors"
        _save_safetensors(model, ckpt_path)
        _append_registry(registry_path, ckpt_path, epoch, val_metrics)

    # 5. provenance 出力（Ch4 canonical fields）
    return {
        "weights_uri": str(ckpt_path),
        "weights_sha256": _sha256(ckpt_path),
        "agent_authorization": {
            "level": cfg["agent_level"],
            "training_job_approval": {
                "required": True,
                "approval_record_id": approval_record["approval_record_id"],
            },
            "checkpoint_overwrite_policy": "append_only",
        },
        "gpu_backend": torch.version.cuda,
        "cudnn_deterministic": True,
        "seed": cfg["seed"],
    }
```

### A.2.3 チェックリスト（PR 前）

- [ ] `approved_hp_range` 内のハイパーパラメータで学習
- [ ] `training_job_approval.approval_record_id` が provenance にある
- [ ] `checkpoints/checkpoint_registry.jsonl` が append-only、best は不変 checkpoint への署名付き pointer
- [ ] `class_weights_source` / `class_weights_hash` を provenance に記録
- [ ] `augmentation_policy_hash` が学習・推論で一致
- [ ] Gate1 承認済み（Ch13）
- [ ] Baseline 相対改善・per-class recall・per-group recall・ECE すべて閾値クリア
- [ ] Grad-CAM で少なくとも 3 サンプル可視化（Ch10）

---

## A.3 画像回帰 Skill テンプレート（第7章・第8章）

### A.3.1 `SKILL.md` 雛形（差分のみ）

```yaml
skill_name: "image_regression_skill"
version: "0.1.0"
agent_tier_required: "L2"
purpose: >
  画像から連続値（粒径・膜厚・組成比・応力値等）を回帰する Skill。
  vol-02 の校正曲線と直列に接続可（校正済み画像特徴 → PyMC 回帰）。

outputs:
  predicted_value: "float"
  prediction_sigma: "float"                  # 予測分散、Ch8/9 継承
  uncertainty_source: "enum: aleatoric|epistemic|both"

success_criteria:
  r2_baseline_relative_improvement_min: 0.05  # ベースライン R^2 対比の絶対差
  r2_absolute_min: "task-defined (例: 0.70-0.85)"
  rmse_max_relative_to_range: 0.15           # 目標値レンジの 15% 以内
  per_group_rmse_max: "task-defined"         # 施設 / 装置別
  calibration_of_prediction_interval:
    coverage_90pct_min: 0.85
    coverage_90pct_max: 0.95                 # 過大 / 過小どちらも fail

forbidden:
  - action: "report_point_estimate_without_sigma"
    severity: "CRITICAL"
    failure_pattern_ids: [MX-01, AG-09]
  - action: "collapse_sigma_to_zero_for_pretty_plots"
    severity: "CRITICAL"
    failure_pattern_ids: [MX-01, AG-09]
  - action: "silent_stop_threshold_change"
    severity: "CRITICAL"
    failure_pattern_ids: [AG-07]

agent_authorization:                         # Ch4 canonical
  level: "L2"
  training_job_approval:
    required: true
    approval_record_id: "sha256"
  checkpoint_overwrite_policy: "append_only"
```

### A.3.2 最小コード骨格

```python
# 損失は Gaussian NLL で aleatoric uncertainty を捕捉
import math
import torch
import torch.nn.functional as F

def gaussian_nll_loss(pred_mu: torch.Tensor,
                      pred_log_sigma: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
    """
    pred_mu:        shape (B,) or (B, D) — 予測平均
    pred_log_sigma: shape (B,) or (B, D) — log σ を直接学習（σ>0 保証）
    target:         shape 同上
    """
    # log_sigma → sigma、下限クランプで数値安定化
    sigma = torch.exp(pred_log_sigma).clamp(min=1e-6)
    # 正規分布 NLL: 0.5 * [log(2π σ²) + (y-μ)² / σ²]
    loss = 0.5 * (torch.log(2.0 * math.pi * sigma**2)
                  + (target - pred_mu)**2 / sigma**2)
    return loss.mean()

# 代替：`torch.nn.GaussianNLLLoss` を使う場合（var を直接受け取る）
# criterion = torch.nn.GaussianNLLLoss(reduction="mean")
# loss = criterion(pred_mu, target, sigma.pow(2))
```

### A.3.3 チェックリスト（PR 前）

- [ ] `prediction_sigma` を常に出力（点推定のみは禁止）
- [ ] Coverage が 85-95% の範囲
- [ ] Aleatoric / Epistemic の分解を記録（Ch8）
- [ ] `uncertainty_stop_threshold` が provenance に pin、実行時変更なし
- [ ] 校正曲線と直列接続時は Ch13 chain slot に載せる

---

## A.4 スペクトル分類 Skill テンプレート（第6章）

### A.4.1 `SKILL.md` 雛形（差分のみ）

```yaml
skill_name: "spectrum_classification_skill"
version: "0.1.0"
agent_tier_required: "L2"
purpose: >
  Raman / IR / XPS 等の 1D スペクトルから物質種別を分類する Skill。
  1D CNN or 1D Transformer、Raman FM 転移も選択可（第8章）。

inputs:
  data_contract_ref: "data-contract-spectrum-<technique>.yaml"
  wavelength_alignment_required: true        # vol-01 継承
  baseline_correction_applied_in_provenance: true

success_criteria:
  macro_f1_baseline_relative_improvement_pct_min: 5
  macro_f1_absolute_min: "task-defined (例: 0.75-0.90)"
  per_instrument_f1_min: 0.70                # 装置間汎化（Ch5 継承）
  per_class_recall_min: 0.60

forbidden:
  - action: "train_test_split_by_random_shuffle_ignoring_instrument"
    severity: "CRITICAL"
    failure_pattern_ids: [DG-01, DG-03]
  - action: "augmentation_that_shifts_peak_positions"
    severity: "CRITICAL"
    failure_pattern_ids: [DG-08, MX-02]
  - action: "resample_to_uniform_wavenumber_after_augmentation"
    severity: "HIGH"
    failure_pattern_ids: [DG-06]

agentic_contract:
  agent_tier: "L2"
  approved_augmentation_ops:
    - "gaussian_noise (sigma <= 0.05 * intensity_range)"
    - "intensity_scale (factor in [0.9, 1.1])"
    - "baseline_shift (bounded, provenance-recorded)"
  forbidden_augmentation_ops:
    - "wavenumber_shift"                     # 化学的に別物質
    - "peak_deletion"
```

### A.4.2 最小コード骨格

```python
# 1D CNN の最小構造
import torch
import torch.nn as nn

class Spectrum1DCNN(nn.Module):
    def __init__(self, in_length: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, in_length)
        return self.classifier(self.features(x).squeeze(-1))
```

### A.4.3 チェックリスト（PR 前）

- [ ] Split が **装置ごと** に stratify されている（instrument leak 禁止）
- [ ] Augmentation で peak position を移動していない
- [ ] Wavelength alignment が学習・推論で同一
- [ ] Baseline correction hash が provenance に記録

---

## A.5 表形式回帰 Skill テンプレート（第6章・第13章）

### A.5.1 `SKILL.md` 雛形（差分のみ）

```yaml
skill_name: "tabular_regression_skill"
version: "0.1.0"
agent_tier_required: "L2"
purpose: >
  実験条件・組成・プロセスパラメータ等の表形式データから
  物性値を回帰する Skill。TabNet / FT-Transformer / GBM(vol-02) の比較を含む。

inputs:
  data_contract_ref: "data-contract-tabular-<domain>.yaml"
  categorical_encoding: "target_encoding|onehot|embedding"
  categorical_encoding_fit_on_train_only: true            # data leak 禁止

success_criteria:
  r2_baseline_relative_improvement_min: 0.05              # 直近ベースライン対比の絶対差
  r2_absolute_min: "task-defined (例: 0.65-0.80)"
  vs_gbm_baseline_improvement_pct_min: 5                   # GBM 比 5% 以上改善
  # 改善が 5% 未満なら深層を選ばず GBM を採用（第13章）

forbidden:
  - action: "target_encoding_fitted_on_full_dataset"
    severity: "CRITICAL"
    failure_pattern_ids: [DG-03]                       # data leakage
  - action: "adopt_deep_when_gbm_baseline_within_5pct"
    severity: "MEDIUM"
    failure_pattern_ids: [MX-04, ORG-06]
  - action: "silent_categorical_reencoding_at_inference"
    severity: "CRITICAL"
    failure_pattern_ids: [DG-06, AG-05]

agent_authorization:                                        # Ch4 canonical
  level: "L2"
  training_job_approval:
    required: true
  checkpoint_overwrite_policy: "append_only"
```

### A.5.2 チェックリスト（PR 前）

- [ ] GBM ベースラインと比較（vol-02 継承）、5% 未満改善なら深層を採用しない
- [ ] Target encoding が train のみで fit、encoding table を provenance に hash 記録
- [ ] Feature importance を SHAP / permutation で記録（Ch10）

---

## A.6 転移学習 Skill テンプレート（第8章・第12章）

### A.6.1 `SKILL.md` 雛形

```yaml
skill_name: "foundation_model_transfer_skill"
version: "0.1.0"
agent_tier_required: "L2"                              # L3 は施設承認委員会経由
purpose: >
  Materials FM（例：MatBERT, MolFormer, MatSciBERT。SEM-FM は本書で仮想例として登場、
  実運用時は実在する検証済みモデルに置き換える）を凍結 or 部分 fine-tune して下流タスクに転移する Skill。

inputs:
  foundation_model_ref: "hf_hub_repo_or_local_uri"
  revision_commit_hash: "40hex sha"                    # Ch11、必ず pin
  fm_update_gate_provenance: "id"

success_criteria:
  transfer_improvement_over_scratch:
    baseline_relative_improvement_pct_min: 15          # scratch 対 relative 改善
    per_class_or_per_group_min_recall: 0.60            # 少数クラス性能床
    domain_relevant_metric_min: "task-defined"         # e.g., MAE, R^2, macro-F1
  no_forgetting_on_original_domain:
    required_if_fine_tune: true
    check_metric: "task-defined pre/post drop <= 3%"

forbidden:
  - action: "load_fm_without_hash_verify"
    severity: "CRITICAL"
    failure_pattern_ids: [AG-08]                       # FM signature verification skip
  - action: "change_default_fm_version_without_gate"
    severity: "CRITICAL"
    failure_pattern_ids: [AG-02, ORG-04]
  - action: "self_sign_as_fm_update_gate_approver"
    severity: "CRITICAL"
    failure_pattern_ids: [AG-06]
  - action: "silent_short_sha_or_branch_revision"
    severity: "CRITICAL"
    failure_pattern_ids: [AG-08]

reproducibility:
  # Ch11 canonical field names — 別名で書き換えない
  foundation_model_provenance:
    used: true
    repo_id: "org/repo"                                # Ch11 canonical
    revision_commit_hash: "40-hex SHA"                 # Ch11 canonical
    hub_api_resolved_sha: "40-hex SHA — HfApi.model_info().sha と一致検証済み"
    fetched_at_utc: "ISO8601"
    declared_license: "from manifest (not live API)"   # Ch11 canonical
    pretraining_data_license: "e.g., cc-by-4.0"        # Ch11 canonical
    pretraining_data_summary: "short text"             # Ch11 canonical
    model_family: "matbert | crystallm | chemberta | other"  # Ch11 canonical
    safetensors_files_with_sha256:                     # Ch11 canonical, file-level
      - file: "model.safetensors"
        sha256: "..."
      - file: "tokenizer.json"
        sha256: "..."
    fetch_method: "snapshot_download"                  # Ch11 canonical
    upstream_license_compatibility_verified: true

agent_authorization:                                   # Ch4 canonical
  level: "L2"                                          # Ch4 canonical field name
  fine_tune_scope_declared:
    layers_frozen: []                                  # e.g., ["backbone.stages.0-2"]
    layers_trainable: []
    parameter_count_trainable: "int"
  scope_change_requires_new_gate1: true                # Ch13
```

### A.6.2 最小コード骨格（Hugging Face — Ch11 安全設計準拠）

> [!IMPORTANT]
> **`AutoModel.from_pretrained(revision=...)` 単独では不足**。理由：
> - Hub の revision 名は tag / branch / 短縮 SHA も受理してしまう
> - state_dict の tensor 列挙順・dtype・device 依存で hash が不安定
> - safetensors 以外の形式（pickle 等）だと deserialize でコード実行の恐れ
>
> 以下は Ch11 の `snapshot_download + approved manifest + file-level sha256` 方式。

```python
import hashlib
import re
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi

def load_fm_with_verify(repo_id: str, revision: str, expected_manifest: dict):
    """
    expected_manifest: 承認済み manifest（Ch11 §12.4）
      - revision_commit_hash: "40-hex SHA"
      - declared_license: "..."
      - pretraining_data_license: "..."
      - safetensors_files_with_sha256: [{file, sha256}, ...]
    """
    # 1. revision が 40-hex commit SHA であることを検証（tag/branch/短縮禁止）
    assert re.fullmatch(r"[0-9a-f]{40}", revision), \
        f"revision must be 40-hex commit sha, got: {revision}"
    assert revision == expected_manifest["revision_commit_hash"], \
        "revision mismatch with approved manifest"

    # 2. Hub API の resolved sha が revision と一致
    api = HfApi()
    model_info = api.model_info(repo_id=repo_id, revision=revision)
    assert model_info.sha == revision, \
        f"hub resolved sha ({model_info.sha}) != approved revision ({revision})"

    # 3. hub のライブライセンスと manifest ライセンスの一致
    hub_license = (getattr(model_info, "cardData", {}) or {}).get("license")
    if hub_license is not None:
        assert hub_license == expected_manifest["declared_license"], \
            f"license mismatch: hub={hub_license}, manifest={expected_manifest['declared_license']}"

    # 4. snapshot_download で allow_patterns を safetensors + tokenizer + config に限定
    local_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=["*.safetensors", "tokenizer.json", "tokenizer_config.json",
                        "config.json", "generation_config.json"],
    )

    # 5. file-level sha256 を manifest と 1:1 で照合
    for entry in expected_manifest["safetensors_files_with_sha256"]:
        p = Path(local_dir) / entry["file"]
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == entry["sha256"], \
            f"file hash mismatch for {entry['file']}: got {h}, expected {entry['sha256']}"

    # 6. 検証済みの local_dir から safetensors 経路のみでロード（pickle 経路禁止）
    from transformers import AutoModel
    model = AutoModel.from_pretrained(local_dir, use_safetensors=True)
    return model, local_dir
```

### A.6.3 チェックリスト（PR 前）

- [ ] `revision_commit_hash` が 40-hex（tag / branch / 短縮 SHA 一切禁止）
- [ ] `hub_api_resolved_sha == revision_commit_hash` の一致検証済み
- [ ] `snapshot_download` の `allow_patterns` で safetensors + tokenizer/config のみ許可
- [ ] `safetensors_files_with_sha256` が manifest と 1:1 一致
- [ ] `declared_license` が manifest ソース（live API ではない）
- [ ] `pretraining_data_license` と下流 license が互換、`commercial_use_allowed` 判定と一致
- [ ] `fm_update_gate_provenance` が存在、approver が non-agent
- [ ] Fine-tune scope（frozen / trainable 層）を provenance に記録
- [ ] Scratch ベースラインに対し `baseline_relative_improvement_pct_min` を満たす

---

## A.7 自己教師あり事前学習 Skill テンプレート（第8章）

### A.7.1 `SKILL.md` 雛形（差分のみ）

```yaml
skill_name: "ssl_pretraining_skill"
version: "0.1.0"
agent_tier_required: "L3"                              # 長時間学習、承認委員会
purpose: >
  ラベルなし材料データで SimCLR / MoCo / MAE 等の SSL を行い、
  下流 fine-tune 用の表現を作る Skill。

inputs:
  unlabeled_data_size_min: 10000                       # SSL は大量データ前提
  labeled_downstream_data_ref: "for probe evaluation"

success_criteria:
  linear_probe_evaluation:
    baseline_relative_improvement_pct_min: 15          # random init 対比
    absolute_accuracy_or_metric_min: "task-defined"    # 絶対値の閾値はタスク毎に定義
    per_class_or_per_group_min: "task-defined"
  downstream_finetune_beats_scratch:
    baseline_relative_improvement_pct_min: 10

forbidden:
  - action: "leakage_of_downstream_labels_into_ssl_augmentation"
    severity: "CRITICAL"
    failure_pattern_ids: [DG-05, MX-03]
  - action: "reuse_ssl_representation_for_new_downstream_without_probe_eval"
    severity: "HIGH"
    failure_pattern_ids: [MX-04]
  - action: "use_pretrained_backbone_without_ssl_augmentation_policy_hash"
    severity: "HIGH"
    failure_pattern_ids: [DG-08]

agent_authorization:                                   # Ch4 canonical
  level: "L3"
  gpu_hours_estimated: "int"                           # 事前見積り、施設承認
  early_stop_on_probe_plateau: true                    # 探索の萎縮回避
  ssl_augmentation_policy_hash: "sha256"
```

### A.7.2 チェックリスト（PR 前）

- [ ] 事前学習と下流評価で **データが完全分離**
- [ ] Linear probe で表現品質を評価
- [ ] GPU quota snapshot が承認済み
- [ ] SSL augmentation が「下流に必要な変動」を保持している

---

## A.8 不確かさ推定 Skill テンプレート（第9章・第10章）

### A.8.1 `SKILL.md` 雛形

```yaml
skill_name: "uncertainty_estimation_skill"
version: "0.1.0"
agent_tier_required: "L2"
purpose: >
  Deep Ensemble / MC-Dropout / BNN (variational) で予測不確かさを
  推定し、Aleatoric / Epistemic に分解する Skill。

variant:
  choose_one:
    - "deep_ensemble (N >= 5)"
    - "mc_dropout (T >= 20)"
    - "variational_bnn"

outputs:
  prediction_mu: "float"
  prediction_sigma_aleatoric: "float"
  prediction_sigma_epistemic: "float"
  prediction_sigma_total: "float"

success_criteria:
  ece_calibration_max: 0.05                            # expected calibration error
  reliability_diagram_recorded: true
  coverage_at_90pct_within: [0.85, 0.95]

forbidden:
  - action: "report_variance_as_sigma_to_pm_normal"    # DG-08、単位混同
    severity: "CRITICAL"
  - action: "hide_epistemic_when_ood_detected"         # 「知らない」を隠す
    severity: "CRITICAL"
  - action: "collapse_ensemble_to_mean_only"
    severity: "HIGH"
forbidden:
  - action: "report_variance_as_sigma_to_pm_normal"
    severity: "CRITICAL"
    failure_pattern_ids: [DG-08, MX-01]
  - action: "hide_epistemic_when_ood_detected"
    severity: "CRITICAL"
    failure_pattern_ids: [AG-09, MX-01]
  - action: "collapse_ensemble_to_mean_only"
    severity: "HIGH"
    failure_pattern_ids: [MX-01]
  - action: "silent_change_of_stop_threshold"
    severity: "CRITICAL"
    failure_pattern_ids: [AG-07]

agent_authorization:                                   # Ch4 canonical
  level: "L2"
  autonomous_stop_threshold_defined_by: "human_analyst"    # 閾値は人が決める
  autonomous_stop_action: "route_to_human_gate2"       # Ch13
  threshold_change_requires_gate3: true                # Ch13
  training_job_approval:
    required: true
  checkpoint_overwrite_policy: "append_only"
```

### A.8.2 最小コード骨格（Deep Ensemble — 回帰版）

> [!NOTE]
> Deep Ensemble の epistemic uncertainty は **予測分布の標準偏差**として測る。**分類タスクでは logits に `.std(dim=0)` は不適切**（softmax 前の値のばらつきは意味を持たない）。分類では **平均 softmax エントロピー** と **predictive mutual information** を epistemic の指標とする。

```python
import torch
import torch.nn as nn

class DeepEnsemble:
    """回帰タスク向け Deep Ensemble。"""
    def __init__(self, model_class, n_members: int = 5, **kwargs):
        assert n_members >= 5, "Deep Ensemble は N>=5 推奨"
        self.members = [model_class(**kwargs) for _ in range(n_members)]

    def predict(self, x: torch.Tensor):
        # 各 member の予測 (回帰値) を (N_members, B, D) で積む
        with torch.no_grad():
            preds = torch.stack([m(x) for m in self.members], dim=0)
        mu = preds.mean(dim=0)
        sigma_epistemic = preds.std(dim=0)             # 標準偏差 (分散ではない)
        return mu, sigma_epistemic


# 分類版は softmax 確率のばらつきを取る
def classification_ensemble_predict(members, x):
    with torch.no_grad():
        probs = torch.stack([torch.softmax(m(x), dim=-1) for m in members], dim=0)
    mean_probs = probs.mean(dim=0)
    # 予測エントロピー (total uncertainty)
    predictive_entropy = -(mean_probs * mean_probs.clamp(min=1e-12).log()).sum(-1)
    # 個別 member のエントロピー平均 (aleatoric)
    per_member_entropy = -(probs * probs.clamp(min=1e-12).log()).sum(-1)
    expected_entropy = per_member_entropy.mean(dim=0)
    # Mutual information (epistemic)
    mutual_information = predictive_entropy - expected_entropy
    return mean_probs, predictive_entropy, mutual_information
```

### A.8.3 チェックリスト（PR 前）

- [ ] `sigma` は **標準偏差**であり分散ではない（DG-08 対策）
- [ ] 分類タスクでは logits の std ではなく **mutual information** で epistemic を測る
- [ ] ECE が 0.05 以下
- [ ] Reliability diagram を artifact に保存
- [ ] Aleatoric / Epistemic が別々に出力
- [ ] 停止閾値が Human 定義、`threshold_provenance_hash` が実行時変更を検知できる
- [ ] 停止閾値変更は Gate3 必須（Ch13）

---

## A.9 深層 × Agentic 拡張 provenance 完全スキーマ

以下は vol-01 §A.4・vol-02 §A.7 の provenance schema を **深層 × Agentic 拡張**した完全版です。すべての深層 Skill はこのスキーマの subset を生成することが期待されます。

> [!IMPORTANT]
> **重複回避原則**：vol-02 §A.7 で定義済みのフィールド（`skill_version`, `input_sha`, `random_seed`, `package_versions`, `output_artifacts.model_sha256` 等）は本 schema では**再定義しない**。vol-03 は「深層特有の追加」のみを定義する。canonical source が競合する場合の解決は下記 mapping 表を参照：
>
> | vol-02 canonical | vol-03 で拡張する場合 | 解決 |
> |---|---|---|
> | `output_artifacts.model_sha256` | `weights_sha256` | 同一値、vol-02 命名を優先 |
> | `package_versions.python` | `framework_versions.pytorch` 等 | vol-03 は framework 詳細のみ追加 |
> | `random_seed` | `random_seed_per_worker`, `numpy_seed`, `python_hash_seed` | vol-03 は per-worker と補助 seed のみ追加 |
> | `data_contract_ref` | `augmentation_provenance` | vol-03 は augmentation の provenance を分離追加 |

```yaml
# vol-03 deep_agentic_provenance_schema.yaml
schema_version: "vol-03/1.0"
extends: "vol-02/A.7"                                  # vol-02 fields inherited, not repeated

# ============================================================
# Section 1: Reproducibility (deep-specific additions)
# vol-02 は python / package_versions / random_seed をカバー済み。
# 以下は GPU 実行環境と framework 詳細のみ追加。
# ============================================================
gpu_backend:
  vendor: "nvidia | amd | apple | cpu_only"
  driver_version: "string"
  cuda_or_rocm_version: "string"
  device_model: "e.g., A100-80GB"
  device_count: "int"
cudnn_deterministic: "bool"
cudnn_benchmark: "bool"                                # false 推奨
random_seed_per_worker: "int (torch DataLoader workers)"
numpy_seed: "int"                                      # vol-02 random_seed とは別種
python_hash_seed: "int"                                # PYTHONHASHSEED
framework_versions:
  pytorch: "string or null"
  jax: "string or null"
  transformers: "string or null"
  timm: "string or null"

# ============================================================
# Section 2: Data & augmentation (Ch5)
# ============================================================
input_shape: "list[int]"
augmentation_provenance:
  policy_name: "string"
  policy_hash: "sha256"                                # 学習・推論で一致
  augmentation_applied_only_on_train: true             # Ch5
  test_time_augmentation:
    enabled: "bool"
    declared_ops: "list"
    forbidden_ops_verified: true
class_imbalance_handling:
  ratio_max_observed: "float"
  class_weights_source: "declared_in_skill | computed_from_train | null"
  class_weights_hash: "sha256 or null"
  loss_fn: "cross_entropy | weighted_ce | focal | null"
  sampler: "random | weighted | class_balanced | null"
  stratified_or_group_split: "stratified | group | random"
  minority_augmentation_policy: "string or null"

# ============================================================
# Section 3: Model & weights
# ============================================================
model_architecture:
  family: "cnn | vit | transformer | tabnet | ft_transformer | 1d_cnn | bnn | ensemble | seg | detection"
  name: "e.g., resnet50, vit_base_patch16_224"
  parameter_count: "int"
  parameter_count_trainable: "int"
weights_uri: "content-addressable URI"
weights_sha256: "sha256"                               # vol-02 output_artifacts.model_sha256 と同一値
weights_format: "safetensors | pt_state_dict"          # safetensors 推奨

# --- Foundation Model provenance (Ch11 canonical field names) ---
foundation_model_provenance:
  used: "bool"
  repo_id: "org/repo (Ch11 canonical)"
  revision_commit_hash: "40-hex SHA (Ch11 canonical)"
  hub_api_resolved_sha: "40-hex SHA — HfApi.model_info().sha == revision の検証済み値 (Ch11)"
  fetched_at_utc: "ISO8601"
  declared_license: "from manifest, not live API (Ch11)"
  pretraining_data_license: "e.g., cc-by-4.0 (Ch11 canonical)"
  pretraining_data_summary: "short text (Ch11)"
  model_family: "matbert | crystallm | chemberta | other (Ch11)"
  # 正本：weights + tokenizer + config + generation_config を kind タグ付きで統一（付録B B.4.4）
  files_with_sha256: "list[{file, kind, sha256}]  (Ch11 canonical, kind ∈ {weights, tokenizer, config, generation_config})"
  # 後方互換の派生ビュー：files_with_sha256 の kind='weights' サブセット
  safetensors_files_with_sha256: "derived from files_with_sha256 where kind='weights'"
  fetch_method: "snapshot_download | direct (snapshot_download 推奨 Ch11)"
  upstream_license_compatibility_verified: "bool"

# ============================================================
# Section 4: Training config
# ============================================================
finetune_config:
  layers_frozen: "list of paths"
  layers_trainable: "list of paths"
  lr_schedule: "string"
  batch_size: "int"
  epochs: "int"
  optimizer: "e.g., adamw"
  loss: "string"
  mixed_precision: "bool"
  gradient_clipping: "float or null"
pretraining_config:                                    # SSL 用（finetune_config とは別）
  method: "simclr | moco | mae | null"
  contrastive_temperature: "float or null"
  masking_ratio: "float or null"
  effective_batch_size: "int"
  epochs: "int"
  ssl_augmentation_policy_hash: "sha256"

# ============================================================
# Section 5: Uncertainty (Ch8-9)
# ============================================================
uncertainty:
  method: "deep_ensemble | mc_dropout | vi_bnn | none"
  ensemble_size: "int or null"
  dropout_samples: "int or null"
  aleatoric_captured: "bool"
  epistemic_captured: "bool"
  ece_measured: "float"
  reliability_diagram_uri: "URI or null"
  classification_uncertainty_metrics:                  # 分類の場合
    predictive_entropy_recorded: "bool"
    mutual_information_recorded: "bool"
  regression_uncertainty_metrics:                      # 回帰の場合
    coverage_at_90pct: "float"
    prediction_interval_calibration_pass: "bool"

# ============================================================
# Section 6: Agent authorization (Ch4 canonical field names)
# ============================================================
agent_authorization:
  level: "L1 | L2 | L3"                                # Ch4 canonical (agent_tier ではない)
  approved_hp_range: "dict (Ch4 canonical, inline or ref)"
  hp_range_exceedance_detected: "bool"
  on_exceedance_action_taken: "block | route_to_L3_approval | null"
  training_job_approval:                               # Ch4 canonical: nested under agent_authorization
    required: "bool"
    approval_record_id: "sha256"                       # Ch4 canonical
    approver_identity: "opaque_id_or_pubkey"
    approver_role: "e.g., pi, ml_lead, facility_committee"
    approver_is_non_agent: "bool"                      # true 必須
    approved_at_utc: "ISO8601"
    approval_scope: "individual | batched"
    approved_hp_range_snapshot: "dict"                 # Ch4 canonical
  checkpoint_overwrite_policy: "append_only | overwrite_with_approval"    # Ch4 canonical string
  checkpoint_registry_uri: "path to checkpoint_registry.jsonl"
  checkpoint_silent_overwrite_detected: false

# ============================================================
# Section 7: FM update gate (Ch11)
# ============================================================
fm_update_gate:
  applied: "bool"
  gate_id: "sha256 or null"
  approver_role: "e.g., project_lead + facility"
  approver_is_non_agent: "bool"
  hub_api_resolved_sha_at_approval: "40-hex"

# ============================================================
# Section 8: Uncertainty stop (Ch8-9-13)
# ============================================================
uncertainty_stop_threshold:
  threshold_value: "float"
  threshold_defined_by: "role (statistician|analyst|pi)"
  threshold_provenance_hash: "sha256"                  # 実行時変更検知
  autonomous_stop_triggered: "bool"
  stop_routed_to_human_gate2: "bool"

# ============================================================
# Section 9: Gate state machine (Ch13 canonical)
# ============================================================
integrated_provenance_chain_ref: "content-addressable URI (Ch13)"
gate_state_machine:
  gate_definitions:                                    # Ch13 canonical
    gate1_fine_tune_launch:
      state: "pending | approved | rejected | expired | disputed"
      required_roles: "any_of: [ml_lead, pi]"
      approval_record_id: "sha256 or null"
      payload_hash: "sha256"                           # 承認対象内容の hash
      approver_identity: "opaque_id_or_pubkey"
      approver_is_non_agent: true
      expiry_at_utc: "ISO8601 or null"
    gate2_post_stop_disposition:
      state: "pending | approved | rejected | expired"
      required_roles: "statistician AND pi"
      approval_record_id: "sha256 or null"
      payload_hash: "sha256"
      disposition_type: "resume | additional_measurement | threshold_review | sample_exclusion"
    gate3_diagnostic_relaxation_or_pooling_change:
      state: "pending | approved | rejected | expired"
      required_roles: "statistician AND pi"
      reviewers_min: 2                                 # Ch13 canonical
      approval_record_id: "sha256 or null"
      payload_hash: "sha256"
      exception_scope: "diagnostic_relaxation | pooling_structure_change"

# ============================================================
# Section 10: Audit (Ch14 canonical, expanded)
# ============================================================
audit_result_provenance:
  audit_id: "sha256"                                   # Ch14 §15.8
  registry_hash_at_run_start: "sha256"                 # Ch14 registry version
  registry_version_at_run_start: "e.g., v2.0"
  effective_registry_hash_from_chain: "sha256"         # Ch14 registry_versioning
  decision: "pass | warn | block"
  blocking_decision_signature: "signature over decision field (Ch14)"
  cross_reference_findings: "list of finding_id + status (Ch14 §15.8)"
  non_agent_verifier_signatures:                       # Ch14 canonical
    verifiers: "list of {pubkey, signature, timestamp}"
    quorum_size_for_pass: 1                            # 最低 1
    quorum_size_for_block_exception: 2                 # blocking 例外は 2 以上
  runtime_and_periodic_audit_guards_ref:               # Ch14 §15.8 canonical
    inference_time_guard_active: "bool"
    drift_guard_active: "bool"
    rollback_guard_active: "bool"
    continual_learning_guard_active: "bool"
  external_ledger_matches:                             # Ch14 canonical
    scheduler_ledger_match: "bool"
    inference_ledger_match: "bool"
    model_registry_ledger_match: "bool"

# ============================================================
# Section 11: Distribution & Ledger (Ch15 canonical)
# ============================================================
model_distribution_contract_ref:                       # Ch15 canonical
  bundle_id: "sha256 or null"
  usage_restrictions_declared: "bool"
  originating_org_trust_bundle:                        # Ch15 canonical, required
    non_agent_verifier_pubkeys: "list"
    registry_hash_at_distribution: "sha256"
    audit_report_signature_chain: "dict"
    transparency_log_inclusion_proofs: "list"
  license:
    weights_license: "e.g., CC-BY-NC-4.0"
    commercial_use_allowed: "yes | no | conditional"   # Ch15 canonical
    downstream_distribution_allowed: "yes | no | conditional"
    patent_sensitive_until_date: "ISO8601 or null"
    export_control_review_required: "bool"
  revocation:
    check_endpoint: "url or null"
    on_revocation_check_failure:
      security_scope: "fail_closed"                    # Ch15 canonical
      non_security_scope: "grace_period_with_cached_last_good"
  non_skill_recipient_profile:
    model_card_included: "bool"
    signed_manifest_verifiable_without_full_skill: "bool"
    manual_checklist_included: "bool"
  hard_expiry_date: "ISO8601 or null"

responsibility_ledger_entry:                           # Ch15 canonical, split projection
  canonical_projection_hash: "sha256"                  # deterministic 再生成対象
  attached_metadata_ref: "content-addressable URI"     # 別保管
  layer_1_entry_signature: "signature by actor"
  layer_2_batch_signature_ref: "monthly Merkle root snapshot ref (Ch15 terminal signer)"
```

### A.9.1 Skill 種別 × 必須項目マトリクス

> [!NOTE]
> 「✅」= 常に必須。「if_*」= 条件付き必須（該当条件が成立する場合のみ必須）。「任意」= 任意記載。「–」= 対象外。
>
> **条件列の意味**：
> - **if_using_fm** — Foundation Model backbone を使う場合（SSL/uncertainty でも FM を使えば必須）
> - **if_training** — 学習ジョブを起動する場合（推論専用ならスキップ可）
> - **if_distributed** — 配布・共有する場合
> - **if_L3** — agent tier L3（scope 外自律 + Human 承認）で運用する場合

| 項目 | 画像分類 A.2 | 画像回帰 A.3 | スペクトル A.4 | 表形式 A.5 | 転移学習 A.6 | SSL A.7 | 不確かさ A.8 | 条件 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `gpu_backend` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 常時 |
| `cudnn_deterministic` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 常時 |
| `random_seed_per_worker` | ✅ | ✅ | ✅ | – | ✅ | ✅ | ✅ | if_training |
| `augmentation_provenance` | ✅ | ✅ | ✅ | – | ✅ | ✅ | ✅ | 学習/推論で aug 使う場合 |
| `class_imbalance_handling` | ✅ | – | ✅ | ✅ | if_classification | – | if_classification | 分類タスク |
| `weights_sha256` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 常時 |
| `foundation_model_provenance` | if_using_fm | if_using_fm | if_using_fm | – | ✅ | if_using_fm | if_using_fm | FM 使う場合 |
| `finetune_config` | if_training | if_training | if_training | – | ✅ | – | if_training | 学習時 |
| `pretraining_config` | – | – | – | – | – | ✅ | – | SSL 事前学習 |
| `uncertainty` | 任意 | ✅ | 任意 | 任意 | 任意 | – | ✅ | 不確かさ出す場合 |
| `agent_authorization.level` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 常時 |
| `agent_authorization.training_job_approval` | if_training | if_training | if_training | if_training | ✅ | ✅ | if_training | 学習ジョブ起動時 |
| `agent_authorization.checkpoint_overwrite_policy` | if_training | if_training | if_training | if_training | ✅ | ✅ | if_training | 学習時 |
| `fm_update_gate` | – | – | – | – | ✅ | if_using_fm | – | FM 更新時 |
| `uncertainty_stop_threshold` | 任意 | ✅ | 任意 | 任意 | 任意 | – | ✅ | 自律停止使う場合 |
| `gate_state_machine.gate1` | if_training | if_training | if_training | if_training | ✅ | ✅ | if_training | fine-tune 起動 |
| `gate_state_machine.gate2` | if_autonomy_stop | ✅ | if_autonomy_stop | if_autonomy_stop | if_autonomy_stop | – | ✅ | 自律停止発火時 |
| `gate_state_machine.gate3` | if_L3 | if_L3 | if_L3 | if_L3 | if_L3 | if_L3 | if_L3 | 診断緩和・pool 変更時 |
| `integrated_provenance_chain_ref` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Ch13 |
| `audit_result_provenance` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Ch14 |
| `model_distribution_contract_ref` | if_distributed | if_distributed | if_distributed | if_distributed | if_distributed | if_distributed | if_distributed | Ch15 |
| `responsibility_ledger_entry` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Ch15 |

---

## A.10 テンプレを新規 Skill に適用する手順

1. **目的 1 文で書く** — テンプレの `purpose` を書き換える
2. **agent tier を決める** — L1/L2/L3、迷ったら低い方（L2）から
3. **data_contract_ref を作る or 参照** — vol-01/02 の contract に準拠
4. **成功条件を数値で書く** — 「良さそう」ではなく閾値で。単独の絶対値だけでなく **baseline 相対改善**と **per-class / per-group 最低性能**の両方を書く
5. **禁止事項を severity 付きで列挙** — CRITICAL は必ず具体的に。各項に `failure_pattern_ids: [DG-xx, AG-xx, MX-xx, ORG-xx]` を紐付け（Ch14 registry）
6. **Agentic 契約を pin** — Ch4 canonical field 名で書く（`agent_authorization.level`, `training_job_approval.approval_record_id`, `checkpoint_overwrite_policy: append_only`）
7. **Ch13 chain context を準備** — `integrated_provenance_chain_ref` に紐づく上流 provenance（data / augmentation / weights / FM / gate1-3）が chain として揃うことを確認
8. **Ch14 registry version/hash を pin** — `registry_version_at_run_start` と `registry_hash_at_run_start` を Skill 実行開始時にキャプチャ
9. **外部 append-only ledger 参照を用意** — scheduler / inference / model-registry ledger の突合対象を Skill が読み出せる形にする
10. **cross_reference_checks を宣言** — audit と ledger の突合、gate と inference の突合など Ch14 §15.8 の 4 種類を Skill 契約に列挙
11. **provenance schema の subset を必須項目マトリクスで確認**
12. **PR 前チェックリストを埋める**
13. **Ch14 audit を pre-merge で 1 回走らせる**（`agentic_deep_failure_audit`）— `non_agent_verifier >= 1`、blocking 例外時は `>= 2`

## A.11 まとめ

- 深層 × Agentic Skill は「モデルコード」よりも「契約・provenance・承認・停止」の言語化が本体
- テンプレは 6 + 1（Agentic 契約）要素で構成
- `augmentation_provenance` `foundation_model_provenance` `training_job_approval` `uncertainty` `audit_result_provenance` は事実上ほぼ全 Skill で必須
- **Ch4/Ch11/Ch13/Ch14/Ch15 の canonical field 名**で書くことが最重要（別名を作らない）
- Ch14 audit をテンプレ完成後に必ず 1 回走らせる

### A.11.1 未収録テンプレ（今後追加候補）

本付録は代表的な 7 テンプレのみを収録している。以下は ARIM データ・共同利用施設で頻出だが本版では未収録：

| 未収録テンプレ | 適用場面 | 追加検討 |
|---|---|---|
| **SEM / TEM セグメンテーション** | 粒子・欠陥・粒界の pixel-wise 分類 | vol-03 v1.1 予定 |
| **物体検出（object detection）** | 触媒粒子・欠陥・回折スポット検出 | vol-03 v1.1 予定 |
| **時系列予測（クロマトグラム）** | HPLC/GC ピーク予測、劣化予測 | vol-03 v1.2 予定 |
| **画像 + 表形式 マルチモーダル** | 顕微鏡像 + 合成条件からの物性予測 | vol-03 v1.2 予定 |
| **Rietveld ハイブリッド（古典解析 + 深層）** | XRD で古典 Rietveld と深層事前確率の組合せ | vol-04 予定 |
| **拡散モデル・生成モデル** | 逆設計（inverse design）、条件付き結晶生成 | vol-04 予定 |

これらを追加する場合、本付録 A.9 の schema を extend し、A.9.1 の必須項目マトリクスに列を追加する。

## 参考資料

### 本書内の該当章

- 第5章：Skill 契約 6 要素、agent tier、approved_hp_range、training_job_approval
- 第6章：Augmentation contract、augmentation_provenance
- 第6-7章：CNN/ViT/1D CNN 実装、SSL、転移学習
- 第8-9章：Deep Ensemble / MC-Dropout / BNN、不確かさ停止
- 第11章：Grad-CAM、feature attribution、diagnostics
- 第12章：Foundation Model provenance、fm_update_gate
- 第13章：checkpoint registry、GBM vs 深層の比較
- 第14章：Gate1/2/3、integrated_provenance_chain
- 第15章：failure_pattern_registry、agentic_deep_failure_audit、audit_id
- 第16章：organization_gpu_resource_policy、model_distribution_contract、responsibility_ledger

### 外部参考

- Paszke et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.
- Wightman. timm: PyTorch Image Models. https://github.com/rwightman/pytorch-image-models
- Wolf et al. (2020). Transformers: State-of-the-Art Natural Language Processing. EMNLP.
- Lakshminarayanan et al. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS.
- Gal & Ghahramani (2016). Dropout as a Bayesian Approximation. ICML.
- Blundell et al. (2015). Weight Uncertainty in Neural Networks. ICML.
- He et al. (2022). Masked Autoencoders Are Scalable Vision Learners. CVPR.
- Chen et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML.
- Arik & Pfister (2021). TabNet: Attentive Interpretable Tabular Learning. AAAI.
- Sigstore project: https://www.sigstore.dev/（provenance signing 参考）
