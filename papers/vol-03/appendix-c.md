# 付録C　GPU トラブルシューティングと深層 × Agentic 演習データ候補

> **本付録の目的**：本書の演習を **手元の GPU / CPU 環境で実際に走らせる**ために必要な運用知識を集約します。前半（§C.1〜§C.5）は GPU トラブルシューティング、後半（§C.6〜§C.8）は演習に使えるデータ候補——**ARIM 風合成データ生成スクリプト**と、**公開ベンチマークを「エージェントに触らせてよいか」の観点で選ぶガイド**——を扱います。

> [!IMPORTANT]
> 深層学習の演習は「動くこと」よりも「**動いた記録が provenance として再現できること**」の方が本書では重要です。したがって本付録のトラブルシューティングは、単にエラーを消すのではなく、**そのエラーが provenance / audit に何を追記させるか**をセットで説明します。付録A（テンプレート）・付録B（チートシート）と併用してください。

---

## C.1 本付録の使い方

| こんなときは | 参照節 | 関連章 |
|---|---|---|
| `CUDA out of memory` が出る | §C.2.1〜C.2.4 | 第5章・第7章 |
| 「同じ seed なのに結果が違う」 | §C.3 | 第5章・第15章 DG-01 |
| CI で GPU 前提コードが落ちる | §C.4 | 第5章（CPU fallback） |
| Mixed precision で NaN が出る | §C.5.1 | 第7章・第9章 |
| DataLoader が遅い / hang する | §C.5.2 | 付録B B.2.4 |
| Hugging Face Hub 認証で 401/403 | §C.5.4 | 第12章・付録B B.4 |
| ARIM 実データにアクセスできない | §C.6 | 第3章・第5章 |
| 公開ベンチマークを使いたい | §C.7〜C.8 | 第8章・第14章 |

> [!TIP]
> **§C.2〜§C.5 のすべてのトラブルシューティングは、「まず provenance / audit ledger に "何が起きたか" を残す」がゴール**です。エラーを消すことがゴールではありません。第15章 §15.5 の failure_pattern_registry と対応させて読んでください。

---

## C.2 GPU メモリ関連

深層学習演習で最も頻出するのが `CUDA out of memory`（OOM）です。原因は 4 種類に分類できます。

### C.2.1 OOM の 4 分類

| 分類 | 症状 | 典型原因 | 対処節 |
|---|---|---|---|
| **B: バッチ過大** | 学習開始直後に OOM | `batch_size` が VRAM に対して大きすぎる | §C.2.2 |
| **A: activation 蓄積** | forward 中盤で OOM | 中間 tensor が保持されている | §C.2.2 |
| **G: gradient 蓄積** | backward で OOM | `optimizer.zero_grad()` 忘れ、`retain_graph=True` | §C.2.3 |
| **F: 断片化（fragmentation）** | 「reserved but unallocated」が大きい | 動的サイズ変化、`torch.cuda.empty_cache()` 濫用 | §C.2.4 |

診断は必ず**この順**で行います：**B → A → G → F**。

```python
# scripts/gpu_diag/oom_probe.py
import torch

def dump_memory_summary(tag: str) -> None:
    """OOM 発生前後で呼び、reserved / allocated / max_reserved を audit に残す。"""
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA unavailable"); return
    stats = {
        "allocated_MB":       torch.cuda.memory_allocated()    / 1024**2,
        "reserved_MB":        torch.cuda.memory_reserved()     / 1024**2,
        "max_allocated_MB":   torch.cuda.max_memory_allocated()/ 1024**2,
        "max_reserved_MB":    torch.cuda.max_memory_reserved() / 1024**2,
    }
    frag = stats["reserved_MB"] - stats["allocated_MB"]
    stats["fragmentation_MB"] = frag
    print(f"[{tag}] " + " ".join(f"{k}={v:.1f}" for k, v in stats.items()))
    # 監査に残す（第15章 audit ledger）
    return stats
```

> [!WARNING]
> **`torch.cuda.empty_cache()` は学習の inner loop では呼ばない**。fragmentation を一時的に緩和しますが、PyTorch のアロケータがキャッシュを再構築するオーバーヘッドで学習が遅くなる/不安定になります。**呼んでよい状況**：validation 終了後 / 巨大モデル切替時 / 他プロセスに VRAM を返却する必要がある時 / `memory_stats()` で fragmentation が有意と診断された時。詳細は §C.2.4。

### C.2.2 分類 B / A：バッチ過大・activation 蓄積

**最初に試すこと**：`batch_size` を半分にし、`gradient_accumulation_steps` を倍にする。**有効バッチサイズ**（`batch_size × grad_accum × world_size`）は不変なので学習ダイナミクスは概ね変わりません（BatchNorm 使用時は例外——§C.2.5）。

```python
# 有効バッチサイズを維持しながら VRAM を半減
effective_batch = 64
micro_batch     = 16                  # VRAM に収まる値
grad_accum      = effective_batch // micro_batch   # = 4

for step, batch in enumerate(loader):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(batch).loss / grad_accum
    loss.backward()
    if (step + 1) % grad_accum == 0:
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
```

**activation 蓄積対策（分類 A）**：

1. `torch.utils.checkpoint.checkpoint` で activation を再計算（メモリ ↓ / 計算 ↑ 30〜40%）
2. `set_to_none=True` を必ず付ける（`zero_grad()` 単独より軽量）
3. ループ外で参照が残っていないか確認（`losses.append(loss)` は `.item()` してから）

> [!NOTE]
> **`losses.append(loss)` は分類 A の典型的な原因**です。`loss` は autograd グラフ全体を参照するため、リストに保持すると **1 ステップ分の activation がずっと解放されません**。必ず `losses.append(loss.item())` にしてください。（第15章 canonical registry には該当 ID なし——実装レベルの一般的バグとして扱う。）

### C.2.3 分類 G：gradient 蓄積

| 症状 | 原因 | 対処 |
|---|---|---|
| 2 step 目以降で OOM | `optimizer.zero_grad()` 忘れ | 毎 step 実行 |
| RNN / GNN で OOM | `retain_graph=True` の残置 | 必要な箇所のみに限定 |
| `.backward()` を複数回呼ぶ | 意図せぬ二重逆伝播 | 明示的に `create_graph=False` |

```python
# 悪例：zero_grad 忘れ
for batch in loader:
    loss = model(batch).loss
    loss.backward()          # ← 勾配が蓄積し続ける
    optimizer.step()

# 良例
for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
```

### C.2.4 分類 F：断片化

`torch.cuda.memory_summary()` の出力で **「reserved: X MB / allocated: Y MB」で `X - Y` が大きい**（例：reserved 20 GB / allocated 12 GB）場合は断片化を疑います。
**対処**（優先順位）：

1. まず `torch.cuda.memory_stats()` の `inactive_split_bytes` / `num_alloc_retries` を確認して fragmentation の有意性を判定
2. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** を **`import torch` より前**に設定（PyTorch / CUDA バージョン依存、benchmark で確認してから採用）
3. 動的サイズ入力（可変長シーケンス、`torch.nested`）を極力使わない
4. batch 内で長さをそろえる（padding or bucketing）
5. `max_split_size_mb` / `garbage_collection_threshold` の調整（allocator backend 依存）
6. **どうしても駄目な場合のみ** `torch.cuda.empty_cache()`

```bash
# scripts/run_train.sh（推奨）
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # 決定論も同時に（付録B B.2.1）
export PYTHONHASHSEED=42                 # 決定論に必須（付録B B.2.1）
python -m scripts.train
```

### C.2.5 有効バッチサイズと BatchNorm の相互作用

`gradient_accumulation_steps` で有効バッチを維持しても、**BatchNorm の統計量は micro_batch 単位で更新される**ため、`micro_batch` が極端に小さいと分散推定が不安定になります。以下は経験的ヒューリスティック（空間次元・チャネル数・タスクにより変動）です：

| micro_batch | 推奨対応（ヒューリスティック） |
|---|---|
| ≥ 32 | BatchNorm 可 |
| 8〜32 | GroupNorm / LayerNorm 推奨 |
| < 8 | InstanceNorm または SyncBatchNorm（DDP、`effective_bn_batch = micro_batch × world_size` で判定）|

> [!IMPORTANT]
> BatchNorm を使うモデルで `micro_batch` を小さくした場合、**provenance に `micro_batch_per_gpu` と `effective_bn_batch = micro_batch × world_size` の両方を記録**してください（Ch14 DG-01 canonical signal と整合）。DDP で SyncBN を使う場合はさらに `sync_bn_enabled: true` を記録。

---

## C.3 決定論・再現性

「同じ seed で走らせたのに結果が違う」は深層 × Agentic で最も監査困難な問題です。

### C.3.1 診断チェックリスト（この順で確認）

| # | チェック項目 | コマンド | 期待値 |
|---|---|---|---|
| 1 | `CUBLAS_WORKSPACE_CONFIG` 環境変数 | `echo $CUBLAS_WORKSPACE_CONFIG` | `:4096:8` or `:16:8` |
| 2 | `PYTHONHASHSEED` 環境変数（付録A/B） | `echo $PYTHONHASHSEED` | 明示された数値 |
| 3 | `torch.backends.cudnn.deterministic` | `torch.backends.cudnn.deterministic` | `True` |
| 4 | `torch.backends.cudnn.benchmark` | `torch.backends.cudnn.benchmark` | `False` |
| 5 | `torch.use_deterministic_algorithms` | `torch.are_deterministic_algorithms_enabled()` | `True` |
| 6 | **TF32 制御**（Ch14 DG-01 canonical signal） | `torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32` | 期待値と provenance の `tf32_allowed` が一致 |
| 7 | Python / NumPy / torch seed | `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all` | 全て同一 seed |
| 8 | DataLoader `generator` と `worker_init_fn` | 付録B B.2.4 参照 | 明示設定済み |
| 9 | GPU 型番 | `nvidia-smi -L` | 学習・検証で同一 |
| 10 | CUDA / cuDNN / driver / PyTorch バージョン | `python -c "import torch; print(torch.version.cuda, torch.backends.cudnn.version(), torch.__version__)"` + `nvidia-smi --query-gpu=driver_version --format=csv` | 学習・検証で同一 |

> [!WARNING]
> **bit-exact 再現性は「PyTorch の保証範囲外」の op を含むため、原則として「許容誤差ベース」で運用**します。bit-exact を要求する場合でも、**同一 GPU 型番・同一 driver/CUDA/cuDNN/PyTorch・同一 TF32 / AMP 設定・全 seed 固定**まで揃える必要があり、それでも一部の op（scatter/gather 系、`atomicAdd` を使う ops、cuBLAS のバージョン差）は `torch.use_deterministic_algorithms(True)` により **fail-fast** します。A100 で学習した重みを H100 で bit-exact に再現できるとは限りません（Ch14 DG-01）。provenance には `gpu_model`, `driver_version`, `cuda_version`, `cudnn_version`, `torch_version`, `tf32_allowed`, `python_hash_seed` の 7 点を必ず記録してください（付録A §A.9 完全スキーマ参照）。

### C.3.2 「動いたつもり」を防ぐアサート

```python
# scripts/gpu_diag/assert_determinism.py
import os, random, numpy as np, torch

def assert_deterministic_ready(expected_seed: int,
                               expect_tf32: bool = False) -> None:
    # 1. Launcher で先に設定すべき環境変数
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"), (
        "CUBLAS_WORKSPACE_CONFIG must be set BEFORE `import torch` "
        "(usually in launcher shell script). Add "
        "`export CUBLAS_WORKSPACE_CONFIG=:4096:8` and re-run."
    )
    assert os.environ.get("PYTHONHASHSEED") == str(expected_seed), (
        f"PYTHONHASHSEED must be exported as '{expected_seed}' before "
        "python starts (see appendix B B.2.1)."
    )
    # 2. cuDNN / deterministic algorithms
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled(), (
        "Call torch.use_deterministic_algorithms(True) at process start."
    )
    # 3. TF32（Ch14 DG-01 canonical signal）
    tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    tf32_cudnn = torch.backends.cudnn.allow_tf32
    assert tf32_matmul == expect_tf32, (
        f"TF32 matmul is {tf32_matmul}, expected {expect_tf32}. "
        "Provenance field `tf32_allowed` must match this."
    )
    assert tf32_cudnn == expect_tf32
```

このアサートを **Skill の入口（`run_train` の先頭）**に置き、失敗したら第15章 **DG-01（GPU 非決定性による結果ずれ）** として `audit.log` に記録するのが本書の運用モデルです。

### C.3.3 「決定論 OFF が正当化される」場合

すべての演習を決定論 ON で走らせると、モデルによっては 20〜40% 遅くなります。**性能ベンチマークだけを目的とする場合は OFF が許容**されますが、その場合は provenance に以下を明示：

```yaml
determinism:
  status: "disabled"
  reason: "throughput_benchmark"
  approved_by: "u_alice"     # 人間の承認（Ch13 Gate1）
  approved_at: "2026-07-10T14:22:00+09:00"
  affected_metrics: ["val_accuracy", "val_loss"]   # 影響評価
  bit_exact_reproducibility: false
```

**エージェントが自律的に決定論を OFF にすることは禁止**です（Ch14 **AG-06 Human-in-the-loop バイパス**に相当——Gate1 の人間承認を経ずに provenance を書き換える行為として扱う）。必ず人間承認を経てください。

---

## C.4 CPU fallback と CI 実行

本書の演習は **CPU でも「動くこと」を第一目標**にしています（時間は掛かる）。CI（GitHub Actions 等）で GPU 前提コードが落ちる典型パターンを整理します。

### C.4.1 device 選択のイディオム

```python
# scripts/utils/device.py
import os, torch

def pick_device() -> torch.device:
    """CI / ローカル CPU / GPU を透過的に扱う。"""
    if os.environ.get("FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():   # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")

def scale_for_ci(config: dict) -> dict:
    """CI では epoch / batch を縮退して "smoke test" のみ通す。"""
    if os.environ.get("CI") == "true":
        config["epochs"] = min(config.get("epochs", 1), 1)
        config["batch_size"] = min(config.get("batch_size", 4), 4)
        config["num_workers"] = 0
        config["max_steps_per_epoch"] = 3
    return config
```

### C.4.2 GPU 前提コードが CI で落ちる典型 4 パターン

| # | 症状 | 原因 | 対処 |
|---|---|---|---|
| 1 | `AssertionError: Torch not compiled with CUDA` | `.cuda()` 決め打ち | `.to(device)` に統一 |
| 2 | `RuntimeError: mps not implemented` | Apple Silicon 未対応 op | `fallback_to_cpu` decorator |
| 3 | `Timeout after 6h` | CPU で full epoch | `scale_for_ci()` で縮退 |
| 4 | Bit-exact regression test fail | CPU/GPU で数値差 | 許容誤差ベース比較（`torch.allclose(rtol=1e-4)`）|

> [!IMPORTANT]
> CI での smoke test は **「Skill が起動する」「provenance が正しい schema で出力される」ことを検証**するもので、**性能検証ではありません**。第5章の Skill 定義では `ci_smoke_test` と `production_run` を明示的に分ける運用を推奨します。

---

## C.5 その他の頻出トラブル

### C.5.1 Mixed precision（AMP）で NaN が出る

```python
# 診断シーケンス（順に試す）
# 1. bfloat16 に切り替え（Ampere / Ada / Hopper）
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    ...

# 2. GradScaler の状態確認（fp16 のみ、bf16 では通常不要）
scaler = torch.amp.GradScaler("cuda")
print(scaler.get_scale())    # 極端に大きい/小さいなら overflow

# 3. loss そのものが inf / nan でないか
assert torch.isfinite(loss).all(), "loss diverged before AMP"

# 4. 特定 op を fp32 に強制
with torch.amp.autocast("cuda", enabled=False):
    logits = model.classifier(features.float())   # 敏感な層だけ fp32
```

**fp16 の最小学習ループ**（bf16 では GradScaler 不要）：

```python
scaler = torch.amp.GradScaler("cuda")

for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss = model(batch).loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)                                 # clip 前に必須
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)                                     # inf 検出時は skip
    scaler.update()                                            # scale 動的調整
```

対処優先順位：**bfloat16 に切替 → 学習率を下げる → 特定層を fp32 → AMP OFF**。**AMP OFF は最終手段**で、その場合 provenance に `amp: {status: "disabled", reason: "nan_divergence", ...}` を明示してください。

### C.5.2 DataLoader が遅い / hang する

| 症状 | 原因 | 対処 |
|---|---|---|
| GPU utilization < 30% | I/O bound | `num_workers` を CPU コア数 × 0.75 に |
| `num_workers > 0` で hang | fork + CUDA 初期化競合 | `multiprocessing_context="spawn"` |
| worker が死ぬ | メモリ不足 / 例外 | `worker_init_fn` にログ、`persistent_workers=True` |
| **同じ seed でも結果が違う** | worker seed 未設定 | 付録B B.2.4 の `worker_init_fn` 必須 |

### C.5.3 DDP / マルチ GPU の hang（NCCL）

| 症状 | 原因 | 対処 |
|---|---|---|
| `NCCL timeout` | rank 間の step 数不一致（データ枯渇、`drop_last=False`）| `DistributedSampler(drop_last=True)` + `set_epoch(epoch)` を毎エポック呼ぶ |
| rank 0 のみ OOM | rank 0 に集約する処理（`.item()`, log） | rank 0 集約は最小限に |
| grad が同期されないパラメータ | 条件分岐で forward スキップ | `DDP(..., find_unused_parameters=True)`（性能低下と引き換え） |
| 診断情報不足 | NCCL ログ抑制 | `export NCCL_DEBUG=INFO`、`NCCL_ASYNC_ERROR_HANDLING=1` |

> [!IMPORTANT]
> DDP 実行時は provenance に **`world_size`, `rank`, `gpu_model_per_rank` の 3 項目**を必ず記録してください。rank 間で `gpu_model_per_rank` が異なる（例：A100 + H100 の混在）と Ch14 DG-01 に該当します。

### C.5.4 Hugging Face Hub 認証と Foundation Model 取得

| ステータス | 意味 | 対処 |
|---|---|---|
| 401 Unauthorized | トークン未設定 or 期限切れ | `export HF_TOKEN=hf_...`（`login()` は使わない、付録B B.4.1）|
| 403 Forbidden | gated model の同意未了 | Hub UI で "Access repository" を人間が承認 |
| 404 Not Found | repo 名タイポ or private | 40-hex revision と repo_id を確認 |
| SHA256 mismatch | 途中で weight 差し替え | **Ch14 AG-08（Foundation Model の署名検証スキップ）**、**即座に停止し audit へ**|

> [!WARNING]
> **Foundation Model の重み配布・ライセンスは執筆時点から変わり得ます**。演習で使う repo_id / revision は **本書執筆時点（2026-07）** のものです。実行時に：
> 1. Hub 上に存在するか（404 なら fallback を検討）
> 2. ライセンスが変わっていないか（`declared_license` を毎回取得して provenance と比較）
> 3. 40-hex revision が本書記載と一致するか（差分があれば Ch11 provenance に "upstream drift" として記録）
> の 3 点を **Skill 実行前に必ず確認**してください（付録B B.4 の `emit_fm_provenance` ヘルパで自動化可能）。

---

## C.6 演習データ方針：ARIM 風合成データを主軸に

本書の演習は **「ARIM 風合成階層データを主軸、公開ベンチマークは対比・スケール確認に限定」**という方針を取ります（章構成 v0.2 A5）。

### C.6.1 なぜ合成データか

| 論点 | 実 ARIM データ | 公開ベンチマーク | ARIM 風合成 |
|---|---|---|---|
| **入手の容易さ** | 会員のみ / 案件依存 | 公開 | 生成スクリプトで即時 |
| **エージェントに触らせる可否** | ★機密度に依存 | ライセンス次第 | ✅ 自由 |
| **階層構造（装置・ロット・研究室）** | ✅ 現実的 | △ 弱いか無い | ✅ 明示的に設計 |
| **少データ性** | ✅ 現実的 | △ 大規模が多い | ✅ 設計可能 |
| **provenance の学習** | ○ 会員向けドキュメント要 | × ベンチ用途で薄い | ✅ 生成時に完全刻印 |

### C.6.2 データ配置規約

- `data/synthetic-deep/` — vol-03 用の生成データ（vol-02 の `data/synthetic-hier/` を深層向けに拡張）
- `data/anonymized-arim/` — ARIM 実データの匿名化サンプル（会員向けに別途配布、本リポジトリには含めない）
- `data/public-benchmark/` — 公開ベンチマークのローカルコピー（`.gitignore` 対象、Hub / 公式サイトから取得）

---

## C.7 ARIM 風合成データ生成スクリプト

以下は本書 §C.6 の合成データ生成の最小骨格です。フルスクリプトは `scripts/synth/gen_deep_hierarchical.py` に配置します。

### C.7.1 階層構造の設計

vol-02 §13 の階層合成データ（研究室 → 装置 → ロット → サンプル）を深層向けに拡張し、**「エージェントが階層を認識して fine-tune 戦略を切り替える」演習**（Ch13 capstone）が可能な構造にします。

```python
# scripts/synth/gen_deep_hierarchical.py
"""
ARIM 風合成階層データ生成

階層：lab (L=3) -> instrument (I=2/lab) -> lot (K=4/instrument)
      -> sample (N=20-40/lot, ラプラス分布)

各 sample には次の 2 モダリティが付く：
  (a) 疑似 SEM 画像（256x256 grayscale, Perlin noise + 装置固有 PSF）
  (b) 疑似 XRD スペクトル（1024 dim, ピーク位置 = f(lot × material))

装置固有バイアスと lot 内相関を明示的に注入し、
「装置間で共有可能な特徴」と「ロット固有ノイズ」を教師的に分離できるようにする。
"""
from __future__ import annotations
import hashlib, json, os
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import yaml

@dataclass
class SynthConfig:
    seed: int = 20260705
    n_labs: int = 3
    n_instruments_per_lab: int = 2
    n_lots_per_instrument: int = 4
    lot_size_mean: int = 25
    lot_size_std: float = 5.0
    lot_size_min: int = 20
    lot_size_max: int = 40
    image_size: int = 256
    spectrum_dim: int = 1024
    # 装置固有バイアス（truth manifest にのみ記録、学習には見せない）
    instrument_psf_sigma_range: tuple = (0.6, 1.8)
    instrument_bias_range: tuple = (-0.05, 0.05)
    output_root: str = "data/synthetic-deep"

def sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _gen_sem_image(rng: np.random.Generator, size: int,
                   psf_sigma: float, bias: float) -> np.ndarray:
    """疑似 SEM 相当の低周波テクスチャ：Gaussian ノイズを FFT で smoothing、
    装置固有 PSF sigma で低周波成分の広がりを、bias で正規化後の contrast 差を制御。

    NOTE: これは実際の SEM 物理（grain / edge / charging / shot noise）を
    再現するものではなく、「装置固有性を学ばせるための toy signal」。
    §C.7.3 のワークで grain / Poisson noise / scale-bar mask を追加すると
    より現実に近づく。
    """
    base = rng.standard_normal((size, size)).astype(np.float32)
    freqs = np.fft.fftfreq(size)[:, None]**2 + np.fft.fftfreq(size)[None, :]**2
    kernel = np.exp(-freqs * (psf_sigma * size) ** 2)
    smoothed = np.real(np.fft.ifft2(np.fft.fft2(base) * kernel))
    # 正規化してから bias を加える（bias が min-max で吸収されないように）
    img = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)
    img = np.clip(img + bias, 0.0, 1.0)
    return (img * 255).astype(np.uint8)

def _gen_xrd_spectrum(rng: np.random.Generator, dim: int,
                      peaks: list[tuple[float, float]]) -> np.ndarray:
    """疑似 XRD：ガウシアンピーク合成。"""
    x = np.linspace(0, 1, dim, dtype=np.float32)
    y = np.zeros(dim, dtype=np.float32)
    for center, height in peaks:
        y += height * np.exp(-((x - center) ** 2) / (2 * 0.005 ** 2))
    y += rng.normal(0, 0.01, size=dim).astype(np.float32)
    return np.clip(y, 0, None)

def generate(cfg: SynthConfig) -> tuple[dict, dict]:
    """returns (manifest_public, manifest_truth)

    manifest_public: 学習プロセスに公開してよい情報のみ（サンプル ID / 階層 ID /
        ファイル hash / split ID）。エージェントはこちらのみアクセス可。
    manifest_truth: 装置バイアス真値・ピーク真値等の「答え」。
        評価器 / Gate のみアクセス可、学習プロセスに渡してはならない（Ch14 DG-03）。
    """
    root = Path(cfg.output_root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    generator_hash = sha256_of_bytes(Path(__file__).read_bytes())
    manifest_public = {
        "schema_version": "arim-synth-deep-1.0",
        "generator_config_public": {                  # truth は含めない
            k: v for k, v in asdict(cfg).items()
            if k not in ("instrument_psf_sigma_range",
                         "instrument_bias_range")
        },
        "generator_source_sha256": generator_hash,
        "files_with_sha256": [],                      # Ch11/付録B canonical
        "hierarchy": [],                              # ID / split のみ
    }
    manifest_truth = {
        "schema_version": "arim-synth-deep-truth-1.0",
        "generator_source_sha256": generator_hash,
        "truths_by_instrument": [],
        "truths_by_lot": [],
    }

    for l in range(cfg.n_labs):
        lab_id = f"lab_{l:02d}"
        instruments_pub = []
        for i in range(cfg.n_instruments_per_lab):
            inst_id = f"{lab_id}_inst_{i}"
            psf = rng.uniform(*cfg.instrument_psf_sigma_range)
            bias = rng.uniform(*cfg.instrument_bias_range)
            manifest_truth["truths_by_instrument"].append({
                "instrument_id": inst_id,
                "psf_sigma_truth": float(psf),
                "bias_truth": float(bias),
            })
            lots_pub = []
            for k in range(cfg.n_lots_per_instrument):
                # 上下限で clip した正規分布（20-40 のロットサイズを保証）
                n = int(np.clip(
                    rng.normal(cfg.lot_size_mean, cfg.lot_size_std),
                    cfg.lot_size_min, cfg.lot_size_max))
                lot_id = f"{inst_id}_lot_{k}"
                peaks = [(0.2 + 0.02 * k, 1.0), (0.7 - 0.01 * k, 0.6)]
                manifest_truth["truths_by_lot"].append({
                    "lot_id": lot_id,
                    "peaks_truth": peaks,
                })
                samples_pub = []
                for s in range(n):
                    img  = _gen_sem_image(rng, cfg.image_size, psf, bias)
                    spec = _gen_xrd_spectrum(rng, cfg.spectrum_dim, peaks)
                    sid = f"{lot_id}_s{s:03d}"
                    filename = f"{sid}.npz"
                    np.savez_compressed(root / filename,
                                        image=img, spectrum=spec)
                    file_hash = sha256_of_bytes((root / filename).read_bytes())
                    # Ch11 / 付録B canonical schema
                    manifest_public["files_with_sha256"].append({
                        "file":   filename,
                        "kind":   "sample_npz",
                        "sha256": file_hash,
                    })
                    samples_pub.append({"sample_id": sid, "file": filename})
                lots_pub.append({"lot_id": lot_id,
                                 "n_samples": n,
                                 "samples": samples_pub})
            instruments_pub.append({"instrument_id": inst_id, "lots": lots_pub})
        manifest_public["hierarchy"].append({"lab_id": lab_id,
                                             "instruments": instruments_pub})

    # 2 つの manifest を分けて出力
    #   - manifest_public.yaml : 学習プロセスに公開してよい情報
    #   - manifest_truth.yaml  : 評価器 / Gate のみが読む「答え」
    with open(root / "manifest_public.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest_public, f, sort_keys=False, allow_unicode=True)
    with open(root / "manifest_truth.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest_truth, f, sort_keys=False, allow_unicode=True)
    return manifest_public, manifest_truth
```

> [!IMPORTANT]
> **`manifest_truth.yaml` は学習プロセスに絶対に見せない**でください。装置バイアス・ピーク位置の真値が学習に入ると Ch14 **DG-03（Fine-tune のデータリーク）** に該当します。運用上は `manifest_truth.yaml` を別ディレクトリ / 別 IAM ロールに置き、評価器と Gate だけが読める権限にします。

### C.7.2 生成データの provenance 刻印

生成されたデータには **`manifest_public.yaml`（学習用）**と **`manifest_truth.yaml`（評価専用）**が同梱され、以下を含みます：

| フィールド | 出力先 | 用途 | 対応章 |
|---|---|---|---|
| `schema_version` | public / truth 両方 | 生成器スキーマ ID | Ch4 canonical |
| `generator_config_public` | public のみ | 学習に公開してよいハイパラ | Ch4 §5.5 |
| `generator_source_sha256` | 両方 | 生成スクリプトの hash | Ch11 §12.4 |
| `files_with_sha256[].{file,kind,sha256}` | public のみ | Ch11 / 付録B canonical schema | Ch11 §12.4 integrity（改ざん検知は AG-08 と同族の signature 系脅威）|
| `hierarchy` (lab/instrument/lot/sample ID) | public のみ | Ch13 階層構造 | Ch13 capstone |
| `truths_by_instrument[].{psf_sigma_truth, bias_truth}` | **truth のみ** | 学習後の bias 復元検証 | Ch13 Gate2 |
| `truths_by_lot[].peaks_truth` | **truth のみ** | ピーク位置の復元検証 | Ch13 Gate2 |
| `psf_sigma_truth` / `bias_truth` | 装置固有真値 | Ch13 capstone 教師 |
| `files_with_sha256[].sha256` | 個別ファイルの integrity | Ch11 §12.4 |

> [!TIP]
> `manifest_public.yaml` は学習・Skill 実行時に参照し、`manifest_truth.yaml` は評価器 / Ch13 Gate2 のみが読みます。エージェントが `lab_00` で学習した重みを `lab_02` に転移するとき、Gate 側で truth と学習結果を照合し、Ch13 Gate2 の判断材料にします。

### C.7.3 split integrity と自主ワーク

train / val / test の split integrity は Ch14 DG-03 の中心的関心事です。**split 定義そのものを hash 化**して provenance に刻印してください：

```python
def make_split_manifest(sample_ids: list[str], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ids = list(sample_ids); rng.shuffle(ids)
    n = len(ids); n_tr = int(n*0.7); n_va = int(n*0.15)
    tr, va, te = ids[:n_tr], ids[n_tr:n_tr+n_va], ids[n_tr+n_va:]
    assert set(tr) & set(va) == set() and set(va) & set(te) == set()
    def h(x): return sha256_of_bytes(json.dumps(sorted(x)).encode())
    return {"train_ids": tr, "val_ids": va, "test_ids": te,
            "train_split_hash": h(tr), "val_split_hash": h(va),
            "test_split_hash": h(te)}
```

**自主ワーク（現実性を上げる）**：§C.7.1 の `_gen_sem_image` は toy signal です。より SEM 現実に近づけるには、grain 楕円の重ね合わせ / edge enhancement / Poisson noise / contrast 変動 / scale bar mask を追加してみてください。

### C.7.4 実行例

```bash
python -m scripts.synth.gen_deep_hierarchical \
    --seed 20260705 \
    --n-labs 3 \
    --output-root data/synthetic-deep

# 期待出力：
#   data/synthetic-deep/manifest_public.yaml   ← 学習に公開
#   data/synthetic-deep/manifest_truth.yaml    ← 評価器のみ（別 IAM 推奨）
#   data/synthetic-deep/lab_00_inst_0_lot_0_s000.npz
#   ...  (合計 3 × 2 × 4 × ~25 ≒ 600 サンプル)
```

---

## C.8 公開ベンチマーク：ライセンスと「エージェントに触らせてよいか」

演習の**対比・スケール確認**として使える公開ベンチマークを 6 件挙げます。**すべてのベンチマークについて、本書執筆時点（2026-07）の情報**であり、実行時はライセンスが変わり得ることに注意してください。

### C.8.1 判断ガイド

| 質問 | Yes | No | 不明 |
|---|---|---|---|
| ライセンスに商用利用可条項がある | ✅ 使用可 | 教育目的のみ限定 | 使用前に確認 |
| 再配布可条項がある | ✅ Skill 同梱可 | ダウンロードのみ | 使用前に確認 |
| 派生モデルの配布制限がない | ✅ Skill 出力を配布可 | 配布不可 | 使用前に確認 |
| **attribution / citation 義務**がある | ✅ paper / README に記載 | — | 使用前に確認 |
| **API ToS が data license と別**にある | ✅ 両方の条件を満たす | — | 使用前に確認 |
| **bulk download / rate-limit** が許諾されている | ✅ 事前 DL 可 | ストリーミングのみ | 使用前に確認 |
| ローカルキャッシュの**再配布**可 | ✅ 組織内共有可 | ダウンロード都度 | 使用前に確認 |
| **学習済みモデル・出力の再配布**が制限されない | ✅ 派生物公開可 | 制限あり | 使用前に確認 |
| PII / 機微データを含む | ❌ 使用不可 | ✅ 使用可 | 使用前に確認 |
| エージェント経由の**自動アクセス**を許諾している | ✅ 自動 fetch 可 | 手動 fetch のみ | 手動を推奨 |

> [!IMPORTANT]
> 「ライセンス条項の解釈」は法務判断です。組織内で使う場合は必ず所属機関の法務・データガバナンス窓口に確認してください。本書のガイドは **技術的な観点でのチェックリスト** であり、法的助言ではありません。特に**「データそのもののライセンス」と「API 経由でのアクセス条件（Terms of Use）」は別**であることが多く（例：Materials Project）、両方を独立に確認する必要があります。

### C.8.2 ベンチマーク一覧（2026-07 時点）

| 名称 | ドメイン | ライセンス | エージェント自動 fetch | 本書での位置づけ |
|---|---|---|---|---|
| **NFFA-EUROPE SEM** | SEM 画像分類 | CC BY 4.0（要確認）| ✅ 可（要 credit）| §C.7 合成データと対比、Ch7 |
| **UHCS microstructure** | 微細組織 SEM | CC BY 4.0（要確認）| ✅ 可 | Ch6 画像分類の実データ検証 |
| **Materials Project** | 結晶構造・物性 | データ CC BY 4.0（要確認）／**API は別 ToS** | △ API rate limit 注意 | Ch11 FM 事前学習の対比 |
| **Open Catalyst 2020/2022** | 触媒構造・エネルギー | 配布時点の LICENSE を要確認（CC BY 4.0 に類する条件が多いが版により変動）| ✅ 可（大容量、事前 DL 推奨）| Ch7 転移学習 |
| **AFLOW** | 結晶物性 | **要確認**（利用条件が明示ライセンスで示されない場合あり、商用・再配布は事前照会）| △ 手動 fetch 推奨 | 参照のみ、Skill 埋め込みは慎重に |
| **NOMAD** | マルチスケール実験・計算 | 記録ごとに個別ライセンス（多くは CC BY 4.0 系）| ✅ 可（メタデータ豊富）| Ch11 provenance の実例 |

> [!IMPORTANT]
> 上表のライセンス欄は **2026-07 執筆時点の技術者向けメモ**であり、**法的助言ではありません**。特に AFLOW / Materials Project は「データ本体のライセンス」と「API 利用規約」が別条件で、片方だけの確認では不十分です。実利用前に必ず各プロジェクトの公式ページで **現時点の LICENSE / Terms of Use** を再確認してください。

### C.8.3 各ベンチマークの取得コマンド（例）

```bash
# 1. NFFA-EUROPE SEM
#    (実 URL は執筆時点のもの、変わり得る)
mkdir -p data/public-benchmark/nffa-europe
# 公式配布ページから手動 DL 後、
sha256sum data/public-benchmark/nffa-europe/*.tar.gz > \
    data/public-benchmark/nffa-europe/CHECKSUMS.txt

# 2. Materials Project (要 API key)
pip install mp-api
python -c "
from mp_api.client import MPRester
import os
with MPRester(os.environ['MP_API_KEY']) as m:
    docs = m.summary.search(elements=['Fe','O'], num_chunks=1)
    print(f'fetched {len(docs)} materials')
"

# 3. Open Catalyst (S3 / Hugging Face Datasets)
python -c "
from datasets import load_dataset
ds = load_dataset('open-catalyst-project/oc20', split='train[:1%]',
                  token=os.environ.get('HF_TOKEN'))  # 付録B B.4
print(ds)
"
```

> [!WARNING]
> **エージェントに自動 fetch させる場合の必須要件**：
> 1. データ取得元を **`dataset_fetch_allowlist`** に事前登録（付録B B.5.3 の URI allowlist（`s3/gs/abfs`）とは別の allowlist として管理）：
>    ```yaml
>    dataset_fetch_allowlist:
>      - type: hf_dataset
>        repo_id: "open-catalyst-project/oc20"
>        revision: "<40-hex-commit>"
>      - type: https
>        domain: "b2share.eudat.eu"
>        path_prefix: "/records/nffa-europe-sem"
>      - type: mp_api
>        endpoint: "https://api.materialsproject.org"
>    ```
> 2. ダウンロード後に **file-level sha256 検証**（付録B B.4.4 の `files_with_sha256` パターンを流用）
> 3. ライセンス文字列と取得時刻を provenance の `dataset_license` / `dataset_fetched_at` に刻印（Ch11 §12.4）
> 4. **rate limit / 大容量 DL の再試行ポリシーを Skill 定義に明記**（Ch4）
>
> **`dataset_fetch_allowlist` にない取得元へエージェントが自律的にアクセスすることは禁止**（Ch14 **AG-06 Human-in-the-loop バイパス**に該当）。取得時に signature / hash 検証を省略した場合は **AG-08（署名検証スキップ）**として扱います。

### C.8.4 ライセンス変更に備えたフォールバック

**Foundation Model と同様に、データセットのライセンス条件も変更され得ます**（章構成 v0.2 の失敗パターン §Foundation Model の重み配布・ライセンス変更）。以下を Skill 実行前に確認：

```python
# scripts/utils/dataset_license_check.py
import hashlib, requests
from datetime import date

# ここでは公式ページの LICENSE ファイル/条項文字列のスナップショットを
# hash 化して保存し、drift を検出する運用を推奨（法的解釈は入れない）。
EXPECTED_LICENSES = {
    "nffa-europe":  ("CC BY 4.0 (as noted on B2SHARE 2026-07-01)", "2026-07-01"),
    "oc20":         ("Open Catalyst project LICENSE snapshot 2026-07-01", "2026-07-01"),
    "aflow":        ("AFLOW terms snapshot 2026-07-01 (verify commercial use)", "2026-07-01"),
}

def check_dataset_license(name: str,
                          fetched_license: str,
                          fetched_at: str) -> dict:
    exp_lic, exp_date = EXPECTED_LICENSES[name]
    drift = fetched_license != exp_lic
    return {
        "dataset": name,
        "expected_license": exp_lic,
        "expected_as_of": exp_date,
        "fetched_license": fetched_license,
        "fetched_at": fetched_at,
        "license_drift": drift,
        # drift 時は Ch14 AG-06 として audit に記録し、Skill を停止
    }
```

---

## C.9 まとめ

- **GPU トラブル**は「エラーを消す」ではなく **「provenance / audit に記録を残す」**をゴールにする
- **OOM は B → A → G → F の順に診断**。`torch.cuda.empty_cache()` は基本使わない
- **決定論は同一 GPU 型番 + 同一 CUDA/cuDNN + 同一 PyTorch でのみ**保証。**エージェントが自律的に決定論 OFF にすることは禁止**
- **CPU fallback / CI smoke test** は「動く」ではなく「provenance が正しい schema で出る」ことを確認するもの
- **演習データは ARIM 風合成データを主軸**にし、公開ベンチマークは対比用途に留める。生成データは **`manifest_public` / `manifest_truth` を分離**し、truth を学習に見せない
- **公開ベンチマークの自動 fetch は `dataset_fetch_allowlist` + file-level sha256 + ライセンス drift 検出**の 3 点セット
- **すべてのトラブルシューティングは第15章 failure_pattern_registry（DG-01〜08 / AG-01〜09）と対応**させ、audit ledger の入力にする

---

## C.10 参考資料

- **PyTorch Reproducibility**: https://pytorch.org/docs/stable/notes/randomness.html
- **PyTorch Memory Management**: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
- **NVIDIA CUDA Environment Variables**: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ENV.html
- **Hugging Face Hub Authentication**: https://huggingface.co/docs/huggingface_hub/quick-start#authentication
- **NFFA-EUROPE SEM dataset**: https://b2share.eudat.eu/records/nffa-europe-sem
- **Materials Project API**: https://next-gen.materialsproject.org/api
- **Open Catalyst Project**: https://opencatalystproject.org/
- **NOMAD Laboratory**: https://nomad-lab.eu/
- 本書付録A（テンプレート集）／付録B（PyTorch/JAX/HF/MCP チートシート）
- 本書第5章（Skill 定義）／第12章（Foundation Model provenance）／第14章（Gate state machine）／第15章（failure_pattern_registry）
