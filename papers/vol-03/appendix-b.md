# 付録B　PyTorch / JAX / Hugging Face チートシートと MCP サーバ実装

> **本付録の目的**：第4-15章で頻出する PyTorch / JAX (Flax) / Hugging Face の API を **Agentic Skill の provenance・承認ゲート要件**と紐付けたチートシートにまとめ、続いて **MCP (Model Context Protocol) サーバ Python SDK による組織内サーバの最小実装**を提示します。vol-02 §15 で「MCP を作るべきかどうか」の判断基準までを扱ったのに対し、本付録は「作ると決めた後に、Agentic 承認ゲートを MCP レベルで組み込む方法」を扱います。

> [!IMPORTANT]
> チートシートは「動くコード」の羅列ではなく、**「Skill が Agentic 契約を満たすために必要な API 呼び出しと provenance 記録」**の対応表です。全コードは第5章の agent tier / 第12章の FM 検証 / 第14章の Gate1-3 / 第15章の audit を **API レベルで満たす形**に揃えています。

---

## B.1 本付録の使い方

| 目的 | 参照するセクション | 該当章 |
|---|---|---|
| 決定論的な学習ループを PyTorch で書く | B.2.1 - B.2.3 | 第5章・第15章 |
| DataLoader worker のシード伝播を正しく行う | B.2.4 | 第15章 (DG-01) |
| Mixed Precision / DDP を有効にしつつ再現性を保つ | B.2.5 - B.2.6 | 第15章 |
| Checkpoint を append-only で保存 | B.2.7 | 第5章・付録A |
| JAX / Flax で決定論的学習を書く | B.3 | 第6-9章 |
| Hugging Face から FM を安全に取得する | B.4.1 - B.4.3 | 第12章 |
| safetensors のみを許可する読み込みパスを組む | B.4.4 | 第12章 (AG-08) |
| MCP サーバを組織内で最小構成で立てる | B.5.1 - B.5.3 | 第5章・vol-02 §15 |
| MCP レベルで train / infer 権限を分離する | B.5.4 | 第5章 |
| MCP に Human 承認 hook を組み込む | B.5.5 | 第14章 |
| MCP サーバの provenance / audit 対応 | B.5.6 | 第15章 |

> [!NOTE]
> 本付録のコードは **PyTorch 2.x / JAX 0.4.x / transformers 4.35+ / huggingface_hub 0.20+ / mcp SDK 1.0+** を想定しています。バージョンは変わるため、実運用時は該当時点の公式ドキュメントで API 名称と署名を確認してください（変更が破壊的な場合は Skill の `framework_versions.*` を必ず provenance に記録）。

---

## B.2 PyTorch チートシート

### B.2.1 決定論設定（起動時 1 回）

> [!WARNING]
> **`CUBLAS_WORKSPACE_CONFIG` は必ずプロセス起動時、`import torch` **より前**に設定してください**。cuBLAS は最初の CUDA op でハンドルを初期化した瞬間に環境変数を読み、それ以降の変更は無視されます。`set_deterministic()` を先に呼んでも、モジュールインポートで既に CUDA が触られていれば `torch.use_deterministic_algorithms(True)` は後の matmul で例外を出します（サイレントに非決定に戻るのではなく **fail-fast**）。**推奨**：launcher shell / systemd unit / `torchrun` の env 経由で `CUBLAS_WORKSPACE_CONFIG=:4096:8` と `PYTHONHASHSEED=<seed>` を渡す。

```bash
# scripts/run_train.sh — Python プロセス起動より前に env を確定
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1     # B.4.1 参照
exec torchrun --nproc_per_node=4 scripts/train.py --seed=42
```

```python
# scripts/_deterministic_setup.py
import os
import random
import numpy as np
import torch

def set_deterministic(seed: int) -> None:
    """
    第5章 §5.5 の再現性条件と Ch14 DG-01 (GPU 非決定性 / cuDNN 非決定性等) 対策。
    provenance には次の 7 項目を必ず記録：
      - random_seed (vol-02)
      - numpy_seed / python_hash_seed / random_seed_per_worker (vol-03)
      - cudnn_deterministic / cudnn_benchmark / gpu_backend

    重要: CUBLAS_WORKSPACE_CONFIG と PYTHONHASHSEED は launcher で
    `import torch` 前に設定済みであることを本関数の冒頭で assert する。
    """
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", (
        "CUBLAS_WORKSPACE_CONFIG must be set to ':4096:8' before importing torch "
        "(export in launcher shell)."
    )
    assert os.environ.get("PYTHONHASHSEED") == str(seed), (
        f"PYTHONHASHSEED must be exported as '{seed}' before python starts."
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False               # true にすると非決定
    torch.use_deterministic_algorithms(True, warn_only=False)
```

> [!WARNING]
> `torch.use_deterministic_algorithms(True)` は一部の CUDA カーネル（`bmm`, `scatter_add` の一部）で例外を発生させます。回避策として `warn_only=True` を使うと再現性が失われるため、Skill 契約側で「決定論違反時は fail」と宣言し、`warn_only` は使わないのが基本です。

### B.2.2 学習ループの最小骨格

```python
# scripts/train_loop.py
from pathlib import Path
import torch
from torch.utils.data import DataLoader

def train_epoch(model, loader, optimizer, loss_fn, device) -> dict:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x, y = batch["x"].to(device, non_blocking=True), batch["y"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        # gradient clipping は provenance に閾値を記録
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    return {"train_loss": total_loss / max(n, 1)}
```

### B.2.3 検証ループ（不確かさ含む）

```python
@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_probs, all_targets = [], []
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs)
        all_targets.append(y)
    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets)
    # 予測エントロピー（total uncertainty）— provenance に記録
    entropy = -(probs * probs.clamp(min=1e-12).log()).sum(-1).mean().item()
    return {"probs": probs, "targets": targets, "predictive_entropy_mean": entropy}
```

### B.2.4 DataLoader worker のシード伝播

```python
# Ch14 DG-01：worker 内での numpy.random / random は再シードしないと非決定
def _worker_init_fn(worker_id: int) -> None:
    # torch.initial_seed() は worker ごとに一意な値を返す（seed + worker_id 由来）
    base_seed = torch.initial_seed() % (2**32)
    np.random.seed(base_seed)
    random.seed(base_seed)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    worker_init_fn=_worker_init_fn,
    generator=torch.Generator().manual_seed(cfg["seed"]),
    persistent_workers=True,        # プロセス再起動による初期化順変動を回避
    pin_memory=True,
)

# provenance:
#   random_seed_per_worker: True
#   worker_init_fn_hash: sha256 of scripts/train_loop.py:_worker_init_fn
```

### B.2.5 Mixed Precision（AMP）と再現性の両立

```python
# PyTorch 2.x: torch.amp.* に統一（torch.cuda.amp.* は deprecated）
import torch
from torch.amp import autocast, GradScaler

scaler = GradScaler("cuda")
for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    with autocast("cuda", dtype=torch.float16):     # bf16 の方が数値安定性が高い場合あり
        logits = model(batch["x"].to(device))
        loss = loss_fn(logits, batch["y"].to(device))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

# provenance:
#   mixed_precision: true
#   amp_dtype: "float16 | bfloat16"
#   grad_scaler_used: true
#   note: fp16 は cudnn_deterministic=True でも
#         reduction 順序で微小な数値差が出うる（1e-6 レベル）
```

### B.2.6 分散学習（DDP）の provenance

```python
# torchrun 起動：torchrun --nproc_per_node=4 scripts/train.py
import torch.distributed as dist

def setup_ddp() -> tuple[int, int]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size

# provenance に必ず記録：
#   ddp:
#     world_size: <N>
#     backend: "nccl"
#     effective_batch_size: <batch_size * world_size>   # ロス曲線比較用
#     all_reduce_algorithm: "ring | tree"                # NCCL_ALGO
#     nccl_version: <str>
#   注意: DDP は BatchNorm を全 rank で同期しない限り非決定。
#         SyncBatchNorm or GroupNorm を使い、provenance に明記。
```

### B.2.7 Checkpoint（append-only + safetensors）

```python
# scripts/_checkpoint.py
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
from safetensors.torch import save_model

def save_checkpoint(model, ckpt_dir: Path, epoch: int, metrics: dict,
                    approval_record_id: str) -> Path:
    """
    Ch4 checkpoint_overwrite_policy: append_only の実装。
    ファイル名にエポック番号を含め、既存ファイルへの上書きを拒否。

    注意: `safetensors.torch.save_file(model.state_dict())` は
    tied weights（embedding と lm_head の共有等、HF LLM で頻出）で
    失敗する。`save_model` は shared tensors を正しく処理するので
    深層モデルでは save_model を推奨。
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fpath = ckpt_dir / f"epoch_{epoch:03d}.safetensors"
    if fpath.exists():
        raise RuntimeError(f"AG-03 (silent overwrite) prevented: {fpath}")

    save_model(model, str(fpath))

    # append-only registry
    sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
    entry = {
        "epoch": epoch,
        "path": fpath.name,
        "sha256": sha,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "approval_record_id": approval_record_id,       # Ch4 canonical
        "metrics": metrics,
    }
    reg = ckpt_dir / "checkpoint_registry.jsonl"
    with reg.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return fpath

def mark_as_best(ckpt_dir: Path, epoch: int, signer_pubkey: str,
                 signature: str) -> None:
    """
    「best」は上書きファイルではなく、不変 checkpoint への signed pointer。
    """
    reg = ckpt_dir / "checkpoint_registry.jsonl"
    entry = {
        "kind": "best_pointer",
        "points_to": f"epoch_{epoch:03d}.safetensors",
        "signer_pubkey": signer_pubkey,
        "signature": signature,
    }
    with reg.open("a") as f:
        f.write(json.dumps(entry) + "\n")
```

### B.2.8 主要 API 早見表

| 目的 | API | provenance に記録すべき項目 |
|---|---|---|
| モデル構築（画像） | `timm.create_model(name, pretrained, num_classes)` | `model_architecture.name`, `parameter_count`, `parameter_count_trainable` |
| Optimizer | `torch.optim.AdamW(params, lr, weight_decay)` | `optimizer`, `lr_schedule`, `weight_decay` |
| Scheduler | `torch.optim.lr_scheduler.CosineAnnealingLR(...)` | `lr_schedule`（enum + 引数 hash） |
| Loss（回帰・不確かさ） | `torch.nn.GaussianNLLLoss(reduction)` | `loss` |
| データ拡張 | `torchvision.transforms.v2.*` or `albumentations` | `augmentation_provenance.policy_hash` |
| 保存 | `safetensors.torch.save_file` | `weights_format: "safetensors"`, `weights_sha256` |
| 読込 | `safetensors.torch.load_file` | 起動時 hash 検証 |

---

## B.3 JAX / Flax チートシート

### B.3.1 決定論と PRNGKey 管理

```python
import jax
import jax.numpy as jnp
from flax import linen as nn

# JAX は「毎回 PRNGKey を明示的に split する」ため決定論性が組み込み。
# vol-03 では PRNGKey chain 全体を provenance に記録。
key = jax.random.PRNGKey(seed=42)
key, init_key, dropout_key, aug_key = jax.random.split(key, 4)

# provenance:
#   prng_seed: 42
#   prng_key_split_chain_hash: sha256 of "42|init|dropout|aug"
#   注意: JAX 側で "非決定" が入りうるのは XLA compilation のみ。
#         xla_flags を provenance に記録。
```

### B.3.2 学習 step の pure function 化

```python
from flax.training import train_state
import optax

def create_state(rng, model, lr):
    params = model.init(rng, jnp.ones((1, 224, 224, 3)))
    tx = optax.adamw(learning_rate=lr, weight_decay=1e-4)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch, dropout_rng):
    def loss_fn(params):
        logits = state.apply_fn(params, batch["x"], train=True,
                                rngs={"dropout": dropout_rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["y"]).mean()
        return loss, logits
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), {"loss": loss}
```

### B.3.3 pmap による multi-device 分散

```python
from flax.jax_utils import replicate, unreplicate

state = replicate(state)                       # 各 device に複製
p_train_step = jax.pmap(train_step, axis_name="batch")

# provenance:
#   distributed_backend: "jax_pmap"
#   n_devices: jax.local_device_count()
#   注意: pmap 内 all_reduce は `axis_name` で明示、
#         全 rank BatchNorm 同期は `flax.linen.BatchNorm(use_running_average=..., axis_name="batch")`
```

### B.3.4 早見表

| PyTorch | JAX / Flax |
|---|---|
| `torch.manual_seed(s)` | `jax.random.PRNGKey(s)` + `split` chain |
| `nn.Dropout(p=0.5)` | `flax.linen.Dropout(rate=0.5, deterministic=not train)` |
| `torch.optim.AdamW` | `optax.adamw` |
| `torch.nn.functional.cross_entropy` | `optax.softmax_cross_entropy_with_integer_labels` |
| `DataLoader(num_workers=N)` | `tf.data` or `grain` を JAX へ pipeline |
| `DDP` | `pmap` + `pjit` (JAX 0.4+) |

---

## B.4 Hugging Face チートシート（Ch11 安全設計準拠）

### B.4.1 認証トークンの取り扱い

> [!WARNING]
> **`huggingface_hub.login()` は避ける**。`login()` は `add_to_git_credential=False` を渡しても **`~/.cache/huggingface/token` にトークンをファイル書き出しする**。HPC / 共有 FS ではトークン漏洩の温床です（credential exposure — 本書 Ch14 canonical registry には該当 ID なし、組織のセキュリティ規程に準拠して扱う）。**推奨は `token=os.environ["HF_TOKEN"]` を `HfApi` / `snapshot_download` / `from_pretrained` に直接渡すこと**、そして launcher で `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` を設定してトークンファイルの暗黙読み込みも遮断すること。

```python
# 環境変数のみ許可、コード内 hardcode 禁止（credential exposure 対策）
# login() は呼ばない。個別 API に token を直接渡す。
import os
from huggingface_hub import HfApi, snapshot_download

_HF_TOKEN = os.environ.get("HF_TOKEN")
assert _HF_TOKEN is not None, "HF_TOKEN must be set via env, never hardcoded"

api = HfApi(token=_HF_TOKEN)

# provenance:
#   hf_auth_method: "env_var_per_call"
#   hf_token_scope_hash: sha256 of scope string (実トークンは書かない)
#   hf_hub_disable_implicit_token: "1"  (launcher で export 済み)
```

### B.4.2 revision pin + Hub resolved SHA 一致検証

```python
import re
from huggingface_hub import HfApi

def verify_revision(repo_id: str, revision: str) -> None:
    # 1. 40-hex commit SHA のみ許容（tag / branch / 短縮 SHA 全禁止）
    assert re.fullmatch(r"[0-9a-f]{40}", revision), \
        f"revision must be 40-hex commit sha, got: {revision}"
    # 2. Hub API の resolved sha と一致
    api = HfApi()
    info = api.model_info(repo_id=repo_id, revision=revision)
    assert info.sha == revision, \
        f"hub resolved sha ({info.sha}) != approved revision ({revision})"
```

### B.4.3 snapshot_download で allow_patterns を絞る

```python
import os
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id=repo_id,
    revision=revision,
    allow_patterns=[
        "*.safetensors",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "config.json", "generation_config.json",
    ],
    # ignore_patterns で pickle / .bin / .msgpack を明示除外
    ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.pkl", "*.pt"],
    token=os.environ["HF_TOKEN"],       # B.4.1、login() を使わない
)
```

### B.4.4 safetensors のみを許可し、tokenizer / config も hash 検証するロードパス

> [!IMPORTANT]
> **hash 検証は weights だけでは不十分**。`tokenizer.json`（BPE merges、追加 special tokens）や `config.json`（vocab_size, rope_theta, num_hidden_layers）、`generation_config.json`（sampling defaults）は改ざんで振る舞いが変わる。**Ch11 §12.4 の canonical manifest は weights + tokenizer + config + generation_config を全て file-level sha256 で pin する**。

```python
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoConfig

# Ch11 canonical: manifest schema
# manifest = {
#   "repo_id": "org/repo",
#   "revision_commit_hash": "<40hex>",
#   "hub_api_resolved_sha": "<40hex>",
#   "declared_license": "cc-by-4.0",
#   "pretraining_data_license": "cc-by-4.0",
#   "pretraining_data_summary": "...",
#   "model_family": "matbert",
#   "fetch_method": "snapshot_download",
#   "files_with_sha256": [                       # weights + tokenizer + config を全て
#       {"file": "model.safetensors",       "kind": "weights",    "sha256": "..."},
#       {"file": "tokenizer.json",          "kind": "tokenizer",  "sha256": "..."},
#       {"file": "tokenizer_config.json",   "kind": "tokenizer",  "sha256": "..."},
#       {"file": "special_tokens_map.json", "kind": "tokenizer",  "sha256": "..."},
#       {"file": "config.json",             "kind": "config",     "sha256": "..."},
#       {"file": "generation_config.json",  "kind": "config",     "sha256": "..."},
#   ],
# }

def load_verified_fm(local_dir: str, manifest: dict):
    """全ファイル hash 検証 + safetensors 経路限定ロード。"""
    local_path = Path(local_dir).resolve()

    # 1. file-level sha256 を weights + tokenizer + config で全件照合
    seen_kinds = set()
    for entry in manifest["files_with_sha256"]:
        p = (local_path / entry["file"]).resolve()
        # path traversal 防御: 解決後のパスが local_dir 配下にあること
        assert str(p).startswith(str(local_path) + os.sep), \
            f"path traversal blocked: {entry['file']}"
        assert p.is_file(), f"missing manifest file: {entry['file']}"
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == entry["sha256"], (
            f"hash mismatch for {entry['file']}: got {h}, expected {entry['sha256']}"
        )
        seen_kinds.add(entry["kind"])

    # 2. weights / tokenizer / config の全 kind が manifest に含まれていること
    for required_kind in ("weights", "tokenizer", "config"):
        assert required_kind in seen_kinds, \
            f"manifest missing required kind: {required_kind}"

    # 3. safetensors 経路でロード（pickle 経路を明示遮断）
    model = AutoModel.from_pretrained(local_dir, use_safetensors=True,
                                      trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=False)
    config = AutoConfig.from_pretrained(local_dir, trust_remote_code=False)
    return model, tokenizer, config


def emit_fm_provenance(manifest: dict, local_dir: str) -> dict:
    """
    Ch11 canonical field 名で foundation_model_provenance を出力する共通ヘルパ。
    Skill 側 provenance の該当セクションにそのまま入れる。
    """
    return {
        "foundation_model_provenance": {
            "used": True,
            "repo_id": manifest["repo_id"],
            "revision_commit_hash": manifest["revision_commit_hash"],
            "hub_api_resolved_sha": manifest["hub_api_resolved_sha"],
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "declared_license": manifest["declared_license"],
            "pretraining_data_license": manifest["pretraining_data_license"],
            "pretraining_data_summary": manifest["pretraining_data_summary"],
            "model_family": manifest["model_family"],
            "safetensors_files_with_sha256": [
                e for e in manifest["files_with_sha256"] if e["kind"] == "weights"
            ],
            "tokenizer_files_with_sha256": [
                e for e in manifest["files_with_sha256"] if e["kind"] == "tokenizer"
            ],
            "config_files_with_sha256": [
                e for e in manifest["files_with_sha256"] if e["kind"] == "config"
            ],
            "fetch_method": manifest["fetch_method"],
            "upstream_license_compatibility_verified": True,
        }
    }
```

### B.4.5 datasets 経由でのデータ取得

```python
from datasets import load_dataset

# revision（コミット SHA）を必ず pin
ds = load_dataset(
    "some-org/some-dataset",
    revision="<40-hex sha>",
    split="train",
    token=os.environ["HF_TOKEN"],
    trust_remote_code=False,          # 任意コード実行を禁止（Ch14 AG-08 関連: repo に紛れ込んだ任意コードが hash 検証をすり抜ける経路）
)

# provenance:
#   dataset_repo_id: "some-org/some-dataset"
#   dataset_revision: "<40-hex sha>"
#   trust_remote_code: false
#   dataset_size_bytes: sum of parquet files
#   dataset_license: from dataset card
```

---

## B.5 MCP サーバ実装（Python SDK 版）

> [!IMPORTANT]
> **MCP サーバは Skill から呼ばれる能力の中で「特に権限を要する部分」を集約する場所**です。ここに承認ゲート・監査 hook・機密資源への窓口を集中させると、Ch4 の agent tier 制御が **API レベルで実装**できます。vol-02 §15 で「作るべきかどうか」を判断した後、本節はその実装編にあたります。

### B.5.1 対象範囲と非対象

**対象**：
- 組織内 GPU クラスタ / MLflow / DVC / データレイクへの窓口を MCP サーバ 1 個に集約
- Train / Infer / Metadata の 3 権限を tool レベルで分離
- Human 承認 hook を tool 呼び出し前に差し込む
- Ch14 監査に必要な cross_reference log を出力

**非対象**（本書では扱わない）：
- MCP transport（stdio / HTTP / WebSocket）の詳細プロトコル
- 認証（SSO / OAuth）連携の全体像 — mTLS / OIDC など既存基盤に委譲
- 高可用性・水平スケールの実装

### B.5.2 最小サーバ骨格

```python
# scripts/mcp_server/server.py
from mcp.server import Server
from mcp.server.models import InitializationOptions, NotificationOptions
import mcp.server.stdio
import mcp.types as types
from typing import Any

server = Server("arim-agentic-mcp")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="submit_training_job",
            description=(
                "Submit a fine-tune job to the GPU cluster. "
                "Requires L2+ agent tier and a valid training_job_approval.approval_record_id. "
                "NOTE: caller identity is derived from the transport session (mTLS peer cert / "
                "verified OIDC token), NOT from any argument."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string"},
                    "skill_version": {"type": "string"},
                    "approval_record_id": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "hp_range_snapshot_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "job_payload_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
                "required": ["skill_name", "skill_version", "approval_record_id",
                             "hp_range_snapshot_hash", "job_payload_hash"],
            },
        ),
        types.Tool(
            name="run_inference",
            description="Run inference on a trusted checkpoint (L1+). URI scheme allowlist enforced.",
            inputSchema={"type": "object", "properties": {
                "checkpoint_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "input_batch_uri": {"type": "string"},
            }, "required": ["checkpoint_sha256", "input_batch_uri"]},
        ),
        types.Tool(
            name="fetch_model_metadata",
            description="Read-only. Fetch skill / checkpoint metadata (L1+).",
            inputSchema={"type": "object", "properties": {
                "query": {"type": "string"},
            }, "required": ["query"]},
        ),
    ]
```

> [!IMPORTANT]
> **`requester_pubkey` を入力スキーマに含めてはいけない**。呼び出し側自身が入力する識別子を信頼すると、L1 agent が「L3 の pubkey を書いた JSON」を送るだけで tier チェックを bypass できます。**caller identity は必ずトランスポート層（mTLS peer cert / 検証済み OIDC token / MCP session context）から取り、`arguments` からは取らない**。以下 B.5.3 の `_resolve_caller_context_from_transport()` はその実装です。

### B.5.3 tool dispatch と権限チェック

> [!WARNING]
> **caller identity は transport から取る**。以下のコードは MCP SDK の `RequestContext` から peer 情報を取り出す想定です（実装は transport 実装依存 — stdio では TLS が使えないので mTLS で HTTP/WS transport を使う、あるいは OIDC 検証を server 側で行う）。

```python
# scripts/mcp_server/dispatch.py
from dataclasses import dataclass
import mcp.types as types

@dataclass
class SubmitResult:
    job_id: str
    scheduler_ledger_entry_id: str
    accepted_at_utc: str
    def to_json(self) -> str:
        import json, dataclasses
        return json.dumps(dataclasses.asdict(self))


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    # 1. caller identity は transport から取る（arguments からは取らない）
    ctx = _resolve_caller_context_from_transport()
    _enforce_tier_for_tool(name, ctx.agent_tier)     # B.5.4

    if name == "submit_training_job":
        # 2. approval → payload binding（Ch13 Gate1 core invariant）
        _require_non_agent_approval(
            approval_record_id=arguments["approval_record_id"],
            skill_name=arguments["skill_name"],
            skill_version=arguments["skill_version"],
            job_payload_hash=arguments["job_payload_hash"],
            hp_range_snapshot_hash=arguments["hp_range_snapshot_hash"],
        )
        _log_audit_event(name, arguments, ctx)                         # B.5.6
        result = _submit_to_scheduler(arguments, ctx)
        return [types.TextContent(type="text", text=result.to_json())]

    elif name == "run_inference":
        _verify_checkpoint_trust(arguments["checkpoint_sha256"])
        _validate_uri_allowlist(arguments["input_batch_uri"])          # 下記
        _log_audit_event(name, arguments, ctx)
        result = _run_inference(arguments)
        return [types.TextContent(type="text", text=result.to_json())]

    elif name == "fetch_model_metadata":
        _log_audit_event(name, arguments, ctx)
        return [types.TextContent(type="text", text=_query_metadata(arguments["query"]))]

    else:
        raise ValueError(f"unknown tool: {name}")


# --- URI allowlist（SSRF / path traversal 防御）---
_ALLOWED_URI_SCHEMES = {"s3", "gs", "abfs"}          # インハウス storage のみ
_ALLOWED_BUCKET_PREFIXES = ("arim-inference-inputs/", "arim-lab-inputs/")

def _validate_uri_allowlist(uri: str) -> None:
    from urllib.parse import urlparse
    parsed = urlparse(uri)
    if parsed.scheme not in _ALLOWED_URI_SCHEMES:
        raise PermissionError(f"scheme not allowed: {parsed.scheme}")
    # path traversal 防御
    if ".." in parsed.path.split("/"):
        raise PermissionError("path traversal blocked")
    # bucket / prefix allowlist
    canonical = f"{parsed.netloc}{parsed.path}".lstrip("/")
    if not any(canonical.startswith(p) for p in _ALLOWED_BUCKET_PREFIXES):
        raise PermissionError(f"bucket/prefix not allowlisted: {canonical}")
```

### B.5.4 tool レベルでの train / infer 権限分離

```python
# scripts/mcp_server/authz.py
from dataclasses import dataclass

@dataclass
class CallerContext:
    principal_id: str
    agent_tier: str              # "L1" | "L2" | "L3"
    is_human: bool
    org_scope: str               # "L-Self" | "L-Lab" | "L-Org" (Ch15)
    transport_verified: bool     # mTLS/OIDC で確認済みか

def _resolve_caller_context_from_transport() -> CallerContext:
    """
    caller identity をトランスポート層から取り出す。
    - stdio: 起動 shell の環境変数 + peer PID 検証（限定的）
    - HTTP/WS: mTLS peer certificate の subject DN、または OIDC access token を
              MCP サーバ側の JWKS 検証で principal_id / tier を得る
    - arguments 由来の identity は絶対に使わない
    """
    # 実装例（transport 実装依存、擬似コード）：
    peer = _current_mcp_session_peer()          # SDK が提供 or transport 側 middleware
    assert peer.verified, "peer identity not verified by transport"
    return CallerContext(
        principal_id=peer.subject_id,
        agent_tier=peer.claims["agent_tier"],   # OIDC claim / cert extension
        is_human=peer.claims["is_human"],
        org_scope=peer.claims["org_scope"],
        transport_verified=True,
    )

TOOL_MIN_TIER = {
    "submit_training_job":  "L2",   # 学習ジョブは L2 以上
    "run_inference":        "L1",   # 推論は L1 でも可
    "fetch_model_metadata": "L1",
    "approve_gate":         "human_only",   # Gate 承認は非 agent 限定 (Ch13)
}

def _enforce_tier_for_tool(tool_name: str, caller_tier: str) -> None:
    required = TOOL_MIN_TIER.get(tool_name)
    if required is None:
        raise PermissionError(f"tool not registered in authz table: {tool_name}")
    if required == "human_only":
        # human_only は本サーバの MCP tool として露出させない（外部承認 UI 経由）
        raise PermissionError("gate approval must go through external Human UI, not MCP")
    order = {"L1": 1, "L2": 2, "L3": 3}
    if order[caller_tier] < order[required]:
        raise PermissionError(
            f"tier {caller_tier} cannot call {tool_name} (requires {required})"
        )
```

### B.5.5 Human 承認 hook（Gate1 の MCP 実装、payload binding 込み）

> [!IMPORTANT]
> **approval を単に「存在する」で通してはいけない**。承認は「特定の skill + version + job payload + hp_range」に紐づくものであり、これらの hash 全てが承認記録側と一致することを確認しないと、benign job の承認を別 job で使い回す攻撃（approval replay）が成立します。**Ch13 Gate1 core invariant = payload binding**。

```python
# scripts/mcp_server/human_gate.py
import re
from datetime import datetime, timezone

def _require_non_agent_approval(
    approval_record_id: str,
    skill_name: str,
    skill_version: str,
    job_payload_hash: str,
    hp_range_snapshot_hash: str,
) -> None:
    """
    第14章 Gate1 を MCP レベルで強制。
    - approval_record_id は 64-hex sha256（形式検証）
    - approval store から non-agent 承認済みかを取得
    - 期限内であること
    - skill_name / skill_version / job_payload_hash / hp_range_snapshot_hash が
      承認記録と全て一致すること（payload binding, Ch13 core invariant）
    """
    assert re.fullmatch(r"[0-9a-f]{64}", approval_record_id), \
        "approval_record_id must be 64-hex sha256"
    assert re.fullmatch(r"[0-9a-f]{64}", job_payload_hash), \
        "job_payload_hash must be 64-hex sha256"
    assert re.fullmatch(r"[0-9a-f]{64}", hp_range_snapshot_hash), \
        "hp_range_snapshot_hash must be 64-hex sha256"

    record = _approval_store.get(approval_record_id)
    if record is None:
        raise PermissionError(f"approval not found: {approval_record_id}")
    if not record["approver_is_non_agent"]:
        raise PermissionError("approver must be a non-agent principal (Gate1 / Ch13)")
    if record["state"] != "approved":
        raise PermissionError(f"approval state = {record['state']} (must be 'approved')")
    now = datetime.now(timezone.utc)
    expiry = datetime.fromisoformat(record["expiry_at_utc"])
    if now > expiry:
        raise PermissionError(f"approval expired at {expiry}")

    # ★ payload binding — Ch13 Gate1 の core invariant
    if record["approved_skill_name"] != skill_name:
        raise PermissionError(
            f"skill_name mismatch: approved={record['approved_skill_name']}, got={skill_name}"
        )
    if record["approved_skill_version"] != skill_version:
        raise PermissionError(
            f"skill_version mismatch: approved={record['approved_skill_version']}, got={skill_version}"
        )
    if record["approved_payload_hash"] != job_payload_hash:
        raise PermissionError(
            "job_payload_hash mismatch — approval was for a different job payload "
            "(approval replay prevented, Ch13 Gate1)"
        )
    if record["approved_hp_range_hash"] != hp_range_snapshot_hash:
        raise PermissionError(
            "hp_range_snapshot_hash mismatch — hp range differs from what was approved "
            "(Ch4 approved_hp_range binding)"
        )
```

> [!TIP]
> Human 承認 UI 自体は本サーバの外部（Web UI + SSO + 電子署名）に置くのが自然です。MCP サーバは「承認済み記録の検証」だけを担当し、「承認手続きそのもの」を担わせないことで責務境界が明確になります（第16章 §16.6 の責任分担 3 分割と一致）。

### B.5.6 監査 log と cross-reference

> [!WARNING]
> **`prev_hash` chain には TOCTOU race がある**。asyncio の並行 `call_tool` や multi-process デプロイで、2 つのエントリが同じ `prev_hash` を読んで両方 append すると chain が silently fork します。**必ずファイルロック（`fcntl.flock` LOCK_EX）または単一 writer タスクで直列化**してください。

```python
# scripts/mcp_server/audit.py
import fcntl
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

AUDIT_LOG = Path("/var/log/arim-mcp/audit.jsonl")  # append-only（ext4 の chattr +a 推奨）

def _log_audit_event(tool_name: str, arguments: dict, ctx: "CallerContext") -> None:
    """
    Ch14 §15.8 audit_result_provenance の元データを吐き出す。
    - append-only ファイル、prev_hash chain（tamper-evident）
    - fcntl.flock で read-prev + append の critical section を直列化
    - Ch14 canonical: registry_hash_at_run_start を出す
    """
    entry_core = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "principal_id": ctx.principal_id,
        "agent_tier": ctx.agent_tier,
        "is_human": ctx.is_human,
        "transport_verified": ctx.transport_verified,
        "arguments_hash": hashlib.sha256(
            json.dumps(arguments, sort_keys=True).encode()
        ).hexdigest(),
        "server_version": _SERVER_VERSION,
        "registry_hash_at_run_start": _current_registry_hash(),   # Ch14 canonical
        "registry_version_at_run_start": _current_registry_version(),
    }

    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG.open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)          # 排他ロック
        try:
            # ロック取得後に prev_hash を読み直す（race 解消）
            entry_core["prev_hash"] = _last_entry_hash(AUDIT_LOG)
            entry_core["self_hash"] = hashlib.sha256(
                json.dumps(entry_core, sort_keys=True).encode()
            ).hexdigest()
            f.write(json.dumps(entry_core) + "\n")
            f.flush()
            import os
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

**5 種の外部 ledger（Ch14 canonical）と MCP tool の対応**：

| Ledger | MCP tool で生成 | Ch14 で cross-reference |
|---|---|---|
| scheduler_ledger | `submit_training_job` | Skill provenance の `training_job_approval.approval_record_id` と突合 |
| inference_ledger | `run_inference` | Gate1/2/3 発火時系列と突合 |
| model_registry_ledger | `submit_training_job` の成果 | `weights_sha256` と `checkpoint_registry.jsonl` の突合 |
| gate_approval_ledger | `_require_non_agent_approval` 呼出 | provenance `gate_state_machine` と突合 |
| revocation_ledger | separate (Ch15) | `model_distribution_contract_ref.revocation` と突合 |

### B.5.7 起動と client 側からの利用

```python
# scripts/mcp_server/__main__.py
import asyncio
import mcp.server.stdio
from mcp.server.models import InitializationOptions, NotificationOptions

from .server import server                       # B.5.2 で定義した Server

async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(
            read, write,
            InitializationOptions(
                server_name="arim-agentic-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
```

Client 側（agentic runner）は MCP プロトコルで tool を呼び出します。**agent は `approve_gate` を呼べない**（tier 制約）、**`submit_training_job` は事前に取得した `approval_record_id` を必ず渡す**、という制約が API レベルで強制されます。

### B.5.8 テスト観点

**基本の正常系 / 認可系**：

- [ ] `submit_training_job` に古い `approval_record_id`（期限切れ）を渡すと `PermissionError`
- [ ] `submit_training_job` の `job_payload_hash` と approval の `approved_payload_hash` が一致しない場合に拒否（**approval replay 防止 / Blocking-2 対応**）
- [ ] `submit_training_job` の `hp_range_snapshot_hash` と approval の `approved_hp_range_hash` が一致しない場合に拒否
- [ ] approval の `approved_skill_name` / `approved_skill_version` と一致しない場合に拒否
- [ ] L1 agent が `submit_training_job` を呼ぶと `PermissionError`
- [ ] `run_inference` に未登録 `checkpoint_sha256` を渡すと拒否

**トランスポート識別のなりすまし系（Blocking-1 対応）**：

- [ ] Client が JSON に細工した `requester_pubkey` / `agent_tier` を含めても、`_resolve_caller_context_from_transport()` の結果が優先され、`arguments` 由来の identity は完全に無視される
- [ ] transport 検証に失敗した peer（自己署名 cert, 期限切れ OIDC token）が呼ぶと全 tool で拒否
- [ ] L1 client が `submit_training_job` の tool 名だけを偽装しても tier チェックで塞がれる

**URI allowlist（Non-Blocking-10 対応）**：

- [ ] `run_inference` に `file:///etc/passwd` を渡すと拒否
- [ ] `run_inference` に `s3://other-tenant/xxx` を渡すと bucket prefix allowlist で拒否
- [ ] `../` を含む path を渡すと拒否

**Audit log 並行 append race（Non-Blocking-6 対応）**：

- [ ] N 並列 `call_tool` を発行して audit log entry が N 個生成され、`prev_hash` chain が単一列として完全であること（fork なし）
- [ ] flock を無効化した control 実装では chain fork が検出できることを確認（テストで実装のガードを検証）
- [ ] audit log を後から編集すると `self_hash` 再計算で検知される

**FM 取得（B.4 の Blocking-3 対応）**：

- [ ] manifest の `tokenizer.json` sha256 を 1 bit 変えて配置すると `load_verified_fm` が拒否
- [ ] `config.json` を差し替えると拒否
- [ ] `files_with_sha256` に `weights` / `tokenizer` / `config` の 3 kind すべてが揃っていないと拒否
- [ ] `revision` が短縮 SHA / tag 名の場合に `verify_revision` が拒否

---

## B.6 まとめ

- **PyTorch / JAX の決定論設定**は起動時 1 回で終わらせ、provenance に 7 項目（seed, worker seed, cudnn_deterministic 等）を必ず記録。`CUBLAS_WORKSPACE_CONFIG` は **`import torch` より前**に launcher で export
- **PyTorch 2.x では `torch.amp.*`** を使う（`torch.cuda.amp.*` は deprecated）
- **DataLoader worker のシード伝播**が抜けると Ch14 DG-01 に該当
- **Hugging Face から FM を取得**する場合は `40-hex revision` + `HfApi resolved sha` + `snapshot_download(allow_patterns)` + `weights + tokenizer + config` を全て file-level sha256 で検証 + `safetensors のみ` の 5 点セットが最小
- **`huggingface_hub.login()` は避ける**（`~/.cache/huggingface/token` に書き出す）。`token=os.environ["HF_TOKEN"]` を個別 API に直接渡す
- **MCP サーバ**は組織内の「特に権限を要する能力」を集約する場所として設計する。tool 名で train / infer / metadata を明確に分け、tier 制約を authz テーブルで宣言的に管理
- **caller identity は transport から取る**（arguments 由来の pubkey は絶対に信頼しない）
- **Human 承認 hook** は MCP tool の入口に置き、UI 手続きは外部委譲。**承認は skill_name + version + payload_hash + hp_range_hash に binding**（approval replay 防止、Ch13 Gate1 core invariant）
- **audit log** は tamper-evident（append-only + prev_hash chain + `fcntl.flock`）にし、5 種の ledger を Ch14 cross-reference の元データにする

## 参考資料

### 本書内の該当章

- 第5章：agent tier 定義、`agent_authorization`, `training_job_approval`, `approved_hp_range`
- 第12章：Foundation Model 取得、`fm_fetch_and_verify`, `snapshot_download`
- 第14章：Gate1/2/3 の役割、`integrated_provenance_chain`
- 第15章：`agentic_deep_failure_audit`, `cross_reference_findings`, 5 種の外部 ledger
- 第16章：組織 GPU 政策、`model_distribution_contract`, `responsibility_ledger`
- 付録A：provenance 完全スキーマ、`SKILL.md` テンプレート

### 外部参考

- PyTorch reproducibility notes: https://pytorch.org/docs/stable/notes/randomness.html
- PyTorch Distributed (DDP): https://pytorch.org/docs/stable/distributed.html
- JAX pseudo random numbers: https://jax.readthedocs.io/en/latest/random-numbers.html
- Flax documentation: https://flax.readthedocs.io/
- Optax: https://optax.readthedocs.io/
- safetensors: https://github.com/huggingface/safetensors
- huggingface_hub client: https://huggingface.co/docs/huggingface_hub/index
- Hugging Face Transformers: https://huggingface.co/docs/transformers/index
- Model Context Protocol (MCP) specification: https://spec.modelcontextprotocol.io/
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- Sigstore（承認・監査 log の署名参考）: https://www.sigstore.dev/
