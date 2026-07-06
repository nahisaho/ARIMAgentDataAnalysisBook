# 第12章　Batch BO と並列実験の Skill 化

> [!IMPORTANT]
> **本章の到達目標**
>
> 1. Batch BO (q>1) の骨格（q-EI / q-EHVI / local penalization）を Skill 契約の言葉で説明できる
> 2. **エージェントが同時に $q$ 個提案するときの多様性保証** を書ける
> 3. `batch_size` と `pending_experiments` の provenance を operational 化できる
> 4. **BO vs BOCS**（組合せ空間で BO 直接適用の限界）の判断表を書ける
> 5. Ax による experiment tracking の Skill 化（BoTorch の上に Ax を組む場合）
>
> **本章で扱わないこと**
>
> - 単目的 acquisition の骨格 → **第7章**
> - 多目的 / 制約 → **第9章 / 第10章**（本章で batch と組み合わせる）
> - Multi-task GP → **第11章**（本章 `cross_facility` を継承）
> - Active Learning → **第13章**
> - Advanced Capstone → **第14章**

---

## 12.1 なぜ Batch BO か

材料実験には **並列度** があります。3 台の装置、8 サンプルの multi-well plate、24 時間で複数実験——**次の 1 点だけ提案する sequential BO はもったいない**。同時に $q$ 個提案する **batch BO** が必要です。

課題：$q$ 個の候補は **お互いに独立に選んではいけない**。全部同じ場所を提案しては情報が重複する。**多様性を保証**する必要がある。

Batch BO の骨格：

- **joint acquisition**：$q$ 個の候補を joint に最大化（$\mathbf{X}_q$ を同時に最適化）
- **sequential greedy**：1 個ずつ選び、選んだ点を pending として次の acquisition に fantasize（第11章 §11.5 の cross_facility と同じ考え方）
- **local penalization**：既に選んだ点の近傍で acquisition を penalize
- **Thompson sampling**：$q$ 個の posterior sample を引き、それぞれの argmax を候補に

**BoTorch の既定**：q-Log-EI / q-Log-EHVI が joint acquisition を **`sample_shape` で並列サンプリング**、`optimize_acqf(q=Q)` で joint 最適化。

---

## 12.2 多様性保証の Skill 契約

エージェントが $q$ 個を提案するとき、**多様性がない**（全部似た点）と情報獲得が減ります。契約で保証：

```yaml
batch_diversity_policy:
  version: "v0.2"
  batch_size: 4                                # q
  strategy: "joint_acquisition"                # enum: joint_acquisition | sequential_greedy | thompson_sampling | local_penalization
  minimum_pairwise_distance:                   # 候補間の最小距離（feature space 単位）
    metric: "standardized_euclidean"           # enum: standardized_euclidean | mahalanobis | gower
    threshold: 0.3                             # per-dim std の 0.3 倍以上
    threshold_source: "human_declared"         # enum: human_declared | derived_from_length_scale
    enforcement: "post_hoc_skill_validation"   # enum: post_hoc_skill_validation | inline_penalization
  fallback_on_violation: "sequential_greedy_replan"  # enum: sequential_greedy_replan | fatal | shrink_batch
  diversity_metrics_logged: true               # Group C provenance へ
```

**運用ルール**：

- **`joint_acquisition` 自体は多様性を保証しない**。BoTorch の joint q-acquisition は flat/noisy/discrete な landscape で近接点を返し得る。したがって Skill は **候補生成後に pairwise distance を post-hoc 検証**し、閾値違反なら `fallback_on_violation` に従う
- `local_penalization` / `sequential_greedy` は多様性を高めるが hard guarantee ではない。最終的な閾値検証は Skill 側の責務
- joint acquisition が閾値を守れなかった場合、`fallback_on_violation` に応じて再計画
- 候補間距離は Group C の `batch_diversity_metrics` に記録、audit 可能

> [!IMPORTANT]
> **`minimum_pairwise_distance` を Skill が勝手に緩めるのは fatal**（第3章 §3.5 逸脱と同種）。閾値変更は新 run。

---

## 12.3 pending_experiments の provenance

**pending experiments**：候補として選ばれたが、まだ実験結果が返ってきていない点。次回 acquisition では **fantasize**（posterior 期待値で仮の結果を挿入）して重複提案を避ける。

```yaml
pending_experiments:
  version: "v0.2"
  entries:                                      # immutable submission records（append-only、in-place の status 変更不可）
    - experiment_id: "exp-2026-1122-A01"
      facility_id: "facility_A"
      x:                                        # search_space_bounds.variables 由来の human-readable キー
        x_temperature: 550.0
        x_pressure: 2.3
      submitted_at: "2026-11-22T09:00:00Z"
      expected_completion: "2026-11-22T15:00:00Z"
      authorization_id: "approval:HITL-20261122-A01"
    - experiment_id: "exp-2026-1122-A02"
      facility_id: "facility_B"
      x:
        x_temperature: 620.0
        x_pressure: 3.1
      submitted_at: "2026-11-22T09:15:00Z"
      expected_completion: "2026-11-22T16:00:00Z"
      authorization_id: "approval:HITL-20261122-A02"
  # 現在の status は entries から derive せず、append-only の event streams から算出：
  #   pending_status_events (in_progress→completed)、cancellation_events、failure_events
  fantasize_policy:
    method: "botorch_internal_fantasize"       # enum: botorch_internal_fantasize | posterior_sample | posterior_mean | fixed_value
    # botorch_internal_fantasize: X_pending を acquisition に渡し、BoTorch 内部の MC sampler で fantasize（qNEI/qLogNEI/qLogNEHVI の既定）
    # posterior_sample:            Skill が事前に posterior sample を引き、pseudo-observation として materialize
    # posterior_mean:              constant-liar（近似、bias 大）
    # fixed_value:                 debugging 用
    num_samples: 128                            # posterior_sample の場合の MC サンプル数
  cancellation_policy: "requires_human_approval"  # enum: requires_human_approval | fatal
  tensor_encoding_contract:
    # entries[*].x は human-readable。Skill が X_pending_tensor を構築するとき、以下を deterministic に生成：
    #   - facility_id → task_index_spec.task_map[facility_id]（Ch11 §11.3）
    #   - categorical → one-hot（Ch6/Ch8 の混合カーネル契約）
    #   - task feature は連続 bounds normalization から除外、fixed/discrete として扱う
    derivation_deterministic: true
```

### 12.3.1 pending 実験のキャンセルは fatal 相当

**Agentic 特有の失敗パターン**：エージェントが「実験が遅い」と判断して pending 実験を勝手にキャンセル → provenance の不整合。

- **契約**：`pending_experiments.entries[*].status: in_progress` を Skill が `cancelled` に変更するのは fatal
- **Human キャンセル**：`cancellation_policy: requires_human_approval` の場合のみ、`cancellation_events` に append-only 記録：

```yaml
cancellation_events:                             # append-only、hash chain で改竄防止（Ch10 event schema 準拠）
  - event_id: "evt-cancel-20261122-001"
    event_hash: "sha256:aaaa..."                # canonical serialization: sort_keys=True, UTF-8, no trailing newline
    previous_event_hash: "sha256:0000..."       # 直前の event_hash（chain の起点は all-zeros）
    experiment_id: "exp-2026-1122-A01"
    cancelled_at: "2026-11-22T14:00:00Z"
    cancelled_by: "human:staff_id_XXXX"
    reason: "装置トラブルで実験中断"
    approval_id: "approval:HITL-20261122-C03"
    run_id: "run-2026-1122-01"
    skill_execution_id: "skill-exec-XYZ"
    pending_contract_hash: "sha256:..."         # 対象 pending_experiments (v0.2) の hash
```

---

## 12.4 BoTorch q-EI / q-EHVI

### 12.4.1 qLogNEI (joint acquisition with pending fantasize)

qLogNEI（qLogNoisyExpectedImprovement）は `X_pending` を受け取り、pending 点を **BoTorch 内部で fantasize**（posterior sample を pseudo-observation として MC 積分）する。`X_baseline` は既観測点。true qLogEI（`best_f` 指定、pending 非対応）とは異なり、noisy かつ pending ありの batch BO で既定推奨。

```python
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf

# X_pending は tensor_encoding_contract に従って構築：
#   - facility_id → task_index_spec.task_map[facility_id] の整数
#   - 連続 bounds normalization から task 次元を除外
X_pending_tensor = build_pending_tensor(pending_experiments.entries, task_index_spec)

acq = qLogNoisyExpectedImprovement(
    model=model,
    X_baseline=X_obs,
    X_pending=X_pending_tensor,     # 未完了の候補（BoTorch 内部で fantasize）
    prune_baseline=True,
)

candidates, acq_value = optimize_acqf(
    acq_function=acq,
    bounds=bounds,                  # task 次元は fixed_features で別扱い
    q=4,                            # batch size
    num_restarts=10,
    raw_samples=512,
)
# candidates: (q, d)
```

> [!NOTE]
> canonical enum: `acquisition_spec.name = "qLogNEI"`（第9章と統一）。true qLogEI（pending 非対応、`best_f` 指定）を使う場合は `qLogExpectedImprovement`。

### 12.4.2 Sequential greedy (fantasize per candidate)

BoTorch の `optimize_acqf(q=Q, sequential=True)`：内部で 1 個ずつ選び、fantasize する。joint より高速、joint より局所最適に落ちやすい。

> [!NOTE]
> `sequential=True` は多様性の hard guarantee ではなく、acquisition 種別によって挙動が異なる（qLogNEI/qLogNEHVI は対応、qKG は非対応など）。Skill は §12.2 の post-hoc 検証を必ず実施。

```python
candidates, _ = optimize_acqf(
    acq_function=acq,
    bounds=bounds,
    q=4,
    num_restarts=10,
    raw_samples=512,
    sequential=True,         # sequential greedy
)
```

### 12.4.3 選択判断表

```
[Q1] Batch size q は？
  ├─ q <= 3   → joint acquisition（q-EI / q-EHVI）
  ├─ q 4-8    → joint (慎重に num_restarts と raw_samples を増やす)
  └─ q >= 10  → sequential greedy or Thompson sampling

[Q2] 制約あり？
  → 第10章 constraints callable を acq に渡す

[Q3] 多目的？
  → 第9章 qLogNEHVI（joint / sequential 両対応）

[Q4] Multi-task（複数装置）？
  → 第11章 MTGP を surrogate に、pending_experiments.entries[*].facility_id を tensor_encoding_contract に従って task_index に変換して X_pending に含める（Ch11 §11.5）
```

---

## 12.5 BO vs BOCS — 組合せ空間での限界

BO は **連続 or 混合空間** を得意とします。**純粋な組合せ空間（binary vectors、permutations、分子グラフ）** では、BOCS (Bayesian Optimization of Combinatorial Structures) のような専用手法の方が良いことが多い。

### 12.5.1 判断表

| 探索空間 | 推奨手法 | 理由 |
|---|---|---|
| 連続変数のみ | BO (Matérn 5/2 GP) | 標準的な適用場面 |
| 連続 + 少数カテゴリ (K<10) | BO (MixedSingleTaskGP + Hamming) | Ch6/Ch8 の混合カーネル |
| 高次元 binary ($d \geq 20$) | **BOCS** or SMAC | BO の GP はスケールしない、二次モデルベースの方が高速 |
| Permutations | **専用 GA / Bayesian Kernel over permutations** | 標準 kernel が定義しにくい |
| 分子グラフ | **Graph BO (JT-VAE + GP on latent) / MolBO** | 構造情報を捉える必要 |

**Skill 契約**：`search_space_bounds` の型から、BO が適用可能か **決定論的規則** で自動判定：

- **判定規則（優先順、上から順に評価）**：
  1. **permutation / graph 検出** → `alternative: permutation_kernel_bo` または `graph_bo`、`status: not_recommended`
  2. **binary_vector dim ≥ 20** → `alternative: bocs`（dim ≥ 50 なら `smac`）、`status: not_recommended`
  3. **categorical 累積 cardinality ≥ 50**（複数 categorical の choices 積算）→ `alternative: specialized_combinatorial_optimizer` または `genetic_algorithm`、`status: not_recommended`
  4. **連続 + 少数 categorical**（各 K < 10、累積 < 50、binary_vector dim < 10）→ `status: mixed_supported`（MixedSingleTaskGP + Hamming、Ch6/Ch8）
  5. **連続のみ**（effective 次元 ≤ 30）→ `status: recommended`
  6. **連続のみ・高次元**（effective 次元 > 30）→ `status: mixed_supported`（TuRBO / SAAS-BO を推奨、Ch8）

```yaml
search_space_bounds:
  variables:
    x_temperature: {type: "continuous", low: 200, high: 800}
    x_material_class: {type: "categorical", choices: ["A", "B", "C", "D"]}
    x_dopant_pattern: {type: "binary_vector", dim: 32}   # ← 高次元 binary
  bo_applicability:
    status: "not_recommended"                    # enum: recommended | mixed_supported | not_recommended
    reason: "binary_vector dim=32 exceeds typical GP capacity (rule 2); consider BOCS"
    alternative: "bocs"                          # enum: bocs | smac | molbo | graph_bo | permutation_kernel_bo | genetic_algorithm | specialized_combinatorial_optimizer | none
    rule_applied: 2                              # 判定規則の番号（provenance）
```

`bo_applicability.status: not_recommended` の場合、Skill は BO 実行を停止し Human に代替提案。

---

## 12.6 Ax による experiment tracking

**Ax** は Facebook 製の experiment management layer。BoTorch を optimization engine として使いつつ、trial / experiment / metric / arm を **DB 的に管理**する。

### 12.6.1 Ax vs BoTorch 単体

| 観点 | BoTorch 単体 | Ax (BoTorch 上) |
|---|---|---|
| optimization | native | Ax の Modelbridge 経由 |
| experiment tracking | 自前実装 | Ax の experiment DB |
| trial 管理 | provenance に手書き | `ax.core.Experiment` |
| Human interaction | 自前 | Ax Service API（HTTP / Python） |
| 学習コスト | 低 | 中（Ax の抽象化を学ぶ） |
| Skill 契約との相性 | 直接的 | Ax の trial ↔ pending_experiments の対応が必要 |

**推奨**：本書の Skill 契約は BoTorch 単体で書きやすいが、**チーム運用で複数実験を管理する場合は Ax を推奨**。Ax の `Trial` を `pending_experiments.entries` に対応させる bridge を設計する。

### 12.6.2 Ax の最小骨格

```python
from ax.service.ax_client import AxClient, ObjectiveProperties

ax_client = AxClient()
ax_client.create_experiment(
    name="hardness_density_mtgp",
    parameters=[
        {"name": "x_temperature", "type": "range", "bounds": [200.0, 800.0]},
        {"name": "x_pressure", "type": "range", "bounds": [0.1, 5.0]},
        {"name": "facility_id", "type": "choice", "values": ["A", "B", "C"]},
    ],
    objectives={
        "hardness": ObjectiveProperties(minimize=False),
        "density": ObjectiveProperties(minimize=True),
    },
    outcome_constraints=["yield_pct >= 60"],
)

# get next trial
parameters, trial_index = ax_client.get_next_trial()
# ... run experiment, get result ...
ax_client.complete_trial(trial_index=trial_index, raw_data={
    "hardness": (12.5, 0.3),   # (mean, sem)
    "density":  (4.2, 0.1),
    "yield_pct":(78.0, 2.0),
})
```

> [!NOTE]
> Ax の `outcome_constraints` は **モデルベースの soft constraint**（cEI 相当、violation を許容しつつ acquisition で penalize）。**Ax の parameter constraints**（`ParameterConstraint`）は search space 上の線形制約で BoTorch/Ax の候補生成側で処理されるが、これも **物理・法的な hard constraint の防護には不十分**。Ch10 の hard_constraints（物理・法的な違反禁止）は Ax の外側、Skill の `pre_authorization_filter` で担当する。**Ax は「実験管理と soft/parameter constraint」を担い、hard constraint 防護は Skill 側の責任**。

---

## 12.7 Agentic 特有の失敗パターン契約

Batch × pending × multi-task が絡む場面で、エージェント自律実行が生む固有の落とし穴：

- **stale pending TTL**：pending 実験が `expected_completion` を超過しても Skill は自動キャンセルしない。TTL 超過は `stale_pending_events` に記録し、**Human へエスカレーション**（Ch10 の Layer 3 相当）。TTL の緩和は fatal
- **existing-pending との重複検査**：多様性検査は batch 内 pairwise だけでなく、`pending_experiments.entries[*].x` との pairwise も含める。同装置で既に走っている点の近傍を新規提案しない
- **batch drift**：候補生成（`optimize_acqf`）と実験投入（authorization_id 発行）の間で model / observed data / constraints が変わっていないことを検証。変化していたら再計画（`batch_drift_events` に記録）
- **Ax trial desync**：Ax `Trial` と `pending_experiments.entries` は bijective。片側だけ増える／減る状態は fatal
- **duplicate candidates 検出**：joint acquisition が同一点を返した場合、§12.2 の fallback を実行し、`duplicate_candidate_events` に記録（Ch15 参照）

---

## 12.8 Skill 契約チェックリスト

- [ ] `batch_diversity_policy.batch_size` が明示
- [ ] `strategy` が canonical enum
- [ ] `minimum_pairwise_distance.threshold` が Human 承認済み
- [ ] Skill が候補生成後に **post-hoc pairwise distance 検証** を実施
- [ ] pairwise 検証は batch 内 + 既存 `pending_experiments.entries` との比較両方を含む
- [ ] `pending_experiments` v0.2：entries は immutable、status は event streams から derive
- [ ] `cancellation_events` に `event_hash` + `previous_event_hash` の hash chain
- [ ] pending 実験の Skill 自律キャンセルは fatal
- [ ] `fantasize_policy.method` が明示（既定 `botorch_internal_fantasize`）
- [ ] `tensor_encoding_contract.derivation_deterministic: true`（facility_id → task_index、categorical → one-hot が deterministic）
- [ ] BoTorch call で `X_pending` が pending_experiments から tensor_encoding_contract 経由で正しく供給される
- [ ] `bo_applicability.status` が `not_recommended` なら Skill が BO 停止、`rule_applied` が記録
- [ ] Ax 併用時、Ax `Trial` と `pending_experiments.entries` が bijective に対応
- [ ] Ax の `outcome_constraints` / parameter constraints を hard constraint と誤用しない
- [ ] Multi-task と併用時、`X_pending_tensor` に task_index を含める（Ch11 §11.5）
- [ ] `stale_pending_events` / `batch_drift_events` / `duplicate_candidate_events` を Group C に記録

---

## 12.9 章末演習

**問 1**：`batch_size: 4`、joint acquisition が同じ点を 2 個提案した。Skill の正しい応答は？

**問 2**：pending 実験が 24 時間経っても completed にならない。エージェントは何をすべきか。

**問 3**：`x_dopant_pattern: binary_vector, dim: 40` に対して Skill はどう振る舞うべきか。

**問 4**：Ax の `outcome_constraints` を hard constraint に流用しようとするエージェント。契約上の正しい対応は？

**問 5**：q=8 の joint acquisition が num_restarts=10 で不安定。設定変更の Human 承認プロセスは？

---

## 12.10 参考資料

### 本書内

- 第3章 §3.5：Skill 逸脱パターン（pending キャンセル、閾値緩め）
- 第5章 §5.2：`batch_size`, `pending_experiments`, `search_space_bounds` の初出
- 第6章 / 第8章：混合カーネル（Matérn + Hamming）と one-hot エンコーディング契約
- 第7章：acquisition 骨格
- 第9章：多目的 (q-EHVI / qLogNEHVI との組み合わせ)
- 第10章：制約付き BO (constraints callable / event_hash schema を batch でも共有)
- 第11章：Multi-task GP (cross_facility pending, task_index_spec)
- 第13章（planned）：Active Learning（BO と使い分け）
- 第14章（planned）：capstone
- 第15章（planned）：duplicate candidates、stale pending、batch drift の失敗パターン

### 外部参考

- Wilson, Hutter, Deisenroth, "Maximizing acquisition functions for Bayesian optimization", NeurIPS 2018 — q-EI の joint 最適化
- Balandat et al., "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization", NeurIPS 2020 — BoTorch の q-acquisition の理論的基盤
- Baptista, Poloczek, "Bayesian Optimization of Combinatorial Structures", ICML 2018 — BOCS 原論文
- Bakshy et al., "AE: A domain-agnostic platform for adaptive experimentation", NeurIPS 2018 — Ax の前身
- Ax documentation / BoTorch tutorials

> [!NOTE]
> 次章（第13章）では、Active Learning と BO の使い分けを扱います。「最適化」と「学習」の目的分離、uncertainty sampling、Emukit / modAL の Skill 化。
