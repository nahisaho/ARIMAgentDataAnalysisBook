# 第11章　マルチ装置・マルチタスク BO — 階層 GP を Skill 化する

> [!IMPORTANT]
> **本章の到達目標**
>
> 1. Multi-task GP と hierarchical GP の違いを Skill 契約の言葉で説明できる
> 2. **装置ごとに異なる目的関数を階層事前分布で共有する** Skill を設計できる
> 3. `surrogate_model_family: multi_task_gp | hierarchical_gp` の operational 契約を書ける
> 4. ARIM 施設内の複数装置での並行探索の設計判断ができる
>
> **本章で扱わないこと**
>
> - GP の基本 → **第6章**
> - Random Forest / BNN / DKL → **第8章**
> - Batch BO（q>1 の並列 acquisition）→ **第12章**
> - 装置差 × 制約 × 多目的の統合ケース → **第14章 capstone**
>
> **前提**：vol-02 第11章の階層モデル（partial pooling）に慣れていること

---

## 11.1 なぜ装置ごとに GP を分けるのか / 共有するのか

ARIM 施設には**同種類だが機差のある複数装置**があります。例：3 台の PLD（Pulsed Laser Deposition）装置——同じレシピでも装置 A / B / C で微妙に結果が異なる（レーザ強度、基板ホルダー角度、真空度の個体差）。

3 つのアプローチ：

| アプローチ | 特徴 | 適用場面 | 欠点 |
|---|---|---|---|
| **装置ごとに独立 GP** | 装置 A の観測は装置 B に一切影響しない | データが装置ごとに豊富、装置差が構造化されていない | 早期は各装置でデータ不足、cold start |
| **単一 GP（装置差無視）** | 装置 A/B/C の観測を pool、`facility_id` は無視 or one-hot | 装置差が小さい / データ極少 | 装置差を学習不能、バイアス発生 |
| **Multi-task / Hierarchical GP** | 装置間で **partial pooling**：一部の構造は共有、装置固有の deviation は個別 | 装置差はあるが同種装置、データが中程度 | 実装が複雑、prior 設計が必要 |

**Multi-task / hierarchical GP は、cold start と装置差学習を両立**する Skill 設計上の主要選択肢です。

---

## 11.2 Multi-task GP と Hierarchical GP の違い

**両者は連続的だが、実装と Skill 契約が異なります**。

### 11.2.1 Multi-task GP (MTGP / ICM)

**Intrinsic Coregionalization Model (ICM)**：$M$ 個の task（装置）を kernel レベルで結合。

$$
k_{\text{MTGP}}\bigl((x, i), (x', j)\bigr) = k_{\text{input}}(x, x') \cdot B_{ij}
$$

- $k_{\text{input}}$：入力 $x$ に関する共通 kernel（Matérn 5/2 など）
- $B$：**task-task correlation matrix** $M \times M$、学習可能パラメータ（正定値）
- $i, j$：task index（`facility_id`）

**強み**：装置間相関 $B_{ij}$ を明示的に学習、対称的な情報共有
**弱み**：$B$ の学習は task 数 $M$ が大きいと不安定、階層構造を強制できない

### 11.2.2 Hierarchical GP

**装置固有の GP + 上位共通 mean/kernel の階層事前分布**：

$$
f_i(x) = f_{\text{global}}(x) + \delta_i(x), \quad \delta_i(x) \sim \mathcal{GP}(0, k_{\text{dev}})
$$

- $f_{\text{global}}$：全装置共通の trend GP
- $\delta_i$：装置 $i$ 固有の deviation GP
- Length scale $\ell_{\text{dev}}$ は $\ell_{\text{global}}$ から階層事前分布で導出

**強み**：階層構造を明示、装置固有 deviation の大きさを制御可能
**弱み**：BoTorch native ではなく、GPyTorch カスタム実装が主流

### 11.2.3 選択判断表

```
[Q1] 装置数 M は？
  ├─ 2-3      → Hierarchical GP（明示的階層で管理しやすい）
  ├─ 4-10     → Multi-task GP (ICM)（相関行列の学習が現実的）
  └─ 10 以上  → 追加の階層化 or 装置クラスタリング（本章スコープ外）

[Q2] 装置差の大きさが事前に分かっているか？
  ├─ YES → Hierarchical GP（deviation prior に事前情報を入れやすい）
  └─ NO  → Multi-task GP（B を学習）

[Q3] 装置ごとの目的関数が同じ（同じ y）か違うか？
  ├─ 同じ y、装置差のみ → 本章の MTGP / HierGP
  ├─ 装置ごとに別 y     → 各 y に独立 MTGP、または vector-valued GP（本章スコープ外）
  └─ 装置ごとに部分共通 → Human 承認付き対応表を設計
```

---

## 11.3 Skill 契約 — `surrogate_model_family` の operational 化

### 11.3.1 Multi-task GP

```yaml
surrogate_model_family: "multi_task_gp"        # canonical enum

kernel_spec:
  input_kernel:
    type: "matern_5_2"
    ard: true
    length_scale_prior:
      type: "gamma"
      concentration: 3.0
      rate: 6.0
  task_kernel:
    type: "icm"                                # enum: icm | lcm
    num_tasks: 3                               # facility_id 数
    rank: 2                                    # low-rank approximation、rank<=num_tasks
    task_correlation_prior:
      type: "lkj"
      shape: 2.0
  mean_function:
    type: "constant"
  likelihood:
    type: "gaussian"
    noise_prior:
      type: "gamma"
      concentration: 1.1
      rate: 0.05

task_index_spec:
  variable: "facility_id"                      # 入力 x のうち task index を担う変数
  encoding: "integer"                          # enum: integer | one_hot（ICM は integer）
  task_map:                                    # facility_id と整数 task index の対応
    "facility_A": 0
    "facility_B": 1
    "facility_C": 2
  new_task_policy: "requires_human_approval"   # enum: requires_human_approval | fatal
```

### 11.3.2 Hierarchical GP

```yaml
surrogate_model_family: "hierarchical_gp"

kernel_spec:
  global_kernel:
    type: "matern_5_2"
    ard: true
    length_scale_prior:
      type: "gamma"
      concentration: 3.0
      rate: 6.0
  deviation_kernel:
    type: "matern_5_2"
    ard: true
    length_scale_prior:
      type: "hierarchical_from_global"         # 上位 global length scale から派生
      shrinkage_alpha: 0.5                     # deviation length scale の shrinkage
    output_scale_prior:
      type: "half_normal"
      scale: 0.3                               # deviation の振幅制限、事前情報
  mean_function:
    type: "shared_constant"                    # 全 task 共通の constant
  likelihood:
    type: "gaussian"
    noise_prior:
      type: "gamma"
      concentration: 1.1
      rate: 0.05

task_index_spec:
  variable: "facility_id"
  encoding: "integer"
  task_map:
    "facility_A": 0
    "facility_B": 1
    "facility_C": 2
  new_task_policy: "requires_human_approval"
```

### 11.3.3 新装置の追加ポリシー

**装置 D を新規追加**するとき：

- `new_task_policy: "requires_human_approval"` の場合：
  - Skill 実行中の追加は fatal（context 変更 = 新 run）
  - 追加時は `task_map` に entry を追加、新 run で開始
- `new_task_policy: "fatal"` の場合：
  - 追加は許可されず、Skill が停止

> [!IMPORTANT]
> `task_map` の変更は第3章 §3.5 逸脱 3 (context 変更) と同種。Skill 自律での追加は fatal。

---

## 11.4 BoTorch / GPyTorch 実装

### 11.4.1 MultiTaskGP (ICM) — BoTorch native

```python
import torch
from botorch.models import MultiTaskGP
from gpytorch.priors import GammaPrior

# train_X: (N, d+1) 最後の列が task index (integer)
# train_Y: (N, 1)
model = MultiTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    task_feature=-1,                # 最後の列が task index
    output_tasks=[0, 1, 2],         # 予測したい task の集合
    rank=2,                         # ICM low-rank
)

# fit
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)
```

### 11.4.2 Hierarchical GP — GPyTorch カスタム

BoTorch native では未対応、GPyTorch でカスタム構築：

```python
import gpytorch
import torch

class HierarchicalGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super().__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks
        # global mean (shared)
        self.mean_module = gpytorch.means.ConstantMean()
        # global kernel (shared trend)
        self.global_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1] - 1)
        )
        # deviation kernels (per task)
        self.deviation_kernels = torch.nn.ModuleList([
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1] - 1)
            )
            for _ in range(num_tasks)
        ])

    def forward(self, x):
        # x[..., :-1] = input, x[..., -1] = task index
        x_input = x[..., :-1]
        task_idx = x[..., -1].long()
        mean = self.mean_module(x_input)
        cov_global = self.global_kernel(x_input)
        # task 固有 deviation は kernel を task-block で足し合わせ
        # （簡易表現、実装は task_idx で mask する）
        cov = cov_global  # + block_diag(deviation_kernels[i](x_input[task_idx==i]))
        return gpytorch.distributions.MultivariateNormal(mean, cov)
```

> [!NOTE]
> 上記は骨格のみ。実運用では **partial pooling** の実装（deviation kernel を task index で block-diag マスク、length scale の階層事前分布）が必要。GPyTorch example の `heteroskedastic_gp` や PyMC の階層モデル実装を参考にする。

### 11.4.3 vol-02 との橋渡し

vol-02 第11章の PyMC 階層モデル：

```python
# vol-02 相当の骨格（PyMC で書くとこう）
with pm.Model() as hier_bo_model:
    # 上位事前分布
    mu_global = pm.Normal("mu_global", 0, 1)
    log_ell_global = pm.Normal("log_ell_global", 0, 1)
    # 装置固有パラメータ（partial pooling）
    log_ell_dev = pm.Normal("log_ell_dev", mu=log_ell_global,
                            sigma=0.5, shape=num_tasks)
    # ...
```

Bayesian surrogate として PyMC / NumPyro で書くと **事前分布の解釈が明示的**。ただし BO の acquisition 最適化とループさせるには BoTorch との bridge が必要（`gpytorch.likelihoods.GaussianLikelihood` + カスタム）。

---

## 11.5 並行探索の Skill 契約

複数装置で同時に候補提案する場合、**装置ごとの pending experiment の扱い**が第12章 batch BO と絡みます。本章では装置差の扱いに焦点：

```yaml
parallel_facility_policy:
  version: "v0.1"
  mode: "shared_surrogate"                # enum: shared_surrogate | independent_per_facility
  per_facility_batch_size:
    facility_A: 2
    facility_B: 1
    facility_C: 2
  pending_experiments_visibility: "cross_facility"   # enum: cross_facility | per_facility_only
  # cross_facility: 装置 A の pending experiment も装置 B の候補選択時に考慮（fantasize）
```

> [!IMPORTANT]
> `pending_experiments_visibility: cross_facility` を使う場合、装置 A の未完了実験の予測値を装置 B の GP で fantasize する必要がある。第12章の `fantasize` API を参照。

---

## 11.6 装置差を「無視」する誤り

**Agentic 特有の失敗パターン**：エージェントが「同じレシピなら装置に依存しない」と誤仮定し、`facility_id` を入力から drop する。

- **検知**：`kernel_spec` に `task_kernel` / `deviation_kernel` が明示されているのに、実際の学習で $B$ が単位行列に collapse（全装置が完全相関）していないか監視
- **予防**：`task_correlation_matrix` を Group C provenance に記録、毎 iteration の値を audit
- **契約**：`facility_id` は `search_space_bounds` の一部として immutable、Skill が drop すれば fatal

---

## 11.7 Skill 契約チェックリスト

- [ ] `surrogate_model_family` が `multi_task_gp` または `hierarchical_gp`
- [ ] `kernel_spec` に task_kernel（MTGP）または deviation_kernel（HierGP）が明示
- [ ] `task_index_spec.task_map` が明示、Skill 自律変更が禁止
- [ ] `new_task_policy` が指定されている（`requires_human_approval` または `fatal`）
- [ ] HierGP の場合、deviation の `output_scale_prior` が事前情報付き（暴走防止）
- [ ] `task_correlation_matrix` が Group C provenance に記録される
- [ ] HRD (Ch7/8) の scale_source が `gp_lengthscale` の場合、input_kernel の length scale を使う（task_kernel ではない）
- [ ] 並行探索時、`parallel_facility_policy` が明示、`pending_experiments_visibility` が意図的
- [ ] `facility_id` の drop は fatal

---

## 11.8 章末演習

**問 1**：3 装置、各装置 5 観測ずつのデータ。MTGP と HierGP のどちらを選ぶか、根拠付きで判断せよ。

**問 2**：`task_correlation_matrix` が iteration 3 で単位行列に近づいた（全装置がほぼ独立）。原因の可能性と Skill の対応は？

**問 3**：装置 D を追加したい。`new_task_policy: requires_human_approval` の場合の手順を書け。

**問 4**：HierGP の `deviation.output_scale` に事前情報がない場合のリスクは？

**問 5**：装置 A で 100 観測、装置 B/C で各 5 観測。この不均衡がどの surrogate に不利か、対策を挙げよ。

---

## 11.9 参考資料

### 本書内

- vol-02 第11章：階層モデル（partial pooling）
- 第5章 §5.2：`surrogate_model_family` の初出、`facility_id` の位置付け
- 第6章：GP の基本、kernel 選択
- 第7章 / 第8章：HRD の 3 判定（本章の task_kernel と整合させる）
- 第10章：制約付き BO（本章の task 別 constraint と組み合わせる）
- 第12章（planned）：Batch × multi-task（`pending_experiments` の cross_facility 可視性）
- 第14章（planned）：capstone（因果 × HierGP × constrained × batch）
- 第15章（planned）：装置差の勝手な無視パターン

### 外部参考

- Bonilla, Chai, Williams, "Multi-task Gaussian Process Prediction", NeurIPS 2007 — ICM 原論文
- Álvarez, Rosasco, Lawrence, "Kernels for Vector-Valued Functions: A Review", 2011 — LCM / ICM の総括
- Swersky, Snoek, Adams, "Multi-Task Bayesian Optimization", NeurIPS 2013 — MTBO の BO 応用
- BoTorch `MultiTaskGP` tutorial

> [!NOTE]
> 次章（第12章）では、Batch BO と並列実験の Skill 化を扱います。q-EI / q-EHVI、local penalization、`pending_experiments` の provenance、Ax による experiment tracking。
