# 第6章　Gaussian Process surrogate の Skill 化

> [!IMPORTANT]
> **本章の到達目標**
>
> 1. Gaussian Process (GP) surrogate の骨格（mean function + covariance kernel + likelihood）を Skill 契約の言葉で説明できる
> 2. 材料 BO で使う 4 系統の kernel — RBF / Matérn 3/2 / Matérn 5/2 / composite (product / additive) — を判断表から選べる
> 3. **ARD (Automatic Relevance Determination)** と length scale の事前分布を、探索空間の物理量スケールから設定できる
> 4. 材料 BO における **推奨 kernel / prior 既定値表**（連続変数、離散/カテゴリ変数、装置差）を Skill 契約に貼れる
> 5. `kernel_spec` フィールド（第5章 §5.2 Group A）を GPyTorch/BoTorch 実装と結び付けて記述できる
>
> **本章で扱わないこと**
>
> - **階層 GP（Multi-task GP、装置差の hierarchical prior）は第11章に完全委譲**。本章では言及のみ
> - Acquisition function（EI / UCB / KG / MES / TS）→ **第7章**
> - 外挿検知の operational 定義（予測分散閾値 + Mahalanobis + length scale）→ **第7章 §7.7**
> - GP 以外の surrogate（Random Forest, BNN, Deep Kernel Learning）→ **第8章**
> - 多目的 / 制約 / batch → 第9-12章

---

## 6.1 なぜ GP を "Skill の心臓" として選ぶのか

BO ループにおいて surrogate はエンジンです。ここが暴走すると acquisition も stop condition も無意味になります。材料 BO で GP が第一選択になる理由は 4 つ：

1. **予測に不確実性が付く**：`mean` と `variance` の両方が出るので、acquisition が信頼領域を意識できる
2. **少ないデータで動く**：10-100 点の観測でも安定して posterior が得られる（vol-04 第11章の smt Kriging と同じ理屈）
3. **kernel を変えて事前知識を注入できる**：滑らかさ、変数間の相関、装置差の階層構造を kernel design で表現可能
4. **`hallucination` を分散から検出できる**：予測分散が閾値を超えた候補は「外挿」として第7章 §7.7 で flag できる

一方、以下は本章の GP では扱えず、後続章に委譲します：

| 弱点 | 対処章 |
|---|---|
| 高次元入力（数十次元以上）| 第8章（Deep Kernel）、第11章末尾で言及 |
| 離散/組合せ空間の広大な探索 | 第12章（BO vs BOCS 判断表）|
| 装置間の階層構造 | 第11章（階層 GP フル展開）|
| 非定常性（領域ごとに滑らかさが変わる）| 第8章（BNN / RF）|

> [!NOTE]
> **vol-04 第11章の smt Kriging との違い**：vol-04 は「一括 DoE 後に応答曲面を張る」用途で、seed 固定・fit 1 回で終わり。vol-05 の GP は「iteration ごとに再 fit / hyperparameter 更新」を前提とし、`retraining_policy`（第5章 §5.4）を明示的に契約化します。

---

## 6.2 GP の骨格を Skill 契約の言葉で書く

GP は次の 3 要素で完全に規定されます：

- **Mean function** $m(x)$：通常はゼロ（`ConstantMean` の zero-init）、事前知識があれば線形/多項式
- **Covariance kernel** $k(x, x'; \theta)$：関数空間の「滑らかさ」と「相関構造」を決める
- **Likelihood**：観測ノイズ $\sigma_n^2$ を含む GaussianLikelihood（通常 homoscedastic、装置差があれば §6.6 で拡張検討）

これを第5章の `kernel_spec` と組み合わせると、Skill 契約はこう書けます：

```yaml
surrogate_model_family: "single_task_gp"  # または fixed_noise_gp / heteroskedastic_gp
mean_function:
  type: "constant"
  initial: 0.0
  learnable: true
kernel_spec:
  continuous:
    type: "matern52"      # RBF | matern32 | matern52 | linear | rq
    ard: true             # per-dimension length scale
    length_scale_prior:
      distribution: "gamma"
      params: { concentration: 3.0, rate: 6.0 }  # 材料 BO 既定
    output_scale_prior:
      distribution: "gamma"
      params: { concentration: 2.0, rate: 0.15 }
  categorical:
    type: "hamming"       # または one_hot + RBF
    variables: ["atmosphere", "facility_id"]
  composite: "product"    # continuous × categorical
likelihood:
  type: "gaussian"
  noise_prior:
    distribution: "gamma"
    params: { concentration: 1.1, rate: 0.05 }
  noise_lower_bound: 1e-4
```

このブロックが Group A（immutable during Skill run）です。**iteration の途中で kernel を切り替えるのは第3章 §3.5 逸脱 2（acquisition/surrogate drift）**——Skill fatal で止めます。kernel 変更は Human 判断で新 run として扱います。

---

## 6.3 材料 BO で使う 4 系統の kernel

### 6.3.1 RBF (Squared Exponential)

$$
k_{\text{RBF}}(x, x') = \sigma_f^2 \exp\!\left(-\tfrac{1}{2} \sum_{i=1}^d \frac{(x_i - x'_i)^2}{\ell_i^2}\right)
$$

- **滑らかさ**：$C^\infty$（無限回微分可能）。滑らかすぎて材料の応答関数には強すぎることが多い
- **推奨場面**：目的関数が「本当に」滑らかであると事前知識で言える場合のみ
- **リスク**：外挿が過剰に自信を持つ → hallucination のリスク

### 6.3.2 Matérn 3/2

$$
k_{3/2}(r) = \sigma_f^2 \left(1 + \sqrt{3}\,\tfrac{r}{\ell}\right) \exp\!\left(-\sqrt{3}\,\tfrac{r}{\ell}\right), \quad r = \|x - x'\|
$$

- **滑らかさ**：1 階微分可能
- **推奨場面**：目的関数が比較的粗い、閾値挙動を持つ、相転移的な変化がある
- **材料 BO での使い所**：反応速度、収率など、局所的な非滑らかさが疑われる場合

### 6.3.3 Matérn 5/2（**本書の第一選択**）

$$
k_{5/2}(r) = \sigma_f^2 \left(1 + \sqrt{5}\,\tfrac{r}{\ell} + \tfrac{5 r^2}{3 \ell^2}\right) \exp\!\left(-\sqrt{5}\,\tfrac{r}{\ell}\right)
$$

- **滑らかさ**：2 階微分可能
- **推奨場面**：材料 BO のデフォルト。RBF より外挿がおとなしく、Matérn 3/2 より応答が滑らか
- **BoTorch/Ax 標準**：`SingleTaskGP` の default が Matérn 5/2 + ARD

### 6.3.4 Composite kernel — product / additive

- **Product kernel**：$k(x, x') = k_{\text{cont}}(x_{\text{cont}}) \cdot k_{\text{cat}}(x_{\text{cat}})$
  - 変数群が「相互作用する」場合（例：温度と雰囲気の組合せで結果が変わる）
- **Additive kernel**：$k(x, x') = \sum_j k_j(x^{(j)}, x'^{(j)})$
  - 変数群が「独立に効く」場合（次元の呪いを緩和）
  - 高次元（10 次元以上）で有効

材料 BO での既定は **product kernel（continuous Matérn 5/2 × categorical Hamming）**。

---

## 6.4 ARD と length scale の設定

### 6.4.1 ARD (Automatic Relevance Determination)

- **やっていること**：kernel の length scale $\ell$ を**次元ごとに独立**に持たせ、hyperparameter 最適化で自動決定する
- **効果**：関係の薄い次元は $\ell_i$ が大きくなり（$\to \infty$ で「効かない」）、実質的な変数選択が起きる
- **契約表現**：`kernel_spec.continuous.ard: true`

### 6.4.2 length scale の事前分布

length scale $\ell$ に flat prior を置くと **暴走する**——観測点近傍にゼロ collapse するか、境界で発散するか。**材料 BO では必ず weak informative prior を置く**。

- **推奨既定**：`Gamma(concentration=3.0, rate=6.0)`（BoTorch/GPyTorch の一般的既定）
- **物理量スケールへの正規化**：Skill 実行前に入力を `Normalize` transform（[0, 1] へ）で揃える → prior が変数依存にならない
- **BoTorch 標準**：`SingleTaskGP` に `input_transform=Normalize(d)` と `outcome_transform=Standardize(m=1)` を渡すのが定石（第4章 §4.6 で既出）

### 6.4.3 length scale が「暴走した」ときの検出

Skill 契約の Group C `surrogate_diagnostics.length_scale` は per-ARD-dim ベクトルで記録します。次のパターンは fatal ではないが **要 Human review**：

| 症状 | 意味 | 対処 |
|---|---|---|
| `length_scale[i]` が bounds 幅の $10\times$ 超 | 次元 i は無関係と判定 | 変数選択の再確認、第10章 constraints_declared に反映 |
| `length_scale[i]` が bounds 幅の $0.01\times$ 未満 | 観測点近傍にゼロ collapse | prior の再検討、noise_lower_bound 増加 |
| iteration ごとに length_scale が振動 | posterior が multimodal | fully Bayesian 化を検討、第11章の階層 prior へ |

---

## 6.5 材料 BO における推奨 kernel / prior 既定値表

Skill 契約に貼るための「既定値表」。**この表がある章の存在**が vol-05 の実務貢献です。

### 6.5.1 連続変数の既定

| 状況 | kernel | length scale prior | 備考 |
|---|---|---|---|
| 温度・圧力・時間など標準的な連続変数 | Matérn 5/2 + ARD | Gamma(3.0, 6.0) | 入力は Normalize([0, 1]) |
| 化学組成比（連続、和が 1 の制約あり）| Matérn 5/2 + ARD | Gamma(3.0, 6.0) | simplex 制約は第10章で reparameterize |
| 相転移が疑われる制御変数 | Matérn 3/2 + ARD | Gamma(2.0, 4.0)（やや長めを許容）| 局所非滑らかを許容 |
| 反応速度・収率（対数変換したい）| Matérn 5/2 + ARD | Gamma(3.0, 6.0) | outcome_transform で log 変換 |

### 6.5.2 離散/カテゴリ変数の既定

| 状況 | 表現 | 備考 |
|---|---|---|
| 順序なしカテゴリ（雰囲気、触媒種）| Hamming kernel（or one-hot + RBF）| product で連続 kernel と結合 |
| 順序ありカテゴリ（グレード A/B/C）| 順序尺度を数値化 → 連続 Matérn 5/2 | 変換の妥当性は Human 判断 |
| 装置 ID（`facility_id`）| **第5章 §5.2 `surrogate_treatment_of_facility_variance` の enum で決定** | 詳細は §6.6 |

### 6.5.3 装置差の扱い（4 選択肢の詳細は第11章）

| enum 値 | 本章での扱い | 完全展開 |
|---|---|---|
| `categorical_input` | Hamming で product kernel に入れる（**本章の既定**）| §6.6 |
| `hierarchical_shrinkage` | 装置ごとに hierarchical prior を組む | **第11章** |
| `blocked_out` | 装置ごとに独立 GP を持つ | **第11章** |
| `not_applicable` | 単一装置のみ、装置次元なし | 何もしない |

### 6.5.4 Likelihood（観測ノイズ）の既定

| 状況 | likelihood | noise prior | 備考 |
|---|---|---|---|
| 観測ノイズが一定と仮定できる | GaussianLikelihood（homoscedastic）| Gamma(1.1, 0.05) | 本章の既定 |
| 観測ノイズが入力依存で変わる | `HeteroskedasticSingleTaskGP` | データ依存 | 第8章で言及、実装は付録 B |
| 装置ごとにノイズが違う | 装置別 likelihood | 装置別 prior | 第11章の階層 GP と組み合わせ |

---

## 6.6 装置差を kernel の中でどう扱うか（本章スコープ）

第5章 §5.2 で導入した `surrogate_treatment_of_facility_variance` の 4 選択肢のうち、本章では **`categorical_input`** の実装を扱います。他 3 選択肢は第11章。

### 6.6.1 categorical_input の実装

`facility_id` を categorical 変数として product kernel に入れる：

$$
k((x_{\text{cont}}, f), (x'_{\text{cont}}, f'))
 = k_{\text{Matern52}}(x_{\text{cont}}, x'_{\text{cont}}; \ell)\ \cdot\ k_{\text{Hamming}}(f, f')
$$

Hamming kernel は：

$$
k_{\text{Hamming}}(f, f') = \sigma_c^2 \cdot \mathbb{1}[f = f']  + \rho \cdot \mathbb{1}[f \neq f']
$$

- $\rho \in [0, 1]$：装置間の相関を表す hyperparameter
- $\rho = 1$：装置差なし（=装置を無視）
- $\rho = 0$：装置ごとに完全独立（=`blocked_out` に近い）

### 6.6.2 categorical_input の限界

- 装置が 2-5 台程度なら OK。10 台以上になると相関パラメータの推定が不安定
- 装置ごとの目的関数の**形が違う**（optimum の位置が違う）と、product kernel では表現しきれない
- どちらかで詰まったら **第11章の hierarchical GP** に切り替える

> [!TIP]
> **判断の目安**：装置 3 台までは categorical_input、4 台以上または「明らかに装置ごとに optimum が違う」なら hierarchical GP を検討。

---

## 6.7 GPyTorch/BoTorch 最小実装

BoTorch 標準 API（v0.11 系）で GP fit → posterior まで：

```python
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

# X: (N, d) tensor、Y: (N, 1) tensor
d = X.shape[-1]
covar_module = ScaleKernel(
    MaternKernel(nu=2.5, ard_num_dims=d)  # Matérn 5/2 + ARD
)
model = SingleTaskGP(
    train_X=X,
    train_Y=Y,
    covar_module=covar_module,
    input_transform=Normalize(d=d),
    outcome_transform=Standardize(m=1),
)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# posterior at test points
posterior = model.posterior(X_test)
mean = posterior.mean         # (M, 1)
variance = posterior.variance # (M, 1)  ← Ch7 §7.7 の外挿検知で使う
```

Skill 契約との対応：

| Skill 契約フィールド | 実装での対応 |
|---|---|
| `kernel_spec.continuous.type: matern52` | `MaternKernel(nu=2.5)` |
| `kernel_spec.continuous.ard: true` | `ard_num_dims=d` |
| `kernel_spec.continuous.length_scale_prior` | `MaternKernel(..., lengthscale_prior=GammaPrior(3, 6))` |
| Normalize / Standardize | `input_transform`, `outcome_transform` |
| `retraining_policy: every_iteration` | 毎 iteration `fit_gpytorch_mll(mll)` |
| `surrogate_diagnostics.length_scale` | `model.covar_module.base_kernel.lengthscale.detach()` |
| `surrogate_diagnostics.noise_variance` | `model.likelihood.noise.detach()` |

### 6.7.1 categorical_input（Hamming kernel）を混ぜる場合

BoTorch には `MixedSingleTaskGP` があり、continuous + categorical を扱えます：

```python
from botorch.models import MixedSingleTaskGP

# cat_dims: categorical 次元のインデックスリスト
model = MixedSingleTaskGP(
    train_X=X,           # (N, d_cont + d_cat)
    train_Y=Y,
    cat_dims=[d_cont + i for i in range(d_cat)],
    input_transform=Normalize(d=d_cont + d_cat, indices=list(range(d_cont))),
    outcome_transform=Standardize(m=1),
)
```

`MixedSingleTaskGP` は内部で continuous 部分に Matérn、categorical 部分に categorical kernel を組んで product を取ります。

---

## 6.8 Skill としての kernel 選択判断表

Skill 契約に貼る kernel 選定フロー：

```
[Q1] 入力次元は？
  ├─ 1-5 次元 → Matérn 5/2 + ARD（product with categorical if needed）
  ├─ 6-15 次元 → Matérn 5/2 + ARD、additive kernel も検討
  └─ 16 次元以上 → 第8章 Deep Kernel / 第11章 hierarchical GP へ

[Q2] 目的関数の性質は？
  ├─ 滑らか（酸化・還元反応の熱力学的量など）→ Matérn 5/2
  ├─ 局所非滑らか（相転移・閾値挙動）→ Matérn 3/2
  └─ 非定常（領域で滑らかさが変わる）→ 第8章 (BNN / DKL)

[Q3] 変数間の相互作用は？
  ├─ 強い（温度×雰囲気で結果が反転など）→ product kernel
  ├─ 独立（各変数が個別に効く）→ additive kernel
  └─ 未知 → product kernel（デフォルト、ARD が実質的に効きの差を吸収）

[Q4] 装置差は？
  ├─ 装置 3 台まで → categorical_input（本章）
  ├─ 装置 4 台以上 or optimum が装置で違う → hierarchical_shrinkage（第11章）
  └─ 装置ごとに独立で構わない → blocked_out（第11章）

[Q5] 観測ノイズは？
  ├─ 一定と仮定できる → GaussianLikelihood（homoscedastic）
  └─ 入力依存 → HeteroskedasticSingleTaskGP（第8章で言及、実装は付録 B）
```

---

## 6.9 GP surrogate の Skill 契約チェックリスト

- [ ] `kernel_spec.continuous.type` が §6.3 の 4 系統から明示的に選ばれている
- [ ] `kernel_spec.continuous.ard: true` が設定されている（連続次元 > 1 の場合）
- [ ] `kernel_spec.continuous.length_scale_prior` に **weak informative prior** が設定されている（flat prior は禁止）
- [ ] 入力に `Normalize`、出力に `Standardize` が適用されている
- [ ] `surrogate_treatment_of_facility_variance` の値と kernel 実装が整合している
- [ ] `retraining_policy` に従って fit のタイミングが決まっている
- [ ] `surrogate_diagnostics.length_scale` が iteration ごとに Group C に記録される
- [ ] length_scale が bounds の $10\times$ 超 / $0.01\times$ 未満で warning を出す仕組みがある

---

## 6.10 章末演習

**問 1**：自分のケースで、連続変数 / カテゴリ変数 / 装置差の 3 種を含む `kernel_spec` を §6.5 の既定値表から選んで書いてください。

**問 2**：入力次元が 12 のケースで、product kernel と additive kernel のどちらを選びますか？ 判断根拠を書けますか？

**問 3**：length_scale が iteration 5 で bounds 幅の $15\times$ になった。この状況で Skill が取るべき次のアクションは何ですか？（fatal / warning / Human review request のどれか、理由付きで）

**問 4**：`categorical_input` から `hierarchical_shrinkage`（第11章）に切り替えるべき兆候を 2 つ挙げてください。

**問 5**：BoTorch の `SingleTaskGP` に `Normalize` を渡さずに fit したら何が起きますか？ length_scale prior との相互作用を説明してください。

---

## 6.11 参考資料

### 本書内

- [第3章](./chapter-03.md) §3.3：装置差の 3 選択肢の初出（本章 §6.6 と §6.5.3）
- [第4章](./chapter-04.md) §4.6：BoTorch API（`fit_gpytorch_mll`, `Normalize`, `Standardize`）
- [第5章](./chapter-05.md) §5.2：`kernel_spec` フィールドの Skill 契約における位置（Group A immutable）
- [第7章](./chapter-07.md)（次章）：acquisition function と外挿検知の operational 定義
- [第8章](./chapter-08.md)（planned）：GP の限界と代替 surrogate
- [第11章](./chapter-11.md)（planned）：hierarchical GP のフル展開

### vol-04 との連携

- vol-04 第11章：応答曲面法 + タグチメソッド + **smt Kriging surrogate**（一括 DoE 用の GP、本章の iteration 版との違いは §6.1 参照）
- vol-04 第5章：Skill 契約テンプレート（本章の `kernel_spec` は BO 拡張）

### 外部参考

- Rasmussen & Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006 — GP の標準教科書。第4章の Matérn 族、第5章の hyperparameter 最適化を参照
- BoTorch Documentation — `SingleTaskGP`, `MixedSingleTaskGP`, `HeteroskedasticSingleTaskGP` の API
- GPyTorch Documentation — kernel の合成、prior の設定
- Frazier, P. I., *A Tutorial on Bayesian Optimization*, arXiv:1807.02811, 2018 — 材料 BO の kernel 選定に関する実務的議論

> [!NOTE]
> 次章（第7章）では、本章の GP surrogate に **acquisition function** を組み合わせて次候補を提案する Skill の設計と、`hallucinated_recommendation_detection` の operational 定義（予測分散 + Mahalanobis + length scale の合成判定）を扱います。
