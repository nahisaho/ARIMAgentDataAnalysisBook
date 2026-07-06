# 第9章　解釈可能性とレポート化

> [!IMPORTANT]
> **本章の到達目標**
>
> - **なぜ解釈するのか**を、Human-in-the-loop の質を上げる視点で言語化できる
> - **permutation importance / PDP / SHAP** の 3 手法の使い分けと落とし穴を説明できる
> - **物理知見との突き合わせ**を仕様書に組み込める
> - **統計的有意 ≠ 物理的意義**の区別を Skill レベルで担保できる
> - 解釈可能性 Skill の 6 要素仕様を書ける
> - 再現可能な解釈レポートを Provenance 付きで生成できる

## 本章で扱わないこと

- **因果推論・因果効果推定**：SHAP 値は因果効果ではない。因果は vol-04 で扱う
- **深層学習モデルの解釈**（Grad-CAM, Attention 可視化等）：vol-03 の範疇
- **LIME**：SHAP に統合されている実務では優先度が低い（比較のみ触れる）
- **公平性・バイアス検知**：関連するが独立トピックで、本書では扱わない

---

## 9.1 なぜ解釈するのか——Human-in-the-loop の質

材料研究で ML を使う目的は多くの場合、**予測そのものではなく「なぜそうなるか」を人間が判断・議論するため**です。予測が当たっても、次の実験計画・論文執筆・共同研究者との合意形成には**理由**が必要です。

エージェント時代には、この「解釈」自体が新しいリスクを持ちます：

| リスク | 例 |
|---|---|
| **もっともらしい解釈のハルシネーション** | 「SHAP 値が高い＝物理的に重要」とエージェントが断言 |
| **解釈手法の選択自体が結果を歪める** | permutation vs SHAP で「重要な特徴」が違う |
| **統計的有意を物理的意義と混同** | p 値が小さい ≠ 材料設計に使える寄与 |
| **解釈が意思決定を後押しする方向にドリフト** | 「解釈できたから採用」の逆算 |

本章の目的は、**解釈手法を Skill として仕様化することで、これらのリスクを事前に封じる**ことです。

---

## 9.2 3 つの解釈手法：使い分けマップ

実務で最初に覚える 3 手法：

| 手法 | 何を測るか | 計算コスト | 主な用途 | 主な落とし穴 |
|---|---|---|---|---|
| **Permutation Importance** | 特徴量をシャッフルした際の予測性能低下 | O(n_features × n_repeats) | どの特徴量がモデルに効いているかの全体像 | 特徴量間の相関に弱い（相関特徴量は互いに importance を下げ合う） |
| **PDP（Partial Dependence Plot）** | 特徴量を変化させたときの平均予測 | O(n_features × grid) | 「その特徴量を増やすとどう変わるか」の可視化 | 特徴量間の相関を無視した外挿を含む |
| **SHAP** | 各サンプル・各特徴量の寄与を Shapley 値で分解 | O(n_samples × モデル依存) | 局所（1 サンプルごと）と大域の両方 | 計算コスト大、tree ベース以外は近似が必要 |

### 選択マップ

```mermaid
flowchart TD
    A[解釈したい] --> B{何を知りたい?}
    B -->|全体で効く特徴量| C{相関特徴量が多い?}
    C -->|Yes| D[グループ化 permutation importance]
    C -->|No| E[permutation importance]
    B -->|特徴量→予測の関数形| F{特徴量間相関が弱い?}
    F -->|Yes| G[PDP]
    F -->|No| H[ALE Accumulated Local Effects]
    B -->|1 サンプルごとの説明| I[SHAP local]
    B -->|大域の要約| J[SHAP global mean-abs]
```

> [!NOTE]
> **ALE**（Accumulated Local Effects）は相関特徴量に強く PDP の代替として推奨されますが、scikit-learn 標準ではないため、本書では概念のみ紹介し、実装は SHAP または `alibi` パッケージを推奨します。

### 3 手法は「一致すべき」ではない

読者の直感に反しますが、**3 手法の結果は必ずしも一致しません**。それぞれ**測っているものが違う**からです：

- **permutation importance**：「その特徴量が使えなくなったらどれだけ性能が落ちるか」（モデル性能への貢献）
- **PDP**：「その特徴量を変えたら予測平均はどう動くか」（関数形）
- **SHAP**：「予測値のうちその特徴量が寄与している分」（予測値への貢献）

3 手法で異なる結論が出た場合、それは**モデルが特徴量間の複雑な相互作用を捉えている**サインで、無理に一致させるのではなく**それぞれの解釈**を仕様書に書き込むのが正しい対応です。

---

## 9.3 permutation importance の Skill 化

### 基本と落とし穴

scikit-learn 標準の `permutation_importance`：

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    pipeline, X_test, y_test,
    n_repeats=30, random_state=42, n_jobs=-1,
    scoring="neg_root_mean_squared_error",
)

importances = pd.DataFrame({
    "feature":       X_test.columns,
    "importance":    result.importances_mean,
    "std":           result.importances_std,
}).sort_values("importance", ascending=False)
```

**落とし穴**：

1. **test セットで計算する**（train で計算するとリークした特徴量が過大評価される）
2. **`n_repeats`**：scikit-learn デフォルトは 5 ですが、安定した順位を得るには `n_repeats=30` 程度を実務標準として推奨します（データサイズ・モデル計算コストと相談）
3. **相関特徴量問題**：`x1` と `x2` が強く相関している場合、片方をシャッフルしても他方から情報が復元でき、両方の importance が過小評価される
4. **符号の読み方**：`permutation_importance` は「baseline_score - permuted_score」を返します。`scoring="neg_root_mean_squared_error"` の場合、`baseline=-1.0, permuted=-1.4` なら `importance = -1.0 - (-1.4) = +0.4`。**重要な特徴量ほど `importances_mean` は正に大きい**ため、順位付けはそのまま `importances_mean` 降順で行います（符号反転しない）
5. **CV との併用**：CV の各 fold で計算し、fold 間で集約するのが最も安定（第8章 §8.5 と同じ発想）
6. **不確かさの表現**：`importances_std` に加えて、bootstrap で 95% CI を出す（サンプル bootstrap を n_repeats と組み合わせる、または重複を許した resample）。第10章への橋渡し

### 相関特徴量問題の対処

```python
# 相関の強い特徴量をグループ化
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

corr = np.abs(X_test.corr().fillna(0).values)   # 定数列で NaN が出る場合の保険
dist = 1 - corr
np.fill_diagonal(dist, 0)
Z = linkage(squareform(dist), method="average")
cluster_ids = fcluster(Z, t=0.3, criterion="distance")
# ↑ t=0.3 は「結合距離の目安」であり、|r|>=0.7 相当を狙う設定。
#   ただし method="average" では、クラスタ内全ペアが |r|>=0.7 になることは
#   保証しない（平均距離での結合のため低相関ペアが混入しうる）。
#   保証したい場合は method="complete" を使う。

# クラスタごとに 1 つずつ「代表特徴量」を選び、それで importance を計算
```

またはシンプルに、**相関 |r| > 0.7 の特徴量ペアがある場合、片方を落としてから計算**する方針もあります。この判断は仕様書に書き込みます。

### Skill 仕様書（抜粋、①〜⑥ は第5章 §5.3 と対応）

```yaml
skill: permutation_importance_v1

# ① 目的
purpose: モデルの大域的な特徴量重要度を、test set で評価する

# ② 入力条件
inputs:
  fitted_pipeline: sklearn Pipeline （fit 済み）
  X_test:  pandas.DataFrame
  y_test:  pandas.Series
  correlation_policy:
    threshold: 0.7
    action:    drop_one_of_pair   # または: group_cluster / warn_only

# ③ 出力形式
outputs:
  importance_table:  CSV（feature, importance_mean, importance_std, ci_low, ci_high, rank）
  importance_plot:   PNG
  provenance:        YAML（後述）

# ④ 成功条件（method-agreement ではなく method-internal な安定性で判定）
success_criteria:
  primary:
    - 上位 k 特徴量の Spearman's ρ ≥ 0.7 (seed 3 通り以上での順位安定性)
    - importance の 95% CI を bootstrap で算出済み
  secondary:
    - top-5 Jaccard overlap ≥ 0.6 (補助指標)
    - importance_std / |importance_mean| < 0.3 (変動係数)

# ⑤ 禁止事項（severity 付き）
forbidden:
  - item: train セットでの計算
    severity: fatal
  - item: 相関 0.7 超のペアを未処理のまま計算
    severity: audit_violation
  - item: baseline_score を provenance に記録せず importance のみ報告
    severity: audit_violation

# ⑥ 再現性条件
reproducibility:
  n_repeats:    30    # 標準推奨。小データ・重いモデルでは 10 まで許容
  random_seed:  仕様書 ⑥ に固定
  bootstrap:
    n_bootstrap: 1000
    ci_level:    0.95
  package_versions: scikit-learn>=1.3
```

---

## 9.4 PDP の Skill 化

### 基本

```python
from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
PartialDependenceDisplay.from_estimator(
    pipeline, X_test, features=["feat_a", "feat_b", ("feat_a", "feat_b")],
    ax=ax, grid_resolution=50,
)
```

### 落とし穴

1. **外挿問題**：`feat_a` のグリッドを [min, max] で切ると、実際には出現しない組み合わせでの予測を含む
2. **相関特徴量問題**：`feat_a` を変えるとき `feat_b` は固定される → 相関があると非現実的な組み合わせを評価
3. **ICE（Individual Conditional Expectation）との併用**：平均だけ見ると Simpson's paradox 的な誤解を招く
4. **`grid_resolution`**：デフォルト 100 は多すぎることが多い、50〜100 で十分

### ICE との併用が必須の場面

**「特徴量の効果はサンプルによって符号が違う」**ような相互作用が強い場合、PDP の平均線だけでは真実を隠します：

```python
PartialDependenceDisplay.from_estimator(
    pipeline, X_test, features=["feat_a"],
    kind="both",   # "average" (PDP) + "individual" (ICE) を両方描く
    subsample=200,
)
```

**ICE 線が交差する**なら相互作用あり、**全 ICE 線が平行**なら平均代表性が高い、というのが実務判断です。

> [!NOTE]
> `kind="both"` および `kind="individual"` は **1 特徴量 PDP のみ** で有効です。2 特徴量 PDP（`("feat_a", "feat_b")`）では `kind="average"` のみサポートされます。

### PDP の代替：ALE

相関特徴量が多い場合、**ALE（Accumulated Local Effects）** の方が現実的な解釈を与えます。scikit-learn 標準にはないため、以下のいずれかを推奨：

- `alibi` パッケージ：`from alibi.explainers import ALE`
- `PyALE` パッケージ：軽量な独立実装

「PDP と ALE の結果が乖離する ＝ 相関特徴量の影響が強い」という診断としても使えます。

---

## 9.5 SHAP の Skill 化

### 基本

```python
import shap

# 前処理と model を分離
preprocessor = pipeline[:-1]
model        = pipeline.named_steps["model"]

# 前処理後空間に移してから explainer に渡す
X_test_transformed = preprocessor.transform(X_test)
# ColumnTransformer / OneHotEncoder を含む場合、列数は元の X_test と一致しないため
# get_feature_names_out() で正しい名前を取得する
feature_names = preprocessor.get_feature_names_out()

# tree ベースモデルなら TreeExplainer（厳密・高速）
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_transformed)

# 大域：feature importance
shap.summary_plot(shap_values, X_test_transformed,
                  feature_names=feature_names, plot_type="bar")

# 局所：1 サンプル（shap_values の形状に応じて index する。次の落とし穴 4 も参照）
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0], base_values=explainer.expected_value,
    data=X_test_transformed[0], feature_names=feature_names,
))
```

> [!TIP]
> scikit-learn 1.2+ では `pipeline.set_output(transform="pandas")` を使うと transform 結果が DataFrame になり、列名が保持されます。sparse 出力が必要な場合（OneHot が多い等）は pandas 化できないため、`get_feature_names_out()` で明示的に取得します。

### 落とし穴と実務判断

1. **モデルタイプで explainer を選ぶ**：
   - tree（RF, GBM, XGBoost, LightGBM, CatBoost）→ `TreeExplainer`（厳密・高速）
   - 線形 → `LinearExplainer`
   - それ以外（NN, SVM, カスタム）→ `KernelExplainer`（近似・遅い）または `Explainer`（自動選択）
2. **前処理後の空間で計算する**：`preprocessor.transform(X_test)` で前処理後空間に移す。**feature name の取得は `preprocessor.get_feature_names_out()` を必ず使う**（OneHot 展開で列数が変わるため、元の `X_test.columns` を渡すとサイレントに誤名が付く）
3. **相関特徴量問題は SHAP でも残る**：
   - `feature_perturbation="tree_path_dependent"`：**背景データ不要**、tree の構造を使うので相関に堅牢
   - `feature_perturbation="interventional"`：**背景データ必須**（100〜1000 サンプル程度）、独立性を仮定
   ```python
   background = shap.sample(X_train_transformed, 100, random_state=42)
   explainer  = shap.TreeExplainer(
       model, data=background,
       feature_perturbation="interventional",
       model_output="raw",   # or "probability" (binary tree のみ、近似)
   )
   ```
4. **`shap_values` の形状**（SHAP 0.45+ 前提。バージョンで挙動が変わるため、実行時に `np.asarray(shap_values).shape` を provenance に必ず記録）：

   | モデル | 返り値の形状 |
   |---|---|
   | 回帰 | `(n_samples, n_features)` |
   | 二値分類（sklearn tree） | `(n_samples, n_features, 2)` になる場合あり |
   | 二値分類（XGBoost / LightGBM raw） | `(n_samples, n_features)`（logit 空間） |
   | 多クラス分類 | `(n_samples, n_features, n_classes)` |

5. **加法性（additivity）は `model_output` の空間で成立**：
   - 回帰：`shap_values.sum(axis=-1) + expected_value == model.predict(X)`（厳密一致）
   - 二値分類 raw：logit 空間で加法性、`predict_proba` の確率とは直接一致しない（sigmoid 変換が必要）
   - 「SHAP 値の合計 = 予測値 - base_value」は**説明対象の output space に依存する**
6. **符号の意味**：`model_output="raw"` の場合、SHAP 値は生スコア（回帰なら y の単位、分類なら logit）への寄与。確率の解釈をしたい場合は `model_output="probability"` を明示的に指定（対応モデル限定）

### Skill 仕様書（抜粋、①〜⑥ は第5章 §5.3 と対応）

```yaml
skill: shap_interpretation_v1

# ① 目的
purpose: モデルの局所・大域の解釈を SHAP 値で生成する

# ② 入力条件
inputs:
  fitted_pipeline: sklearn Pipeline （fit 済み）
  X_test:                  pandas.DataFrame
  model_type:              "tree" | "linear" | "kernel"   # explainer 選択根拠
  feature_perturbation:    "tree_path_dependent" | "interventional"
  background_data:         X_train_transformed のサンプル（interventional 時必須）
  model_output:            "raw" | "probability"          # additivity の成立空間

# ③ 出力形式
outputs:
  shap_values:      npz（形状 = np.asarray(shap_values).shape を provenance に記録）
  feature_names:    preprocessor.get_feature_names_out() の結果
  summary_plot:     PNG（bar + beeswarm）
  waterfall_plots:  PNG × N（代表 N サンプル）
  provenance:       YAML

# ④ 成功条件（method-agreement は成功条件にしない：§9.2 参照）
success_criteria:
  primary:
    - shap_values の形状と model_output 空間が provenance に記録済み
    - waterfall で報告される特徴量の符号と絶対値順位を出力
    - additivity が model_output 空間で検証済み（回帰なら誤差 < 1e-6）
  secondary:
    - test サンプルの bootstrap で SHAP mean-abs の 95% CI を算出

# ⑤ 禁止事項（severity 付き）
forbidden:
  - item: 前処理を含まない空間で explainer に渡す
    severity: fatal
  - item: get_feature_names_out() を使わずに元の X_test.columns を渡す
    severity: fatal
  - item: feature_perturbation="interventional" を背景データなしで指定
    severity: fatal
  - item: モデルタイプ未確認のまま KernelExplainer を default で使う
    severity: audit_violation
  - item: shap_values の平均絶対値だけで「重要度」と単独で呼ぶ（分布を無視）
    severity: audit_violation
  - item: 二値分類で logit 空間の SHAP を「確率への寄与」と説明する
    severity: audit_violation

# ⑥ 再現性条件
reproducibility:
  random_seed:      42
  shap_version:     ">=0.45"  # 形状が list → ndarray に変わったため
  sample_selection: 独立テストから固定 index（seed 42, 100 サンプル）
  background_seed:  42
```

---

## 9.6 物理知見との突き合わせ

材料科学の解釈は、**統計的重要度と物理的意義の合致・乖離**の両方が意味を持ちます：

| 状況 | 意味 | 対応 |
|---|---|---|
| 統計的に重要 + 物理的に既知 | 妥当性の確認 | 「モデルは既知物理を捉えている」と報告 |
| 統計的に重要 + 物理的に未知 | **仮説発見**の候補 | 追実験・文献調査で検証 |
| 統計的に非重要 + 物理的に既知 | **モデル設計の問題** | 特徴量表現・データ量・モデル種類を再検討 |
| 統計的に非重要 + 物理的に非重要 | ノイズ / 無関係 | 特徴量から除外を検討 |

> [!NOTE]
> 二値（重要/非重要、既知/未知）の分類は初期整理には有効ですが、実務では**確信度（confidence / evidence level）**を持たせるとより実効的です。仕様書では次のように書きます：
> ```yaml
> physical_interpretation:
>   - feature: feat_a
>     statistical_importance:   high   # high / medium / low
>     statistical_ci_width:     0.06 eV
>     physical_prior_strength:  medium
>     literature_support:       weak
>     decision:                 hypothesis_candidate
> ```

**エージェントに解釈を丸投げしない**ためのプロトコル：

1. 解釈結果を**まず数値・図として提示させる**（言葉での「この特徴量が重要」を先に出させない）
2. 人間が**物理的意義との照合**を行う
3. エージェントには**照合結果を仕様書に書く支援**を依頼する
4. **統計的有意を「物理的意義」と読み替える解釈をエージェントが提示したら訂正**する。Skill spec で半自動化可能：
   ```yaml
   forbidden:
     - item: '"statistical importance implies physical importance" 型の文を未レビューで出力'
       severity: audit_violation
     - item: reviewed_by が未記録のまま report を final 扱いにする
       severity: audit_violation
   ```

> [!IMPORTANT]
> **解釈は final locked model の説明であり、解釈をもとに再設計する場合は Skill version を上げ、別 split / ネスト CV に戻る**（第8章 §8.7 の「test スコア駆動探索」と同じ発想）。test set 上での解釈結果で特徴量削除・モデル変更を行うと、独立テストへの過剰適応になります。

### 解釈レポートの書き方

解釈結果を書くときは、**測ったもの・測っていないものを明示**します：

- ❌ 「`bandgap_expt` が最も重要な特徴量」
- ✅ 「`bandgap_expt` は permutation importance で 1 位（±0.03 eV、95% CI [0.24, 0.32]）だが、`element_composition` との相関が |r|=0.72 と強く、この 2 つは分離できていない可能性がある」

---

## 9.7 統計的有意 ≠ 物理的意義

第10章の入り口として、本章でも触れておくべき論点です。**エージェントが「有意差あり」を「意味がある」に読み替える**のは、実務で最頻の失敗です：

| 概念 | 定義 | 材料研究での意味 |
|---|---|---|
| **統計的有意** | 帰無仮説の下でその差以上が観測される確率が低い | 「偶然ではなさそう」 |
| **効果量** | 差の大きさ（Cohen's d, r², 説明分散等） | 「どれくらい大きい差か」 |
| **物理的意義** | 材料設計・実験計画への影響 | 「その差を活かせるか」 |

**Skill 仕様書 ⑤ 禁止事項**に次を必ず入れます：

- p 値のみで「意味がある」と結論しない
- 効果量とその不確かさを併記する
- 物理的閾値（例：「合成再現性の範囲は ±0.1 eV」）を仕様書 ④ の成功条件に書く

### 頻度論での効果量 CI の算出（第10章への橋渡し）

効果量 + 95% CI を要求するからには、頻度論の枠内でも計算手段を明示します：

| 対象 | CI 算出方法 |
|---|---|
| permutation importance | サンプル bootstrap で `n_bootstrap=1000` を回し、`importances_mean` の 95% 分位点 |
| PDP の関数値 | fold / bootstrap resample で PDP を複数本描き、各 grid 点の 95% 分位点 |
| SHAP global mean-abs | test サンプルの bootstrap で `mean(\|shap\|)` の 95% 分位点 |
| 予測誤差（RMSE 等） | K-fold の fold ごとの値 + `t.interval` または bootstrap |

**ここでの CI は「サンプリング不確かさ」のみ**で、モデル選択・前処理選択の不確かさは含まない、という限界を第10章で Bayesian 版と対比します。

### 有意 vs 効果量のワークシート

解釈レポートには、次の 3 列を必ず併記します：

| 特徴量 | 効果推定値 | 95% CI | 物理的閾値 | 判断 |
|---|---|---|---|---|
| `feat_a` | +0.05 eV | [+0.02, +0.08] | ±0.10 eV | 有意だが閾値未満 → 参考 |
| `feat_b` | +0.30 eV | [+0.20, +0.40] | ±0.10 eV | 有意かつ閾値超 → **設計に反映** |
| `feat_c` | +0.15 eV | [-0.05, +0.35] | ±0.10 eV | 閾値超だが不確か → 追実験 |

---

## 9.8 再現可能な解釈レポート

解釈は **Skill 実行の provenance に必ず添付**されるべき成果物です。Appendix A の provenance スキーマ拡張：

```yaml
interpretation_report:
  methods_used: [permutation_importance, pdp, shap]        # ALE を実施した場合は追加
  method_versions:
    scikit_learn: 1.5.0
    shap:         0.45.0
    ale_package:  PyALE 1.2.0   # ALE 実施時

  permutation_importance:
    file:                       artifacts/permimp.csv     # .github/skills/<name>/artifacts/ 配下（付録A §A.1.1）
    n_repeats:                  30
    scoring:                    neg_root_mean_squared_error
    baseline_score:             -0.85     # 符号込みで記録（§9.3 落とし穴 4）
    correlation_policy_applied: drop_one_of_pair
    bootstrap:
      n_bootstrap: 1000
      ci_level:    0.95

  pdp:
    features:          [feat_a, feat_b, "(feat_a, feat_b)"]
    ice_included:      true
    grid_resolution:   50

  ale:
    used:              true                # false の場合は ale_not_run_reason を必須
    features:          [feat_a, feat_b]
    package:           PyALE
    rationale:         correlation_detected_|r|>0.7

  shap:
    explainer:                 TreeExplainer
    feature_perturbation:      tree_path_dependent
    model_output:              raw
    additivity_verified:       true
    shap_values_shape:         [100, 15]   # np.asarray(shap_values).shape を記録
    feature_names_source:      preprocessor.get_feature_names_out()
    n_samples:                 100
    sample_selection_seed:     42
    background_data_seed:      42          # interventional 時のみ

  cross_method_agreement:
    top5_jaccard_permimp_shap:            0.60
    spearman_rho_full_rank_permimp_shap:  0.72
    note: "disagreement is treated as diagnostic (§9.2), NOT as failure"

  physical_interpretation:
    - feature:                  feat_a
      statistical_importance:   high
      statistical_ci_width:     0.06
      physical_prior_strength:  medium
      literature_support:       weak
      decision:                 hypothesis_candidate
    reviewed_by:                "PI name"
    review_date:                2026-XX-XX
    review_status:              final       # draft の場合は final 扱い禁止
```

> [!IMPORTANT]
> **cross_method_agreement は「診断シグナル」であり、成功条件ではありません**（§9.2 参照）。3 手法が食い違うのは、モデルが特徴量間の相互作用や相関を捉えているサインで、無理に一致させるのではなく **「なぜ乖離しているか」を分析する材料** として使います。

**ランキング一致度の指標選択**：

| 状況 | 推奨指標 |
|---|---|
| 全特徴量の順位が得られる（数十程度） | **Spearman's ρ** または Kendall's τ（tie に強いのは τ） |
| top-k のみ比較（k=5〜10） | **Jaccard overlap**（top-k の集合類似度） + rank-biased overlap（RBO） |
| tie が多い / 重複順位あり | Kendall's τ_b（tie 補正版） |

Kendall's τ を top-k に単独適用すると、非共通特徴量や tie に弱く解釈が不安定になるため、**Jaccard + Spearman の併記**を推奨します。

### レポート生成の Skill 化

解釈レポート生成自体を Skill にすることで、**エージェントに毎回同じ形式で書かせる**：

```yaml
skill: interpretability_report_v1
purpose: 3+ 手法（permutation / PDP / (ALE) / SHAP）の結果と物理知見の照合を統合レポートにする

inputs:
  permimp_result:    permutation_importance_v1 の出力
  pdp_result:        pdp_v1 の出力
  shap_result:       shap_interpretation_v1 の出力
  ale_result:        ale_v1 の出力（オプション、相関検知時は必須）
  physical_priors:   YAML（既知物理・閾値）

outputs:
  interpretation_report:  Markdown + PDF
  cross_method_table:     CSV（特徴量 × 手法の順位表、Jaccard + Spearman）
  discrepancy_flags:      YAML（手法間乖離 / 物理知見乖離 / diagnostic 扱い）

success_criteria:
  primary:
    - 各手法が測っている量が明記され、順位・符号・不確かさ(CI)が provenance に記録
    - 手法間の乖離があれば、相関・相互作用・外挿・前処理影響のいずれかを候補として記録
    - 効果量 + 95% CI + 物理的閾値が全上位特徴量で埋まっている
  secondary:
    - 上記 4 象限（統計 × 物理）のすべてに 1 件以上が分類されている（偏りは分析不足のサイン）
    - PI レビュー済み（review_date と review_status: final が記録されている）

forbidden:
  - item: p 値のみで「意味がある」と結論する
    severity: audit_violation
  - item: SHAP の絶対値平均を「特徴量重要度」と単独で呼ぶ
    severity: audit_violation
  - item: 効果量と不確かさを併記せずに「重要度ランキング」を出す
    severity: audit_violation
  - item: 手法間乖離を無視して単一ランキングに統合する
    severity: audit_violation
  - item: test set 上の解釈結果を根拠に特徴量削除・モデル変更を行う（別 split に戻らずに）
    severity: fatal
```

---

## 9.9 章末チェックリスト

**解釈手法選択チェック**（severity 付き）：

- [ ] 🔴 test セットで計算した（train ではない）
- [ ] 🔴 前処理後の空間で SHAP に渡した（`preprocessor.transform`）
- [ ] 🔴 **前処理後特徴量名 `get_feature_names_out()` と SHAP 配列の列数が一致**している
- [ ] 🔴 SHAP `feature_perturbation="interventional"` を使う場合、背景データを指定した
- [ ] 🟡 permutation importance の `n_repeats` を仕様書で凍結（標準推奨 30、小データでは 10 まで許容）
- [ ] 🟡 相関特徴量ペア（|r| > 0.7）の処理方針を仕様書に書いた
- [ ] 🟡 PDP と ICE を併用（相互作用が疑われる場合、1 特徴量 PDP のみで有効）
- [ ] 🟡 SHAP explainer をモデルタイプに応じて選んだ（`Explainer` の自動選択に頼らない）

**解釈結果の書き方チェック**：

- [ ] 🔴 効果量と 95% CI を必ず併記（bootstrap による CI 算出方法を provenance に記録）
- [ ] 🔴 物理的閾値を仕様書 ④ の成功条件に書いた
- [ ] 🔴 SHAP の `model_output` 空間と additivity 検証結果を provenance に記録した
- [ ] 🔴 **手法間乖離を無視して単一ランキングに統合していない**（乖離は diagnostic として記録）
- [ ] 🟡 手法間一致度を Jaccard overlap + Spearman's ρ の両方で算出（Kendall's τ 単独は避ける）
- [ ] 🟡 統計 × 物理の 4 象限 + confidence（high/medium/low）で特徴量を分類した
- [ ] 🟡 PI / 共同研究者のレビュー日付を provenance に記録

**エージェント運用チェック**：

- [ ] 🔴 エージェントに数値・図を先に出させ、言葉での「重要」判定は最後
- [ ] 🔴 **test set 上の解釈結果で特徴量削除・モデル変更を行わない**（Ch7 §8.7 test 駆動探索禁止と整合）
- [ ] 🟡 統計的有意を「物理的意義」に読み替える解釈を検知したら訂正（Skill spec で禁止文検出可能）
- [ ] 🟡 OneHot / scaling / imputation 後の feature lineage を provenance に保存
- [ ] 🟡 SHAP values / transformed X / feature_names を cache 化し、再描画で再計算しない
- [ ] 🟡 解釈レポート生成自体を Skill 化した

---

## 9.10 章末ワーク

1. **手法間乖離の測定**：第6章 §6.7（MatBench）または §6.8（RRUFF）のデータで、permutation / PDP / SHAP の上位 5 特徴量ランキングを Jaccard overlap + Spearman's ρ で比較。乖離が大きい場合、その原因候補（相関・相互作用・外挿・前処理）を分析して provenance の `discrepancy_flags` に記録
2. **相関特徴量ペアの検知**：`|r| > 0.7` の特徴量ペアを列挙し、片方削除 or クラスタ化 or 併存の 3 案の効果を比較。`method="average"` と `"complete"` のクラスタ化結果の違いも確認
3. **統計 × 物理 4 象限 + confidence 分類**：自分のドメインの既知物理を持ち込み、モデルの上位特徴量を 4 象限に振り分け、さらに `statistical_importance` / `physical_prior_strength` の high/medium/low を付与
4. **効果量 + 95% CI + 物理的閾値のワークシート**：§9.7 の表を、自分のプロジェクトの 3 特徴量で埋める（bootstrap CI も含めて）
5. **解釈レポート Skill を実装**：§9.8 の `interpretability_report_v1` を実装し、第6章のいずれかのハンズオンに適用。`review_status: final` になるまで PI レビュー loop を回す

---

## 9.11 本章のまとめ

- 解釈の目的は **Human-in-the-loop の質を上げる**こと。予測性能とは独立の Skill 責務
- **3 手法**（permutation / PDP / SHAP、必要に応じて ALE）は測っているものが違うため、一致しないのが自然。乖離は **診断シグナル（diagnostic）** として記録し、成功条件にはしない
- **相関特徴量問題**は 3 手法すべてに影響。事前の処理方針を仕様書で凍結
- **feature name の保持**（`get_feature_names_out()`）が SHAP + Pipeline で最も壊れやすい実装上の落とし穴
- **物理知見との突き合わせ**を統計 × 物理 × confidence（high/medium/low）で行う。エージェント任せにしない
- **統計的有意 ≠ 物理的意義**。効果量 + 95% CI + 物理的閾値の 3 点セットを仕様書 ④ に組み込む
- **頻度論の CI（bootstrap）とその限界**を明示し、次章（第10章）で Bayesian による不確かさ表現に橋渡しする
- 解釈レポート自体を Skill 化し、`provenance.interpretation_report` に完全記録する（review_status: final まで）
- **test set 上の解釈結果で再設計をしない**（Ch7 §8.7 の test 駆動探索禁止と整合）

---

## 参考資料

### 本書内の該当章

- 第5章 §5.5：Skill 設計の禁止事項（解釈手法にも適用）
- 第8章 §8.5, §8.7：Pipeline による前処理リーク防止、エージェント特有リーク（test 駆動探索）
- 第10章：頻度論と Bayesian の橋、効果量の不確かさ表現へ
- 第15章：解釈の失敗パターン（vol-01 第14章の統計版）
- 付録A：provenance スキーマ拡張（`interpretation_report` 含む）

### 外部参考

<a id="ref-8-1">[8-1]</a> Molnar, C. (2022). *Interpretable Machine Learning*. 2nd ed. [https://christophm.github.io/interpretable-ml-book/](https://christophm.github.io/interpretable-ml-book/) — 本章の理論的土台
<a id="ref-8-2">[8-2]</a> Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*, 30. — SHAP 原論文
<a id="ref-8-3">[8-3]</a> Fisher, A., Rudin, C., & Dominici, F. (2019). All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously. *JMLR*, 20(177), 1–81. — permutation importance の理論
<a id="ref-8-4">[8-4]</a> Apley, D. W., & Zhu, J. (2020). Visualizing the effects of predictor variables in black box supervised learning models. *JRSS-B*, 82(4), 1059–1086. — ALE の原典
<a id="ref-8-5">[8-5]</a> Wasserstein, R. L., & Lazar, N. A. (2016). The ASA Statement on p-Values: Context, Process, and Purpose. *The American Statistician*, 70(2), 129–133. — 統計的有意の限界
<a id="ref-8-6">[8-6]</a> SHAP 公式ドキュメント：[https://shap.readthedocs.io/](https://shap.readthedocs.io/)
<a id="ref-8-7">[8-7]</a> SHAP Release Notes（形状・API 変更の追跡）：[https://github.com/shap/shap/releases](https://github.com/shap/shap/releases) — 0.42+ で multi-output の返り値が list → ndarray に変更
<a id="ref-8-8">[8-8]</a> scikit-learn Inspection モジュール：[https://scikit-learn.org/stable/modules/inspection.html](https://scikit-learn.org/stable/modules/inspection.html)
<a id="ref-8-9">[8-9]</a> Webber, W., Moffat, A., & Zobel, J. (2010). A similarity measure for indefinite rankings. *ACM TOIS*, 28(4), 1–38. — Rank-biased overlap の原典
