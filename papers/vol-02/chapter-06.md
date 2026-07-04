# 第6章 教師なし学習を Skill 化する

> **本章の到達目標**
> - 第5章の Skill 構造を、**教師なし学習**（PCA・クラスタリング・異常検知）に拡張できる
> - 目的変数がないゆえに難しくなる「成功条件」を、`task_type` ごとに書き下せる
> - PCA・k-means・HDBSCAN・Isolation Forest を、目的とデータ性質から選び分けられる
> - 3 つのハンズオン（PCA でスペクトル群、クラスタリングで回折パターン群、異常検知で顕微鏡画像特徴量）を、共通構造で実装できる
> - 教師なし学習特有の落とし穴（**「見えたパターン」の後付け解釈**、クラスタ数の後付け変更）を回避できる
>
> **本章で扱わないこと**
> - 詳細な CV 設計・bootstrap 安定性評価 → **第7章**
> - SHAP・PDP 等の教師あり用解釈手法 → **第8章**（PCA loading の解釈は本章で扱う）
> - 深層 embedding（オートエンコーダ・自己教師あり学習） → 本書外（vol-03 予定）
> - 統計的検定を伴う異常検知（外れ値検定など） → 統計学の教科書に譲る

---

## 6.1 教師なし学習と Skill 化の難しさ

教師あり学習には「予測誤差」という揺るぎない評価軸がありました。**教師なし学習にはそれがありません**——正解ラベルがないため、「何をもって成功とするか」を自分で定義する必要があります。この違いが Skill 化を難しくします。

第5章から本章に持ち越すもの・変わるもの：

| 要素 | 第5章（教師あり） | 第6章（教師なし） |
|---|---|---|
| ①目的 | 「予測/分類する」 | 「**構造を抽出する** / 異常を検出する / 次元を減らす」 |
| ②入力 | `X` + `y` + 識別子 | `X` + 識別子（`y` は評価用に持てば持つ、無くてもよい） |
| ③出力 | 予測値・確率 + provenance | **潜在表現 / クラスタ ID / 異常度スコア** + 可視化 |
| ④成功条件 | 予測誤差の 3 点セット | **タスク種別に応じた内部指標 + 安定性 + ドメイン整合** |
| ⑤禁止事項 | データリーク・後付け変更 | 上記 + **「見えたパターン」の後付け解釈**・クラスタ数後付け変更 |
| ⑥再現性 | `data_split` / `cv_scheme` 等 | 上記 + **初期化 seed（k-means）・並び順（HDBSCAN）** |

**特に難しいのは④成功条件**です。第4章 §4.3 で `task_type` ごとの成功条件表を示しましたが、本章ではそれを実装レベルまで具体化します。

> [!IMPORTANT]
> **教師なし学習では、Skill 仕様書の④成功条件を書けないまま実装に進むと、後で「なんとなく良さそう」で採用してしまいます**。本章の 3 ハンズオンは、成功条件を先に書き下すことで、この落とし穴を避けます。

---

## 6.2 3 つのハンズオン概観

| ハンズオン | データ | task_type | 主手法 | 該当節 |
|---|---|---|---|---|
| A. PCA でスペクトル群 | RRUFF Raman など | `dimensionality_reduction` | PCA、Kernel PCA | §6.3 |
| B. クラスタリングで回折パターン群 | XRD パターン群（合成 or ARIM 抜粋） | `clustering` | k-means, HDBSCAN | §6.4 |
| C. 異常検知で顕微鏡画像特徴量 | 事前抽出した画像特徴量（Magpie 相当） | `anomaly_detection` | Isolation Forest, One-Class SVM | §6.5 |

**共通構造**は第5章 §5.4 の 5 ブロック（入力検証 / Pipeline / 学習・評価 / 予測・可視化 / provenance）を継承します。教師なしでは train/test 分割の概念が薄い（あるいは異なる）ため、**§5.2 の anti-leakage split contract は「安定性評価のための resample split」用**に置き換えて使います（§6.4 で詳述）。

---

## 6.3 ハンズオン A：PCA でスペクトル群を要約する

**目的**：Raman スペクトル群を数個の主成分に要約し、①群構造の可視化、②後続 Skill 用の低次元特徴量、③異常サンプルの一次スクリーニングに用いる。

### Skill 仕様書（要約）

| 要素 | 内容 |
|---|---|
| ① 目的 | スペクトル群を線形部分空間で要約。`task_type=dimensionality_reduction` |
| ② 入力 | 波数（列）× 試料（行）の tabular。前処理済み（ベースライン補正・正規化）を前提 |
| ③ 出力 | `PCA` オブジェクト、成分行列、寄与率、スコアプロット（可視化）、provenance |
| ④ 成功条件 | ①**累積寄与率**：第 K 主成分で ≥ 0.90（K は仕様書で固定）／②**成分の安定性**：bootstrap 100 回で loading の符号一致率 ≥ 0.9 ／③**既知群整合**：ラベルがあれば、スコアプロットで既知群が視覚的に分離 |
| ⑤ 禁止事項 | 全データで前処理してから PCA、CV や bootstrap を通さない安定性宣言、成分の物理的意味の後付け決定 |
| ⑥ 再現性 | `n_components`、`whiten`、`svd_solver`、`random_state`（`svd_solver="randomized"` 使用時）、前処理チェイン |

### 実装スケルトン

```python
from sklearn.decomposition import PCA

pipeline = Pipeline([
    # 前処理は必ず Pipeline 内で
    ("scaler", StandardScaler(with_std=False)),  # 平均のみ中心化
    ("pca",    PCA(n_components=10, svd_solver="full", random_state=42)),
])
pipeline.fit(X_train)
Z_train = pipeline.transform(X_train)
explained = pipeline.named_steps["pca"].explained_variance_ratio_
```

### 成分の安定性を bootstrap で評価

**「たまたま第 1 成分が出た」ではなく「サンプルが多少違っても同じ第 1 成分が出る」ことを確認**します：

```python
def bootstrap_pca_stability(X, n_components=3, n_boot=100, seed=0):
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    ref_pca = PCA(n_components=n_components).fit(X)
    sign_agree = np.zeros(n_components)
    for _ in range(n_boot):
        idx = rng.choice(len(X), len(X), replace=True)
        boot = PCA(n_components=n_components).fit(X[idx])
        # 符号反転を許容した上で loading の一致を測る
        for k in range(n_components):
            sim = np.abs(np.dot(ref_pca.components_[k], boot.components_[k]))
            sign_agree[k] += int(sim > 0.9)
    return sign_agree / n_boot   # 各成分の安定率
```

**成分の符号は PCA の非決定性**（符号反転は解のうち等価な自由度）なので、`|dot|` で判定します。

### 可視化

- **スクリー図**（`explained_variance_ratio_` の棒グラフ）：K の選択根拠
- **スコアプロット**（PC1 vs PC2）：既知群があれば色分け
- **loading プロット**：各成分の物理的解釈（例：Raman 第 1 成分の高波数ピーク）

> [!WARNING]
> **loading の物理的解釈は、成分安定性が確認できてから**行います。安定率 < 0.9 の成分について「この成分は結晶構造の秩序を表す」などと語るのは、統計版の p-hacking（第4章 §4.4）に近づきます。

---

## 6.4 ハンズオン B：クラスタリングで回折パターン群を分類する

**目的**：XRD パターン群を教師なしでクラスタリングし、①既知相の再確認、②未知相の候補抽出、③次のラベル付け対象の優先順位付けに用いる。

### Skill 仕様書（要約 + A との差分）

| 要素 | A との差分 |
|---|---|
| ① 目的 | パターン群のクラスタ構造抽出。`task_type=clustering` |
| ② 入力 | 2θ 軸を統一補間済みの XRD パターン（tabular）、または PCA 済み低次元表現 |
| ③ 出力 | クラスタラベル、代表パターン、silhouette 等の内部指標、可視化、provenance |
| ④ 成功条件 | ①**内部指標**：silhouette ≥ 0.5（k-means の場合、目安）／②**安定性**：resample を含む bootstrap で ARI ≥ 0.7 ／③**既知ラベルがあれば** ARI / NMI で外部評価 |
| ⑤ 禁止事項 | クラスタ数 K を結果を見た後で変更、silhouette 最大化 K を「発見された K」と主張、外れ値の後付け除外 |
| ⑥ 再現性 | アルゴリズム名、`n_clusters`（k-means）、`min_cluster_size` / `min_samples`（HDBSCAN）、`random_state` |

### k-means と HDBSCAN の使い分け

| 観点 | k-means | HDBSCAN |
|---|---|---|
| クラスタ数 | 事前指定（K が既知の時） | 自動決定（未知の時） |
| クラスタ形状 | 凸・球状 | 任意形状 |
| ノイズ扱い | なし（全点をどこかに割り当て） | ノイズ点を `-1` にラベリング |
| 密度差 | 苦手 | 得意 |
| ARIM でよくある使い方 | 既知相の 5〜10 クラス分類 | 未知相探索・外れ試料抽出 |

**両方を実行して比較する**のは Skill 増殖ですが、**目的が違えば別 Skill として並列運用**するのは妥当です。仕様書 ①目的にどちらを選ぶかを明記します。

### 実装スケルトン（HDBSCAN）

```python
from sklearn.cluster import HDBSCAN  # scikit-learn 1.3+ で組込み

pipeline = Pipeline([
    ("pca",     PCA(n_components=10, random_state=42)),  # 次元削減
    ("cluster", HDBSCAN(min_cluster_size=5, min_samples=3)),
])
pipeline.fit(X_train)
labels = pipeline.named_steps["cluster"].labels_
n_noise = np.sum(labels == -1)
```

HDBSCAN は `predict` を持たないため、**新規サンプルへの割り当ては別 Skill**（`approximate_predict` を別ライブラリで使うか、代表点への距離ベースで割り当て）にします。

### 安定性評価：bootstrap ARI

**「同じデータ・同じ乱数」で同じクラスタが出るのは当たり前**。教師なしで問うべきは「サンプルが少し変わっても同じクラスタが出るか」です：

```python
from sklearn.metrics import adjusted_rand_score

def bootstrap_cluster_stability(pipeline, X, n_boot=50, seed=0):
    rng = np.random.default_rng(seed)
    ref = pipeline.fit(X).named_steps["cluster"].labels_
    aris = []
    for _ in range(n_boot):
        idx = rng.choice(len(X), size=int(len(X) * 0.8), replace=False)
        boot = pipeline.fit(X[idx]).named_steps["cluster"].labels_
        # 共通部分の index で ARI を計算
        common = np.intersect1d(np.arange(len(X)), idx)
        aris.append(adjusted_rand_score(ref[common], boot))
    return np.mean(aris), np.std(aris)
```

**ARI < 0.7 なら「見えたクラスタは偶発的である可能性」を仕様書に明記**して警告を出します。

### 教師なしにおける「anti-leakage split contract」の読み替え

第5章 §5.2 の契約は、教師なしでは次の 3 点に置き換えます：

1. **前処理を全データで実施しない**：bootstrap の各 resample 内で PCA / 標準化を実施
2. **同一標本の複数測定を bootstrap の resample 単位で分離**：`specimen_id` を group にした group-bootstrap
3. **既知ラベルがある場合、それをクラスタリングの入力に混ぜない**（あくまで事後評価用）

**教師なしでも fold 内前処理は必須**です。

---

## 6.5 ハンズオン C：異常検知で顕微鏡画像特徴量から外れ試料を抽出

**目的**：顕微鏡画像から事前抽出した特徴量（形状記述子・テクスチャ記述子）から、「他と異なる」試料を検出し、目視確認優先度を付ける。

### Skill 仕様書（要約 + A/B との差分）

| 要素 | A/B との差分 |
|---|---|
| ① 目的 | 外れ試料の検出。`task_type=anomaly_detection` |
| ② 入力 | 事前抽出済み画像特徴量（形状・テクスチャ・PCA 済み低次元表現） |
| ③ 出力 | 異常度スコア（連続値）、閾値超過フラグ、優先度ランキング、可視化 |
| ④ 成功条件 | ①**既知異常があれば検出率 ≥ 0.8**、偽陽性率 ≤ 0.1 ／②**既知異常が無ければ**：目視確認上位 K 件のうち専門家判定で 5 割以上が「確かに異常」（人間評価が必須）／③スコア分布が単峰・単一の裾を持つ |
| ⑤ 禁止事項 | 異常度分布を見た後で閾値を緩める、既知異常でチューニングした後に同じ異常でテストする、目視確認結果を Skill 実行後に契約に追加 |
| ⑥ 再現性 | 手法、`contamination`（Isolation Forest）、`nu`（One-Class SVM）、`random_state`、特徴量スケール |

### Isolation Forest vs One-Class SVM

| 観点 | Isolation Forest | One-Class SVM |
|---|---|---|
| 前提 | 「異常はランダム分割で早く isolate される」 | 「正常はカーネル空間で 1 つの領域に収まる」 |
| データ量 | 大規模に強い | 小〜中規模、O(n²) メモリ |
| 特徴量スケール | 比較的頑健 | 敏感（標準化必須） |
| 解釈 | パス長ベース、直感的 | サポートベクター経由、間接的 |
| ARIM での目安 | 数千サンプル以上 | 数十〜数百サンプル |

### 実装スケルトン（Isolation Forest）

```python
from sklearn.ensemble import IsolationForest

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  IsolationForest(
        n_estimators=200, contamination=0.05,
        random_state=42, n_jobs=-1,
    )),
])
pipeline.fit(X_train)
scores = -pipeline.named_steps["model"].score_samples(X_train)  # 大きいほど異常
is_anomaly = pipeline.named_steps["model"].predict(X_train) == -1
```

### `contamination` パラメータの扱いは要注意

`contamination`（想定異常割合）は **仕様書段階で固定**します。**結果を見て `contamination` を変えるのは第4章 §4.4 の循環設計**です。既知異常がない場合、`contamination="auto"` または「最小可視化のための 5%」を仕様書で書いておき、独立評価では変更しません。

---

## 6.6 教師なし学習特有の落とし穴

第5章 §5.9 の失敗表に、教師なし特有の項目を追加します：

| 症状 | 原因 | 改善策 | 該当節 |
|---|---|---|---|
| 「良さそうなクラスタが見えた」で採用 | 内部指標・安定性・外部評価を省略 | 3 点評価（内部指標・bootstrap 安定性・可能ならドメイン評価）を仕様書 ④ に明記 | §6.4 |
| クラスタ数 K を silhouette 最大で選ぶ → それを「発見」と主張 | 探索と本評価が未分離 | K を仕様書で固定するか、探索用と本評価用の Skill を分離 | §6.4, §4.4 |
| PCA 第 1 成分に「結晶秩序」などと名前を付ける | 成分安定性が未検証 | bootstrap 安定率 ≥ 0.9 の成分にのみ解釈を許可 | §6.3 |
| 異常検知の閾値を、結果を見た後に緩めて「異常なし」と主張 | 循環設計 | 閾値・`contamination` を仕様書で固定、監査ログで検証 | §6.5, §4.4 |
| 特徴量スケールを揃えず k-means を回す → クラスタが変数のスケールで決まる | 前処理漏れ | 標準化を Pipeline 内に | §6.4 |
| 「異常」と「珍しい」を区別せずに報告 | 用語の未定義 | 仕様書 ① で「珍しさ vs 物理的異常」を定義、可能ならラベル付けサイクルへ | §6.5 |

---

## 6.7 可視化ガイド

教師なし学習の Skill では、**可視化が成功条件の一部**です（数字だけでは判断できないため）。共通ルール：

- **スクリー図**（PCA）：K 選択の根拠を残す
- **スコアプロット**（PCA / UMAP 等）：既知群があれば色分け、無ければクラスタラベルで色分け
- **loading プロット**（PCA）：物理的解釈は安定成分のみ
- **代表パターン**（クラスタリング）：各クラスタの中央値・平均・medoid を重ねて表示
- **異常度分布**（異常検知）：ヒストグラム + 閾値線
- **優先度ランキング**（異常検知）：上位 K 件のサムネイル（画像なら画像、パターンならプロット）

**可視化ファイルも provenance の一部**として保存します（`figures/pca_scree_v0.1.0.png` のようにバージョン付き）。エージェントに「良さそうな可視化を作って」と依頼した場合、**その可視化バージョンも仕様書で凍結**します（後付けの見栄え変更禁止）。

---

## 6.8 章末チェックリスト・ワーク

### セルフチェックリスト

- [ ] ①目的で `task_type` が明記されているか？（dimensionality_reduction / clustering / anomaly_detection）
- [ ] ④成功条件が「内部指標 + 安定性 + ドメイン評価 or 人間評価」の 3 点を含むか？
- [ ] クラスタ数・成分数・`contamination` などのハイパーパラメータが仕様書で固定されているか？
- [ ] bootstrap 安定性評価が実装されているか？
- [ ] 前処理が Pipeline 内にあり、bootstrap の各 resample 内で fit されているか？
- [ ] 可視化がバージョン付きで保存され、provenance に記録されているか？
- [ ] 教師あり評価用の既知ラベルを、教師なし学習の**入力に混ぜていない**か？

### ワーク

1. 自分の実験データで、A / B / C のどれか 1 つを選び、Skill 仕様書を書く
2. bootstrap 安定性評価を実装し、成分・クラスタが安定でない場合の警告メッセージを設計する
3. 「見えたパターン」に名前を付けたくなったとき、**安定性を確認するまで名付けを保留する**運用ルールを Skill 仕様書 ⑤ に追記する
4. 教師なし Skill の出力（潜在表現・クラスタラベル・異常度）を、**次の Skill の入力**として使う場面を 1 つ設計する（例：PCA の潜在表現を第5章の教師あり Skill に投入）
5. 可視化のバージョン管理ルールを仕様書 ⑥ に追記する

---

## 6.9 本章のまとめ

- 教師なし学習の Skill 化で最も難しいのは④**成功条件**：`task_type` ごとに「内部指標・安定性・ドメイン評価」の 3 点で書く
- PCA では **成分安定性の bootstrap 評価**、クラスタリングでは **ARI の bootstrap 評価**が必須。ここを省くと「見えたパターン」の後付け解釈になる
- k-means / HDBSCAN / Isolation Forest / One-Class SVM を、**目的とデータ性質から選び分ける**。全部試して一番良いのを採用するのは循環設計
- 教師なしでも、`anti-leakage split contract` は resample 単位で読み替えて適用する（fold 内前処理・同一標本の分離・ラベル混入禁止）
- **可視化も Skill 成果物**：バージョン付きで保存、後付け変更禁止
- 次章（第7章）では、教師あり・教師なしの両方に共通する **CV 設計とリーク検知** を深掘りする

---

## 参考資料

### 本書内の該当章
- [第2章 ARIM データに現れる統計的課題](./chapter-02.md)
- [第3章 Scikit-learn と PyMC の全体像・使い分け](./chapter-03.md)
- [第4章 統計/ML 分析用 Skill の設計原則](./chapter-04.md)
- [第5章 教師あり学習を Skill 化する](./chapter-05.md)
- 第7章 モデル選択・交差検証・データリーク検知（bootstrap 詳細）
- 第8章 解釈可能性とレポート化
- 第14章 統計/ML 特有の失敗パターン（教師なしの失敗事例）
- 付録B Scikit-learn チートシート

### 外部参考
- scikit-learn User Guide - Unsupervised learning: <https://scikit-learn.org/stable/unsupervised_learning.html>
- Campello, R. J. G. B., Moulavi, D., & Sander, J. "Density-Based Clustering Based on Hierarchical Density Estimates." *PAKDD 2013*. — HDBSCAN の原論文
- McInnes, L., Healy, J., & Astels, S. "hdbscan: Hierarchical density based clustering." *Journal of Open Source Software* **2**, 205 (2017).
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. "Isolation Forest." *ICDM 2008*.
- Hubert, L., & Arabie, P. "Comparing partitions." *Journal of Classification* **2**, 193–218 (1985). — Adjusted Rand Index の原典
- Rousseeuw, P. J. "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis." *Journal of Computational and Applied Mathematics* **20**, 53–65 (1987).
