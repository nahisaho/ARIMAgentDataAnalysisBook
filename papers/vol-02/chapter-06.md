# 第6章 教師あり学習を Skill 化する

> **本章の到達目標**
> - 第5章の Skill 仕様書テンプレートを、**教師あり学習 Skill**（scikit-learn ベース）として実装できる
> - 章冒頭の **anti-leakage split contract**（同一試料・同一ロットの分割違反禁止）を、Skill の入り口で機械的に検査できる
> - 線形回帰・**PLS**・ランダムフォレスト・勾配ブースティングを、目的とデータ性質から選び分けられる
> - 3 つのハンズオン（校正曲線・物性予測・スペクトル分類）を、共通の Skill 構造で実装できる
> - 失敗例（リーク混入・過学習・特徴量スケール未処理）を検知し、改善版を書ける
>
> **本章で扱わないこと**
> - k-fold / grouped / time-series の詳細な CV 設計、ネスト CV → **第8章**
> - SHAP・PDP・permutation importance などの解釈可能性 → **第9章**
> - PyMC・階層モデル → **第9〜11章**
> - 深層学習（NN, CNN, Transformer 系） → 本書外（vol-03 予定）
> - 実装コードの完全形（本章は Skill 構造とキーとなる契約チェックに集中、完全なノートブックは付属リポジトリ）

---

## 6.1 この章で作る Skill

本章で作る **教師あり学習 Skill** は、次の 3 つのハンズオンを共通構造で扱います：

| ハンズオン | データ | 目的変数 | 主に使う手法 | 該当節 |
|---|---|---|---|---|
| A. 校正曲線 | 標準試料の測定値と真値ペア（vol-01 由来の合成データ） | 濃度・強度など連続値 | **線形回帰**（単純・多項式） | §6.6 |
| B. 物性予測 | MatBench `matbench_expt_gap` (n=4,604) | バンドギャップ (eV) | **ランダムフォレスト / 勾配ブースティング** | §6.7 |
| C. スペクトル分類 | RRUFF Raman データ | 鉱物種ラベル | **PLS-DA / ロジスティック回帰** | §6.8 |

**共通構造**は第5章のテンプレートに準拠します：①目的 / ②入力条件 / ③出力形式 / ④成功条件（3 点セット）/ ⑤禁止事項（severity 付き）/ ⑥再現性条件。**本章では、この 6 要素にコードを対応させる**ことに集中し、CV 設計の細部（第8章）や解釈（第9章）は後段に譲ります。

> [!IMPORTANT]
> **第6章は「型を作る章」です**。第7章（教師なし）、第10・11章（ベイズ）、第14章（capstone）は、本章で作った Skill 構造を差分で拡張していきます。ここで手を抜くと後の章が全部やり直しになります。

---

## 6.2 章冒頭の anti-leakage split contract

**本章のすべてのハンズオンは、次の契約を通過してから始めます**。詳細な CV 設計は第8章ですが、**最低限これだけは第6章の入り口で機械的に検査**します：

### 契約項目（最低限）

1. **同一試料が train / test の両方に入っていない**
   - 「同一試料」の定義：`sample_id` 列で判定（無ければ Skill 実行前に付与を必須化）
2. **同一ロット / バッチが train / test の両方に入っていない**（該当データのみ）
   - 例：合成条件が同じロット、同一実験セッションの繰り返し測定
   - `group_id` 列で判定
3. **時系列データでは train が test より過去である**
   - `timestamp` 列で判定、未来を先に見ない
4. **全データでの正規化・スケーリングを行っていない**
   - `StandardScaler.fit` は train のみで実施、test には `transform` のみを適用（`Pipeline` 経由で自動化）
5. **目的変数派生量を特徴量に含めていない**
   - 特徴量列名と目的変数の系譜（provenance）を突き合わせて確認

### 実装スケルトン（第6章共通の契約チェッカ）

```python
class LeakageError(RuntimeError):
    """契約違反時の fatal 例外（第5章 §5.5 分類 A に対応）。"""

def check_split_contract(
    X_train, X_test, y_train, y_test,
    *,
    id_train=None, id_test=None,          # sample_id の値配列
    group_train=None, group_test=None,    # group_id の値配列
    ts_train=None, ts_test=None,          # timestamp の値配列
    target_lineage: set[str] = frozenset(),
    requires_sample_id: bool = True,       # デフォルトで必須
    requires_group_id: bool = False,       # 該当データのみ True
    requires_timestamp: bool = False,      # 時系列データのみ True
) -> None:
    """全ハンズオンの入り口で呼ぶ。契約違反があれば raise（fatal）。"""
    # 1. sample_id 重複禁止
    if requires_sample_id:
        if id_train is None or id_test is None:
            raise LeakageError("sample_id が必須だが与えられていない")
        overlap = set(id_train) & set(id_test)
        if overlap:
            raise LeakageError(f"sample_id overlap: {len(overlap)} 件")
    # 2. group_id 重複禁止
    if requires_group_id:
        if group_train is None or group_test is None:
            raise LeakageError("group_id が必須だが与えられていない")
        overlap = set(group_train) & set(group_test)
        if overlap:
            raise LeakageError(f"group_id overlap: {len(overlap)} 件")
    # 3. 時系列順序（本章では最小 holdout 用のみ扱う）
    if requires_timestamp:
        if ts_train is None or ts_test is None:
            raise LeakageError("timestamp が必須だが与えられていない")
        if pd.isna(ts_train).any() or pd.isna(ts_test).any():
            raise LeakageError("timestamp に欠損あり")
        if ts_train.max() >= ts_test.min():
            raise LeakageError("train が test より未来を含む、または同時刻を共有")
    # 4. 目的変数派生量チェック
    illegal = set(X_train.columns) & target_lineage
    if illegal:
        raise LeakageError(f"目的変数派生量が特徴量に混入: {illegal}")
    # 5. スケーリング契約は Pipeline で担保 (§6.4)
```

**設計上のポイント**：

- **`sample_id` はデフォルトで必須**。指定を忘れると即 `LeakageError`
- **特徴量 `X` と識別子 `id_*` は分離**（`X_train` に `sample_id` 列を含めると Pipeline が特徴量として学習してしまう）
- **`requires_timestamp=True` は本章の最小 holdout 用**。rolling / expanding CV は第8章
- **同一 timestamp 内リーク** は `>=` で拾う（等号を許すと同時刻測定が train/test に分散するため）


> [!WARNING]
> `LeakageError` は **fatal 例外**として扱い、Skill 実行を停止します（第5章 §5.5 分類 A）。「警告だけ出して続行」は禁止です——エージェントが自動継続してしまうと、後段で「動いた」ように見えるからです。

### この契約でカバーできないこと

- **train 内での CV split の分割違反**（例：CV の各 fold で group leak が起きる）→ 第8章の `GroupKFold`・`GroupShuffleSplit` で対応
- **前処理の順序による微妙なリーク**（例：欠損補完を全データで実施）→ Pipeline 化で担保（§6.4）
- **時系列の遅延特徴量作成時の未来参照** → 第8章のケーススタディで扱う

本章では **「大きな 3 つの穴」だけを塞ぎ**、細部は第8章に任せる方針です。

---

## 6.3 3 つのハンズオンの位置づけ

3 ハンズオンは、**難易度**と**手法**の観点で意図的に段階を付けています：

```mermaid
flowchart LR
    A[A. 校正曲線<br/>線形回帰] --> B[B. 物性予測<br/>RF / GBM]
    B --> C[C. スペクトル分類<br/>PLS-DA / ロジスティック]
    A -.->|型の完成| B
    B -.->|CV導入| C
```

| 観点 | A. 校正曲線 | B. 物性予測 | C. スペクトル分類 |
|---|---|---|---|
| データ数 | 数十点（小 n） | 数千点（MatBench） | 数百〜数千点（RRUFF） |
| 手法 | 線形回帰・多項式 | RF・GBM（LightGBM 系） | PLS-DA・ロジスティック |
| 主な学び | Skill 構造の完成 | ハイパーパラメータ・CV 導入 | 高次元・多クラス・PLS の妥当性 |
| 難所 | 過学習（多項式次数） | 特徴量重要度の誤読 | 波長間の強い相関 |
| 章間接続 | 第5章のテンプレを最小例で完成 | 第8章 CV / 第9章 解釈可能性の予告 | 第7章 PCA との対比 |

**A から順に読むことを推奨**しますが、時間が限られていれば **A + B** だけでも Skill 構造の理解には十分です。C は第7章 PCA と対比しながら読むと理解が深まります。

---

## 6.4 教師あり学習 Skill の共通構造

第5章の仕様書テンプレートを、実装レベルで次の 5 ブロックに展開します。**全ハンズオンで同じ構造を守る**のが本章のポイントです：

```
Skill 実行の流れ:
  1. 入力検証     → schema チェック + anti-leakage split contract
  2. Pipeline 構築 → 前処理 + モデル（sklearn.pipeline.Pipeline）
  3. 学習・評価  → CV スコア + 独立テストスコア
  4. 予測（任意）→ 予測値 + 外挿警告
  5. 記録        → provenance 保存
```

### ブロック 2 の要：`Pipeline` を必ず使う

**scikit-learn の `Pipeline` は「前処理を含めた fit / predict を 1 オブジェクトにまとめる」道具**で、リーク防止に必須です。**Pipeline を使わずに前処理を手書きすると、CV の各 fold で「全データで fit した scaler」が使われて微妙なリークが起きます**（第8章で再説）。

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipeline = Pipeline([
    ("scaler", StandardScaler()),   # fit は fold の train のみで行われる
    ("model",  Ridge(alpha=1.0)),
])
pipeline.fit(X_train, y_train)      # 前処理 + 学習が一気通貫
```

### ブロック 3 の要：`cross_val_score` の落とし穴と代替

`cross_val_score` は簡単ですが、**各 fold のスコアしか返らず、fold ごとの予測値・分割 index が取れない**のが弱点です。本章では、**再現性を担保するために `cross_validate` + 明示的 splitter** を推奨します：

```python
from sklearn.model_selection import cross_validate, KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(
    pipeline, X_train, y_train, cv=cv,
    scoring=["neg_root_mean_squared_error", "r2"],
    return_train_score=True, return_indices=True,  # 分割 index を残す
)
```

`return_indices=True`（scikit-learn 1.3+ で利用可能）を使うと、`data_split`（第5章 §5.6）を provenance にそのまま保存できます。古い環境では splitter の `split()` 結果を自前保存してください。返り値の `scores["indices"]["train"]` / `scores["indices"]["test"]` は numpy 配列なので、JSON/YAML 保存時は `.tolist()` 化が必要です。

> [!WARNING]
> ここでは説明のために `KFold` を素で使っていますが、**反復測定・同一ロットが含まれるデータでは `GroupKFold`（第8章）が本来の姿**です。本章のスケルトンは「型のデモ」であり、実データで運用する際は必ずグループ構造を確認します。

### ブロック 5 の要：provenance の最小セット

第5章の追加推奨フィールドに、実行結果を書き込みます：

```python
provenance = {
    # vol-01 継承
    "input_sha256": sha256_of(X, y),
    "skill_version": "0.1.0",
    "run_datetime_utc": now_utc_isoformat(),
    "package_versions": {"scikit-learn": sklearn.__version__, ...},
    "random_seed": 42,
    # vol-02 outline canonical
    "cv_scheme": {"type": "KFold", "n_splits": 5, "shuffle": True, "seed": 42},
    "data_split": scores["indices"],  # 各 fold の train/test index
    "model_config": pipeline.get_params(),
    # 第5章追加推奨
    "feature_columns": list(X_train.columns),
    "target_column": {"name": y.name, "unit": "eV"},
    "metric_definition": {
        "primary": "RMSE = sqrt(mean((y_true - y_pred)^2))",
        "unit": "eV",
        "aggregation": "fold-mean",
    },
    "training_score": scores["train_neg_root_mean_squared_error"].tolist(),
    "cv_score": scores["test_neg_root_mean_squared_error"].tolist(),
}
```

**この構造は、全ハンズオン共通**です。§6.6〜§6.8 では、この構造への差分だけを説明します。

---

## 6.5 手法の選び分け（線形・PLS・RF・GBM）

第4章 §4.2 の選択マップを、教師あり回帰・分類に絞って具体化します：

| 手法 | 得意領域 | 苦手 | 本書での位置づけ |
|---|---|---|---|
| **線形回帰・Ridge・Lasso** | 特徴量が少なく解釈重視、校正曲線 | 非線形関係、変数間強相関 | 校正曲線・ベースライン |
| **PLS（部分最小二乗）** | 特徴量が多く（p > n）、変数間相関が強い（スペクトル） | 極端な非線形 | 分光データの標準手法 |
| **ランダムフォレスト (RF)** | 非線形、変数間相互作用、頑健性 | 外挿、過学習の予防が甘い | 中規模 tabular の第一選択 |
| **勾配ブースティング（LightGBM / XGBoost）** | 予測性能最重視、大規模 tabular | ハイパーパラメータ調整が繊細、外挿 | ベンチマーク・性能上限確認 |

### 選び方の 3 質問

1. **特徴量数 p は n より大きいか？** → Yes なら PLS / Ridge が候補
2. **変数間に強い相関があるか？** → Yes なら Lasso（変数選択）か PLS（合成成分）
3. **物理的な線形性の根拠があるか？** → Yes なら線形、No なら RF / GBM

**「とりあえず全部試して一番良かったのを採用」は第5章 §5.4 の循環設計問題**にあたります。Skill 仕様書段階で **1〜2 手法に絞り**、選ばなかった手法は「なぜ選ばなかったか」を仕様書 ① 目的の下に書きます。

> [!TIP]
> どうしても複数試したい場合は、**独立テストを見る前に決着させる**か、**「探索用 Skill」と「本評価 Skill」を分ける**（探索は CV 内、本評価は 1 回のみ）ようにします。第5章 §5.4 の対話例で示した「テスト後の後付け探索」は禁止です。

---

## 6.6 ハンズオン A：校正曲線 Skill

**目的**：標準試料の測定値と真値のペアから、未知試料の測定値を真値に変換する校正関数を作る。

### Skill 仕様書（要約）

| 要素 | 内容 |
|---|---|
| ① 目的 | 単変量校正曲線を学習し、未知試料の測定値を物理量に変換する。`task_type=regression` |
| ② 入力 | tabular（`measured`, `true_value`, `sample_id`）、n≈30〜100 |
| ③ 出力 | 学習済み Pipeline、CV スコア、外挿警告フラグ、provenance |
| ④ 成功条件 | 評価指標 = RMSE (単位は目的変数と同じ)、目的関数 = MSE、5-fold CV RMSE ≤ 目的変数レンジの 5%、R² ≥ 0.95 |
| ⑤ 禁止事項 | 標準試料の重複投入（同一 sample_id の train/test 混在）、多項式次数の後付け引き上げ、外挿域予測の警告なし |
| ⑥ 再現性 | 上記共通セット + `polynomial_degree` |

### 実装スケルトン

```python
FEATURES = ["measured"]
METADATA = ["sample_id"]

def fit_calibration(df, degree=1, cv_splits=5, random_state=42):
    # 特徴量と識別子を分離（sample_id を特徴量にしない）
    X = df[FEATURES]
    y = df["true_value"]
    ids = df["sample_id"]
    # 1. 分割 → 入力検証（sample_id 契約）
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=random_state,
    )
    check_split_contract(
        X_train, X_test, y_train, y_test,
        id_train=id_train, id_test=id_test,
        requires_sample_id=True,
    )
    # 2. Pipeline
    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    # 3. 学習 + CV（同一 sample_id が反復測定される場合は GroupKFold 推奨、第8章）
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scores = cross_validate(pipeline, X_train, y_train, cv=cv,
                            scoring=["neg_root_mean_squared_error", "r2"],
                            return_indices=True)
    pipeline.fit(X_train, y_train)
    # 4. 独立テスト（1 回のみ）
    test_rmse = root_mean_squared_error(y_test, pipeline.predict(X_test))
    # 5. provenance 記録（省略）
    return pipeline, scores, test_rmse, provenance
```

### 典型的な失敗と修正

| 失敗 | 症状 | 修正 |
|---|---|---|
| 多項式次数を試して一番良いのを採用 | CV スコアと独立テストが大きく乖離 | 次数を仕様書で 1 or 2 に固定、探索は別 Skill |
| 標準試料の反復測定を分割せず投入 | R²=0.99 だが実装現場で使うと精度悪化 | `sample_id` 契約チェッカで拒否 |
| 校正範囲外の測定値を予測 | 外挿域で非現実的な値 | 予測時に `X` の値域と学習範囲を比較して警告 |

---

## 6.7 ハンズオン B：物性予測 Skill

**目的**：組成・記述子から物性値（バンドギャップ等）を予測する。MatBench `matbench_expt_gap` (n=4,604) を使う。

### Skill 仕様書（要約 + A との差分）

| 要素 | A との差分 |
|---|---|
| ① 目的 | 多変量回帰、`task_type=regression` |
| ② 入力 | 組成記述子（Magpie 等で生成）、または CIF から特徴量抽出。n≈数千 |
| ③ 出力 | RF or GBM の学習済み Pipeline、特徴量重要度（第9章で扱う）、外挿警告 |
| ④ 成功条件 | RF/Magpie ベースラインの目安：**MatBench 公式 5-fold で RMSE ≈ 0.8–0.9 eV**。RMSE ≤ 0.5 eV は上位モデル・高度特徴量・専用モデルの目標であり本章の合格基準にはしない。詳細な数値は本節末尾の外部参考を参照 |
| ⑤ 禁止事項 | 特徴量重要度で選んだ変数だけで再学習して独立テストを見る（後付け特徴量選択） |
| ⑥ 再現性 | RF/GBM のハイパーパラメータ（`n_estimators`, `max_depth`, `learning_rate` 等）全指定 |

> [!NOTE]
> **MatBench は公式 5-fold の nested-CV benchmark** です。leaderboard 数値と比較したい場合は、任意の `KFold` ではなく **MatBench 公式 fold（`matbench` パッケージ）を使う**必要があります。本章は Skill 構造を学ぶための練習として `KFold` を使いますが、公式リーダーボード比較は付録B / 第8章で扱います。

### 実装スケルトン（RF）

```python
from sklearn.ensemble import RandomForestRegressor

pipeline = Pipeline([
    ("model", RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=42,
    )),
])
# 以降の cross_validate / provenance は §6.4 と同じ
```

### RF/GBM 特有の注意

- **`n_estimators` を後から増やして CV スコアを改善する**のは仕様書違反（後付け）。仕様書で範囲を固定
- **特徴量重要度は「予測に効いた」であって「物理的に効いた」ではない**（第9章）
- **外挿は苦手**：学習データが取っていない組成領域の予測は、`applicability_domain_check`（最近傍距離チェック等、実装は§6.10）で warning を返す

### LightGBM を使う場合

LightGBM は sklearn API 互換なので、`Pipeline` にそのまま入ります：

```python
from lightgbm import LGBMRegressor
pipeline = Pipeline([
    ("model", LGBMRegressor(
        n_estimators=500, num_leaves=31, learning_rate=0.05,
        random_state=42, n_jobs=-1,
    )),
])
```

**RF と LightGBM の比較は「性能上限確認」が目的**であり、両方を並行運用するのは Skill 増殖の元です。運用 Skill は 1 つに絞ります。

---

## 6.8 ハンズオン C：スペクトル分類 Skill

**目的**：Raman スペクトル（数百〜数千の波数点）から鉱物種を分類する。RRUFF データを使用。

### Skill 仕様書（要約 + A/B との差分）

| 要素 | A/B との差分 |
|---|---|
| ① 目的 | 多クラス分類、`task_type=classification` |
| ② 入力 | 波数（列）× 試料（行）の tabular（p ≈ 500〜2000、n ≈ 数百〜千） |
| ③ 出力 | 学習済み Pipeline、**PLS 潜在空間でのクラススコア + argmax 判別**（校正確率が必要なら `CalibratedClassifierCV` を後段に）、混同行列 |
| ④ 成功条件 | macro F1 ≥ 0.85、各クラス recall ≥ 0.75、クラス数 K で K が大きい場合は per-class 集計 |
| ⑤ 禁止事項 | ベースライン補正・正規化を全データで実施（Pipeline 化必須）、同一標本（specimen）の複数スペクトルを train/test 両方に分散配置 |
| ⑥ 再現性 | 前処理チェイン（ベースライン推定法、平滑化、L2 正規化）を仕様書で固定 |

### PLS-DA（PLS 判別分析）を使う理由

Raman スペクトルは **p >> n（波長点 > 試料数）** かつ **隣接波長間で強い相関**があります。この状況では：

- **単純ロジスティック回帰**：p > n で解が発散、L2 正則化必須
- **PLS-DA**：合成成分（潜在変数）を作って回帰し、判別する。**分光分野の標準的第一選択**
- **RF / GBM**：波長間の順序情報を活かしにくく、通常 PLS より劣る

### 実装スケルトン（PLS-DA）

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer

# PLS-DA は「PLS 回帰 + one-hot 目的変数 + argmax 判別」
lb = LabelBinarizer()
Y_train = lb.fit_transform(y_train)   # (n, K)

# PLSRegression は内部で中心化・スケーリングを行うため
# 外側の StandardScaler は使わない（二重処理を防ぐ）
pipeline = Pipeline([
    ("pls", PLSRegression(n_components=10, scale=True)),
])
pipeline.fit(X_train, Y_train)
Y_score = pipeline.predict(X_test)           # クラス score（確率ではない）
y_pred = lb.classes_[np.argmax(Y_score, 1)]  # argmax で判別
```

**PLS-DA の出力は「クラス score」であり、確率校正されていない点に注意**します。ROC/PR 曲線や意思決定閾値の設計が必要な場合は、後段に `sklearn.calibration.CalibratedClassifierCV` を置くか、L2 正則化ロジスティック回帰に切り替えます。`n_components`（潜在変数の数）は **CV で選ぶ**が、独立テストを見た後の後付け変更は禁止（§5.4 循環設計）。

### Raman 前処理の一般論

上記スケルトンは PLS-DA の最小例です。実運用では、次を Pipeline 内に組み込みます：

- **ベースライン補正**（Asymmetric Least Squares 等、`FunctionTransformer` 経由）
- **強度正規化**（vector normalization、area normalization、SNV のいずれか、目的に応じて選択）
- **平滑化**（Savitzky-Golay filter、必要に応じて微分）
- **波長軸の統一補間**（試料ごとに波長点が異なる場合）

前処理はすべて Pipeline 内で実施し、fold ごとに fit されることを担保します。前処理の選択自体を CV で最適化する場合は、これも独立テストを見る前に固定します。

### group leak と `GroupKFold`

「サンプル数を稼ぐために同一標本の複数スペクトルを重複投入」すると、`sample_id` レベルで leak します。RRUFF では **同一標本（specimen）** に複数の測定エントリがある場合があり、この場合は **`specimen_id`（標本 ID）を `group_id` にして `GroupKFold`（第8章）** を使うのが本来の姿です。

> [!WARNING]
> **鉱物種ラベル（分類の目的変数）を group に使うのは誤り**です。それをすると、fold ごとに特定クラスが train から消え、通常の多クラス分類として評価が成立しません（それは open-set / zero-shot 分類という別問題で、本書では扱いません）。group は「同一標本」「同一測定セッション」「同一 RRUFF entry」など、**目的変数と直交する識別子**を選びます。

本章では最低限「同一 `specimen_id` が train/test 両方に無い」ことを §6.2 の契約チェッカで担保します。波長軸の不揃いは事前補間を Pipeline 内に組み込み（`FunctionTransformer` 経由）、fold 内で実施します。

---

## 6.9 失敗パターンと改善版

3 ハンズオン共通で頻出する失敗を、統合表としてまとめます（第15章の予告編）：

| 症状 | 原因 | 改善策 | 該当節 |
|---|---|---|---|
| CV スコアが良いのに独立テストが悪い | 前処理を Pipeline 化せず全データで実施 | `Pipeline` 経由に変更 | §6.4 |
| CV スコアが良いのに実運用で悪化 | 同一試料/ロットの train/test 混在 | `check_split_contract` を fatal で拒否 | §6.2 |
| ハイパーパラメータを増やすと CV スコアが上がり続ける | 探索空間が広すぎる + テストを繰り返し見ている | ネスト CV（第8章）、独立テストは 1 回のみ | §6.5, 第8章 |
| 結果を見た後で指標・閾値・除外条件を変更した（p-hacking の統計版） | 探索と本評価が分離されていない | 仕様書に frozen date と diff を記録、audit violation として扱う | §5.4, §5.5 |
| 特徴量重要度で選んだ変数だけで再学習 → 精度改善 | 特徴量選択のリーク | 特徴量選択も Pipeline 内で fold ごとに実施 | §6.7, 第8章 |
| 学習データに無い組成/波長域で予測 → 非現実的な値 | 外挿域予測の警告なし | applicability domain チェックを予測時に組み込み | §6.10 |

**改善版はコード差分ではなく、Skill 仕様書の差分として管理**します。「なぜ v0.2 で追加したか」を仕様書のバージョン履歴に書き残すのが provenance 運用の基本です。

---

## 6.10 他データ型への転用と applicability domain

本章の 3 ハンズオンは Raman・組成が中心ですが、**同じ Skill 構造で他のデータ型に転用可能**です。転用時のチェックポイント：

| データ型 | 追加チェック | 参照 |
|---|---|---|
| XRD パターン | 2θ 軸のズレ補正、ピーク検出との併用 | vol-01 第13章 |
| 顕微鏡画像 | 特徴量抽出（PCA / 深層 embedding）→ 本 Skill に投入 | 本書外（vol-03） |
| 時系列（in-situ 測定） | 時系列 CV（第8章）必須、遅延特徴量作成時の未来参照禁止 | 第8章 |
| 多目的最適化 | 本 Skill は単目的想定。多目的は別 Skill で構成 | 本書外 |

### applicability domain チェックの最小版

```python
def check_applicability_domain(X_train_transformed, X_pred_transformed,
                               k=5, threshold_quantile=0.99):
    """予測点が学習データの近傍にあるかを最近傍距離で判定。

    重要：Pipeline の前処理を通した後の空間で距離計算すること。
    生の組成特徴量やスペクトル強度をそのまま渡すと、
    列ごとのスケール差で距離の意味が壊れる。
    """
    from sklearn.neighbors import NearestNeighbors
    # train 側：自己近傍を含めないよう k+1 を取り、先頭（距離 0）を捨てる
    nn_train = NearestNeighbors(n_neighbors=k + 1).fit(X_train_transformed)
    d_train, _ = nn_train.kneighbors(X_train_transformed)
    d_train = d_train[:, 1:]  # 自己距離を除外
    # 予測点：train を基準に k 近傍
    nn_ref = NearestNeighbors(n_neighbors=k).fit(X_train_transformed)
    d_pred, _ = nn_ref.kneighbors(X_pred_transformed)
    threshold = np.quantile(d_train.mean(axis=1), threshold_quantile)
    is_extrapolation = d_pred.mean(axis=1) > threshold
    return is_extrapolation  # True の予測点は warning 対象
```

これは **暫定的な最小実装**です。組成データでは Mahalanobis 距離、スペクトルでは PLS 残差など、より適切な距離が知られています（Roy 2015 参照）。また、Pipeline 前段の前処理が入っている場合は `pipeline[:-1].transform(X)` で最終モデル直前の空間を作り、そこで距離を測ります。詳細は付録B のチートシートに載せます。

---

## 6.11 章末チェックリスト・ワーク

### セルフチェックリスト（Skill 実装後に確認）

- [ ] Skill 仕様書の ①〜⑥ が第5章テンプレに沿って埋まっているか？
- [ ] `check_split_contract` が Skill の入り口で呼ばれ、契約違反時は `LeakageError` で停止するか？
- [ ] 前処理は Pipeline 内に入っており、全データで `fit` を呼ぶ箇所が無いか？
- [ ] `cross_validate` で `return_indices=True` を使い、`data_split` を provenance に保存しているか？
- [ ] 独立テストは 1 回のみ評価しているか？ 複数回参照していないか？
- [ ] 予測時に applicability domain 警告が発火するか？
- [ ] provenance に `feature_columns`, `target_column`, `metric_definition` が入っているか？

### ワーク

1. 自分の実験データで、A / B / C のどれか 1 つの構造を選び、Skill 仕様書を書く（第5章の続き）
2. `check_split_contract` を、自分のデータで意図的にリークを作って呼び出し、fatal で拒否されることを確認
3. ハイパーパラメータを 3 通り試し、**独立テストを見る前に**採用値を仕様書で固定する
4. 独立テストを 1 回だけ評価し、結果を provenance と一緒に保存する
5. 「もっと良いスコアが欲しくなった時に、何をしてはいけないか」を仕様書 ⑤ 禁止事項に追記する（第5章 §5.4 / §5.5 の復習）

---

## 6.12 本章のまとめ

- 教師あり学習 Skill は、第5章の仕様書テンプレを **入力検証 / Pipeline / 学習・評価 / 予測 / provenance の 5 ブロック**で実装する
- **anti-leakage split contract**（同一試料・同一ロット・時系列順・スケーリング・目的変数派生）を章冒頭で機械的に検査。違反は fatal
- **Pipeline を必ず使う**：前処理を含めた fit/predict を 1 オブジェクトに閉じ込め、CV の各 fold で正しく処理されることを保証
- 手法選択は **3 質問（p vs n、変数間相関、線形性根拠）** で絞り、循環設計問題（第5章 §5.4）を避ける
- 3 ハンズオン（校正曲線・物性予測・スペクトル分類）は共通構造で、A から順に難度が上がる
- 詳細な CV 設計・解釈可能性は **第8章・第9章** に譲る。本章は「型」を作る章

---

## 参考資料

### 本書内の該当章
- [第3章 ARIM データに現れる統計的課題](./chapter-03.md)
- [第4章 Scikit-learn と PyMC の全体像・使い分け](./chapter-04.md)
- [第5章 統計/ML 分析用 Skill の設計原則](./chapter-05.md)
- 第7章 教師なし学習を Skill 化する（次章）
- 第8章 モデル選択・交差検証・データリーク検知（詳細 CV）
- 第9章 解釈可能性とレポート化（SHAP / PDP / permutation importance）
- 付録A 統計/ML Skill テンプレート集
- 付録B Scikit-learn チートシート

### 外部参考
- scikit-learn User Guide - Supervised learning: <https://scikit-learn.org/stable/supervised_learning.html>
- scikit-learn Pipeline: <https://scikit-learn.org/stable/modules/compose.html>
- Dunn, A., Wang, Q., Ganose, A. et al. "Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm." *npj Computational Materials* **6**, 138 (2020). <https://doi.org/10.1038/s41524-020-00406-3>
- Wold, S., Sjöström, M., & Eriksson, L. "PLS-regression: a basic tool of chemometrics." *Chemometrics and Intelligent Laboratory Systems* **58**, 109–130 (2001). <https://doi.org/10.1016/S0169-7439(01)00155-1>
- Ke, G., Meng, Q., Finley, T. et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *Advances in Neural Information Processing Systems* **30** (NIPS 2017).
- Roy, K., Kar, S., & Ambure, P. "On a simple approach for determining applicability domain of QSAR models." *Chemometrics and Intelligent Laboratory Systems* **145**, 22–29 (2015).
