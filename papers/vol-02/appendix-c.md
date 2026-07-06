# 付録C　統計/ML トラブルシューティングと階層データの実データ候補

> **本付録の目的**：本文で扱いきれない **運用上の落とし穴**と、**階層構造を持つ実データ候補**（Materials Project ラウンドロビン等）の入手・ライセンスを列挙する。

> [!IMPORTANT]
> **実データ候補は「使える」と保証するものではありません**。ライセンス・アクセス方法・データ品質は本書執筆時点の情報であり、**利用前に必ず一次情報を確認**してください。

---

## C.1 トラブルシューティング一覧

| カテゴリ | 節 | 主な症状 |
|---|---|---|
| MCMC 未収束 | C.2 | R̂ > 1.01, divergences, low ESS |
| 警告地獄 | C.3 | pytensor warning, deprecation, sampler warnings |
| 過学習疑い | C.4 | train R² >> test R², CV fold 分散大 |
| データリーク疑い | C.5 | CV スコアが hold-out より異常に良い |
| 特徴量スケーリング | C.6 | 収束速度異常、係数の解釈不能 |
| NumPyro / JAX 環境 | C.7 | float32 デフォルト、`jit` エラー、GPU 認識問題 |

---

## C.2 MCMC 未収束

### C.2.1 症状パターン早見

| 症状 | 症状レベル | 対処優先度 |
|---|---|---|
| divergences 数 > 総サンプルの 1% | 高 | ★★★ |
| R̂ > 1.05 | 高 | ★★★ |
| ESS_bulk < 100 chain あたり | 中 | ★★ |
| ESS_tail << ESS_bulk | 中 | ★★ |
| BFMI < 0.3 | 中 | ★★ |
| max_treedepth reached > 1% | 低 | ★ |

### C.2.2 対処フローチャート

```
Step 1: 発散点を localize（az.plot_pair で高曲率領域を確認）
  └─ 単純に「深いスケール差」の場合 →
Step 2: 特徴量・ターゲットの標準化 / スケール見直し
  ├─ 改善 → OK
  └─ 未改善 →
Step 3: パラメータの非中心化 (Centered → Non-centered)
  ├─ 改善 → OK
  └─ 未改善 →
Step 4: prior の見直し
  ├─ 広すぎる → 弱情報 prior に絞る
  ├─ 狭すぎる → データが動かせるように広げる
  └─ 数値不安定な尺度 → 標準化前提の prior に変更
Step 5: モデル簡素化（階層を減らす、相関を落とす）
Step 6: 最終手段として target_accept を 0.99 に上げる
Step 7: バックエンド変更（NumPyro CPU → NutPie など）
```

> **`target_accept=0.99` を最初にやらない**：Ch12 §13.6 の梯子どおり **原因の localize → 標準化 → 非中心化** を先に潰す。`target_accept` を先に上げると根本原因（曲率・スケール・パラメータ化）が隠れ、認定 provenance に「発散 0 だが `target_accept=0.99` に依存」の脆いモデルが残る。

### C.2.3 divergences 対処例

```python
# ❌ 発散多数
with pm.Model() as m:
    sigma_group = pm.HalfNormal("sigma_group", 1)
    mu = pm.Normal("mu", 0, sigma_group, dims="group")   # centered
    pm.Normal("y", mu[group_idx], 1, observed=y)
    idata = pm.sample()   # divergences ~ 数百

# ✅ 非中心化 + target_accept 上げ
with pm.Model() as m2:
    sigma_group = pm.HalfNormal("sigma_group", 1)
    mu_raw = pm.Normal("mu_raw", 0, 1, dims="group")
    mu = pm.Deterministic("mu", mu_raw * sigma_group, dims="group")
    pm.Normal("y", mu[group_idx], 1, observed=y)
    idata = pm.sample(target_accept=0.95)   # divergences ~ 0
```

---

## C.3 警告地獄への対処

### C.3.1 頻出警告と対処

| 警告 | 意味 | 対処 |
|---|---|---|
| `PyTensor could not link to BLAS` | BLAS リンクなし（pip 版で頻発） | conda/mamba 環境推奨、または NumPyro/NUTS へ切替 |
| `UserWarning: The rhat statistic is larger than 1.01` | 収束不良 | C.2 参照 |
| `UserWarning: The effective sample size ... is smaller than 100` | サンプル不足 | `draws` 増、非中心化 |
| `SamplingWarning: There were N divergences after tuning` | 発散あり | C.2.3 参照 |
| `DeprecationWarning: sparse=True is deprecated` | sklearn OneHotEncoder | `sparse_output=True` に変更 |
| `FutureWarning: is_categorical_dtype is deprecated` | pandas | `isinstance(dtype, CategoricalDtype)` |
| `SettingWithCopyWarning: A value is trying to be set on a copy` | pandas スライスへの代入 | `.loc[]` 明示、`.copy()` 追加 |

### C.3.2 警告を「一時的にミュート」する場面

**原則**：警告は**根本原因を対処**する。ただし監査ログには残す。

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    # 一時的な処理
```

**ミュートを避けるべき警告**：
- `SamplingWarning`, `UserWarning`（arviz/pymc）：診断関連は必ず対応
- `SettingWithCopyWarning`：silent bug の温床

---

## C.4 過学習疑い

### C.4.1 診断チェック

| 診断 | 判定 |
|---|---|
| train R² − test R² > 0.15 | 明確な過学習 |
| train R² − CV R² > 0.10 | CV でも過学習傾向 |
| CV fold 間 RMSE の変動係数 > 30% | 分割敏感（データ量不足の疑い） |
| feature_importance で 1〜2 個の特徴量が >90% | 単一特徴量依存の疑い |

### C.4.2 対処カタログ

| 状況 | 対処 |
|---|---|
| 特徴量数 >> サンプル数 | 正則化強化 (`alpha` 増), feature selection |
| 木のモデル | `max_depth` 減, `min_samples_leaf` 増, `n_estimators` 減も |
| ニューラル系 | dropout, early stopping, weight decay |
| ハイパーパラ探索が広い | Nested CV（Ch7 §8.6） |
| Bayesian で post が prior に張り付かず prior 支配 | prior を弱く（Ch14 §15.5） |

### C.4.3 sample size learning curve

```python
from sklearn.model_selection import learning_curve
sizes, train_s, test_s = learning_curve(
    pipe, X, y, cv=cv, groups=g,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="neg_root_mean_squared_error", n_jobs=-1)
# train_s と test_s の差が縮まらない → データ量不足
# 両方悪い → モデル表現力不足（underfit）
```

---

## C.5 データリーク疑い

### C.5.1 リーク検知チェックリスト

- [ ] Scaler / feature selection は Pipeline 内で fit しているか？
- [ ] Test 集合を EDA・特徴量エンジニアリングで参照していないか？
- [ ] Group 単位で予測する場合、GroupKFold を使っているか？
- [ ] 時系列で TimeSeriesSplit（gap 付き）を使っているか？
- [ ] Target encoding を全データで fit していないか？
- [ ] 「未来の特徴量」が含まれていないか（feature_availability_time チェック）

### C.5.2 リーク検知テスト（Ch14 §15.10 #1）

```python
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

def audit_scaler_leakage(model, preprocess, X, y, cv, groups=None,
                        threshold=0.05):
    """"Pipeline 内 fit"（正）と "全体で fit 済み特徴量に対する model 単体 CV"（漏洩）
    を比較するスモークテスト。

    - `preprocess` は Scaler / feature selector 等（`fit_transform` を持つ）
    - 「漏洩ケース」は前処理を CV の外で全データにフィットしてしまう典型ミスを再現
    - スコアが「漏洩ケースで有意に高い」なら要調査（下振れは検知対象外）
    """
    from sklearn.pipeline import Pipeline
    pipe_correct = Pipeline([("prep", clone(preprocess)),
                             ("model", clone(model))])
    # 漏洩再現：前処理を X 全体でフィットしてから model 単体を CV
    X_leaked = clone(preprocess).fit_transform(X, y)

    s_correct = cross_val_score(pipe_correct, X, y, cv=cv, groups=groups,
                                scoring="r2").mean()
    s_leaked = cross_val_score(clone(model), X_leaked, y, cv=cv, groups=groups,
                               scoring="r2").mean()

    diff = s_leaked - s_correct   # 向きで判定（漏洩は「上振れ」）
    if diff > threshold:
        raise AssertionError(
            f"leakage suspected: leaked r2 - correct r2 = {diff:+.3f} "
            f"(> {threshold}). preprocess を Pipeline に入れているか確認。")
    return {"correct_r2": s_correct, "leaked_r2": s_leaked, "signed_diff": diff}
```

> **注意（スモークテスト）**：これは「Scaler を CV 外で fit する」典型パターンの再現テストで、あらゆるリーク（target encoding, 特徴量選択, 時系列の未来参照 等）を検知できるわけではない。個別のリークパターンには **Ch14 §15.2** の該当項目のテストを追加すること。判定は `abs()` ではなく **符号付き**（漏洩ケースが有意に良い＝上振れのみ検知）にする。

---

## C.6 特徴量スケーリング問題

| 症状 | 原因 | 対処 |
|---|---|---|
| Ridge / Lasso / SVM で係数が発散 | 特徴量 scale の桁差 | `StandardScaler` in Pipeline |
| PyMC で MCMC が全く進まない | 特徴量が非標準化で prior と乖離 | 特徴量標準化 + Normal(0, 5) prior |
| PLS で `n_components` を増やしても改善しない | 変数選択が必要 | Variable selection PLS or feature engineering |
| PCA で 1〜2 成分に集中 | scale の 1 変数が支配 | 標準化必須 |

**目安**：特徴量の値域は `[-5, 5]` に収めるのが Bayesian の prior 設計上も楽。

---

## C.7 NumPyro / JAX 環境問題

### C.7.1 セットアップチェックリスト

```python
# 最初に必ず実行
import jax
jax.config.update("jax_enable_x64", True)    # ★ 必須
print("JAX version:", jax.__version__)
print("Devices:", jax.devices())              # CPU/GPU 確認
print("x64 enabled:", jax.config.jax_enable_x64)
```

### C.7.2 よくある症状

| 症状 | 原因 | 対処 |
|---|---|---|
| 収束不良で発散多数（PyMC OK, NumPyro NG） | float32 のまま | `jax_enable_x64=True` を **import 直後に** 設定 |
| `TracerBoolConversionError` | jit 内で bool 分岐 | `jnp.where` に置換 |
| `Cannot mix jax and numpy arrays` | 型混在 | `jnp.asarray()` で統一 |
| GPU 認識されない | CUDA 環境未対応の `jaxlib` | `pip install "jax[cuda12]"` |
| `An NVIDIA GPU may be present ... jaxlib is not installed. Falling back to cpu.` | GPU jaxlib 未インストール | CPU で OK / 必要なら CUDA 版インストール |
| メモリ不足（GPU） | chain 全並列で VRAM 消費 | chain 数を減らす（`chains=2`）／CPU バックエンドに固定（`nuts_sampler="pymc"` or `chain_method="sequential"`）／NumPyro/JAX の `chain_method`・`postprocessing_backend` を確認 |

### C.7.3 バックエンド差の許容範囲（第11章 §11.9）

**tolerance の設計**：
```
|post_mean_A - post_mean_B| / max(post_sd_A, post_sd_B) < 0.1
```

パラメータの posterior SD で正規化した差が 10% 未満なら「同じ」と見なす。認定は 1 バックエンドで行い、他バックエンドは差分検証のみ。

### C.7.4 arviz バージョン互換性

**方針**：PyMC の公式インストールメタデータが解決するバージョンを使う（`pip install pymc` に任せる）。CI・執筆再現のため lockfile（`requirements.txt` / `uv.lock` / `poetry.lock`）に固定する。

**参考（本書検証環境）**：

| 環境 | pymc | arviz | 備考 |
|---|---|---|---|
| 本書検証時 | 5.28.5 | 0.23.4 | `numpyro 0.21`, `jax 0.10` と併用 |
| 過渡期の pin 例 | 5.19.x | `arviz<1.0` | 本書コードが `az.concat` を使う場合の互換 pin |

**注意**：
- PyMC 5.x の依存宣言はおおむね `arviz>=0.13`。書籍のバージョン対応表を「厳密な必須ペア」として受け取らない
- arviz 1.0+ で API 変更あり（`az.concat` 削除等）。本書の付録・章のコードで該当 API を使う場合のみ **`arviz<1.0` を pin**
- 環境問題を「PyMC↔ArviZ バージョン非互換」と即断せず、まず本書の lockfile どおりに再構築してから切り分ける

---

## C.8 階層データの実データ候補

第11-13章で使用する **合成階層データ**の実データ替わり候補です。**すべて執筆時点の情報**、利用前に一次情報を確認してください。

### C.8.1 候補一覧（サマリ）

| 名称 | 階層構造 | ライセンス | サイズ | 用途適合度 |
|---|---|---|---|---|
| Materials Project 経由（MPContribs / 論文 supplementary） | 装置間・研究室間（該当データセットに依存） | 個別（要確認） | 小-中 | △〜◎（該当データセット確認要） |
| NIST SRD（Standard Reference Data） | 認証値・不確かさ（raw 階層は SRD 製品による）| Public Domain（一部有償・要ライセンス） | 大 | ◎（Bayesian 校正）／raw 階層は要確認 |
| NOMAD Repository | 計算ラボ × 手法 | メタデータ CC BY / 個別 DS は別 | 巨大 | ◯（計算 vs 実験） |
| RRUFF Raman | 試料 × 装置 | 教育研究用 | 中 | ◯（第14章 Layer 1-2） |
| MatBench | 単一データセット | CC BY（原データ別） | 中 | ×（階層構造なし・校正の認証値でもない） |
| ChemBench | 化学 benchmarks | 個別 | 中 | △ |
| PDB (Protein Data Bank) | 装置 × 手法 | Public Domain | 巨大 | △（材料外） |
| ARIM 公開データ | 装置 × 拠点 | 個別確認（本書合成データは CC BY-NC-SA 4.0）| 中 | ◎（本書想定） |

### C.8.2 Materials Project 経由の階層データ

**概要**：Materials Project (MP) は主に DFT 計算材料 DB。**「Round-Robin」という単一の公式データセット名は存在しない**ため、階層構造を扱う候補としては次を検討する:

- **MPContribs 上の実験データセット**：外部研究者が投稿した実験データ。装置間比較・研究室間比較を含むものが存在
- **論文 supplementary**：MP を参照する round-robin 実験（例: XRD, XPS）の supplementary CSV
- **同一組成 × 複数計算手法**：MP 上の複数計算エントリを「手法階層」として扱う（実験差ではない点に注意）

- **URL**：<https://materialsproject.org/>、<https://contribs.materialsproject.org/>
- **アクセス**：`mp-api` パッケージ（`mpr.summary.search` は MP の要約ドキュメント検索。round-robin 系の階層データは通常 MPContribs や supplementary から個別取得）
- **ライセンス**：**個別データセットごとに要確認**。MP コア DB は CC BY 4.0（一部改定履歴あり）、MPContribs は投稿元ライセンス依存
- **入手方法**：
  ```python
  from mp_api.client import MPRester
  with MPRester(api_key="...") as mpr:
      # 例: DFT 計算エントリの検索。実験 round-robin データは
      # MPContribs / 論文 supplementary から個別取得する
      docs = mpr.summary.search(...)
  ```
- **本書での用途**：第12章 §12.4 の装置効果モデル、第14章 Layer 3 の階層モデル（**要出典確認**）
- **注意点**：
  - `mpr.summary.search` は round-robin 実験データを直接返さない。**MPContribs / 論文 supplementary を明示的に指す**
  - 装置数（inst_count）が 3〜5 と少ない場合、`sigma_inst` の事後が prior に依存（Ch14 §15.5 パターン B）
  - **API key の取り扱い**：commit しない、`.env` で管理

### C.8.3 NIST SRD

**概要**：米国標準技術研究所の Standard Reference Database。標準試料の物性値・評価済みデータのデータベース。

- **URL**：<https://www.nist.gov/srd>
- **ライセンス**：Public Domain（一部要ライセンス、有償の SRD 製品もあり）
- **主要 SRD**：
  - SRD 34: XPS Database
  - SRD 108: Ceramics WebBook
  - SRD 46: Critically Selected Stability Constants of Metal Complexes
- **本書での用途**：Bayesian 校正の「認証値」として利用
- **注意点**：
  - **SRD 全般が「標準試料 × 装置」の raw 階層データを提供するわけではない**。多くは評価済み・critically selected な参照 DB。装置階層を必要とする場合は個別 SRD 製品の raw / metadata 提供状況を確認する
  - **不確かさの定義**（Type A/B）を確認して MCMC 事後と対応させる
  - 一部は有償・登録要

### C.8.4 NOMAD Repository

**概要**：ヨーロッパ主導の材料計算データリポジトリ。DFT / MD 等の計算データが集約。

- **URL**：<https://nomad-lab.eu/>
- **ライセンス**：**CC BY 4.0**（メタデータ）+ 個別データセットのライセンス
- **アクセス**：`nomad` Python パッケージ、または API
- **本書での用途**：**計算コード × 手法**の階層構造（第14章拡張）
- **注意点**：
  - 計算データなので観測誤差は擬似的（数値誤差）
  - 手法（DFT-GGA / DFT-LDA / DFT-hybrid）を階層のレベルとして扱う

### C.8.5 RRUFF Raman Database

**概要**：鉱物の Raman スペクトルデータベース。同一鉱物を異なる励起波長・装置で測定。

- **URL**：<https://rruff.info/>
- **ライセンス**：**教育・研究用途で自由**（詳細は個別確認）
- **入手**：Web からの直接ダウンロード（バルク API なし）
- **本書での用途**：第14章 Layer 1（sklearn 分類）、Layer 2（Bayesian 校正）
- **注意点**：
  - 励起波長・装置ごとにスペクトルの形状が変わる → group_key として `excitation_wavelength` or `instrument_id`
  - **同一鉱物が train/test 両方に入らない**よう GroupKFold 必須（Ch14 §15.2 B）

### C.8.6 ARIM 公開データ（本書の第一候補）

**概要**：文部科学省 ARIM（Advanced Research Infrastructure for Materials and Nanotechnology）事業で公開される装置測定データ。

- **URL**：<https://arim.nims.go.jp/> または各拠点の公開データポータル
- **ライセンス**：**個別データセットごとに一次確認する**（ARIM 全体で単一ライセンスとは限らない）
- **入手**：拠点ごとの公開データセット、ARIM DX 事業のデータ基盤
- **階層構造**：**装置（inst）× 拠点（facility）× 試料（sample）× ロット（lot）** の 4 階層
- **本書での用途**：**第12章・第14章の第一候補**
- **注意点**：
  - **本書の Ch02 §3.9 で CC BY-NC-SA 4.0 と明記しているのは、本書が同梱する「ARIM 風合成階層データ」のライセンス**であって、実 ARIM 公開データのライセンスとは別。実データを使う場合は該当データセット／ポータルの利用規約を必ず一次確認する
  - もし実データが CC BY-NC 系の場合、Skill を商用製品に組み込む用途では利用不可
  - `arim_context.instrument_id`, `facility_id`, `sample_id`, `lot_id` を provenance に記録（Ch15 §16.5.3）
  - **拠点間で単位表記が異なる場合**：Ch4 のデータ契約で単位統一を必須化

### C.8.7 データ選定判断フロー

```
Q1: 商用利用の可能性がある？
  YES → データセット単位でライセンス確認が必須:
        - MatBench (CC BY) は候補（ただし個別データの原ライセンスも確認）
        - NOMAD は「メタデータ CC BY、個別データセットのライセンスは別」なので per-dataset 確認
        - NIST SRD は Public Domain の SRD 製品のみ（有償・要ライセンスの SRD もある）
  NO  →
Q2: 装置間差・研究室間差の階層構造が主目的？
  YES → ARIM 公開データ、MPContribs 上の実験データ、論文 supplementary
  NO  →
Q3: 認証値（reference value）による Bayesian 校正が主目的？
  YES → NIST SRD / SRM 相当（不確かさ Type A/B が明記されているもの）
        ※ MatBench は ML ベンチマーク用データセットで、校正の「認証値」ではない
  NO  →
Q4: 教育・演習が主目的？
  → 本書の合成階層データを推奨（正解が既知、階層設計が明示的）
```

### C.8.7a 実運用向け一次スクリーニング表

| ゲート | 判定基準 |
|---|---|
| ライセンス | 用途（研究／商用／再配布）に対して明示的に許諾されている |
| 階層キー | `inst_id` / `facility_id` / `lot_id` 等が **明示的に取得可能** |
| Group 数 | `n_group ≥ Ch11 §12.4 の applicability domain 最小値`（例: `inst ≥ 3`）|
| 不確かさ | 認証値・観測不確かさが提供されている（Bayesian 校正用途）|
| 取得再現性 | API / 静的ダウンロードで **バージョン付き** に再取得できる |
| Citation | 引用形式（DOI / BibTeX 等）が公開されている |

**運用**：新規データを採用する際は、まずこのゲートを通過するかを確認してから Ch4 のデータ契約に進む。

### C.8.8 実データ利用時の共通チェック

- [ ] **ライセンス**をリポジトリの LICENSE / README で確認
- [ ] **引用（citation）**を README に明記
- [ ] **アクセス制限**（API key, 認証）を `.env` / secret manager で管理
- [ ] **データバージョン**を provenance の `dataset_sha256` に記録（Ch4）
- [ ] **単位・欠損・外れ値**を EDA で確認（Ch4 のデータ契約）
- [ ] **階層構造の group 数**（inst_count, lot_count 等）を確認（Ch11 §12.4 の applicability domain）
- [ ] **合成データではない**ことを `data_source: real` として provenance に記録（Ch14 §15.8）

---

## C.9 まとめ

- **MCMC 未収束**：発散点の localize → 標準化 → 非中心化 → prior 見直し → モデル簡素化 → 最終手段で `target_accept` → バックエンド変更
- **警告地獄**：警告は根本対処。ミュートは監査ログに残す
- **過学習・リーク**：Pipeline 内 fit、学習曲線、リーク検知 audit（スモークテスト）
- **NumPyro / JAX**：`jax_enable_x64=True` を **必ず** import 直後に
- **arviz / pymc 互換性**：厳密なペア表に頼らず lockfile で pin（本書コードが `az.concat` を使う場合のみ `arviz<1.0`）
- **実データ候補**：ARIM 公開データを第一候補、次点で MPContribs / NIST SRD / RRUFF
- **商用転用**：**すべてデータセット単位でライセンス一次確認**（「CC BY-NC を避ければ良い」ではなく個別確認が必須）
- **すべての実データ利用は provenance に一次情報を記録**

vol-02 本編と 3 付録はここで完結です。

---

## 参考資料

### 本書内の該当章

- 第5章：データ契約・provenance
- 第8章 / 第15章 §15.2：データリーク
- 第11章 §11.9 / 第15章 §15.7：バックエンド差
- 第12章 §12.4：階層モデル applicability domain
- 第13章 §13.9：MCMC 診断 14 項目
- 第15章 §15.5：prior 暴走
- 第15章 §15.8：合成 vs 実データ
- 第16章 §16.5.3：arim_context provenance
- 付録A A.7：完全 provenance schema
- 付録B B.3, B.4：バックエンド・MCMC 対処

### 外部参考

- [Materials Project](https://materialsproject.org/)
- [NIST Standard Reference Data](https://www.nist.gov/srd)
- [NOMAD Repository](https://nomad-lab.eu/)
- [RRUFF Database](https://rruff.info/)
- [MatBench](https://matbench.materialsproject.org/)
- [ARIM 公開データ](https://arim.nims.go.jp/)
- [PyMC Troubleshooting](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/troubleshooting.html)
- [JAX FAQ](https://jax.readthedocs.io/en/latest/faq.html)
- Kapoor, S. & Narayanan, A. *Leakage and the Reproducibility Crisis in ML-Based Science*. Patterns (2023). — Leakage 事例
