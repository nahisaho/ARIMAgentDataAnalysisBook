# 第14章　統計/ML 特有の失敗パターン

> [!IMPORTANT]
> **本章の到達目標**
>
> - 統計/ML Skill を壊す **代表的な失敗パターン**を、症状 → 原因 → 予防策 → 検知方法の順に把握する
> - **リーク・p-hacking・過学習・prior 暴走・MCMC 未収束・多重比較・バックエンド差**を、Ch4-13 の設計原則がどこで守っているかで整理する
> - 「動いてしまう」失敗（沈黙する誤りやすい signal）を区別し、**必ず検知できる provenance ゲート**を設計する
> - 発生時の**復旧手順**（rollback / re-fit / Skill バージョンアップ）を運用パターンとして持ち帰る

## 本章で扱わないこと

- **新しいモデル種類**：本章は Ch4-13 の裏返し。新規手法は導入しない
- **セキュリティ**：モデル抽出攻撃・データ毒入れは vol-04 以降
- **人的組織要因**：責任分界・レビュー体制は第15章
- **失敗の統計的検定理論**：Type I/II の理論展開は他書に委ねる

---

## 14.1 失敗パターン俯瞰

vol-02 で導入した Skill が壊れるとき、症状は 7 種類に集約できます。

| 分類 | 主な失敗 | 主に守る章 | 検知の難しさ |
|---|---|---|---|
| **データ準備** | データリーク（fit-time / feature-time / group-time） | Ch4, Ch5, Ch7 | 高（沈黙する） |
| **評価** | p-hacking、多重比較未補正 | Ch7, Ch8 | 中 |
| **モデル** | 過学習、正則化不足 | Ch5, Ch7 | 低（CV で気づく） |
| **Bayesian** | prior 暴走（データ無視 or データ丸乗り） | Ch9, Ch10 | 中（PPC で見える） |
| **MCMC** | 収束不良、divergences 常態化 | Ch10, Ch12 | 低（診断で気づく） |
| **バックエンド** | PyMC ↔ NumPyro 差、GPU 差、乱数実装差 | Ch10, Ch12 | 高（気づきにくい） |
| **統合** | provenance チェイン切れ、合成→実データ誤運用 | Ch13 | 高（hash 検証がないと通る） |

> [!IMPORTANT]
> **「沈黙する失敗」を最も警戒**：CV スコアが良く、$\hat{R}$ も PPC も通ってしまう失敗（リーク・多重比較・チェイン切れ）が最も危険です。**動いた** ≠ **正しい**。本章は「動いてしまう失敗」を優先します。

---

## 14.2 データリーク

### パターン A：fit-time leakage（scaler / feature 選択が CV の外で fit）

**症状**：CV RMSE が異常に良い、test でも大差ない、と思って本番投入したら性能が落ちる。

**原因**：`StandardScaler` や特徴量選択を **CV の外で `fit_transform` してから** `cross_val_score` に投入している：

```python
# ❌ 悪い例
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)         # test 情報が混入
cross_val_score(Ridge(), X_scaled, y_train, cv=5)
```

`fit_transform` は全データから mean/std を計算するため、各 fold の validation 側の情報が train 側に洩れます。

**予防**：`Pipeline` に含める：

```python
# ✅ 正しい例（Ch5-7 の標準）
pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
cross_val_score(pipe, X_train, y_train, cv=5)   # fold ごとに scaler が fit
```

**検知**：CV スコアと **`Pipeline` 外で fit した scaler での CV スコアを比較**する監査テストを作り、両者の差が閾値以下（例えば 1% 以内）でないと fail にする。差が大きいときは Pipeline 外で fit している疑いが強い。

### パターン B：group-time leakage（同一群が train と test に分かれる）

**症状**：装置・ロット・患者などの群単位で予測すべきなのに、通常の `KFold` で分割し、同じ群のサンプルが両側に出る。

**原因**：`GroupKFold` を使うべきところで `KFold` を使用。

**予防**：Ch7 の CV 設計に従い、`group_key`（`inst_id` / `lot_id` / `subject_id` 等）を明示：

```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
cross_val_score(pipe, X_train, y_train, cv=gkf, groups=inst_id_train)
```

**検知**：provenance の `cv_scheme.group_key` フィールド必須化。null で保存された Skill は「group leakage 疑い」ラベルを付ける。

### パターン C：target-time leakage（未来情報が特徴量に混入）

**症状**：時系列で「同じ日の別センサ値」など、予測時点で入手不可能な値が特徴量に紛れる。

**予防**：Ch4 のデータ契約で `feature_availability_time <= prediction_time` を明示する。時系列は `TimeSeriesSplit` を使用。

**検知**：特徴量の生成タイムスタンプを provenance に含める。データセット定義段階で feature の入手時刻をメタとして保持する。

---

## 14.3 p-hacking と多重比較

### 症状

- 「特徴量選択 → CV → 良い組合せを採用」を繰り返して過学習
- 「複数モデル × 複数ハイパラ」で総当たりして偶然の best を採用
- p 値・R² を複数見て「有意」なものを事後選定

### 原因

- 探索空間の広さに応じて **偶然の良い結果** が生まれる確率が上がる
- CV スコアも「事後選定」の対象になれば信頼できない

### 予防

- **hold-out set を最終評価まで開封しない**（Ch13 の 3 分割・test は最後）
- **nested CV** を導入：外側 CV でハイパラ探索、内側 CV で最終評価
- 探索履歴を provenance に完全記録：`hyperparam_search.n_trials`, `search_space`, `selection_metric`
- 多重比較補正（Bonferroni / Holm / BH）を必要な文脈で適用

### 検知

- provenance に `n_trials >= 50` が記録されているのに hold-out 未使用 → 警告
- 同一データセットに対して過去 5 Skill バージョン以上が commit されている → 「過学習リスク」ラベル

> [!TIP]
> **Bayesian は p-hacking の代替ではない**：ベイズ因子や事後確率を「有意っぽい閾値」で切って複数比較を繰り返すと同じ罠にはまります。Bayesian でも hold-out と探索履歴の記録は必要です。

---

## 14.4 過学習と正則化不足

**vol-01 でも扱った古典的失敗**なので、統計/ML 特有の視点だけ再掲：

| 症状 | 診断 |
|---|---|
| 訓練スコア >> CV スコア | 明確な過学習 |
| CV スコアは良いが test で悪化 | データリーク or CV 設計不良 |
| ノイズ的特徴量が高重要度 | 特徴量選択の甘さ |

**予防**：Ch5-8 の設計（Pipeline、正則化、CV、feature importance の SHAP 等での安定性確認）。

**検知**：Skill 認定時に **learning curve**（訓練サイズ vs スコア）を出力し、右肩上がりで飽和していれば過学習リスク低。訓練スコアだけ天井、CV が伸び悩むなら過学習。

---

## 14.5 prior の暴走

### パターン A：情報過多（データを無視する prior）

**症状**：事後平均が prior 平均にほぼ張り付き、データが変わっても事後がほとんど動かない。

**原因**：狭すぎる prior（例：`Normal(0, 0.01)` を弱情報として付けたつもりが実質固定）。

**予防**：Ch9 の prior sensitivity テスト（main / weaker / stronger の 3 分岐）。weaker 版で事後が prior 平均から離れれば prior 支配が確認できる。

**検知**：**prior と posterior の KL divergence** を Skill 出力に含める。閾値以下は「prior 支配」の警告。

### パターン B：情報不足（データに丸乗りされる prior）

**症状**：少数群で `sigma_group` の事後が prior に強く依存する（第11章 §11.4、第13章 §13.5 の警告）。

**予防**：group 数が少ないときは strong prior + 事前選定した理由を provenance に。

**検知**：`hierarchical_structure.applicability_domain.inst_count < 5` のとき、`prior_specification.sigma_group.sensitivity_alternatives` が必須。null なら fail。

### パターン C：Prior predictive がドメイン外

**症状**：prior predictive check が非物理的な範囲（負の濃度、光速超え）を含む。

**予防**：Ch10 §10.3 の prior predictive check を必ず実施し、`prior_predictive_range` を provenance に記録。

**検知**：`prior_predictive_range` が `applicability_domain.y_range` と 20% 以上ずれていれば警告。

---

## 14.6 MCMC 未収束（動くけど正しくない）

**症状**：Skill が結果を出す → 意思決定に使う → 数週間後に「別 seed で走らせたら結果が違う」判明。

**原因**：`diagnostics_pass` を確認せずデプロイ、または閾値を緩めて通した（R-hat 1.05 とか、divergences 10 個くらい）。

**予防**：Ch10-12 の診断ゲート。特に第12章 §12.9 の 14 項目チェックリストで**機械的に**判定。

**検知**：

- provenance の `diagnostics_summary` を必須化し、`diagnostics_pass: true` が false なら Skill 実行時に例外を投げる（`assert diag["diagnostics_pass"]`）
- **seed ロバスト性テスト**：3 seed で走らせて `diagnostics_pass` が全て true でない Skill は認定不可

> [!IMPORTANT]
> **「一度合格したから安心」ではない**：同じ Skill でもデータサイズや値の分布が変わると再度未収束になります。**新しいデータで実行するたび** `check_diagnostics()` を呼び、fail なら**停止**します。エージェントに「保留の判断」を委ねません。

---

## 14.7 バックエンド差による結果ずれ

### 症状

「PyMC pytensor で通ったのに NumPyro に切り替えたら事後が違う」「開発マシンと本番で結果が違う」。

### 原因

- JIT の演算順序差
- 乱数実装差（Threefry vs PCG）
- FP32/FP64 の切替
- CPU/GPU/TPU の並列化差

### 予防（Ch10 §10.9 と Ch12 §12.7 の再掲）

- **認定は CPU + 1 バックエンド固定**
- `backend_reproducibility.tolerance` を **posterior sd 正規化**で設計
- `jax_enable_x64=True` を強制
- 参照バックエンドとの差分テストを CI に含める

### 検知

- `cross_backend_tested: true` が provenance にない Skill は不可
- CI で PyMC / NumPyro 両方で走らせ、tolerance 内かを assert

---

## 14.8 provenance チェイン切れ・合成データ誤運用

### パターン A：チェイン切れ

**症状**：Layer 2 の下流で異常な予測。原因を追うと、Layer 1 のモデルアーティファクトが差し替わっていたが upstream ハッシュが更新されていなかった。

**予防**：第13章 §13.6 の `verify_provenance_chain()` を **Skill 実行前の必須ゲート**にする。ハッシュ不一致で例外。

**検知**：CI で `assert layer_2.inputs.y_pred.sha256 == layer_1.output_artifacts.y_pred_calib.sha256`。

### パターン B：合成データを実データと誤認

**症状**：第13章 §13.5 の Layer 3（合成階層データで fit）を、そのまま**実装置の判定 Skill** として使う。装置差 `sigma_inst` は合成データの真値であって、実装置間差ではない。

**予防**：`data_source: synthetic` を provenance に必須で書き、これがある Skill は「実運用禁止」ラベルを自動付与。

**検知**：Skill 検索インタフェース側で `data_source == "synthetic"` の Skill を production クエリから除外。

---

## 14.9 失敗検知の運用パターン

### 3 層のゲート

| 層 | 内容 | タイミング |
|---|---|---|
| **静的** | provenance schema 検証（フィールド存在、hash 形式） | Skill commit 時 |
| **動的** | Skill 実行時に `check_diagnostics()`, `verify_provenance_chain()` 呼び出し | 毎回の実行 |
| **監査** | 定期的な CI で seed ロバスト性・バックエンド差テスト | 週次 or 月次 |

### 失敗発見時の復旧

1. **即時停止**：Skill を disable にし、下流の意思決定を止める
2. **原因の特定**：どのゲートが本来検知すべきだったかを分析
3. **ゲート追加**：抜けた検知を CI に追加
4. **Skill バージョンアップ**：修正版を新バージョンで公開
5. **provenance に事故記録**：`known_issues` セクションを追記、過去の実行結果に警告を伝播

### 失敗の型に応じた版数運用

| 失敗の種類 | バージョン扱い |
|---|---|
| リーク発覚 → データ分割修正 | Major up（`2.0.0`）：過去結果は無効 |
| Prior 選定変更 | Minor up（`1.1.0`）：過去結果は残す |
| MCMC target_accept 変更 | Patch（`1.0.1`）：sampler_config パッチ |
| バックエンド tolerance 緩和 | Patch（`1.0.1`） |
| provenance schema 拡張 | Minor up（`1.1.0`）：スキーマ後方互換維持 |

> [!TIP]
> **失敗を hide しない文化**：`known_issues` を書ける provenance schema を Ch4-13 の全 Skill に用意しておくと、事故が起きたときに「なかったこと」にできません。むしろ「早く見つけた Skill」を評価する運用に変えます。

---

## 14.10 章末ワーク

1. **リーク検知テストの実装**：`Pipeline` 外で fit した scaler で cross_val_score を回し、`Pipeline` 内 fit との差が閾値超過で fail する pytest を書く
2. **p-hacking シミュレーション**：ランダムに 20 個の特徴量から best 5 を選ぶ操作を 100 回繰り返し、hold-out RMSE の分布を可視化。**hold-out を触らない**運用の重要性を確認
3. **Prior 暴走テスト**：Ch10 の校正モデルで `beta ~ Normal(1, 0.01)` にして事後が prior に張り付くことを確認
4. **未収束 Skill の停止テスト**：`diagnostics_pass: false` の idata を作り、Skill 実行時に例外が飛ぶことを確認
5. **バックエンド差テスト**：同じ seed / 同じモデルを pytensor と numpyro で走らせ、事後平均を sd 正規化した差を出す。Ch12 の tolerance に収まるか
6. **チェイン切れテスト**：Layer 1 の `y_pred_calib` を微改変して Layer 2 を走らせ、`verify_provenance_chain()` が assert で止まることを確認
7. **合成→実運用ガード**：Skill 実行前に `provenance["data_source"] != "synthetic"` を assert する wrapper を書く

---

## 14.11 本章のまとめ

- 統計/ML の失敗は **7 種類**：データリーク・p-hacking・過学習・prior 暴走・MCMC 未収束・バックエンド差・チェイン切れ / 合成データ誤運用
- **「動いた」≠「正しい」**：CV スコアや $\hat{R}$ が良く見える失敗が最も危険
- Ch4-13 の設計は各失敗を**予防的に**塞いでいるが、**運用ゲート**（静的・動的・監査の 3 層）で検知を強制しないと素通りする
- **`known_issues` を書ける provenance schema** で失敗を隠さない
- 失敗のタイプで **Skill 版数を差別化**（Major/Minor/Patch）

vol-02 の**壊れ方**の整理はここで完了です。第15章では、これらを組織で運用する方法と、vol-03 以降への道しるべを示します。

---

## 参考資料

### 本書内の該当章

- 第4章：データ契約と anti-leakage（本章 §14.2 の予防）
- 第5-7章：Pipeline・CV 設計（§14.2 パターン A/B）
- 第8章：SHAP・permutation importance の安定性（§14.4）
- 第9-10章：prior 設計と prior predictive check（§14.5）
- 第10章 §10.4, 第12章 §12.2：診断（§14.6）
- 第10章 §10.9, 第12章 §12.7：バックエンド設計（§14.7）
- 第13章 §13.6：provenance チェイン検証（§14.8）
- 第15章：組織で運用するゲート設計、監査文化

### 外部参考

<a id="ref-14-1">[14-1]</a> Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM TKDD*, 6(4). — データリークの体系化
<a id="ref-14-2">[14-2]</a> Ioannidis, J. P. A. (2005). Why most published research findings are false. *PLOS Medicine*, 2(8), e124. — p-hacking と多重比較の危険性
<a id="ref-14-3">[14-3]</a> Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-Normalization, Folding, and Localization: An Improved $\hat{R}$ for Assessing Convergence of MCMC. *Bayesian Analysis*, 16(2), 667–718. — $\hat{R}$ による未収束検知（§14.6）
<a id="ref-14-4">[14-4]</a> Gabry, J., Simpson, D., Vehtari, A., Betancourt, M., & Gelman, A. (2019). Visualization in Bayesian workflow. *JRSS A*, 182(2), 389–402. — prior predictive check と PPC による誤モデル検知（§14.5）
<a id="ref-14-5">[14-5]</a> Sculley, D., Holt, G., Golovin, D. et al. (2015). Hidden technical debt in machine learning systems. *NeurIPS 2015*. — ML システムの運用負債（§14.9）
<a id="ref-14-6">[14-6]</a> Lakens, D. (2016). The 20% Statistician: Why the 2016 ASA statement on p-values is important. — p 値の運用注意（§14.3）
