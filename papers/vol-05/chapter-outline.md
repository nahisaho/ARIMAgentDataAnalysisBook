# 「AI エージェント時代のベイズ最適化・逐次実験計画入門 — ARIM データで動く Agentic Skill」章構成（v0.1 ドラフト）

> vol-05 は vol-01「AI エージェント時代のデータ分析入門」、vol-02「AI エージェント時代の統計・機械学習分析入門」、vol-03「AI エージェント時代の深層学習分析入門」、vol-04「AI エージェント時代の因果推論・実験計画入門」の続編。
> vol-04 が **「観測データから因果を推定し、DoE で介入計画を設計する」** ラインを整えたのに対し、
> vol-05 は **「次にどの実験を実行するか、どこで止めるか」** を Skill 化する——
> ARIM データポータルの実験データ（少データ・装置固有・実験コスト高）を主戦場に、
> **「Gaussian Process で surrogate を作り、acquisition function で次の候補を提案し、Human-in-the-loop で実験実行を承認する」**
> ループをエージェントの Skill として設計する。

## 版履歴

### v0.2（rubber-duck review 反映）
- **Blocking-1（Ch13 capstone 過密）修正**：Ch13 の 4 Phase × 10 iteration を **Ch13a（基本 BO ループ：single-objective + 3 iteration）と Ch13b（発展：multi-objective + 階層 + iteration 10 まで、そのうち後半 5 iteration は付録 D にスクリプト化）** に分割。各 iteration の Human 承認は Ch13a で丁寧に扱い、Ch13b は差分のみ
- **Blocking-2（`hallucinated_recommendation_detection` 未操作化）修正**：Ch6 に **「外挿検知の operational 定義」節**を新設（GP の予測分散閾値、入力から近傍学習点までの Mahalanobis 距離、length scale 越え検知の合成基準）。Ch7 で BNN / RF surrogate の場合の外挿検知に拡張
- **Blocking-3（provenance field 使用マッピング欠落）修正**：各 provenance field を **章と Skill の operational 使用箇所**に紐付け。`budget_remaining` → Ch13 stop condition、`batch_size` → Ch11、`sequential_seed_provenance` → Ch4 + Ch14、`pending_experiments` → Ch11、`stop_condition` → Ch13 + Ch14
- **Non-Blocking 修正**：
  - **vol-04 Ch12 (Bayesian D/A-optimality) と vol-05 Ch6 (KG/MES) 重複**：Ch6 冒頭で **「情報利得ベース acquisition の位置付け——vol-04 Ch12 は one-shot Bayesian DoE、vol-05 Ch6 は逐次的な次候補選択」** と boundary を明示
  - **Ch5 GP 基礎章の過負荷**：Ch5 を **「GP の骨格 + kernel + ARD + 事前分布 + GPyTorch」** に絞り、**hierarchical GP は Ch10 に完全に移譲**
  - **Ch6 acquisition survey に Thompson sampling / PES 追加**：EI / UCB / PI / KG / MES に加えて **Thompson sampling（batch 拡張への橋渡し）と Predictive Entropy Search (PES)** を追加
  - **Ch8-9-11 の組合せ**：Ch13a/13b capstone で **batch × multi-objective × constrained の統合ケース**を明示的に扱う
  - **環境ライブラリ整理**：`entmoot` は用途不明のため **標準環境から削除**、`modAL`（Ch12 active learning で使用）を **標準環境に追加**
  - **Ch14 failure taxonomy 追加**：stale/pending observations の扱い、duplicate candidates（同一点の重複提案）、unit/bounds drift（search space の単位ずれ）、budget misaccounting（実験コストの合計誤集計）の 4 項を追加
- **Suggestions 反映**：
  - **Suggestion-1**：Ch11 batch BO 節に **BO vs BOCS (Bayesian Optimization of Combinatorial Structures)** の使い分け表を追加（組合せ空間で BO 直接適用の限界）
  - **Suggestion-2**：Ch3 に **BoTorch と Ax の gap** についての実務ノート（BoTorch はライブラリ、Ax は実験管理層；本書は主に BoTorch を扱い Ax は Ch11 で言及）
  - **Suggestion-3**：Ch5 に **「材料 BO における推奨 kernel / prior 既定値表」** を追加（連続変数 Matérn 5/2、離散/カテゴリ Hamming、装置差 hierarchical prior の既定 σ）

### v0.1（初版企画ドラフト）
- vol-03 章構成 v0.2 で「vol-04 候補」として棚上げされていた **ベイズ最適化** と **逐次実験計画** を本巻に分離（vol-04 では古典 DoE、vol-05 では逐次適応型 DoE）
- **分離の理由**：古典 DoE（vol-04 Ch10-11）は計画を一括生成する。BO / active learning は**観測結果を見ながら次の候補を選ぶ**ため、Agentic Skill としてのループ設計が根本的に異なる。特に「エージェントが提案 → Human 承認 → 実験 → データ取り込み → 次の提案」というループ全体を Skill 化する必要がある
- 6 か月・250〜300 ページ規模、本編 15 章 + 付録 3 の骨格を継承
- **エージェントに何を許すか**：surrogate 学習・acquisition 計算は自律、**候補の絞り込み提案は準自律**、**実験実行判断は必ず Human 承認**（材料実験は消耗と時間コストが高いため）

## 前提

- **対象読者**: ARIM データポータル会員のデータ分析者（材料・ナノテク研究者）。Python / Jupyter 経験あり。**vol-01 + vol-02 完読を推奨**、**vol-04 完読を強く推奨**（DoE の骨格と因果推論を前提とする節が複数）、vol-03 は必須ではない
- **vol-01 / vol-02 / vol-03 / vol-04 との関係**:
  - **vol-01 完読推奨**: Skill 設計 6 要素、データ契約、Human-in-the-loop、provenance、6 データ型
  - **vol-02 完読推奨**: 第10-12章 PyMC / MCMC、**第11章の階層モデル**（マルチタスク BO で階層事前分布を使う）
  - **vol-04 強く推奨**: 第10-12章の DoE 章、第4章の介入承認ゲート、第13章の capstone（因果 → DoE のフロー）
  - **vol-03 推奨**（必須ではない）: 第11章の Foundation Model 特徴（BO の入力表現に使う発展）
  - vol-05 の provenance は vol-04 拡張に、**BO × 逐次 × Agentic 特有フィールド**を追加：`surrogate_model_family`（GP / Random Forest / BNN）, `kernel_spec`, `acquisition_function`（EI / UCB / PI / KG / MES）, `iteration_index`, `batch_size`, `search_space_bounds`, `constraints_declared`, `pending_experiments`, `budget_remaining`, `stop_condition`（budget / convergence / regret threshold）, **`experiment_launch_authorization`**, **`hallucinated_recommendation_detection`**（surrogate の外挿誤用検知）, **`sequential_seed_provenance`**（seed の上書き禁止契約）
- **最終ゴール（合格ライン）**:
  - **Pillar 1**: **単目的 BO Skill**（GP surrogate + EI/UCB）を 1 つ以上作れる。**「エージェントが次の 1 点を提案 → Human 承認 → データ取込 → 次の提案」ループが provenance に完全記録される**
  - **Pillar 2**: **多目的 BO / 多制約 BO Skill**（qEHVI, cEI, safe BO のいずれか）を 1 つ以上作れる。**Pareto front の外挿範囲がエージェントに管理されている**
  - **Advanced Capstone**: 上記 2 本を統合した **「vol-04 の因果構造を利用して search space を絞り、階層 GP でマルチ装置に対応し、Human 承認ループで BO を回す」複合 Skill** を、ARIM 風合成実験データで完成させる
- **分量目安**: 実践書（250〜300 ページ、vol-04 と同等）
- **期限**: 6 か月
- **ハンズオン標準環境**（vol-04 に追加）:
  - vol-04 標準 + `botorch`, `gpytorch`（BO のデファクト）
  - `emukit`（active learning / experimental design）
  - `scikit-optimize`（skopt、軽量 BO）
  - `ax-platform`（Facebook Ax、BO の実験管理）
  - `hebo`（BO の代替 surrogate、HEBO ラッパ）
  - `modAL`（active learning、Ch12 で使用）
  - `numpyro` / `pymc`（Bayesian surrogate、vol-02 継承）
  - **CI 環境は CPU で完結**を必須要件（vol-03/04 と同じ）
- **データセット方針**:
  - **主軸**: **ARIM 風合成 BO データ**（真の目的関数を持つ、装置差 × 実験コスト × 制約を再現した合成データ）を本書でリポジトリ配布（`data/synthetic-bo/` 想定）
    - 単目的：物性最大化（例：合成条件 → 熱伝導率）
    - 多目的：Pareto（例：合成条件 → (硬度, 密度)）
    - 制約付き：安全条件（例：温度上限、圧力上限）
  - **対比・スケール確認**: 公開データセット
    - **Olympus benchmark suite**（合成材料実験の BO ベンチマーク）
    - **Bayesian benchmark functions**（Branin, Hartmann, Ackley 等、教科書標準）
    - **Materials Project の DFT データを "測定コスト付き" にラップ**した演習
    - **ARIM 実データ**（読者現場で持ち込む、匿名化ガイドは付録C）
- **参照**:
  - BoTorch: https://botorch.org/
  - GPyTorch: https://gpytorch.ai/
  - Ax: https://ax.dev/
  - Emukit: https://emukit.github.io/
  - scikit-optimize: https://scikit-optimize.github.io/
  - HEBO (Huawei): https://github.com/huawei-noah/HEBO
  - Olympus: https://aspuru-guzik-group.github.io/olympus/
  - Frazier "A Tutorial on Bayesian Optimization" (arXiv:1807.02811)
  - Shahriari et al. "Taking the Human Out of the Loop: A Review of Bayesian Optimization" (Proc. IEEE 2016) — 本書は逆に **Human を戻す** 立場
  - vol-01 / vol-02 / vol-03 / vol-04 リポジトリ（本書の前提）

## 本書で扱わないこと（明示）

vol-05 は **「エージェントが逐次実験計画を提案し、Human が承認する Skill」** に scope を絞る。

| トピック | 扱わない理由 | 想定巻 |
|---|---|---|
| **生成モデル・材料逆設計（VAE / GAN / Diffusion）** | 「潜在空間から材料を生成する」は探索とは別軸。BO の search space は既知の設計変数空間に限定 | vol-06 |
| **深層 BO の全面展開（DKL / neural surrogate）** | GP + BoTorch を中心に扱い、深層 surrogate は 1 節（第7章）で概説のみ | — |
| **強化学習ベースの実験自律実行** | 「エージェントが実験装置を直接叩く」は本書のスコープ外。実験実行は Human 承認必須 | 別書候補 |
| **多階層プロセス最適化（プラント全体設計）** | 単一実験セル〜数装置レベルに絞る | 別書候補 |
| **GP の数学的完全展開（Matérn kernel の理論等）** | 既存教科書 (Rasmussen & Williams) に譲る。本書は Skill 化と provenance に集中 | — |

## 6 データ型と BO × 逐次実験計画 Skill の対応

| データ型 | vol-04 で扱えたこと | vol-05 で扱う Agentic Skill |
|---|---|---|
| スペクトル型 | 前処理条件の因果効果推定、DoE | **合成条件 → スペクトル特徴（ピーク位置・強度）の目的最適化 Skill**、**マルチ装置階層 GP** |
| クロマトグラム・時系列型 | 反応条件の因果推定、DiD | **反応条件 → 収率最大化 Skill**、**時間進行を制約に組み込む逐次計画** |
| 画像・顕微鏡型 | 前処理条件 → 微細構造の因果推定 | **粒径・欠陥密度の目的最適化 Skill**、**画像特徴を BO の応答変数として使う** |
| 回折・散乱パターン型 | 合成条件 → 結晶相の因果推定 | **結晶性最大化・特定相の生成率最大化 Skill** |
| 表形式・プロセス条件型 | 因果推論 + DoE | **BO の主戦場**。単目的 / 多目的 / 制約付き / マルチ装置 の全パターン |
| マルチモーダル統合型 | 多モーダル confounder 調整 | **多モーダル観測を surrogate 入力にする発展**、**制約が多モーダル** |

**共通する Agentic 観点**：Skill は「次の候補を提案する」だけでなく、**「surrogate の外挿誤用がないか、acquisition が真の情報利得を捉えているか、実験予算に対して合理的か」をエージェントが判定し、Human に候補と根拠を提示する契約**。実験実行の GO/NO-GO は常に Human。

## 章構成（案）

**分量目安の合計**：本編 250〜300 ページ + 付録 40〜50 ページ

### 第0章 vol-01〜04 の最小復習（目安 15 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第0章 | vol-01〜04 の最小復習 | Skill / MCP / Human-in-the-loop / provenance / 6 データ型 / PyMC / 深層 Agentic 権限 / **vol-04 の DoE と因果推論** を 15 ページに圧縮 | vol-05 の前提 |

### 第I部　なぜ「エージェント × 逐次実験計画」なのか（目安 30〜35 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第1章 | 予測 → 因果 → 逐次 のラダーで vol-01〜04 に何が足りないか | vol-04 の DoE は計画を一括生成するが、材料実験では **観測後に次を決める** のが実務。BO の位置づけ、Agentic 特有の Human-in-the-loop 必要性、扱わないことの明示 | 本書の到達点、逐次ループの位置付け |
| 第2章 | ARIM データで BO を回すときの Agentic 特有の課題 | **少データ現場での GP の暴走**、**装置差を surrogate の入力に含めるか階層 GP にするか**、**実験コストと BO サイクル時間**、**エージェントが acquisition を "勝手に" 最大化する危険**、演習用データ紹介 | 自分のデータの BO 適用性判定、演習データ入手 |
| 第3章 | BO / active learning ライブラリ地図 — Agentic 使い分け | **BoTorch / GPyTorch / Ax / Emukit / scikit-optimize / HEBO / modAL** の位置づけ、**BoTorch と Ax の gap 実務ノート**（BoTorch はライブラリ、Ax は実験管理層；本書は主に BoTorch を扱い、Ax は Ch11 の batch 管理で言及）、**エージェントがどこまで自律的に叩けるか**、vol-04 の DoE ライブラリとの連携 | ライブラリ使い分けマップ、BoTorch-Ax gap ノート |

### 第II部　単目的 BO の Skill 化（目安 55〜65 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第4章 | BO × Agentic Skill の設計原則 | vol-02/03/04 第4章の BO × Agentic 拡張。**「次の候補を提案する」の provenance**、iteration index、surrogate 再学習ポリシー、acquisition の選択根拠、**新設節：実験実行承認ゲート**（`experiment_launch_authorization`）——**エージェントは候補提示までで、実験実行は Human 承認**。stop condition の設計、外挿検知 (`hallucinated_recommendation_detection` の宣言；operational 定義は Ch6)。**Provenance field 使用マップ**（`sequential_seed_provenance` は本章と Ch14、`batch_size` は Ch11、`pending_experiments` は Ch11、`budget_remaining` と `stop_condition` は Ch13 で operational 化） | BO × Agentic Skill 仕様書テンプレート、provenance field 使用マップ |
| 第5章 | Gaussian Process surrogate の Skill 化 | GP の骨格、kernel 選択（RBF / Matérn 3/2 / 5/2 / product / additive）、length scale の事前分布、**ARD**、GPyTorch/BoTorch 実装、**Skill としての kernel 選択判断表**、**材料 BO における推奨 kernel / prior 既定値表**（連続変数 Matérn 5/2、離散/カテゴリ Hamming、装置差 hierarchical prior の既定 σ）。**hierarchical GP は Ch10 に委譲**（本章では言及のみ） | GP surrogate Skill、既定値表 |
| 第6章 | Acquisition function の Skill 化と外挿検知の operational 定義 | Expected Improvement / Upper Confidence Bound / Probability of Improvement / Knowledge Gradient / Max-Entropy Search / **Thompson Sampling** / **Predictive Entropy Search (PES)** の**判断表**、trade-off（exploration vs exploitation）、**エージェントが acquisition を "動的切替する" 場合の妥当性判定**、**新設節：外挿検知の operational 定義**（`hallucinated_recommendation_detection` を、GP の予測分散閾値・入力から近傍学習点までの Mahalanobis 距離・length scale 越え検知の合成基準として実装）。**vol-04 Ch12 (Bayesian D/A-optimality) との boundary**：Ch12 は one-shot Bayesian DoE、本章 (KG/MES) は逐次的な次候補選択 | Acquisition Skill、判断表、外挿検知 operational 節 |
| 第7章 | Surrogate の代替と拡張 — Random Forest / BNN / Deep Kernel | GP の限界（高次元・離散混在・非定常）、Random Forest surrogate、BNN surrogate（vol-03 第9章の応用）、Deep Kernel Learning の概観、**Skill としての surrogate 選択判断表**、**BNN / RF 使用時の外挿検知**（Ch6 の合成基準を BNN posterior / RF disagreement に拡張） | 代替 surrogate Skill、選択判断表 |

### 第III部　多目的・制約付き・階層 BO（目安 55〜65 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第8章 | 多目的 BO を Skill 化する — qEHVI と Pareto front | 多目的 BO の骨格、Pareto front、qEHVI（expected hypervolume improvement）、**Skill としての目的関数重み付けの明示化**（"エージェントが勝手にスカラー化しない" 契約）、ARIM 材料開発の (硬度, 密度) トレードオフ例 | 多目的 BO Skill |
| 第9章 | 制約付き BO と safe BO — 安全条件を Skill 契約に組み込む | 実行可能領域制約（cEI、feasibility model）、**safe BO**（安全条件を破らない保証つき探索）、**"温度上限を超える提案は Skill が絶対に返さない" 契約**、Skill の失敗モードとしての制約破り | 制約付き BO Skill、safe BO Skill |
| 第10章 | マルチ装置・マルチタスク BO — 階層 GP を Skill 化する | Multi-task GP、**vol-02 第11章の階層モデル**を surrogate に持ち込む、**装置ごとに異なる目的関数を階層事前分布で共有**（Ch5 から本章に移譲された hierarchical GP の完全展開）、ARIM 施設内の複数装置での並行探索 | 階層 BO Skill |
| 第11章 | Batch BO と並列実験の Skill 化 | q-EI、q-EHVI、Local penalization、**"エージェントが同時に n 個提案する" ときの多様性保証**、実験装置の並列度に合わせた `batch_size` の provenance、`pending_experiments` の provenance、**BO vs BOCS（組合せ空間で BO 直接適用の限界）判断表**、**Ax による experiment tracking**（BoTorch の上に組む場合の Ax 呼び出し例） | Batch BO Skill、BO/BOCS 判断表 |

### 第IV部　総合ハンズオンと運用（目安 45〜55 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第12章 | Active Learning と逐次実験計画 — BO 以外の逐次戦略 | 分類 / 回帰の active learning（uncertainty sampling / query by committee / expected model change）、**BO と active learning の判断分岐**（"最適化" vs "学習"）、Emukit / modAL の Skill 化、**エージェントが目的を "最適化" と誤解しない契約** | Active Learning Skill、BO との使い分け表 |
| 第13章 | 総合ハンズオン（Advanced Capstone、2 節構成）— 因果構造で search space を絞り、階層 GP でマルチ装置に対応し、Human 承認ループで BO を回す | **13a：基本 BO ループ**（Phase 1: vol-04 の因果 DAG から effective search space を導出 → Phase 2: 単一装置 GP surrogate → Phase 3: single-objective BO で 3 iteration まで丁寧に回す。各 iteration で Human 承認、`budget_remaining` と `stop_condition` を operational 化）。**13b：発展**（階層 GP でマルチ装置対応、multi-objective + constrained BO で 10 iteration まで拡張、後半 5 iteration のスクリプトは付録 D）。**Batch × Multi-objective × Constrained の統合ケース**を明示的に扱う | 複合 BO Skill (capstone) |
| 第14章 | BO × Agentic 特有の失敗パターンと監査 | **セクション 1（BO 一般の失敗）**：GP の外挿誤用、length scale の暴走、acquisition の局所解、Pareto front の疑似収束、safe BO の制約違反、**stale/pending observations の扱い漏れ**、**duplicate candidates（同一点の重複提案）**、**unit/bounds drift（search space の単位ずれ）**、**budget misaccounting（実験コストの合計誤集計）**。**セクション 2（Agentic 特有）**：**エージェントが acquisition を "動的" に勝手に切り替える、外挿領域を "自信あり" と報告、Human 承認なしに次の候補を実行、iteration seed を上書き、pending experiments を勝手にキャンセル、stop condition を勝手に緩める、多目的スカラー化重みを勝手に変更** | BO × Agentic 失敗チェックリスト |
| 第15章 | 組織展開と終章 — 実験リソースの割当・BO 運用・次巻への道しるべ | **BO サイクルの組織的運用**（Skill 提案 → チーム承認 → 装置予約 → 実行 → データ取込）、**vol-05 の到達点**、**vol-06（生成モデル・逆設計）への道しるべ**、ARIM 施設としての BO 運用（複数研究者間での surrogate 共有、装置予約統合） | 組織運用方針、次巻ロードマップ |

### 付録（目安 40〜50 ページ）

| 付録 | タイトル | 内容 |
|---|---|---|
| 付録A | BO × Agentic Skill テンプレート集 | 単目的 BO Skill、多目的 BO Skill、制約付き BO Skill、safe BO Skill、階層 BO Skill、batch BO Skill、active learning Skill の雛形。**vol-04 の provenance を BO × 逐次 × Agentic 向けに拡張**（`surrogate_model_family`, `kernel_spec`, `acquisition_function`, `iteration_index`, `search_space_bounds`, `constraints_declared`, `pending_experiments`, `stop_condition`, `experiment_launch_authorization`, `hallucinated_recommendation_detection` 等）。**Provenance field と章の使用マップ**を巻頭に掲載 |
| 付録B | BoTorch / GPyTorch / Ax / Emukit / modAL チートシートと **MCP サーバ実装（BO 版）** | よく使う API、GP fit → acquisition → 候補提案の最小コード、Ax による experiment tracking、modAL の active learning、**MCP Python SDK で BO ループを Skill 化するミニマル実装例**、**"候補提示までは自律、実行判断は Human" のゲートを MCP レベルで組む** |
| 付録C | BO 演習データ候補と benchmark 関数 | **ARIM 風合成 BO データの生成スクリプト**（真の目的関数と装置差を持つ）、Olympus benchmark、Bayesian benchmark functions（Branin, Hartmann, Ackley）、**Materials Project を "測定コスト付き" にラップ**する手順、ARIM 実データを持ち込む際の匿名化ガイドと search space 定義のレビュー手順 |
| 付録D | Capstone 発展スクリプト集（v0.2 新設） | Ch13b の後半 5 iteration のスクリプト、batch × multi-objective × constrained の統合ケースのフル実装、`budget_remaining` / `stop_condition` の operational コード、hierarchical GP でマルチ装置対応のフル実装 |

## 責務分離マップ（重複防止）

### vol-05 内での分離

| 論点 | 予防・設計側 | 実行・事例側 |
|---|---|---|
| Surrogate の選択 | 第5章 (GP)・第7章 (代替) | 第13章 capstone |
| Acquisition の選択 | 第6章 | 第8章 (多目的)・第9章 (制約)・第13章 |
| 制約管理 | 第9章 | 第14章 (制約破りの失敗) |
| 階層構造 | 第10章 | 第13章 |
| 並列実験 | 第11章 | 第13章 |
| Agentic 権限 | 第4章 (`experiment_launch_authorization`) | 第14章・付録B (MCP 実装) |
| Active learning vs BO | 第12章 (判断分岐) | 第13章 (BO 側で活用) |

### vol-01〜04 との分離

| 論点 | 既刊側 | vol-05 側 |
|---|---|---|
| 予測 Skill | vol-01/02/03 | 目的関数の proxy として第5章で活用 |
| 統計モデル | vol-02 第10-12章 (PyMC) | Bayesian surrogate として第5-7章で活用、階層 GP として第10章 |
| 深層特徴 | vol-03 第7章 | BO の入力表現として発展的に第7章末で言及 |
| 階層モデル | vol-02 第11章 | Multi-task GP の kernel prior として第10章 |
| 因果推論 | vol-04 全体 | Search space の絞り込みに第13章 capstone で活用 |
| DoE | vol-04 第10-12章 | 初期実験計画として BO 開始前に活用、以降は逐次で置き換え |
| Human-in-the-loop | vol-01 第6章 / vol-04 第4章 | **実験実行の Human 承認**（`experiment_launch_authorization`、第4章）、**iteration ごとの候補承認**（第13章 capstone） |
| Agentic 権限 | vol-03/04 第4章 | **BO ループの 3 段階権限**：surrogate 学習・acquisition 計算は自律 / 候補提案は準自律 / **実験実行判断は必ず Human 承認** |
| 失敗パターン | 各巻 第14章 | BO 一般 + Agentic 特有（第14章 2 セクション構成） |

## 各ハンズオン章の共通構成

vol-03/04 と同形式。**第5章で丁寧に説明し、以降は差分中心**。「エージェント役割」節を各章に必ず含める。

- この章で作る Skill の概要
- **エージェント役割**（Skill は何を計算し、何を Human に投げるか、**実験実行は誰が判断するか**）
- 入力仕様 / 出力仕様 / 制約条件（vol-04 拡張 + vol-05 BO × 逐次 × Agentic 拡張に準拠）
- **BO 特有の評価基準**（regret / hypervolume / calibration / external validity of surrogate / acquisition informativeness）
- 実行例 / 失敗例（**外挿誤用と Agentic 失敗を含む**）/ 改善版
- 他ライブラリ（BoTorch ↔ Ax ↔ Emukit）への転用方法

## 特に注意する重点管理項目

| 項目 | 予防・設計 | 事例・検証 |
|---|---|---|
| **エージェントが acquisition を勝手に切り替える** | 第4章 `acquisition_function` の pin | 第14章 |
| **エージェントが surrogate の外挿を "自信あり" と報告** | 第4章 `hallucinated_recommendation_detection` | 第14章 |
| **safe BO の制約を破る候補提案** | 第9章 制約破り絶対禁止契約 | 第14章 |
| **iteration seed を上書き** | 第4章 `sequential_seed_provenance` | 第14章 |
| **Pareto front の疑似収束を "収束" と誤報告** | 第8章 hypervolume 判定 | 第14章 |
| **多目的スカラー化重みを勝手に変更** | 第8章 重み provenance | 第14章 |
| **Human 承認なしの実験実行** | 第4章 `experiment_launch_authorization` | 第14章・付録B (MCP 実装) |
