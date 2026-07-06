# AI エージェント時代のベイズ最適化・逐次実験計画入門 — ARIM データで動く Agentic Skill

vol-01「AI エージェント時代のデータ分析入門」、vol-02「AI エージェント時代の統計・機械学習分析入門」、vol-03「AI エージェント時代の深層学習分析入門」、vol-04「AI エージェント時代の因果推論・実験計画入門」の続編。vol-01〜03 が **「観測データから予測を作る」**、vol-04 が **「観測データから "なぜ" と "もし" を主張し、一括で介入計画を組む」** ラインを整えたのに対し、vol-05 は **「次にどの実験を実行するか、どこで止めるか」** を Skill 化する実践書です。ARIM データポータルの実験データ（少データ・装置固有・実験コスト高）を主戦場に、**「Gaussian Process で surrogate を作り、acquisition function で次の候補を提案し、Human-in-the-loop で実験実行を承認する」** ループを Skill として設計し、**エージェントは候補提示までで、実験実行判断は必ず Human 承認** という 3 段階権限を provenance に完全記録するまでを扱います。

- **対象読者**: ARIM データポータル会員のデータ分析者（材料・ナノテク研究者。Python/Jupyter 経験あり）。**vol-01 + vol-02 完読を推奨、vol-04 完読を強く推奨**（DoE の骨格と因果推論を前提とする節が複数）、vol-03 は必須ではないが Ch8 で BNN surrogate に接続する節あり
- **最終ゴール（合格ライン）**:
  - **Pillar 1**: **単目的 BO Skill**（GP surrogate + EI/UCB/PI/KG/MES/Thompson Sampling/PES のいずれか）を 1 つ以上作れる。**「エージェントが次の 1 点を提案 → Human 承認 → データ取込 → 次の提案」ループが provenance に完全記録**され、**iteration seed の上書き禁止契約**（`sequential_seed_provenance`）と**外挿検知**（`hallucinated_recommendation_detection`：GP 予測分散閾値 × Mahalanobis 距離 × length scale 越え検知の合成基準）が operational に組まれている
  - **Pillar 2**: **多目的 BO / 制約付き BO / safe BO Skill**（qEHVI, cEI, safe BO のいずれか）を 1 つ以上作れる。**Pareto front の外挿範囲がエージェントに管理され、"温度上限を超える提案は Skill が絶対に返さない" 契約**が制約 gate に組み込まれている
  - **Advanced Capstone**: 上記 2 本を統合した **「vol-04 の因果 DAG を利用して search space を絞り、階層 GP でマルチ装置に対応し、Human 承認ループで batch × multi-objective × constrained BO を回す」複合 Skill** を、ARIM 風合成実験データで完成させる
- **標準環境**（vol-04 に追加）: `botorch`, `gpytorch`（BO のデファクト）, `emukit`（active learning）, `scikit-optimize`, `ax-platform`（実験管理層）, `hebo`, `modAL`（Ch13 active learning）, `numpyro`/`pymc`（vol-02 継承）。**CI は CPU で完結**を必須要件

> [!NOTE]
> 生成モデル・逆設計（generative inverse design）、強化学習ベースの実験計画、LLM ファインチューニングはスコープ外。**BOCS**（Bayesian Optimization of Combinatorial Structures）は Ch12 で BO との使い分け表として言及するのみで、実装は扱いません。組合せ空間で BO 直接適用の限界に踏み込むケースは vol-06 に持ち越します。

> [!TIP]
> データセット方針は **ARIM 風合成 BO データ**（真の目的関数と装置差を持つ）を主軸に、Olympus benchmark suite / Bayesian benchmark functions（Branin, Hartmann, Ackley）/ Materials Project の "測定コスト付き" ラップを対比・スケール確認に位置付けます。付録 C（未執筆）に BO 演習データ生成スクリプトと ARIM 匿名化・search space 定義レビュー手順を収録予定です。

> [!IMPORTANT]
> vol-05 の provenance は vol-04 拡張に加え、**BO × 逐次 × Agentic 特有フィールド** を追加します：`surrogate_model_family`（`single_task_gp` / `fixed_noise_gp` / `heteroskedastic_gp` / `multi_task_gp` / `hierarchical_gp` / `random_forest` / `bayesian_neural_network` / `deep_kernel_learning` の canonical enum、Ch5 §5.2.X）、`kernel_spec`（RBF / Matérn 3/2・5/2 / product / additive / Hamming）、`acquisition_spec.name`（単目的 `qLogEI` / `qLogNEI` / `qUCB` / `qPI` / `qKG` / `qMES` / `qTS` / `qPES`、多目的 `qLogEHVI` / `qLogNEHVI`、スカラー化 `qParEGO` / `qNParEGO` の canonical enum）、`iteration_index`、`batch_size`、`search_space_bounds`、`constraints_declared`、`pending_experiments`、`budget_remaining`、`stop_condition`（budget / convergence / regret threshold）、**`experiment_launch_authorization`**（vol-04 L3 `parent_authorization_id` に chain、Ch5 §5.3）、**`hallucinated_recommendation_detection`**（Ch7 で operational 化）、**`sequential_seed_provenance`**（seed の上書き禁止契約、Ch5 + Ch15）、`tensor_encoding_contract`（Group A、Ch5）、`event_hash` / `previous_event_hash`（RFC 8785 JCS canonical serialization、Ch10 §10.5 anchor `event_canonical_serialization`）、`hard_constraints_ceramic_v0_2`（Ch10 §10.6）、`batch_diversity_policy_v0_2`（Ch12）、`dag_revision_policy` + `experiment_reservation_events`（Ch14 §14.4.0）、`active_learning_acquisition_spec.name`（Ch13 §13.2.1 canonical enum）。全体は **21 fields（Core 18 + 拡張 3：`bo_library_stack_metadata` / `sequential_seed_provenance` / `batch_size`）**、canonical スキーマは **Ch5 §5.2** に一次リファレンスとしてまとめています。

## 目次

> **執筆進捗**: 本編 **15 / 16 章 完了**（Ch16 未執筆）、付録 **0 / 4 未執筆**

### 第0部　前提の再確認

| 章 | タイトル | 状態 |
|---|---|---|
| 第1章 | [vol-01 / vol-02 / vol-03 / vol-04 の最小復習](chapter-01.md) | ✅ 執筆済 |

### 第I部　なぜ「エージェント × 逐次実験計画」なのか

| 章 | タイトル | 状態 |
|---|---|---|
| 第2章 | [vol-01〜04 の Skill に何が足りないのか — 予測から "次にどこを打つか" へ](chapter-02.md) | ✅ 執筆済 |
| 第3章 | [ARIM データで BO を回すときの Agentic 特有の課題](chapter-03.md) | ✅ 執筆済 |
| 第4章 | [BO / active learning ライブラリ地図 — Agentic 使い分け](chapter-04.md) | ✅ 執筆済 |

### 第II部　単目的 BO の Skill 化

| 章 | タイトル | 状態 |
|---|---|---|
| 第5章 | [BO × Agentic Skill の設計原則](chapter-05.md) | ✅ 執筆済 |
| 第6章 | [Gaussian Process surrogate の Skill 化](chapter-06.md) | ✅ 執筆済 |
| 第7章 | [Acquisition function の Skill 化と外挿検知の operational 定義](chapter-07.md) | ✅ 執筆済 |
| 第8章 | [Surrogate の代替と拡張 — Random Forest / BNN / Deep Kernel](chapter-08.md) | ✅ 執筆済 |

### 第III部　多目的・制約付き・階層 BO

| 章 | タイトル | 状態 |
|---|---|---|
| 第9章 | [多目的 BO を Skill 化する — qEHVI と Pareto front](chapter-09.md) | ✅ 執筆済 |
| 第10章 | [制約付き BO と safe BO — 安全条件を Skill 契約に組み込む](chapter-10.md) | ✅ 執筆済 |
| 第11章 | [マルチ装置・マルチタスク BO — 階層 GP を Skill 化する](chapter-11.md) | ✅ 執筆済 |
| 第12章 | [Batch BO と並列実験の Skill 化](chapter-12.md) | ✅ 執筆済 |

### 第IV部　総合ハンズオンと運用

| 章 | タイトル | 状態 |
|---|---|---|
| 第13章 | [Active Learning と逐次実験計画 — BO 以外の逐次戦略](chapter-13.md) | ✅ 執筆済 |
| 第14章 | [総合ハンズオン (Advanced Capstone) — 因果 DAG × 階層 GP × Human 承認 BO](chapter-14.md) | ✅ 執筆済 |
| 第15章 | [BO × Agentic 特有の失敗パターンと監査](chapter-15.md) | ✅ 執筆済 |
| 第16章 | 組織展開と終章 — 実験リソースの割当・BO 運用・次巻への道しるべ | 🚧 未執筆 |

### 付録

| 付録 | タイトル | 状態 |
|---|---|---|
| 付録A | BO × Agentic Skill テンプレート集 | 🚧 未執筆 |
| 付録B | BoTorch / GPyTorch / Ax / Emukit / modAL チートシートと MCP サーバ実装（BO 版） | 🚧 未執筆 |
| 付録C | BO 演習データ候補と benchmark 関数 | 🚧 未執筆 |
| 付録D | Capstone 発展スクリプト集（Ch14 後半 5 iteration） | 🚧 未執筆 |

### 章構成ドラフト

- [章構成ドラフト（v0.2）](chapter-outline.md)：本書全体のスコープ、版履歴、vol-01〜04 との関係、扱わないことの明示、rubber-duck review 反映履歴（Blocking-1〜3 + Non-Blocking 修正 + Suggestions-1〜3）

## 読み進め方

- **vol-04 未読の方**: まず vol-04 を Ch1-4, Ch10-12 だけでも通読することを強く推奨します。本書の Skill 契約は vol-04 の `authorization_gates` L1-L4 と DoE provenance を前提にしています。**vol-04 が本当に完読できない場合**は、第1章の 15 ページ最小復習で vol-04 骨格（DoE 一括生成 vs BO 逐次選択の境界、L3 実験実行承認、`experiment_launch_authorization` の親子関係）を導入してから、第4章に進んでください
- **Pillar 1（単目的 BO Skill）を完成させたい**: 第2〜8章を順に。特に第5章（`experiment_launch_authorization` と `sequential_seed_provenance`）、第6章（GP 骨格 + ARD + 材料 BO 推奨 kernel/prior 既定値表）、第7章（外挿検知の operational 定義：GP 予測分散 × Mahalanobis 距離 × length scale 越え検知の合成基準）は飛ばさない
- **Pillar 2（多目的・制約付き BO Skill）を完成させたい**: 第9〜12章 →（並行して）第5章の Skill 設計。特に第10章の safe BO（"温度上限を超える提案は Skill が絶対に返さない" 契約）と第12章の batch BO（`batch_size` / `pending_experiments` / `batch_diversity_policy_v0_2`）は熟読
- **階層 BO・マルチ装置に挑む**: 第6章（GP 骨格）→ 第11章（階層 GP、Multi-task GP）→ 第14章 Phase 2（マルチ装置対応）
- **Active learning と BO の使い分けを学ぶ**: 第13章（判断分岐：最適化 vs 学習）→ 付録B（modAL の Skill 化、未執筆）
- **Advanced Capstone（複合 Skill）に挑む**: 第9章 → 第11章 → 第12章 → 第14章の順。第14章は 14a（基本 BO ループ：single-objective + 3 iteration、Human 承認を丁寧に扱う）と 14b（発展：階層 GP × multi-objective × constrained BO で 10 iteration まで拡張、後半 5 iteration のスクリプトは付録D）の 2 節構成
- **監査を設計したい**: 第15章（BO 一般の失敗 + Agentic 特有の失敗の 2 セクション構成）→ 付録B（MCP handler 実装、未執筆）→ 第16章（組織責任、未執筆）
- **自分の現場へ展開したい**: 第15章 → 第16章（未執筆）→ 付録A（Skill テンプレート、未執筆）→ 付録C（合成データと ARIM 匿名化、未執筆）

## Skill / リポジトリ配置規約

本書の Skill ディレクトリ構造は **vol-01 付録A §A.2.1 と同一**（`skills/<name>/SKILL.md`, `references/`, `scripts/`, `tests/`, `examples/`）。vol-05 で追加される要素（BO × 逐次 × Agentic provenance の 21 fields、`surrogate_model_family` / `acquisition_spec.name` の canonical enum、`experiment_launch_authorization` の vol-04 L3 chain、`hallucinated_recommendation_detection` の合成基準、`sequential_seed_provenance` の上書き禁止契約、`event_hash` の RFC 8785 JCS canonical serialization）は **Ch5 §5.2** と **Ch10 §10.5-10.6** に一次リファレンスとしてまとめています。canonical fatal 名レジストリと bare-form ↔ `Ch{n}.` prefix 形式のマッピング規約は **Ch15** を正本とします。付録 A の Skill テンプレート集（未執筆）は home chapter の canonical bare-form fatal を使用してください。

## 巻をまたぐ canonical 参照

vol-05 は以下の cross-vol canonical スキーマに依存します（cross-volume audit で確立）:

| canonical | 定義箇所 | vol-05 参照 |
|---|---|---|
| `event_hash` (RFC 8785 JCS + `sha256(json.dumps(sort_keys=True, separators=(',', ':')))`) | vol-05 Ch10 §10.5 anchor `event_canonical_serialization` | Ch12/13/14/15 で全 event stream に採用 |
| Identity `^human:<id>` regex | vol-04 Ch4 §4.6.2 | Ch5 `experiment_launch_authorization.approver_id` |
| `experiment_launch_authorization` の parent chain | vol-05 Ch5 §5.3 → vol-04 L3 | 全 BO iteration の実験実行承認 |
| `causal_dag_ceramic_v1` (DAG artifact) | vol-04 appendix-c §C.12 | Ch14 capstone で search space 絞り込みに使用 |
| enum tables (`surrogate_model_family`, `acquisition_spec.name`) | vol-05 Ch5 §5.2.X | Ch6/8/9/13 で subset として使用 |

## 関連リポジトリ・データセット

- [vol-01 リポジトリ](../vol-01/index.md) / [vol-02 リポジトリ](../vol-02/index.md) / [vol-03 リポジトリ](../vol-03/index.md) / [vol-04 リポジトリ](../vol-04/index.md)（本書の前提）
- **BO デファクト**: [BoTorch](https://botorch.org/) / [GPyTorch](https://gpytorch.ai/) / [Ax](https://ax.dev/)（Facebook Ax、実験管理層）
- **BO 代替・軽量**: [scikit-optimize](https://scikit-optimize.github.io/) / [HEBO (Huawei)](https://github.com/huawei-noah/HEBO) / [Emukit](https://emukit.github.io/)（active learning / experimental design）
- **Active learning**: [modAL](https://modal-python.readthedocs.io/)
- **Bayesian**: [PyMC](https://www.pymc.io/) / [NumPyro](https://num.pyro.ai/)（vol-02 継承、Bayesian surrogate）
- **可視化**: [ArviZ](https://python.arviz.org/) / [Matplotlib](https://matplotlib.org/)
- **MCP**: [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- **BO benchmark データ**: [Olympus](https://aspuru-guzik-group.github.io/olympus/)（合成材料実験の BO ベンチマーク）/ Bayesian benchmark functions（Branin, Hartmann, Ackley）/ [Materials Project](https://materialsproject.org/)（"測定コスト付き" ラップ演習）
- **書籍・論文**: Frazier "A Tutorial on Bayesian Optimization" (arXiv:1807.02811) / Shahriari et al. "Taking the Human Out of the Loop: A Review of Bayesian Optimization" (Proc. IEEE 2016) — 本書は逆に **Human を戻す** 立場 / Garnett "Bayesian Optimization" (Cambridge, 2023)
