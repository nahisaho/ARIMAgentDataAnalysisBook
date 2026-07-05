# 「AI エージェント時代の生成モデル・材料逆設計入門 — ARIM データで動く Agentic Skill」章構成（v0.1 ドラフト）

> vol-06 は vol-01「AI エージェント時代のデータ分析入門」、vol-02「統計・機械学習分析入門」、vol-03「深層学習分析入門」、vol-04「因果推論・実験計画入門」、vol-05「ベイズ最適化・逐次実験計画入門」の続編。
> vol-01〜05 が **「観測 → 予測 → 因果 → 逐次探索」** ラインを整えたのに対し、
> vol-06 は **「潜在空間で仮想の材料を生成し、望みの物性を持つ候補を提案する」** ラインを Skill 化する——
> ARIM データポータルの実験データ（結晶構造・組成・微細構造画像）を主戦場に、
> **「VAE / Diffusion で候補材料を生成し、Skill で妥当性フィルタを通し、Human-in-the-loop で合成試行を承認する」**
> ループをエージェントの Skill として設計する。

## 版履歴

### v0.2（rubber-duck review 反映）
- **Blocking-1（factual error）修正**：CDVAE (`txie-93/cdvae`) と DiffCSP (`jiaor17/DiffCSP`) は Hugging Face Hub に公式重みが存在しないため、「HF Hub 経由 fetch」の記述を修正。CDVAE は GitHub リリース配布、DiffCSP は著者公開の Google Drive 配布、**MatterGen (`microsoft/mattergen`) のみ HF Hub 公式**として明確化
- **Blocking-2 修正**：Ch5 (組成 VAE)・Ch7 (Flow / AR) の「スクラッチ学習」記述を **「教育目的の小規模スクラッチ学習（数百〜数千サンプル、単一 CPU / 単一 GPU で完結）」** と明示。**FM 規模の事前学習は vol-03 継承で禁止**を維持
- **Blocking-3 修正**：前提を「vol-01〜05 全巻完読」から **「vol-03 完読を必須、vol-04 と vol-05 は "いずれか" 完読を推奨」** に緩和。Ch0 は 15 → 12 ページに縮小し、代わりに **付録 D「vol-04/05 未読者のための最小知識（10 ページ）」を新設**
- **Blocking-4 修正**：Ch10 の「因果的妥当性判定」を **OOD detection + distributional coverage** の枠組みに reframe。vol-04 refutation は「参考手法」に降格し、**「生成候補が学習分布内か（IS）」「因果的識別に必要な positivity は成立するか」を分けて評価**
- **Blocking-5 修正**：Ch11 で生成候補を BO の「初期点」ではなく **「離散候補集合の surrogate ランキング」** として扱う（連続 search space の BO と別枠と明示）
- **Blocking-6 修正**：Capstone (Ch12) の Phase 数を **5 phase に統一**（DFT proxy を Phase 3 に統合、Human 承認を最終 Phase として独立化）
- **Blocking-7 修正**：Safety screening を **「静的 filter list」から「Governance レイヤの契約」に格上げ**。第4章・第15章・付録 C で **「filter は必要条件であって十分条件ではない」「dual-use・法的責任は組織レベルの governance で担保」** を明示
- **Non-Blocking 修正**：
  - `dgl` / `torch-geometric` を **optional 環境**に降格（Ch6 結晶グラフ拡張節でのみ言及）
  - Ch5 (組成生成) と Ch8 (物理制約フィルタ) の境界を **「Ch5＝生成器の学習と分布内サンプリング」「Ch8＝生成後の rule-based / model-based スクリーニング」** と明示
  - Ch13 (分子 + 結晶) を **Ch13a (分子・SMILES) と Ch13b (結晶構造) の 2 節構成**に分割
  - provenance field を **`condition_variables_declared` に統一**（v0.1 の `condition_variables` は誤記）
  - Ch11 冒頭に **vol-05 未読者向けの BO 最小知識（3 ページ）**を配置
  - 付録 B を vol-03 付録 B との重複回避のため **「生成モデル固有の MCP パターン」に絞る**（Skill / MCP 一般は vol-03 参照）
  - `hallucinatory_composition_detection` の operational definition を Ch5・Ch10 で明示（学習データからの Mahalanobis 距離 + PCA 再構成誤差 + 物理制約違反率の合成スコア）
- **Suggestions 反映**：
  - **Suggestion-1**：付録 C に **「生成モデル学習に利用可能な open dataset のライセンス比較表」** を追加（MP / OQMD / JARVIS / QM9 の商用可否）
  - **Suggestion-2**：Ch8 に **合成可能性ベンチマーク**（SC-Score, SA-Score, RA-Score for molecules; 熱力学的安定性 hull distance for crystals）の紹介節を追加
  - **Suggestion-3**：Ch6 に **「材料生成における推奨 sampling temperature / guidance scale レンジ」表**を追加（教科書的な既定値を提示）

### v0.1（初版企画ドラフト）
- vol-03 章構成 v0.2 で「vol-05 以降候補」として棚上げされていた **生成モデル** と **材料逆設計** を本巻に統合
- **統合の理由**：ARIM 実データからの生成モデル（vol-03 の材料 Foundation Model と接続）と、生成結果からの逆設計（vol-04 の因果構造で妥当性を判定 + vol-05 の BO で最適化候補を絞り込み）は、Agentic Skill として一体で運用される
- 6 か月・250〜300 ページ規模、本編 15 章 + 付録 3 の骨格を継承
- **エージェントに何を許すか**：潜在空間からの生成・妥当性スクリーニングは自律、**合成試行提案は準自律**、**実験合成の実行判断は必ず Human 承認**（材料合成は不可逆コストが高く、危険物質を生成し得る）

## 前提

- **対象読者**: ARIM データポータル会員のデータ分析者（材料・ナノテク研究者）。Python / Jupyter 経験あり。**vol-03 完読を必須**（生成モデルの骨格は深層学習に依存）、**vol-04 と vol-05 は "いずれか一方" の完読を推奨**（未読巻は付録 D の最小知識で補完可）
- **vol-01 / vol-02 / vol-03 / vol-04 / vol-05 との関係**:
  - **vol-01 完読推奨**: Skill 設計 6 要素、データ契約、Human-in-the-loop、provenance、6 データ型
  - **vol-02 完読推奨**: 第11章の階層モデル（生成モデルの事後解析）、第9-12章の PyMC
  - **vol-03 完読必須**: 第7章の転移学習、第9章の不確かさ、第11章の材料 Foundation Model、**第12章の自己教師あり学習**（生成モデルとの共通哲学）、第14章の Agentic 失敗パターン
  - **vol-04 または vol-05 のいずれか推奨**（両方は不要、未読側は付録 D で補完）:
    - **vol-04 側**：因果構造を「生成候補が学習分布外でないか（distributional coverage）」の判定枠組みとして活用（Ch10）
    - **vol-05 側**：生成候補集合を離散候補として surrogate ランキング（Ch11）
  - vol-06 の provenance は vol-03 拡張に、**生成 × 逆設計 × Agentic 特有フィールド**を追加：`generative_model_family`（VAE / Diffusion / GAN / normalizing flow / autoregressive）, `latent_dim`, `training_data_provenance`, `condition_variables_declared`, `generation_temperature`, `filter_rules_applied`（物理制約・安定性・合成可能性）, `screening_model_family`（DFT proxy / classifier chain）, `top_k_returned`, `inverse_design_authorization`, **`synthesis_launch_authorization`**, **`hallucinatory_composition_detection`**（学習分布からの Mahalanobis 距離 + PCA 再構成誤差 + 物理制約違反率の合成スコア閾値、Ch5・Ch10 で operational 定義）, **`safety_screening_passed`**（毒性・爆発性等のフラグ；ただし **filter は必要条件、十分条件ではない**——governance 契約は Ch4・Ch15・付録 C で規定）
- **最終ゴール（合格ライン）**:
  - **Pillar 1**: **表形式・組成データからの逆設計 Skill**（VAE または Diffusion）を 1 つ以上作れる。生成候補が **物理制約（電荷中性、化学量論、酸化数）を満たすフィルタ**を通り、**エージェントが top-k 候補を提案 → Human 承認**の provenance が残る
  - **Pillar 2**: **画像 / 結晶構造の生成 Skill**（Diffusion または VAE の材料応用）を 1 つ以上作れる。**分布外・非物理的構造の検知**が Skill に組み込まれている
  - **Advanced Capstone**: 上記 2 本を統合した **「Foundation Model 潜在空間で候補生成 → 物理制約フィルタ → 因果的妥当性判定 → BO でランキング → Human 承認」の複合 Skill** を、ARIM 風合成材料データで完成させる
- **分量目安**: 実践書（250〜300 ページ、vol-05 と同等）
- **期限**: 6 か月
- **ハンズオン標準環境**（vol-05 に追加）:
  - vol-03 標準 + `diffusers`（Hugging Face の Diffusion モデル）
  - `pytorch-lightning`（生成モデルの学習管理）
  - `pymatgen`（結晶構造の生成・検証、物理制約チェック）
  - `ase`（原子シミュレーション）
  - `rdkit`（分子構造の生成・検証）
  - `matminer`（組成 → 物性特徴量、生成候補のスクリーニング）
  - **CI 環境は CPU で完結**を必須要件（vol-03〜05 と同じ）
  - **Optional**（Ch6 結晶グラフ拡張節でのみ使用）: `dgl`, `torch-geometric`
- **データセット方針**:
  - **主軸**: **ARIM 風合成材料データ**（真の物性関数を持つ、合成可能性フラグ付きの合成材料データ）を本書でリポジトリ配布（`data/synthetic-generative/` 想定）
    - 組成データ：3 元系・4 元系の合成組成 × 真の物性 × 合成可能性フラグ
    - 結晶構造：格子定数と原子位置を持つ小規模結晶構造
    - 微細構造画像：ARIM 風の合成 SEM/TEM 画像
  - **対比・スケール確認**: 公開データセット
    - **Materials Project**（結晶構造 + 物性、逆設計の教科書ベンチマーク）
    - **OQMD**（Open Quantum Materials Database）
    - **JARVIS-DFT**（NIST の材料 DFT データベース）
    - **AFLOW**（結晶構造 + 熱力学的安定性）
    - **QM9 / OQMD**（分子・小結晶）
    - **付録 C にライセンス比較表**（商用可否・二次配布可否・生成モデル学習利用可否）を掲載
  - **既存 Foundation Model 活用**: 各モデルの実際の配布経路を明示する立場を取る。**スクラッチ事前学習は本書スコープ外**（vol-03 継承）：
    - **MatterGen**：Hugging Face Hub 公式リポジトリ `microsoft/mattergen` 経由で fetch 可
    - **CDVAE**：Hugging Face Hub 公式重みは存在せず、GitHub リポジトリ `txie-93/cdvae` の README で著者配布リンクを参照（本書で fetch スクリプトを提供）
    - **DiffCSP**：Hugging Face Hub 公式重みは存在せず、著者公開の Google Drive で配布（本書で checksum 検証つき fetch スクリプトを提供）
    - **教育目的の小規模スクラッチ学習**（数百〜数千サンプル、単一 CPU / 単一 GPU で完結する組成 VAE・組成 Flow 等）は Ch5・Ch7 で扱う（FM 規模の事前学習ではない）
- **参照**:
  - PyTorch Lightning: https://lightning.ai/
  - Diffusers (Hugging Face): https://huggingface.co/docs/diffusers/
  - Pymatgen: https://pymatgen.org/
  - ASE: https://wiki.fysik.dtu.dk/ase/
  - RDKit: https://www.rdkit.org/
  - matminer: https://hackingmaterials.lbl.gov/matminer/
  - CDVAE: https://github.com/txie-93/cdvae (Xie et al. 2022；著者配布リンクは GitHub README 参照)
  - DiffCSP: https://github.com/jiaor17/DiffCSP (Jiao et al. 2023；重みは著者 Google Drive で配布)
  - MatterGen: https://github.com/microsoft/mattergen かつ Hugging Face `microsoft/mattergen` (Zeni et al. 2023)
  - Materials Project: https://materialsproject.org/
  - OQMD: https://oqmd.org/
  - JARVIS-DFT: https://jarvis.nist.gov/
  - Ho et al. "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
  - Kingma & Welling "Auto-Encoding Variational Bayes" (ICLR 2014)
  - vol-01 〜 vol-05 リポジトリ（本書の前提）

## 本書で扱わないこと（明示）

vol-06 は **「エージェントが材料候補を生成し、Human が合成試行を承認する Skill」** に scope を絞る。

| トピック | 扱わない理由 | 想定巻 |
|---|---|---|
| **FM 規模の生成モデル事前学習** | 材料 Foundation Model（CDVAE / DiffCSP / MatterGen 等）は既存重み利用の立場。**教育目的の小規模スクラッチ学習（Ch5 の組成 VAE・Ch7 の組成 Flow 等、数百〜数千サンプル）は本書内で扱う**が、FM 規模（数十万サンプル・多 GPU）の事前学習は扱わない | vol-03 継承 |
| **強化学習・自律実験ロボティクス** | 「エージェントが合成装置を直接叩く」は本書のスコープ外。合成実行は Human 承認必須 | 別書候補 |
| **理論計算（DFT / MD シミュレーション）の並行実行** | 生成候補のスクリーニングに DFT proxy モデルは使うが、DFT 実行そのものは扱わない | — |
| **生成 AI の LLM 経由での "材料アイデア生成"** | LLM ハルシネーションの妥当性判定が本書の因果的判定と対立するため、本書は数値モデルの生成に集中 | 別書候補 |
| **大規模分散学習・多 GPU での生成モデル学習** | 単一 GPU / 単一ノード規模での fine-tune に絞る（vol-03 と同じ方針） | 別書候補 |
| **生成モデルの数学的完全展開（ELBO の変分展開、SDE 定式化等）** | 既存教科書に譲る。本書は Skill 化と provenance に集中 | — |

## 6 データ型と生成 × 逆設計 Skill の対応

| データ型 | vol-05 で扱えたこと | vol-06 で扱う Agentic Skill |
|---|---|---|
| スペクトル型 | 逐次探索での特徴最適化 | **目標スペクトルからの合成条件逆推定 Skill**、**cVAE で "スペクトル → 組成候補"** |
| クロマトグラム・時系列型 | 反応条件の逐次最適化 | **目標収率プロファイルからの反応条件逆推定** |
| 画像・顕微鏡型 | 粒径・欠陥密度の最適化 | **目標微細構造からの合成条件逆推定 Skill**、**Diffusion で微細構造生成 → 妥当性判定** |
| 回折・散乱パターン型 | 結晶相の最適化 | **目標回折パターンからの結晶構造候補生成 Skill**（CDVAE / DiffCSP 活用） |
| 表形式・プロセス条件型 | 単目的・多目的 BO | **組成 → 物性の逆設計 Skill**（本巻の主戦場）、**cVAE / cDiffusion / autoregressive** の全パターン |
| マルチモーダル統合型 | 多モーダル制約付き BO | **多モーダル条件付き生成**（例：組成 + 目標構造 → プロセス条件） |

**共通する Agentic 観点**：Skill は「候補を生成する」だけでなく、**「候補が物理制約を満たすか、学習分布内か、安全か、実際に合成可能か」を段階的にフィルタし、Human に上位 k 候補と根拠を提示する契約**。合成実行の GO/NO-GO は常に Human。

## 章構成（案）

**分量目安の合計**：本編 250〜300 ページ + 付録 40〜50 ページ

### 第0章 vol-01〜05 の最小復習（目安 12 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第0章 | vol-01〜05 の最小復習 | Skill / provenance / 6 データ型 / 深層 Agentic 権限 の最小前提を 12 ページに圧縮。**vol-04 と vol-05 の詳細は付録 D に切り出し**（片方未読の読者を想定） | vol-06 の前提 |

### 第I部　なぜ「エージェント × 逆設計」なのか（目安 30〜35 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第1章 | 予測 → 因果 → 逐次 → 逆設計 のラダーで vol-01〜05 に何が足りないか | vol-05 の BO は既知の search space を探索するが、逆設計は **新規候補を生成** する。生成モデルの位置づけ、Agentic 特有の hallucination リスク、扱わないことの明示 | 本書の到達点、逆設計の位置付け |
| 第2章 | ARIM データで生成モデルを回すときの Agentic 特有の課題 | **少データ現場での生成の overfitting・mode collapse**、**非物理的候補の生成（電荷不整合・化学量論違反）**、**学習分布外の候補を "自信あり" と誤報告するリスク**、**"合成可能性" の定量化の困難さ**、**エージェントが安全性チェックをスキップする危険**、演習用データ紹介 | 自分のデータの生成モデル適用性判定、演習データ入手 |
| 第3章 | 生成モデルライブラリ地図 — Agentic 使い分け | **Diffusers / PyTorch Lightning / Pymatgen / RDKit / matminer** の位置づけ、**CDVAE / DiffCSP / MatterGen 等の材料 Foundation Model**、**エージェントがどこまで自律的に叩けるか**、vol-03 材料 FM との連携 | ライブラリ使い分けマップ |

### 第II部　生成モデルの Skill 化（目安 60〜70 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第4章 | 生成 × Agentic Skill の設計原則 | vol-03/04/05 第4章の生成 × Agentic 拡張。**生成の provenance**（`generative_model_family`, `latent_dim`, `condition_variables_declared`, `generation_temperature`）、**新設節：合成試行承認ゲート**（`synthesis_launch_authorization`）——**エージェントは候補提示までで、合成実行は Human 承認**。**hallucination 対策**（`hallucinatory_composition_detection` の operational 定義：学習データからの Mahalanobis 距離 + PCA 再構成誤差 + 物理制約違反率の合成スコア閾値）、**安全性スクリーニングの governance レイヤ**（`safety_screening_passed` は必要条件、十分条件は組織 governance で担保；dual-use・法的責任・ARIM 施設ポリシーの参照義務） | 生成 × Agentic Skill 仕様書テンプレート、Safety governance 契約書 |
| 第5章 | VAE を Skill 化する — 潜在空間で組成候補を生成 | VAE の骨格（encoder / decoder / ELBO）、**組成 VAE**（例：3-4 元系合成データで**教育目的の小規模スクラッチ学習**、単一 GPU で完結）、**cVAE（条件付き）**、**分布内サンプリングと分布外候補の区別**（この章は「生成器の学習と分布内サンプリング」に責務を絞り、生成後の物理制約チェックは Ch8 に委譲）、**Skill としての "分布外検知" 契約**（`hallucinatory_composition_detection` の実装例） | 組成 VAE Skill、分布外検知節 |
| 第6章 | Diffusion Model を Skill 化する — 材料生成の主戦場 | DDPM / DDIM の骨格、**組成 Diffusion**、**結晶構造 Diffusion**（MatterGen 経由：HF Hub `microsoft/mattergen` から fetch；DiffCSP は Google Drive、CDVAE は GitHub README リンクからの fetch スクリプトを本書で提供）、**条件付き生成**（classifier-free guidance）、**Skill としての temperature / guidance scale 管理**（"サンプリング温度を勝手に上げない" 契約）、**推奨 sampling temperature / guidance scale レンジ表**（教科書的既定値を提示）、**Optional 節：結晶グラフ拡張**（`dgl` / `torch-geometric` を使う場合、単一 GPU に収める範囲で） | Diffusion Skill、条件付き生成 Skill、サンプリング既定値表 |
| 第7章 | Normalizing Flow と autoregressive を Skill 化する | Normalizing Flow の骨格、**組成 flow**（教育目的の小規模スクラッチ）、**autoregressive（SMILES-like な組成列生成）**、**VAE / Diffusion / Flow / AR の判断表**（データ量 × 制約表現 × 計算コスト） | Flow / AR Skill、生成手法選択判断表 |

### 第III部　スクリーニングと逆設計の Skill 化（目安 55〜65 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第8章 | 物理制約フィルタと合成可能性スクリーニングを Skill 化する | **本章の責務**：Ch5-7 で生成された候補に対する **生成後の rule-based / model-based スクリーニング**。Pymatgen による物理制約チェック（電荷中性、酸化数、化学量論、格子安定性）、matminer による組成 → 物性特徴量、**合成可能性ベンチマーク**（分子：SC-Score / SA-Score / RA-Score、結晶：熱力学的 hull distance と ICSD historical existence）、**Skill としての多段フィルタ契約**（Filter A → Filter B → Filter C の順序と provenance）、**Ch5 の "分布内検知" との責務分離**（Ch5＝生成器内の OOD detection、Ch8＝生成後の物理・合成可能性チェック） | 物理制約フィルタ Skill、スクリーニング Skill、合成可能性ベンチマーク節 |
| 第9章 | DFT proxy と surrogate による物性予測を Skill 化する | 生成候補の物性を **DFT proxy モデル**（MEGNet / M3GNet / MACE 等の既存重み）で高速評価、**vol-03 の Foundation Model 特徴を surrogate 入力に**、**Skill としての予測不確かさの管理**（vol-03 第8-9章の不確かさ手法を再利用） | DFT proxy Skill、物性予測 Skill |
| 第10章 | 生成候補の分布的妥当性判定を Skill に組み込む — OOD detection と distributional coverage | **本章の責務（v0.2 で reframe）**：vol-04 の refutation テストを "生成候補評価" に直接転用するのはカテゴリエラーのため、以下 2 軸で分離する。**軸 A：分布的妥当性**（生成候補が学習分布内か——Mahalanobis 距離・KL divergence・PCA 再構成誤差）、**軸 B：因果的識別条件の positivity**（生成候補の条件範囲が観測データの positivity を満たすか；vol-04 未読者向けの解説を 1 節に）、**参考手法**として vol-04 第9章の refutation を「候補ではなく学習データの identification 妥当性」に適用する使い方を紹介、**Skill としての "妥当性判定失敗時は候補を返さない" 契約** | 分布的妥当性判定 Skill、positivity 診断 Skill |
| 第11章 | 生成候補の surrogate ランキング — 離散候補集合の絞り込み | **本章の責務（v0.2 で reframe）**：生成モデルが出力するのは **離散候補集合**であって連続 search space ではないため、vol-05 の BO を「初期点として利用」ではなく「離散候補集合の surrogate スコアリングと top-k 選抜」として使う。**vol-05 未読者向けの BO 最小知識（3 ページ）**を冒頭に配置。**多目的の場合は Pareto 非劣選抜**、**"生成 → surrogate ランキング → 実験 → 追加学習" ループ**（実験は Human 承認後） | 離散候補ランキング Skill、Pareto 選抜 Skill |

### 第IV部　総合ハンズオンと運用（目安 45〜55 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第12章 | 総合ハンズオン（Advanced Capstone）— Foundation Model → 候補生成 → 物理フィルタ → 分布的妥当性 → Surrogate 選抜 → Human 承認 | ARIM 風合成データで、**Phase 1: 目標物性を条件として cVAE / cDiffusion で候補生成 → Phase 2: Pymatgen で物理制約フィルタ + 合成可能性スコア + DFT proxy による物性予測（Ch8-9 統合） → Phase 3: 分布的妥当性判定（Ch10） → Phase 4: 多目的 surrogate ランキングで top-k 選抜（Ch11） → Phase 5: Human 承認と Safety governance レビュー**（5 phase 構成に統一）。**エージェントは各 Phase で "根拠と不確かさと外挿警告" を提示** | 複合逆設計 Skill (capstone) |
| 第13章 | 分子・結晶構造の逆設計 — 追加ハンズオン（2 節構成） | **13a：分子・SMILES 逆設計**（RDKit + 分子 VAE / SELFIES、QM9 ベンチマーク）、**13b：結晶構造逆設計**（Pymatgen + 結晶 Diffusion、Materials Project ベンチマーク）、各節で "実データを持ち込む場合のガイド" と "既存 FM（MatterGen 等）の fetch 手順" | 分子 Skill、結晶構造 Skill |
| 第14章 | 生成 × Agentic 特有の失敗パターンと監査 | **セクション 1（生成モデル一般の失敗）**：mode collapse、非物理的候補の生成、分布外候補の "自信あり" 報告、条件付き生成の条件無視、Diffusion の temperature 暴走。**セクション 2（Agentic 特有）**：**エージェントが物理制約フィルタをスキップ、"合成可能性" 判定を勝手に緩める、安全性スクリーニングをスキップ、hallucinated 候補を top-k に含める、Human 承認なしに合成試行を推薦、生成 seed を上書き、条件変数を勝手に変更、危険物質を生成候補として返す、Safety governance 契約をバイパスして "filter を通せば OK" と誤解釈する** | 生成 × Agentic 失敗チェックリスト |
| 第15章 | 組織展開と終章 — 逆設計の責任分担・合成試行の管理・Safety governance・シリーズ終章 | **逆設計候補の責任分担**（Skill 提案 vs 研究者判断 vs 合成担当）、**Safety governance の 3 レイヤ**（Skill レイヤの静的 filter、組織レイヤの dual-use review、施設レイヤの法的責任と ARIM ポリシー）、**危険候補の遮断責任**、**vol-06 の到達点**、**シリーズ全体（vol-01〜06）の到達点と接続図**、**次に読むべき既存教科書・論文リスト**、ARIM 施設としての逆設計運用（複数研究者間での生成モデル共有、合成キュー統合） | 組織運用方針、Safety governance 3-layer 契約、シリーズ終章 |

### 付録（目安 40〜50 ページ）

| 付録 | タイトル | 内容 |
|---|---|---|
| 付録A | 生成 × Agentic Skill テンプレート集 | 組成 VAE Skill、cVAE Skill、組成 Diffusion Skill、結晶構造 Diffusion Skill、Flow Skill、AR Skill、物理制約フィルタ Skill、DFT proxy Skill、分布的妥当性判定 Skill、離散候補 surrogate ランキング Skill の雛形。**vol-03 の provenance を生成 × 逆設計 × Agentic 向けに拡張**（`generative_model_family`, `latent_dim`, `condition_variables_declared`, `generation_temperature`, `filter_rules_applied`, `screening_model_family`, `top_k_returned`, `synthesis_launch_authorization`, `hallucinatory_composition_detection`, `safety_screening_passed` 等） |
| 付録B | 生成モデル固有の MCP パターン（vol-03 付録 B との差分に絞る） | **Skill / MCP の一般設計は vol-03 付録 B を参照**とし、本付録は **生成モデル固有の課題**に集中：既存 FM の fetch（HF Hub / GitHub / Google Drive の 3 経路のハンドリング）、checksum 検証、`generation_temperature` / `condition_variables_declared` の MCP ハンドラでの検証、**合成試行承認ゲートの Governance レイヤ実装**（filter passing だけでは insufficient で、dual-use review エンドポイントを組む方針） |
| 付録C | 生成モデル演習データ候補と benchmark、ライセンス比較表、危険物質 filter list、Safety governance テンプレート | **ARIM 風合成材料データの生成スクリプト**（合成可能性フラグと真の物性を持つ）、**公開データセットのライセンス比較表**（Materials Project / OQMD / JARVIS-DFT / AFLOW / QM9 の商用可否・二次配布可否・生成モデル学習利用可否）、**危険物質 filter list**（爆発性・毒性・放射性・dual-use；本 list は必要条件であり十分条件ではないと明示）、**Safety governance 契約書テンプレート**（施設ポリシー参照義務・dual-use review 手順・法的責任範囲）、ARIM 実データを持ち込む際の匿名化ガイドと "生成候補を外部提案する" 際のレビュー手順 |
| 付録D | vol-04 / vol-05 未読者のための最小知識（v0.2 新設、10 ページ） | **vol-04 側**：DAG / confounder / positivity / refutation の 5 ページ最小紹介（Ch10 で使う）、**vol-05 側**：GP surrogate / acquisition function / Pareto 選抜 の 5 ページ最小紹介（Ch11 で使う）。本付録は "vol-04 と vol-05 のいずれか一方のみ読んだ読者" の残り側を補完することを目的とし、深掘りは各巻を参照 |

## 責務分離マップ（重複防止）

### vol-06 内での分離

| 論点 | 予防・設計側 | 実行・事例側 |
|---|---|---|
| 生成モデル選択 | 第5-7章 (VAE / Diffusion / Flow / AR) | 第12章 capstone |
| 物理制約フィルタ | 第8章 | 第14章 (フィルタスキップの失敗) |
| DFT proxy | 第9章 | 第12章 |
| 因果的妥当性判定 | 第10章 | 第12章 |
| 生成候補の surrogate ランキング | 第11章 | 第12章 |
| Agentic 権限 | 第4章 (`synthesis_launch_authorization`) | 第14章・付録B (MCP 実装) |
| 安全性スクリーニング | 第4章 (`safety_screening_passed` 必要条件) + 第15章（Governance 3-layer） | 第14章・付録C (危険物質 filter list + Safety governance テンプレート) |

### vol-01〜05 との分離

| 論点 | 既刊側 | vol-06 側 |
|---|---|---|
| 予測 Skill | vol-01/02/03 | 生成候補の物性予測に第9章で活用 |
| 統計モデル | vol-02 第10-12章 | 生成候補の事後解析に活用（生成 seed の事後分布等） |
| 深層学習 | vol-03 全体 | 生成モデルの骨格として第5-7章で活用 |
| 材料 Foundation Model | vol-03 第11章 | **既存 FM を生成モデルに拡張**（CDVAE / DiffCSP / MatterGen 経由）、第6章 |
| 自己教師あり学習 | vol-03 第12章 | 潜在空間の学習として共通哲学、第7章で言及 |
| 因果推論 | vol-04 全体 | **生成候補の妥当性判定**に第10章で活用 |
| DoE | vol-04 第10-11章 | 生成候補の合成試行を DoE で計画（第12章 capstone Phase 6） |
| Bayesian 最適化 | vol-05 全体 | **生成候補のランキングと絞り込み**に第11章で活用 |
| Human-in-the-loop | vol-01 第6章 / vol-05 第4章 | **合成試行の Human 承認**（`synthesis_launch_authorization`、第4章）、**危険候補の遮断責任**（第15章） |
| Agentic 権限 | vol-03/04/05 第4章 | **生成 → フィルタ → 選抜 の 3 段階権限**：生成・フィルタ・スクリーニングは自律 / top-k 選抜は準自律 / **合成実行は必ず Human 承認** |
| 失敗パターン | 各巻 第14章 | 生成モデル一般 + Agentic 特有（第14章 2 セクション構成） |

## 各ハンズオン章の共通構成

vol-03/04/05 と同形式。**第5章で丁寧に説明し、以降は差分中心**。「エージェント役割」節を各章に必ず含める。

- この章で作る Skill の概要
- **エージェント役割**（Skill は何を生成し、何をフィルタし、何を Human に投げるか、**合成試行は誰が判断するか**）
- 入力仕様 / 出力仕様 / 制約条件（vol-05 拡張 + vol-06 生成 × 逆設計 × Agentic 拡張に準拠）
- **生成モデル特有の評価基準**（validity rate / uniqueness / novelty / condition adherence / physical constraint satisfaction / synthesizability proxy）
- 実行例 / 失敗例（**hallucinated 候補と Agentic 失敗を含む**）/ 改善版
- 他生成モデル（VAE ↔ Diffusion ↔ Flow）への転用方法

## 特に注意する重点管理項目

| 項目 | 予防・設計 | 事例・検証 |
|---|---|---|
| **エージェントが物理制約フィルタをスキップ** | 第4章・第8章 `filter_rules_applied` の必須化 | 第14章 |
| **エージェントが hallucinated 候補を top-k に含める** | 第4章 `hallucinatory_composition_detection`（Mahalanobis + PCA 再構成誤差 + 制約違反率の合成スコア閾値） | 第14章 |
| **危険物質を生成候補として返す** | 第4章 `safety_screening_passed`（必要条件）+ 付録C filter list + 第15章 Safety governance 3-layer | 第14章 |
| **Safety governance を "filter を通せば OK" と誤解釈** | 第4章・第15章の governance 3-layer 契約 | 第14章 |
| **合成試行を Human 未承認で推薦** | 第4章 `synthesis_launch_authorization` | 第14章・付録B (MCP 実装) |
| **Diffusion の temperature を勝手に上げる** | 第6章 temperature pin | 第14章 |
| **条件付き生成の条件を無視** | 第4章 `condition_variables_declared` | 第14章 |
| **合成可能性判定を勝手に緩める** | 第8章 スクリーニング契約 | 第14章 |
| **生成 seed を上書き** | 第4章 seed provenance | 第14章 |

## シリーズ全体の到達点（第15章での接続図）

```
vol-01: 動く Skill を 1 つ自力で作る
   ↓ 統計/ML の厚みと不確かさを積む
vol-02: 統計/ML Skill + PyMC Skill + 階層モデル capstone
   ↓ 深層と Foundation Model を Agentic に扱う
vol-03: 深層 Skill + FM 呼び出し + PyMC 階層 capstone
   ↓ "なぜ" と "もし" を主張する
vol-04: 因果推論 Skill + DoE Skill + 反実仮想 capstone
   ↓ "次にどの実験" を提案する
vol-05: BO Skill + active learning + 階層 BO capstone
   ↓ "新しい材料を提案する"
vol-06: 生成 Skill + 逆設計 Skill + 生成 × BO 統合 capstone
   ↓
【シリーズ完結】ARIM データポータル会員が
    観測 → 予測 → 因果 → 逐次 → 生成
    の全ラダーを Agentic Skill として運用できる
```

**シリーズ通しての一貫哲学**：**エージェントは Skill として提案する。実験実行・合成試行・介入判断は常に Human が承認する。全 provenance は次巻でも参照可能な形で残す**。
