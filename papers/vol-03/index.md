# AI エージェント時代の深層学習分析入門 — ARIM データで動く Agentic Skill

vol-01「AI エージェント時代のデータ分析入門」および vol-02「AI エージェント時代の統計・機械学習分析入門」の続編。vol-01 が「動く Skill を 1 つ自力で作る」まで、vol-02 が「その Skill に統計/ML の厚みと不確かさ」を積み上げたのに対し、vol-03 は **エージェントが深層モデル（Foundation Model 含む）を Skill として安全に扱えるようにする** ための実践書です。ARIM データポータルの実験データ（少データ・装置固有・階層構造）を主戦場に、**「Human-in-the-loop で fine-tune を承認する」「勝手にモデルを更新させない」「学習ジョブの起動権限を Skill で契約する」**までを扱います。

- **対象読者**: ARIM データポータル会員のデータ分析者（材料・ナノテク研究者。Python/Jupyter 経験あり）。vol-01 + vol-02 完読を推奨、未読でも**第1章の最小復習**で読み進められる構造
- **最終ゴール（合格ライン）**:
  - **Pillar 1**: **ARIM 実データ（風の合成/匿名化データ含む）に対して、fine-tune 判断が Human-in-the-loop で承認される「教師あり深層学習 + 転移学習」Skill** を 1 つ以上作れる。単に動くだけでなく、**エージェントが勝手にモデルを更新できない契約**が入っている
  - **Pillar 2**: **不確かさつき深層モデル Skill**（Deep Ensemble / MC-Dropout / BNN のいずれか）を 1 つ以上作れる。calibration（Brier / ECE）と **「不確かさが閾値を超えたらエージェントは自律決定を停止し人間に投げる」ゲート** を持つ
  - **Advanced Capstone**: 上記 2 本を統合した **「Foundation Model → 深層特徴 → PyMC 階層モデル」の複合 Skill**（vol-02 第14章 capstone の深層版）を、**ARIM 風合成階層データ**（装置間・ロット間・研究室間）で完成させる
- **標準環境**（vol-02 に追加）: `torch`, `torchvision`, `torchaudio`, `jax`, `jaxlib`, `flax`, `optax`, `transformers`, `datasets`, `accelerate`, `huggingface_hub`, `captum`, `grad-cam`, `shap`（vol-02 継承）。**GPU**: CUDA / ROCm / MPS 対応、**CI は CPU で完結**を必須要件

> [!NOTE]
> 因果推論・ベイズ最適化・生成モデル（VAE / GAN / Diffusion）・大規模分散学習は本書のスコープ外です（第2章・第16章で道しるべを示すのみ）。強化学習・LLM ファインチューニングもスコープ外。

> [!TIP]
> データセット方針は**反転**：ARIM 風合成データ・匿名化データを主軸、公開ベンチマーク（NFFA-EUROPE SEM / UHCS microstructure / MatBench / RRUFF Raman）は「対比・スケール確認」の位置づけ。Foundation Model は Hugging Face Hub 経由で MatBERT / CrystaLLM / ChemBERTa を利用します（付録C にライセンス・retrieval 手順あり）。

> [!IMPORTANT]
> vol-03 の provenance は vol-02 拡張に加え、**深層 × Agentic 特有フィールド**を追加します：`gpu_environment_provenance`（`cudnn_deterministic`, `random_seed_per_worker`, `tf32_allowed` 等）、`foundation_model_provenance`（`weights_uri`, `files_with_sha256[{file,kind,sha256}]`, `revision_commit_hash`, `pretraining_data_license`）、`agent_authorization`、`training_job_approval`、`checkpoint_overwrite_policy`、`fm_update_gate`。canonical スキーマは**付録A §A.9** に一次リファレンスとしてまとめています。

## 目次

> **執筆進捗**: 本編 **16 / 16 章 完了 ✅**、付録 **3 / 3 完了 ✅**

| 章 | タイトル | 状態 |
|---|---|---|
| 第1章 | [vol-01 / vol-02 の最小復習](chapter-01.md) | ✅ 執筆済 |

### 第I部　なぜ「深層 × Agentic Skill」なのか

| 章 | タイトル | 状態 |
|---|---|---|
| 第2章 | [vol-02 の Skill をエージェントが深層で拡張するとき、ARIM データで何が起きるか](chapter-02.md) | ✅ 執筆済 |
| 第3章 | [ARIM データに深層を持ち込むときの Agentic 特有の課題](chapter-03.md) | ✅ 執筆済 |
| 第4章 | [PyTorch / JAX / Hugging Face の Agentic 使い分け](chapter-04.md) | ✅ 執筆済 |

### 第II部　深層 Agentic Skill の設計と教師あり学習

| 章 | タイトル | 状態 |
|---|---|---|
| 第5章 | [深層 × Agentic Skill の設計原則](chapter-05.md) | ✅ 執筆済 |
| 第6章 | [前処理と Augmentation の Skill 化 — Agentic 契約と深層 anti-leakage](chapter-06.md) | ✅ 執筆済 |
| 第7章 | [教師あり深層 Agentic Skill を作る](chapter-07.md) | ✅ 執筆済 |
| 第8章 | [転移学習 / fine-tuning を Skill 化する — Agentic 判断つき](chapter-08.md) | ✅ 執筆済 |

### 第III部　不確かさと解釈

| 章 | タイトル | 状態 |
|---|---|---|
| 第9章 | [深層モデルの不確かさ入門 — Agentic 停止条件つき](chapter-09.md) | ✅ 執筆済 |
| 第10章 | [MC-Dropout と Bayesian Neural Net を Skill 化する — 手法選択の判断表](chapter-10.md) | ✅ 執筆済 |
| 第11章 | [深層モデルの検証・可視化・レポート化 — Human-in-the-loop 拡張](chapter-11.md) | ✅ 執筆済 |

### 第IV部　Foundation Model と表現学習

| 章 | タイトル | 状態 |
|---|---|---|
| 第12章 | [材料 Foundation Model と MCP 連携 — Agentic 呼び出し契約](chapter-12.md) | ✅ 執筆済 |
| 第13章 | [自己教師あり学習と対比学習を Skill 化する — 「作る側」の判断契約](chapter-13.md) | ✅ 執筆済 |
| 第14章 | [総合ハンズオン（Advanced Capstone）— Foundation Model → 深層特徴 → PyMC 階層モデル](chapter-14.md) | ✅ 執筆済 |

### 第V部　運用・失敗・展望

| 章 | タイトル | 状態 |
|---|---|---|
| 第15章 | [深層 × Agentic 特有の失敗パターンと監査](chapter-15.md) | ✅ 執筆済 |
| 第16章 | [組織展開と終章 — GPU リソース・モデル配布・Agentic 責任分担](chapter-16.md) | ✅ 執筆済 |

### 付録

| 付録 | タイトル | 状態 |
|---|---|---|
| 付録A | [深層 × Agentic Skill テンプレート集](appendix-a.md) | ✅ 執筆済 |
| 付録B | [PyTorch / JAX / Hugging Face チートシートと MCP サーバ実装](appendix-b.md) | ✅ 執筆済 |
| 付録C | [GPU トラブルシューティングと深層 × Agentic 演習データ候補](appendix-c.md) | ✅ 執筆済 |

### 章構成ドラフト

- [章構成ドラフト（v0.2）](chapter-outline.md)：本書全体のスコープ、版履歴、vol-01 / vol-02 との関係、扱わないことの明示

## 読み進め方

- **vol-01 / vol-02 未読の方**: 第1章 → 第2章 → 第5章 →（Pillar 選択）へ。第1章は 15 ページの最小復習で、Skill の 6 要素・Human-in-the-loop・階層モデルの基礎を素早く再導入します
- **Pillar 1（教師あり深層 + 転移学習 Skill）を完成させたい**: 第1〜8章を順に。特に第5章（Agentic 学習権限設計）と第6章（augmentation の anti-leakage 契約）は飛ばさない
- **Pillar 2（不確かさつき深層 Skill）を完成させたい**: 第9〜11章 →（並行して）第5章の Skill 設計。Deep Ensemble / MC-Dropout / BNN の**手法選択判断表**（第10章）を熟読し、自分のデータに合った手法を選ぶ
- **Foundation Model を Skill から呼びたい**: 第12章（材料 FM の MCP 呼び出し契約） → 付録A §A.9 の `foundation_model_provenance` → 付録B B.4（HF Hub 経由の pin 手順）
- **Advanced Capstone（複合 Skill）に挑む**: 第8章 → 第12章 → 第14章の順。第14章は ARIM 風合成階層データを使い、FM 転移 → 深層特徴 → PyMC 階層モデルを 1 つの Skill に統合
- **自分の現場へ展開したい**: 第15章（失敗パターン + 監査 registry）→ 第16章（GPU・モデル配布・組織責任）→ 付録A（テンプレート）→ 付録C §C.9（社内 Skill レジストリ構築）

## Skill / リポジトリ配置規約

本書の Skill ディレクトリ構造は **vol-01 付録A §A.2.1 と同一**（`skills/<name>/SKILL.md`, `references/`, `scripts/`, `tests/`, `examples/`）。vol-03 で追加される要素（GPU 環境 provenance、Foundation Model provenance、Agentic 権限契約、checkpoint 上書きポリシー、FM 更新ゲート等）は**付録A §A.9** に一次リファレンスとしてまとめています。canonical な失敗パターン ID レジストリ（DG-01〜DG-08 / AG-01〜AG-09 / MX-01〜MX-04 / ORG-01〜ORG-07）は**第15章**を正本とします。

## 関連リポジトリ・データセット

- [vol-01 リポジトリ](../vol-01/index.md) / [vol-02 リポジトリ](../vol-02/index.md)（本書の前提）
- [PyTorch](https://pytorch.org/) / [JAX](https://jax.readthedocs.io/) / [Flax](https://flax.readthedocs.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) / [datasets](https://huggingface.co/docs/datasets/) / [huggingface_hub](https://huggingface.co/docs/huggingface_hub/)
- [MatBERT](https://github.com/lbnlp/matbert) / [CrystaLLM](https://github.com/lantunes/CrystaLLM) / [ChemBERTa](https://github.com/seyonechithrananda/bert-loves-chemistry)
- [Captum](https://captum.ai/) / [Grad-CAM (pytorch-grad-cam)](https://github.com/jacobgil/pytorch-grad-cam) / [SHAP](https://shap.readthedocs.io/)
- [MLflow](https://mlflow.org/) / [Weights & Biases](https://wandb.ai/)（optional、第15章で判断基準）
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- 参考ベンチマーク: [NFFA-EUROPE SEM](https://b2share.eudat.eu/records/80df8606fcdb4b2bae1656f0dc6db8ba) / [UHCS microstructure](https://hdl.handle.net/11256/940) / [MatBench](https://matbench.materialsproject.org/) / [RRUFF](https://rruff.info/)
