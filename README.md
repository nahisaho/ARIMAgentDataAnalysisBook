# AI エージェント時代のデータ分析シリーズ

> *An Agentic Data Analysis Cookbook Series —— for materials & nanotech researchers*

AI Agent と MCP（Model Context Protocol）を用いて、実験データの分析を **自然言語で・Human-in-the-loop で・再現可能に** 実行できるようにするための実践書シリーズです。ARIM データポータルの材料・ナノテク研究者を主対象に、Python/Jupyter 経験者が **自分の実験データ向けの「分析用 Skill」を自力で作成できる** 状態まで到達することを目標としています。

シリーズは **予測 → 統計/ML → 深層 → 因果 → 逐次 → 逆設計** の 6 段ラダーで構成され、各巻とも「動く・検証済み・再現できる Skill」を成果物とします。

## 📚 シリーズ構成（全 6 巻）

| 巻 | タイトル | 主テーマ | 状態 |
|---|---|---|---|
| **vol-01** | [AI エージェント時代のデータ分析入門](papers/vol-01/index.md) | Skill / MCP / Human-in-the-loop の基礎、6 データ型による装置横断 | ✅ 本編 15/15 章完了 |
| **vol-02** | [AI エージェント時代の統計・機械学習分析入門](papers/vol-02/chapter-outline.md) | Scikit-learn × PyMC、CV・不確かさ・階層モデル | ✅ 本編 16/16 章完了 |
| **vol-03** | [AI エージェント時代の深層学習分析入門](papers/vol-03/chapter-outline.md) | PyTorch / JAX / Hugging Face、Foundation Model、fine-tune の Agentic 承認 | ✅ 本編 16/16 章完了 |
| **vol-04** | [AI エージェント時代の因果推論・実験計画入門](papers/vol-04/chapter-outline.md) | DAG・介入効果・DoE、"なぜ" と "もし" の Skill 化 | 🚧 執筆中（7/16 章） |
| **vol-05** | [AI エージェント時代のベイズ最適化・逐次実験計画入門](papers/vol-05/chapter-outline.md) | GP surrogate・acquisition・多目的/制約付き BO・Active Learning | 📋 章構成のみ |
| **vol-06** | [AI エージェント時代の生成モデル・材料逆設計入門](papers/vol-06/chapter-outline.md) | VAE / Diffusion / Flow、物理制約・分布的妥当性・逆設計 | 📋 章構成のみ |

> [!NOTE]
> シリーズ全体を通じて、多様な装置カテゴリを **6 つのデータ型**（スペクトル型／クロマトグラム・時系列型／画像・顕微鏡型／回折・散乱パターン型／表形式・プロセス条件型／マルチモーダル統合型）に抽象化して扱います。1 つの型で Skill を作れれば、同じ型の他装置へ骨格を転用できます。

## 📖 本シリーズについて

| 項目 | 内容 |
|---|---|
| **対象読者** | ARIM データポータル会員（材料・ナノテク研究者）／Python・Jupyter 経験あり／MCP・AI Agent 未経験からスタート |
| **合格ライン** | 各巻とも、自分の実験データに対応する Skill を **動く・検証済み・再現できる** の 3 拍子で作れること |
| **標準環境** | Python + JupyterLab + Jupyter MCP + GitHub Copilot CLI + ToolUniverse MCP（+ 各巻の追加ライブラリ） |
| **各巻の分量** | 実践書 200〜290 ページ規模／執筆期間 各 6 か月 |
| **設計原則** | Human-in-the-loop／データ契約／provenance／循環設計問題の回避／責務分離 |

## 🗂 各巻の目次概略

### vol-01　AI エージェント時代のデータ分析入門（本編 15 章）

「動く Skill を 1 つ自力で作る」までを扱うシリーズの土台。

- **第I部** なぜ必要か・何を作るか（第1〜3章）
- **第II部** 最小環境で動かす（第4〜6章：環境構築／Jupyter MCP／安全運用）
- **第III部** Skill を作る（第7〜9章：設計原則／データ契約／単一データ解析）
- **第IV部** Skill を拡張・検証する（第10〜12章：文献照合／マルチモーダル／検証・レポート化）
- **第V部** 横展開と運用（第13〜15章：装置カテゴリ別テンプレート／失敗パターン／組織導入）

➡︎ [vol-01 全 15 章の詳細目次](papers/vol-01/index.md) ｜ [vol-01 章構成計画](papers/vol-01/chapter-outline.md)

### vol-02　統計・機械学習分析入門（全 16 章）

vol-01 の Skill に **統計/ML の厚みと不確かさ** を積む。**Scikit-learn Skill + PyMC Skill + 階層モデル Skill（capstone）** の 2 柱 + 1 発展。

- **第1章** vol-01 最小復習
- **第I部** なぜ「エージェント × 統計・ML」なのか（第2〜4章）
- **第II部** Scikit-learn で足場を築く（第5〜7章：設計原則／教師あり／教師なし）
- **第III部** 落とし穴と検証（第8〜9章：CV・リーク検知／解釈可能性）
- **第IV部** PyMC で不確かさを扱う（第10〜13章：概念／PyMC 入門／階層モデル／MCMC 実務）
- **第V部** 総合・運用・失敗（第14〜16章：capstone／失敗パターン／組織展開）

➡︎ [vol-02 章構成計画](papers/vol-02/chapter-outline.md)

### vol-03　深層学習分析入門（全 16 章）

**エージェントが深層モデル（Foundation Model 含む）を Skill として安全に扱う** ——ARIM データの少データ・装置固有・階層構造を主戦場に、fine-tune の Human 承認・学習ジョブ起動権限の Skill 契約までを扱う。

- **第1章** vol-01/02 最小復習
- **第I部** なぜ「深層 × Agentic Skill」なのか（第2〜4章）
- **第II部** 深層 Agentic Skill の設計と教師あり学習（第5〜8章）
- **第III部** 不確かさと解釈（第9〜11章）
- **第IV部** Foundation Model と表現学習（第12〜14章）
- **第V部** 運用・失敗・展望（第15〜16章）

➡︎ [vol-03 章構成計画](papers/vol-03/chapter-outline.md)

### vol-04　因果推論・実験計画入門（第0章 + 本編 15 章）

予測から **"なぜ" と "もし"** へ。DAG・介入効果・古典 DoE・Bayesian 実験計画を Agentic Skill として扱う。

- **第0章** vol-01〜03 最小復習
- **第I部** なぜ因果と実験計画か（第1〜3章）
- **第II部** 因果 Skill の設計と DAG（第4〜5章）
- **第III部** 介入効果の推定（第6〜9章：Propensity/IPW/DR ／DiD/IV/Synthetic Control ／CATE ／感度分析）
- **第IV部** 実験計画の Skill 化（第10〜12章：Full/Fractional factorial ／応答曲面・タグチ ／Bayesian DoE）
- **第V部** 総合・失敗・運用（第13〜15章：観測→DoE→反実仮想 capstone ／失敗パターン／責任分担）

➡︎ [vol-04 章構成計画](papers/vol-04/chapter-outline.md)

### vol-05　ベイズ最適化・逐次実験計画入門（全 16 章）

「次にどの試料を測るか」の判断を Skill 化。**GP surrogate ・acquisition ・多目的/制約付き BO ・階層 GP・batch BO ・Active Learning** を扱う。

- **第1章** vol-01〜04 最小復習
- **第I部** なぜ「エージェント × 逐次実験計画」なのか（第2〜4章）
- **第II部** 単目的 BO の Skill 化（第5〜8章：設計原則／GP surrogate ／acquisition と外挿検知／代替 surrogate）
- **第III部** 多目的・制約付き・階層 BO（第9〜12章：qEHVI ／制約付き・safe BO ／階層 GP ／batch BO）
- **第IV部** 総合ハンズオンと運用（第13〜16章：Active Learning ／因果 × 階層 × 承認 capstone ／失敗パターン／実験リソース割当）

➡︎ [vol-05 章構成計画](papers/vol-05/chapter-outline.md)

### vol-06　生成モデル・材料逆設計入門（第0章 + 本編 15 章）

シリーズ終章。予測→因果→逐次→**逆設計** の最終ラダー。**VAE / Diffusion / Normalizing Flow** による組成・構造生成と、**物理制約・DFT proxy・分布的妥当性・surrogate ランキング** を通じた Human 承認付き逆設計を扱う。

- **第0章** vol-01〜05 最小復習
- **第I部** なぜ逆設計か（第1〜3章）
- **第II部** 生成モデル Skill（第4〜7章：設計原則／VAE ／Diffusion ／Flow・autoregressive）
- **第III部** フィルタリングと選抜（第8〜11章：物理制約／DFT proxy ／OOD・distributional coverage ／surrogate ランキング）
- **第IV部** 総合・追加ハンズオン・失敗・運用（第12〜15章：Foundation Model → 生成 → 承認 capstone ／分子・結晶逆設計／失敗パターン／Safety governance・シリーズ終章）

➡︎ [vol-06 章構成計画](papers/vol-06/chapter-outline.md)

## 🚀 読み進め方

| 目的 | 推奨ルート |
|---|---|
| **最短で「動く」を体験したい** | vol-01 の第 1 → 3 → 4 → 5 章 |
| **最初の Skill を自力で作りたい** | vol-01 を第 1 〜 9 章まで順に（第 6・8 章は飛ばさない） |
| **統計的厚みと不確かさを扱いたい** | vol-01 完了後、vol-02 を第 1 章から |
| **深層モデル・Foundation Model を扱いたい** | vol-02 完了後、vol-03 を第 1 章から |
| **"なぜ" "もし" を主張したい／DoE を回したい** | vol-03 まで完了後、vol-04 へ |
| **逐次実験（BO・active learning）を回したい** | vol-04 完了後、vol-05 へ |
| **材料の逆設計に踏み込みたい** | vol-05 完了後、vol-06 へ |
| **失敗を先に知っておきたい** | 各巻の第 14 章（失敗パターン）を先読み |

> [!TIP]
> 各巻の **第 0 章（最小復習）** は 10〜15 ページに前巻までの要点を圧縮しているため、途中巻から入ることも可能です。ただし Skill / MCP / Human-in-the-loop / データ契約 / provenance の 5 概念は vol-01 で最も丁寧に扱っています。

## 📁 リポジトリ構成

```
agentic-data-analysis-cookbook/
├── README.md              # 本ファイル（シリーズ全体の入口）
├── LICENSE                # CC BY-NC-SA 4.0
├── manifest.yaml          # プロジェクト管理マニフェスト
├── papers/                # シリーズ全体の原稿
│   ├── vol-01/            # データ分析入門（Skill / MCP の基礎）
│   ├── vol-02/            # 統計・機械学習分析入門（sklearn × PyMC）
│   ├── vol-03/            # 深層学習分析入門（PyTorch / JAX / HF）
│   ├── vol-04/            # 因果推論・実験計画入門（DAG × DoE）
│   ├── vol-05/            # ベイズ最適化・逐次実験計画入門（BO × active）
│   └── vol-06/            # 生成モデル・材料逆設計入門（VAE / Diffusion）
├── research/              # 執筆過程で収集した情報・引用ソース
└── prompts/               # 執筆時のプロンプト履歴
```

各 `papers/vol-XX/` 配下は原則として以下の構成をとります。

- `chapter-outline.md` — 執筆計画・責務分離マップ
- `index.md` — 全章 + 付録の目次（vol-01 で提供、他巻は執筆進捗に応じて）
- `chapter-00.md` 〜 `chapter-15.md` — 本文
- `appendix-a.md` 〜 `appendix-c.md` — テンプレート集・チートシート・トラブルシューティング

## 🛠 執筆・レビュー方針

- **記法**: GitHub Flavored Markdown。注釈は `> [!NOTE] / [!WARNING] / [!TIP] / [!IMPORTANT]`
- **図**: Mermaid を優先（ASCII 図は必要な場合のみ）
- **章末構成**: 到達目標／扱うこと・扱わないこと／章末ワーク／本章のまとめ／参考資料
- **責務分離**: 各巻の `chapter-outline.md` に「責務分離マップ（重複防止）」を明記
- **品質保証サイクル**: 執筆 → 実機テスト → rubber-duck review → 修正 → 再テスト → commit

## 📜 ライセンスと利用について

- 原稿・コード例を含む本リポジトリの著作物: **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ja)**（表示 - 非営利 - 継承 4.0 国際）
  - **表示 (BY)**: 適切なクレジット表示とライセンスへのリンクが必要です
  - **非営利 (NC)**: **商用利用は禁止**です
  - **継承 (SA)**: 改変した場合、同じライセンス（CC BY-NC-SA 4.0）で公開する必要があります
- 引用元の第三者コンテンツ（論文・ツール・データ）は各出典元のライセンスに従います
- 詳細は [`LICENSE`](./LICENSE) を参照してください

> [!WARNING]
> 本シリーズに登場する装置別の物理定数・分解温度・帰属候補値、また合成階層データ・MatBench・RRUFF 等を用いた分析結果は **参考情報** です。実験・論文投稿時は必ず一次文献・自装置校正データ・自組織のガバナンスに従って再確認してください。

## 🤝 コントリビュート

Issue / PR は歓迎します。特に以下は大歓迎です。

- 誤植・技術的誤り・引用漏れの指摘
- 装置カテゴリ別の失敗事例（各巻の第 14 章）や運用パターン（各巻の第 15 章）の追加
- 環境構築の再現ログ（vol-01 第 4 章、vol-03 の GPU 環境など）
- 他分野研究者からの「6 つのデータ型」枠組みへのフィードバック
- 執筆待ちの vol-04〜06 章構成へのレビューコメント

## 🔗 関連リンク

- [ARIM データポータル](https://nanonet.go.jp/data_service/page/textbook.html)
- [Model Context Protocol 仕様](https://modelcontextprotocol.io/)
- [GitHub Copilot CLI](https://github.com/features/copilot)
- [MatBench（vol-02 以降で使用）](https://matbench.materialsproject.org/)
- [RRUFF（vol-02 以降で使用）](https://rruff.info/)

## 👤 著者

- **仁科 直哉**（[@nahisaho](https://github.com/nahisaho)）

---

**Status**: vol-01/02/03 完成 ✅ ｜ vol-04 執筆中 🚧 ｜ vol-05/06 章構成のみ 📋 ｜ Last update: 2026-07-06
