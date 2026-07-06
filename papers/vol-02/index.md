# AI エージェント時代の統計・機械学習分析入門

vol-01「AI エージェント時代のデータ分析入門」の続編。動く Skill に**統計的・機械学習的な厚み**を持たせ、**不確かさまで含めて主張できる分析**を Skill 化するための実践書です。

- **対象読者**: ARIMデータポータル会員のデータ分析者（材料・ナノテク研究者。Python/Jupyter 経験あり）。vol-01 未読でも**第1章の最小復習**で読み進められる構造
- **最終ゴール（合格ライン）**:
  - **Pillar 1**: **Scikit-learn を核にした統計/ML Skill** を 1 つ以上（回帰・分類・PCA・クラスタリングのいずれか）動く・検証済み・再現できる形で作れる
  - **Pillar 2**: **PyMC を核にした Bayesian Skill** を 1 つ以上（事後分布 / 信用区間 / 事後予測チェックまで含む）作れる
  - **Advanced Capstone**: 上記 2 本を統合した **階層モデル拡張の Skill**（反復測定・ロット差・装置間差のいずれかを扱う）を作れる
- **標準環境**（vol-01 に追加）: `scikit-learn`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `pymc`, `arviz`, `numpyro`, `jax`, `shap`, `pdpbox`

> [!NOTE]
> Stan / cmdstanpy は本文で使用しません（付録B の PyMC↔Stan 対応表として参照用のみ）。深層学習・因果推論は本書のスコープ外です（第2章・第16章で道しるべを示すのみ）。

> [!TIP]
> データセット方針は**併用**：章内ハンズオンは vol-01 のサンプルデータを流用、本格演習は **MatBench**（表形式）／**RRUFF Raman**（スペクトル）、階層モデルの Capstone は本書提供の**合成階層データ**（`data/synthetic-hierarchy/`）を使います。実データ候補は付録C に列挙。

## 目次

> **執筆進捗**: 本編 **16 / 16 章 完了 ✅**、付録 **3 / 3 完了 ✅**

| 章 | タイトル | 状態 |
|---|---|---|
| 第1章 | [vol-01 の最小復習](chapter-01.md) | ✅ 執筆済 |

### 第I部　なぜ「エージェント × 統計・ML」なのか

| 章 | タイトル | 状態 |
|---|---|---|
| 第2章 | [vol-01 の Skill に何が足りないのか](chapter-02.md) | ✅ 執筆済 |
| 第3章 | [ARIM データに現れる統計的課題](chapter-03.md) | ✅ 執筆済 |
| 第4章 | [Scikit-learn と PyMC の全体像・使い分け](chapter-04.md) | ✅ 執筆済 |

### 第II部　Scikit-learn で足場を築く

| 章 | タイトル | 状態 |
|---|---|---|
| 第5章 | [統計/ML 分析用 Skill の設計原則](chapter-05.md) | ✅ 執筆済 |
| 第6章 | [教師あり学習を Skill 化する](chapter-06.md) | ✅ 執筆済 |
| 第7章 | [教師なし学習を Skill 化する](chapter-07.md) | ✅ 執筆済 |

### 第III部　落とし穴と検証

| 章 | タイトル | 状態 |
|---|---|---|
| 第8章 | [モデル選択・交差検証・データリーク検知](chapter-08.md) | ✅ 執筆済 |
| 第9章 | [解釈可能性とレポート化](chapter-09.md) | ✅ 執筆済 |

### 第IV部　PyMC で不確かさを扱う

| 章 | タイトル | 状態 |
|---|---|---|
| 第10章 | [不確かさ入門：頻度論の限界と Bayesian への橋](chapter-10.md) | ✅ 執筆済 |
| 第11章 | [PyMC 入門と Skill 化](chapter-11.md) | ✅ 執筆済 |
| 第12章 | [階層モデル：反復測定・ロット差・測定誤差](chapter-12.md) | ✅ 執筆済 |
| 第13章 | [MCMC の実務と限界：判断と修正](chapter-13.md) | ✅ 執筆済 |

### 第V部　総合・運用・失敗

| 章 | タイトル | 状態 |
|---|---|---|
| 第14章 | [総合ハンズオン（Advanced Capstone）：sklearn → PyMC → 階層モデル](chapter-14.md) | ✅ 執筆済 |
| 第15章 | [統計/ML 特有の失敗パターン](chapter-15.md) | ✅ 執筆済 |
| 第16章 | [組織展開と終章 — Skill 共有・監査ログ・次巻への道しるべ](chapter-16.md) | ✅ 執筆済 |

### 付録

| 付録 | タイトル | 状態 |
|---|---|---|
| 付録A | [統計/ML Skill テンプレート集](appendix-a.md) | ✅ 執筆済 |
| 付録B | [Scikit-learn / PyMC チートシートと PyMC↔Stan 対応表](appendix-b.md) | ✅ 執筆済 |
| 付録C | [統計/ML トラブルシューティングと階層データの実データ候補](appendix-c.md) | ✅ 執筆済 |

### 章構成ドラフト

- [章構成ドラフト（v0.3）](chapter-outline.md)：本書全体のスコープ、版履歴、vol-01 との関係、扱わないことの明示

## 読み進め方

- **vol-01 未読の方**: 第1章 → 第2章 → 第4章 → 第5章 → 第6章（Pillar 1 に進む最短経路）
- **Pillar 1（sklearn Skill）を完成させたい**: 第1〜9章を順に（特に第5章の Skill 設計原則、第8章のリーク検知は飛ばさない）
- **Pillar 2（PyMC Skill）を完成させたい**: 第9〜10章 →（並行して）第5章の Skill 設計 → 第13章の MCMC 診断
- **Advanced Capstone（階層モデル）に挑む**: 第10〜13章。特に第12章（反復測定・ロット差・測定誤差）は付録C の実データ候補と組み合わせて演習
- **自分の現場へ展開したい**: 第15章（失敗パターン）→ 第16章（組織展開）→ 付録A（テンプレート）

## Skill / リポジトリ配置規約

本書の Skill ディレクトリ構造は **vol-01 付録A §A.2.1 と同一**です（`skills/<name>/SKILL.md`, `references/`, `scripts/`, `tests/`, `examples/`）。vol-02 で追加される要素（Skill 配下 `artifacts/`・`figures/`、リポジトリルートの合成データ・生成スクリプト、SemVer 付き成果物ファイル名など）は**付録A §A.1.1** に一次リファレンスとしてまとめています。

## 関連リポジトリ・データセット

- [vol-01 リポジトリ](../vol-01/index.md)（本書の前提）
- [PyMC](https://www.pymc.io/) / [ArviZ](https://python.arviz.org/) / [NumPyro](https://num.pyro.ai/)
- [Scikit-learn](https://scikit-learn.org/)
- [MatBench](https://matbench.materialsproject.org/)（表形式・物性予測ベンチマーク）
- [RRUFF](https://rruff.info/)（Raman スペクトルデータベース）
- [Stan](https://mc-stan.org/)（付録B の PyMC↔Stan 対応表参照用のみ）
