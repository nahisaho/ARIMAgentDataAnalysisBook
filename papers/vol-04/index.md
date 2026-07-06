# AI エージェント時代の因果推論・実験計画入門 — ARIM データで動く Agentic Skill

vol-01「AI エージェント時代のデータ分析入門」、vol-02「AI エージェント時代の統計・機械学習分析入門」、vol-03「AI エージェント時代の深層学習分析入門」の続編。vol-01〜03 が **「観測データから予測を作る」** ラインを整えたのに対し、vol-04 は **「観測データから "なぜ" と "もし" を主張する」** ラインを Skill 化する実践書です。ARIM データポータルの実験データ（前処理条件・装置設定・オペレータ差）を主戦場に、**「どの条件が効いたか（因果推論）」「反実仮想的にこの条件を変えていたらどうなったか」「次にどの介入をすべきか（実験計画）」** をエージェントが Skill として提示し、**Human-in-the-loop で介入の因果的妥当性を承認する**までを扱います。

- **対象読者**: ARIM データポータル会員のデータ分析者（材料・ナノテク研究者。Python/Jupyter 経験あり）。vol-01 + vol-02 完読を推奨、vol-03 は必須ではないが Ch8 で深層特徴を CATE に接続する節あり
- **最終ゴール（合格ライン）**:
  - **Pillar 1**: **観測データからの因果効果推定 Skill**（材料実験の前処理条件・装置設定・オペレータ差のいずれかについて）を 1 つ以上作れる。DAG が明示され、identification 戦略が provenance に残り、**unmeasured confounder への感度分析** まで含む
  - **Pillar 2**: **古典的実験計画 (DoE) を Skill 化**したもの（要因計画・分割区画・応答曲面・タグチメソッド・one-shot Bayesian DoE のいずれか）を 1 つ以上作れる。**randomization と blocking が provenance に残る**
  - **Advanced Capstone**: 上記 2 本を統合した **「観測データで因果構造を仮説化 → DoE で介入検証 → 反実仮想シミュレーションで最適条件を提案」の複合 Skill** を、ARIM 風合成実験データで完成させる
- **標準環境**（vol-02/03 に追加）: `dowhy`, `econml`, `causal-learn`, `causalpy`, `pgmpy`, `pyDOE2`, `smt`, `scikit-uplift`, `linearmodels`, PyMC（vol-02 継承）, GraphViz。**CI は CPU で完結**を必須要件

> [!NOTE]
> 逐次 Bayesian Optimization（sequential BO）と active learning は vol-05 に完全移譲します。本書の Ch12 は **one-shot Bayesian DoE（事前分布 → 実験計画一括生成 → 事後解析）** に scope 限定し、逐次化への発展は「橋渡し節」で vol-05 に接続します。強化学習・LLM ファインチューニング・生成モデルはスコープ外。

> [!TIP]
> データセット方針は **ARIM 風合成実験データ**（真の DAG / ATE / CATE を持つ）を主軸に、Materials Project / AFLOW / NOMAD を対比・スケール確認に位置付けます。付録 C に 6 データ型（スペクトル・時系列・画像・回折・表形式・マルチモーダル）の合成データ生成スクリプトと ARIM 匿名化ガイドを収録しています。

> [!IMPORTANT]
> vol-04 の provenance は vol-02/03 拡張に加え、**因果 × Agentic 特有フィールド** を追加します：`dag_of_record_uri` / `dag_of_record_sha256`（旧名 `causal_graph_*` から統一）、`identification_strategy`（backdoor / frontdoor / iv / did / rdd / synthetic_control / g_formula / scm）、`estimand_type` + `mediation_role`、`declared_required_tests`（Ch9 §9.7.1 canonical enum `ch09_v0_3` — 10 tests）、`sensitivity_analysis`（E-value / Rosenbaum bounds + `effect_direction` + `ci_bound_closest_to_null`）、**`authorization_gates` の 4 階層**（L1: dag_authorization / L2: variable_selection_authorization / L3: intervention_execution_authorization / L4: facility_standard_promotion）、**`counterfactual_scope_gate`**（4 判定：Mahalanobis / variance / kNN density / support envelope）、**`experimental_design_provenance`**（DoE、randomization seed、blocking factors、`numpy.random.default_rng` canonical library）、**`audit_manifest_v1`**（Ch14 §14.4 で 19 checks に統合：causal 6 + doe 4 + agentic 9）。canonical スキーマは **Ch4 §4.9** と **Ch15 §15.1.1** に一次リファレンスとしてまとめています。

## 目次

> **執筆進捗**: 本編 **16 / 16 章 完了 ✅**、付録 **4 / 4 完了 ✅**

### 第0章：前提の再確認

| 章 | タイトル | 状態 |
|---|---|---|
| 第0章 | [vol-01 / vol-02 / vol-03 の最小復習](chapter-00.md) | ✅ 執筆済 |

### 第I部　なぜ「因果 × Agentic Skill」なのか

| 章 | タイトル | 状態 |
|---|---|---|
| 第1章 | [vol-01〜03 の Skill に何が足りないのか — 予測から "なぜ" と "もし" へ](chapter-01.md) | ✅ 執筆済 |
| 第2章 | [ARIM データで因果を主張するときの Agentic 特有の課題](chapter-02.md) | ✅ 執筆済 |
| 第3章 | [因果推論と実験計画のライブラリ地図 — Agentic 使い分け](chapter-03.md) | ✅ 執筆済 |
| 第4章 | [因果 × Agentic Skill の設計原則](chapter-04.md) | ✅ 執筆済 |

### 第II部　因果推論の Skill 化

| 章 | タイトル | 状態 |
|---|---|---|
| 第5章 | [DAG と識別戦略 — Skill 化する前に決めること（SCM と反実仮想の骨格を含む）](chapter-05.md) | ✅ 執筆済 |
| 第6章 | [介入効果の推定を Skill 化する（前半：Propensity / IPW / DR）](chapter-06.md) | ✅ 執筆済 |
| 第7章 | [介入効果の推定を Skill 化する（後半：DiD / IV / Synthetic Control）](chapter-07.md) | ✅ 執筆済 |
| 第8章 | [Heterogeneous Treatment Effect と CATE — 「誰に効くか」の Skill 化（g-formula と外挿範囲を含む）](chapter-08.md) | ✅ 執筆済 |
| 第9章 | [感度分析と Refutation — 因果主張の "検算" を Skill 化](chapter-09.md) | ✅ 執筆済 |

### 第III部　実験計画の Skill 化

| 章 | タイトル | 状態 |
|---|---|---|
| 第10章 | [古典的実験計画の Skill 化 — Full factorial / Fractional factorial / Randomization / Blocking](chapter-10.md) | ✅ 執筆済 |
| 第11章 | [応答曲面法とタグチメソッドの Skill 化](chapter-11.md) | ✅ 執筆済 |
| 第12章 | [Bayesian 実験計画（one-shot）— 事前分布と情報利得で Skill 化](chapter-12.md) | ✅ 執筆済 |

### 第IV部　統合と運用

| 章 | タイトル | 状態 |
|---|---|---|
| 第13章 | [総合ハンズオン (Advanced Capstone) — 観測 DAG 仮説化 → DoE 検証 → 反実仮想で最適条件提案](chapter-13.md) | ✅ 執筆済 |
| 第14章 | [因果 × Agentic 特有の失敗パターンと監査（audit_manifest_v1 19 checks）](chapter-14.md) | ✅ 執筆済 |
| 第15章 | [組織展開と終章 — 因果的主張の責任分担・実験計画の組織運用・次巻への道しるべ](chapter-15.md) | ✅ 執筆済 |

### 付録

| 付録 | タイトル | 状態 |
|---|---|---|
| 付録A | [因果 × Agentic Skill テンプレート集 + 介入承認フロー worked example](appendix-a.md) | ✅ 執筆済 |
| 付録B | [因果推論・DoE 固有の MCP パターン](appendix-b.md) | ✅ 執筆済 |
| 付録C | [演習データセット（合成 causal data + ARIM 匿名化ガイド）](appendix-c.md) | ✅ 執筆済 |
| 付録D | [因果推論用語集](appendix-d.md) | ✅ 執筆済 |

### 章構成ドラフト

- [章構成ドラフト（v0.2）](chapter-outline.md)：本書全体のスコープ、版履歴、vol-01〜03 との関係、扱わないことの明示、rubber-duck review 反映履歴

## 読み進め方

- **vol-01/02 未読の方**: 第0章 → 第1章 → 第4章 →（Pillar 選択）へ。第0章は 15 ページの最小復習で、Skill の 6 要素・Human-in-the-loop・階層モデル・不確かさ Gate の基礎を素早く再導入します
- **Pillar 1（因果効果推定 Skill）を完成させたい**: 第1〜9章を順に。特に第4章（authorization_gates L1-L4）、第5章（DAG と SCM）、第9章（refutation_gate 10 tests）は飛ばさない。第6章は Propensity / IPW / DR、第7章は DiD / IV / Synthetic Control、第8章は CATE と g-formula
- **Pillar 2（DoE Skill）を完成させたい**: 第10〜12章 →（並行して）第4章の Skill 設計。特に第10章（randomization seed の 4-stage detection、blocking）と第12章（one-shot Bayesian DoE、EIG）は熟読
- **反実仮想シミュレーションに挑む**: 第5章 §5.7（SCM）→ 第8章（g-formula と外挿範囲）→ 第9章 §9.6（scope_gate 再検証）→ 第13章 Phase 3
- **Advanced Capstone（複合 Skill）に挑む**: 第9章 → 第12章 → 第13章の順。第13章は ARIM 風合成階層データを使い、観測 DAG → DoE → 反実仮想を 1 つの Skill に統合
- **監査を設計したい**: 第14章（audit_manifest_v1 の 19 checks）→ 付録B（MCP handler 実装）→ 第15章（組織責任と registry）
- **自分の現場へ展開したい**: 第14章（失敗パターン + 監査 registry）→ 第15章（責任分担）→ 付録A（Skill テンプレート）→ 付録C（合成データと ARIM 匿名化）

## Skill / リポジトリ配置規約

本書の Skill ディレクトリ構造は **vol-01 付録A §A.2.1 と同一**（`skills/<name>/SKILL.md`, `references/`, `scripts/`, `tests/`, `examples/`）。vol-04 で追加される要素（因果 provenance、DoE provenance、`authorization_gates` L1-L4、`refutation_gate` `ch09_v0_3` enum、`counterfactual_scope_gate` の 4 判定、`audit_manifest_v1` の 19 checks）は **Ch4 §4.9 と Ch15 §15.1.1** に一次リファレンスとしてまとめています。canonical fatal 名レジストリと bare-form ↔ `Ch{n}.` prefix 形式のマッピング規約は **Ch14 §14.6** を正本とします。付録 A の `prohibited_actions` は **概念的グルーピング** として読み、実装 handler は home chapter の canonical bare-form fatal を使用してください（付録 A 冒頭の解釈規約 note 参照）。

## 関連リポジトリ・データセット

- [vol-01 リポジトリ](../vol-01/index.md) / [vol-02 リポジトリ](../vol-02/index.md) / [vol-03 リポジトリ](../vol-03/index.md)（本書の前提）
- **因果推論**: [DoWhy](https://www.pywhy.org/dowhy/) / [EconML](https://econml.azurewebsites.net/) / [causal-learn](https://causal-learn.readthedocs.io/) / [CausalPy](https://causalpy.readthedocs.io/) / [pgmpy](https://pgmpy.org/)
- **DoE / 応答曲面**: [pyDOE2](https://github.com/clicumu/pyDOE2) / [SMT (Surrogate Modeling Toolbox)](https://smt.readthedocs.io/) / [OApackage](https://oapackage.readthedocs.io/)
- **Bayesian**: [PyMC](https://www.pymc.io/) / [ArviZ](https://python.arviz.org/)（vol-02 継承）
- **推定・応用**: [scikit-uplift](https://www.uplift-modeling.com/) / [linearmodels](https://bashtage.github.io/linearmodels/) / [sensemakr](https://cran.r-project.org/package=sensemakr)（Cinelli-Hazlett robustness）
- **可視化**: [GraphViz](https://graphviz.org/) / [NetworkX](https://networkx.org/)
- **MCP**: [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- **公開データ**: [Materials Project](https://materialsproject.org/) / [AFLOW](https://aflowlib.org/) / [NOMAD](https://nomad-lab.eu/) / [MatBench](https://matbench.materialsproject.org/)
- **書籍**: Judea Pearl "The Book of Why" / Hernán & Robins "Causal Inference: What If" / Montgomery "Design and Analysis of Experiments" / Wu & Hamada "Experiments: Planning, Analysis, and Optimization"
