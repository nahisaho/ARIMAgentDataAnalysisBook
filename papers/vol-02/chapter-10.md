# 第10章　不確かさ入門：頻度論の限界と Bayesian への橋

> [!IMPORTANT]
> **本章の到達目標**
>
> - **信頼区間（CI）と信用区間（CrI）**の意味の違いを、材料研究の文脈で説明できる
> - **事前分布**が何を表現するか、なぜ材料科学で活きるかを言語化できる
> - **頻度論の不確かさ表現の限界**（サンプリング不確かさのみ）を、Bayesian の全確率表現と対比できる
> - Skill 仕様書の `uncertainty_scheme` フィールドで、頻度論・Bayesian を明示的に選択できる
> - なぜ第11章以降で PyMC を導入するか、その動機を自分の言葉で言える

## 本章で扱わないこと

- **PyMC の具体的なコード**：概念に絞る。実装は第11章から
- **MCMC 診断**：第11章で扱う
- **階層モデル**：第12章で扱う
- **完全な Bayesian vs 頻度論の哲学論争**：本書は実務書。両者を「使い分けの道具」として扱う
- **仮説検定の詳細**：p 値の意味は第9章 §9.7 で触れた。以降も検定は主題にしない

---

## 10.1 なぜ「不確かさ」を Skill 化するか

材料研究で「予測値 = 3.42 eV」と単独で報告する Skill は、実務では**使い物になりません**。次のような判断ができないからです：

| 判断 | 単一値では不可能 | 不確かさ付きなら可能 |
|---|---|---|
| 実験計画：候補材料を追加合成するか | ❌ | 「±0.5 eV なら合成再現の閾値を超えるので採用」 |
| リスク評価：新規プロセスを本格導入するか | ❌ | 「95% CrI が閾値内なら Go」 |
| 追実験の要否 | ❌ | 「CI 幅がまだ大きいので追試」 |
| モデル間の比較 | 「A の方が良い」しか言えない | 「A と B の予測差は CI 内で有意でない」 |

**不確かさ表現は Skill の副産物ではなく、Skill の主成果物**です。

### 第9章 §9.7 からの続き

第9章で **効果量 + 95% CI + 物理的閾値**の 3 点セットを解釈レポートに書くルールを導入しました。ここでの CI は頻度論的 bootstrap による「サンプリング不確かさ」でしたが、材料研究では以下のような**別種の不確かさ**もあります：

| 不確かさの種類 | 例 | 頻度論 CI で扱えるか |
|---|---|---|
| サンプリング不確かさ | データが有限であることによる | ✅ bootstrap で扱える |
| モデル構造の不確かさ | RF vs GBM vs Ridge のどれが正しいか | ⚠️ ネスト CV で部分的 |
| ハイパーパラメータの不確かさ | 最適とされる λ でも本当に最適か | ⚠️ ネスト CV で部分的 |
| **事前知識の不確かさ** | 「bandgap は 0-10 eV 範囲」という前提の強さ | ❌ 事前分布として明示 |
| **観測誤差の不確かさ** | 装置の測定精度、ラボ間差 | ⚠️ 追加モデリング必要 |
| **系統誤差の不確かさ** | 装置校正のズレ | ❌ 事前分布として明示 |

Bayesian は**これらすべてを「確率分布」として統一的に表現**できます。**頻度論でも**混合効果モデル・測定誤差モデル・sandwich SE・bootstrap 等で多くの不確かさを扱えますが、その都度**別の道具立て**が必要で、**未知パラメータそのものに確率分布を割り当てない**点で表現が非対称になります [[9-4]](#ref-9-4) [[9-5]](#ref-9-5)。

---

## 10.2 信頼区間（CI）と信用区間（CrI）の違い

**この違いを説明できるかで、Bayesian 移行の第一歩が決まります**。

### 定義

| 用語 | 略号 | 意味 |
|---|---|---|
| **信頼区間**（Confidence Interval, CI） | 頻度論 | 「同じ実験を無限回繰り返したら、その 95% で真値を含む区間の作り方」 |
| **信用区間**（Credible Interval, CrI） | Bayesian | 「観測データを踏まえて、真値がこの範囲にある確率が 95%」 |

**両者は同じ「[3.1, 3.7] eV」でも意味が違います**：

- CI（95%）：「95% の**手続き**が真値を含む区間を作る」。個別の区間が真値を含む確率は 0 か 1（既に決まっているが分からない）
- CrI（95%）：「真値がこの区間にある**確率**が 95%」

「95% CI は真値を含む確率が 95%」という説明は**頻度論としては誤り**で、この誤解を体系的に整理した文献が [[9-4]](#ref-9-4) です。Skill 実装ではこの誤解を助長しない解釈テンプレを自動出力します。

### 材料研究者にとってどちらが自然か

日常会話・実験計画では、**CrI の解釈の方が自然**です：

- ✅「この試料の bandgap は 95% の確率で 3.1〜3.7 eV の間にある」
- ❌「同じ実験を無限回繰り返したら 95% で 3.1〜3.7 eV を含む」

しかし論文の慣習・査読者の期待は多くの場合 CI 側です。Skill 実装では**どちらを出すかを仕様書で明示**します：

```yaml
uncertainty_scheme:
  type:                credible_interval   # or confidence_interval
  level:               0.95
  interval_method:     hdi                 # hdi (HPD) or equal_tailed
  posterior_quantity:  parameter           # parameter | prediction | difference | threshold_probability
  interpretation_note: "posterior probability given the model, prior, likelihood, and data"
  reporting_convention: "match domain expectation (journal, reviewer)"
```

> [!NOTE]
> **CrI の型を明示**：等尾区間（equal-tailed, 2.5%〜97.5% 分位点）と HDI（最狭区間）は、事後分布が歪んでいる・多峰な場合に**別の区間を返します**。仕様書で明示しないと、実装者ごとにブレます。
>
> **推論対象の明示**：パラメータの不確かさ（`parameter`）と、新規試料の予測不確かさ（`prediction` = posterior predictive interval）は**別物**です。材料判断では後者が必要になる場面が多い（第11章、第12章）。

### 頻度論と Bayesian で「同じ数値」になる場合

**尤度が正規で、平均のみを推定し、既知分散の下で平坦事前分布を置く**等の限定的条件では、古典的 CI と Bayesian CrI が数値的に一致することがあります [[9-4]](#ref-9-4)。ただし条件は一般に狭く（未知分散・非正規尤度・非線形パラメータ変換では成り立ちません）、**意味は依然として異なる**ため、Skill のドキュメントは常に区別して書きます。

> [!WARNING]
> **エージェントが「CI と CrI は同じ」と言ってきたら訂正**してください。数値は近くても、**書き方・解釈・意思決定の枠組み**が異なります。Skill 仕様書 ⑤ の禁止事項に「CI と CrI を混同する表現」を入れます。

---

## 10.3 事前分布：何を、いつ、どう書くか

### 事前分布は「情報の入り口」

Bayesian の中核概念は **事前分布 × 尤度 ∝ 事後分布**：

$$
p(\theta \mid \text{data}) \propto p(\text{data} \mid \theta) \; p(\theta)
$$

- **事前分布** $p(\theta)$：観測前の知識
- **尤度** $p(\text{data} \mid \theta)$：観測がこのパラメータで生じる確率
- **事後分布** $p(\theta \mid \text{data})$：観測後の知識

**「事前分布を書く」とは、材料研究の既知情報をコード化する行為**です。頻度論はこの入り口を持ちません。

### 材料研究で書きたくなる事前分布の例

| 対象 | 既知情報 | 事前分布の例 |
|---|---|---|
| bandgap（半導体） | 一般に 0〜10 eV、多くは 1〜4 eV | `TruncatedNormal(mu=2.5, sigma=2.0, lower=0, upper=10)`（HalfNormal だと mode=0 になり物性の実感と合わない） |
| Raman ピーク位置 | 装置分解能 ±2 cm⁻¹、既知フォノンモード付近 | `Normal(mu=known_mode, sigma=5)` |
| 反応収率 | 0〜1 の範囲、経験的に多くは 0.3〜0.8 | `Beta(alpha=3, beta=3)` |
| 装置校正定数 | メーカー公称値 ±0.5%、経年で ±2% | 相対係数として `calibration_factor ~ Normal(1.0, 0.02)` を置き、`measured = nominal × factor` と乗法的にモデル化（`Normal(mu=nominal, sigma=nominal*0.02)` は nominal≈0 で破綻） |
| 階層の分散成分 | 装置間ばらつきは装置内より大きい傾向 | `HalfNormal(sigma=large)` on inter, `HalfNormal(sigma=small)` on intra — ただし**構造的に大小を強制しない**。強制したい場合は ordered/ratio parameterization。通常は prior predictive check で妥当性を確認 |

これらは**材料科学者が普段の議論で暗黙に前提としている情報**です。Bayesian はこれを明示的に書きます。

### 事前分布の 3 段階

| 段階 | 名称 | 意味 |
|---|---|---|
| 1 | **無情報事前分布**（uniform, improper） | 「何も知らない」を装う。事後は尤度に近くなる |
| 2 | **弱情報事前分布**（weakly informative） | 「常識的な範囲」を制約する。数値的安定性のためにも推奨 |
| 3 | **情報事前分布**（informative） | 過去実験・文献値・物理制約を強く反映 |

**実務のデフォルトは段階 2（弱情報事前分布）**です。段階 1 は「特殊な理論的正当化がある場合のみ」、段階 3 は「事前情報を明示的に検証する場合のみ」使います（第11章）。

> [!WARNING]
> **「無情報事前分布」は真に無情報ではない**：uniform / improper prior は**パラメータ変換に対して不変ではなく**（対数変換すれば flat でなくなる）、事後分布を improper にすることがあります。Stan/Gelman 系の実務では **「無情報」より「regularizing weakly informative prior」を推奨**しています [[9-6]](#ref-9-6) [[9-5]](#ref-9-5)。理論的正当化なく uniform / flat を選ぶのは避けてください。

### 事前分布を書く際のリスク

| リスク | 例 | 予防 |
|---|---|---|
| **事前分布による結論の暴走** | 強い事前分布で観測を無視した結論 | prior predictive check（第11章） |
| **事前分布の後付け** | 事後を見てから prior を変える | 仕様書 ⑤ で禁止、prior を先に凍結（`data_cutoff_for_prior` を記録） |
| **事前分布のハルシネーション** | エージェントが「その分野の慣例」と誤情報 | 文献引用・PI レビュー必須 |
| **確認バイアスの潜入** | 期待する結果を反映した prior | Sensitivity analysis で複数 prior を比較（第11章の感度分析節） |
| **尤度・パラメータ変換の文脈欠落** | 対数変換や標準化後に flat を仮定 | `parameter_scale` と `support` を仕様書に明記 [[9-5]](#ref-9-5) |

### 事前分布の Skill 仕様書エントリ

```yaml
prior_specification:
  parameter:        bandgap_offset
  distribution:     Normal
  hyperparameters:  {mu: 0.0, sigma: 0.5}
  units:            eV
  parameter_scale:  original         # original | log | standardized
  support:          [-3.0, 3.0]      # 物理制約に基づく定義域
  likelihood_context: |
    観測モデルは Normal(y | mu(theta), sigma_obs)、
    sigma_obs も推定対象。prior はこの尤度前提での意味を持つ。
  justification:    |
    合成再現性の実測分散 (±0.3 eV, 過去 5 年 30 試料) を
    やや保守的に拡張した弱情報事前分布。
    参考: 内部レポート XX (2024)
  literature_ref:              [ref-lab-report-2024]
  justification_document:      docs/priors/bandgap_offset.md
  data_cutoff_for_prior:       2025-12-31   # この日以降のデータは prior 決定に使っていない
  prior_predictive_check_required: true
  sensitivity_alternatives:
    - {distribution: Normal, sigma: 1.0, label: weaker}
    - {distribution: Normal, sigma: 0.2, label: stronger}
  reviewed_by:      "PI name"
  review_date:      2026-XX-XX
```

**`sensitivity_alternatives`** を書く理由：後段の「事前分布への感度分析」（第11章）で必要になるため、Skill 設計段階で「代替候補」を凍結しておきます。`data_cutoff_for_prior` は、事後を見てから prior を修正する「後付け汚染」の防止線です。

---

## 10.4 なぜ材料科学で Bayesian が活きるか

材料科学が Bayesian と特に相性が良い理由：

1. **サンプルサイズが小さい**：合成コスト・装置時間の制約で n=10〜100 が普通。この領域では**漸近近似に頼る頻度論 CI や naive bootstrap CI が不安定になりやすく**、Bayesian は事前情報で情報量を補える（ただし prior sensitivity の管理が前提）
2. **事前知識が豊富**：物性物理・化学の理論、過去実験の再現性、装置仕様など、書ける prior が多い
3. **階層構造がある**：試料内→試料間→ロット間→装置間→ラボ間。partial pooling が自然（第12章）
4. **観測誤差モデルが必要**：測定誤差・装置校正のばらつきを明示的に扱う場面が多い
5. **意思決定と接続する必要**：「この材料を大量合成すべきか」は事後確率で答える方が自然

### vol-01 のデータ契約との接続

vol-01 第7章の 6 要素 Skill 仕様書のうち、Bayesian が主に強化するのは **④成功条件** と **⑥再現性条件** です：

- ④成功条件：「点推定 + CI が閾値内」→「事後分布の 95% CrI が閾値内」
- ⑥再現性条件：`sampler_config` / `posterior_artifact` / `diagnostics_summary` を追加（付録 A）

第5章 §5.3 の `task_type` に `bayesian_inference` を追加した理由もここにあります。**成功条件の書き方が根本的に変わる**ため、独立したタスクタイプとして扱います。

---

## 10.5 頻度論と Bayesian の使い分け

### 使い分けの意思決定表

| 状況 | 推奨 | 理由 |
|---|---|---|
| データ量が大量、事前情報が乏しい | 頻度論 | Bayesian の計算コストを払う価値が小さい |
| データが小さい（n < 50）、事前情報あり | **Bayesian** | 事前分布で情報を補える |
| 階層構造がある（試料 × 装置 × ロット） | **Bayesian** | partial pooling が自然 |
| 論文が既存の頻度論慣習に従う分野 | 頻度論 or 両方 | 査読対応 |
| 意思決定と直結する（合成 Go/No-Go） | **Bayesian** | 事後確率で判断しやすい |
| **標準規格・認証・監査に従う必要**（ASTM/JIS/ISO 等） | 頻度論 or 規格指定手法 | 規格が手法を指定するため選択の余地なし |
| 事後の逐次更新がある（オンライン、iid/交換可能性が成立） | **Bayesian** | Bayes 更新が自然（後述の注意） |
| 装置校正・観測誤差モデルが必要 | **Bayesian** | 誤差の階層化が自然 |
| 説明責任がシンプルであるべき | 頻度論 | 事前分布の説明が不要 |

> [!WARNING]
> **「Bayesian 更新 = オンライン学習」ではない**：Bayes の逐次更新は、尤度の因子分解が正しく、データが（仮定した意味で）**iid または交換可能**、かつモデルと prior が固定されている場合に成立します。**分布ドリフト**や非定常性がある実験では、静的 prior→posterior の逐次更新ではなく、**状態空間モデルや動的 Bayesian モデル**が必要です。オンライン SGD による適応学習とも別物です。

### 併用のパターン

**実務では「両方出す」が多い**です：

- 頻度論 CI で査読者の期待に応え、Bayesian CrI で意思決定に使う
- 頻度論で feature importance を出し、Bayesian で予測不確かさを出す
- 頻度論の予測誤差と Bayesian の事後予測分散を比較して、モデル妥当性を診断する

### Skill 仕様書での明示

```yaml
uncertainty_scheme:
  primary:
    type:            credible_interval
    level:           0.95
    interval_method: hdi
  secondary:
    type:   confidence_interval
    level:  0.95
    method: bootstrap
  divergence_check:
    metric:    interval_width_relative_diff   # |width_CrI - width_CI| / width_CI
    threshold: 0.10                            # heuristic, not a validity test
  rationale: |
    意思決定は CrI で行い、論文報告時は CI も併記する。
    数値差が閾値を超える場合は前提の不一致（prior 強すぎ・尤度誤特定・n 小）を診断する。
```

> [!NOTE]
> **10% 閾値の意味**：これは**妥当性の検定ではなく、診断トリガー**です。閾値超過は「原因を調べよ」というシグナルであり、「Bayesian が間違い」でも「頻度論が間違い」でもありません。閾値そのものはドメインで調整してください。

---

## 10.6 エージェント時代の不確かさ表現

エージェントに任せると生じやすい失敗：

| 失敗パターン | 例 | 予防 |
|---|---|---|
| **不確かさの脱落** | 「予測値は 3.42 eV です」で終わる | Skill 出力に CI/CrI を必須化 |
| **CI と CrI の混同** | エージェントが同じものと言う | 仕様書 ⑤ 禁止事項に明記 |
| **CI の解釈間違い** | 「95% で真値がこの区間にある」と CI について述べる | Skill が自動で解釈テンプレを出力 |
| **事前分布のハルシネーション** | 「この分野の慣例では〜」と根拠なく提示 | literature_ref を必須フィールドに |
| **感度分析の省略** | prior を 1 種類だけで結論を出す | sensitivity_alternatives を仕様書 ⑥ に必須化 |
| **不確かさの過小報告** | 事後分布の一部だけ見せる | posterior_artifact 全体を provenance に保存 |
| **収束診断の省略** | $\hat{R}$ / ESS / divergences / tree depth を見ずに CrI を報告 | Skill 出力に diagnostics_summary を必須化（第11章・第13章） |

### エージェントへの依頼テンプレ

- ❌「予測値と信頼区間を出して」→ 曖昧、CI/CrI の指定なし
- ✅「予測値と 95% CrI を出し、CI との数値差 10% 超なら診断対象として記録して」

Skill 仕様書は**エージェントに任せる余地を減らす**方向に書きます。

---

## 10.7 次章への準備

第11章では最小限の PyMC コードを使い、以下を実践します：

1. `pm.Model` の基本構文（`with pm.Model() as model:` のブロック）
2. **prior predictive check**：観測前に「この prior でどんなデータが生成されうるか」を確認
3. **posterior predictive check**：観測後に**事後分布からパラメータをサンプルし**、そのパラメータで新規データを生成して観測と比較（"事後 prior" ではなく **posterior predictive distribution**）
4. **収束診断**（$\hat{R}$、ESS、divergences）の「見る場所」
5. `校正曲線のベイズ版`ハンズオン
6. **NumPyro / JAX への準備**（第12章の階層モデル向け）

第11章の準備として、以下を確認しておきます：

- Python 環境で `pip install pymc arviz` が通ること
- PyMC のバックエンドはデフォルト（PyTensor）でよいこと（NumPyro/JAX は第12章から）
- vol-01 の環境凍結（`pip freeze > requirements.txt`）の習慣を続けること

---

## 10.8 章末ワーク

1. **自分のプロジェクトの Skill 1 つを取り、`uncertainty_scheme` フィールドを追加する**：type / level / primary vs secondary を明示
2. **CI と CrI の意味の違いを、自分のドメインの実験計画例で説明する**（100 字程度）：査読者に説明する想定
3. **事前分布を書きたくなる場面を 3 つリストアップ**：装置校正、階層分散、経験的範囲、など。各々に対して弱情報事前分布の候補を書く
4. **事前分布のハルシネーション検知チェック**：エージェントに事前分布を提案させ、literature_ref を求めた際に根拠が出てくるかテスト
5. **頻度論と Bayesian の使い分け表**を、自分のプロジェクトの Skill 群に対して埋める（§10.5 の意思決定表を援用）

---

## 10.9 本章のまとめ

- **不確かさは Skill の主成果物**。点推定単独の Skill は実務で使えない
- **信頼区間（CI）と信用区間（CrI）は意味が違う**。数値が近くても解釈・意思決定の枠組みが異なる
- **事前分布は既知情報の入り口**。材料科学では書きたい prior が多い
- 材料科学が Bayesian と相性が良い 5 つの理由：小サンプル / 事前知識豊富 / 階層構造 / 観測誤差モデル / 意思決定接続
- **実務は使い分け**：査読対応で頻度論、意思決定で Bayesian、が典型パターン
- エージェント時代の失敗（不確かさ脱落、CI/CrI 混同、prior ハルシネーション）を Skill 仕様書で予防
- 次章から PyMC を最小限で導入。**概念先行、コードは道具**という順序を保つ

---

## 参考資料

### 本書内の該当章

- 第5章 §5.3：`task_type` に `bayesian_inference` を追加した動機
- 第8章 §8.6：ネスト CV（構造・ハイパーパラメータ不確かさの部分的な扱い）
- 第9章 §9.7：頻度論の効果量 CI 算出方法（本章はこの延長）
- 第11章：PyMC 実装・prior/posterior predictive check・MCMC 診断
- 第12章：階層モデル（Bayesian の真骨頂）
- 第15章：Bayesian の失敗パターン（事前分布暴走、収束不良）
- 付録 A：provenance スキーマ拡張（`sampler_config`, `posterior_artifact`, `diagnostics_summary`）

### 外部参考

<a id="ref-9-1">[9-1]</a> Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis*. 3rd ed. CRC Press. — 本章の理論的土台
<a id="ref-9-2">[9-2]</a> McElreath, R. (2020). *Statistical Rethinking: A Bayesian Course with Examples in R and Stan*. 2nd ed. CRC Press. — 概念の直感的解説、事前分布の書き方
<a id="ref-9-3">[9-3]</a> Kruschke, J. K. (2015). *Doing Bayesian Data Analysis*. 2nd ed. Academic Press. — CI vs CrI の対比が丁寧
<a id="ref-9-4">[9-4]</a> Morey, R. D., Hoekstra, R., Rouder, J. N., Lee, M. D., & Wagenmakers, E.-J. (2016). The fallacy of placing confidence in confidence intervals. *Psychonomic Bulletin & Review*, 23(1), 103–123. — CI の誤解を体系的に整理
<a id="ref-9-5">[9-5]</a> Gelman, A., Simpson, D., & Betancourt, M. (2017). The Prior Can Often Only Be Understood in the Context of the Likelihood. *Entropy*, 19(10), 555. — 事前分布と尤度の相互依存
<a id="ref-9-6">[9-6]</a> Stan Development Team. Prior Choice Recommendations. [https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations) — 弱情報事前分布の実務ガイド
<a id="ref-9-7">[9-7]</a> Bzdok, D., Altman, N., & Krzywinski, M. (2018). Statistics versus machine learning. *Nature Methods*, 15(4), 233–234. — 頻度論・Bayesian・ML の使い分け
