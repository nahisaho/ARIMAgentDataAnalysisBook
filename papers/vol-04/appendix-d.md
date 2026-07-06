# 付録 D：因果推論用語集

> [!NOTE]
> **本付録の位置付け**：Vol-04 で用いる因果推論用語を **1 行定義 + SoT 章 cross-reference** で示す。詳細は各章本編を参照。用語は英語アルファベット順。

---

## D.1 識別・調整（Identification / Adjustment）

**Adjustment set / Adjustment 集合**
: Backdoor path をすべてブロックする観測変数の集合。DAG に対して pgmpy で自動導出可能。→ Ch5 §5.6

**Backdoor path / バックドア経路**
: Treatment T → Outcome Y の因果経路以外で、confounder を経由する経路。ブロックすると identifiability が回復する。→ Ch5 §5.2

**Backdoor criterion / バックドア基準**
: 「Adjustment set が backdoor path をすべてブロックし、かつ Y の子孫を含まない」条件。Pearl (1995)。→ Ch5 §5.3

**Butterfly bias / バタフライバイアス**
: M-bias + confounding が同時に発生し、adjustment が bias を **増やす** ケース。X の adjustment 可否を DAG から判断すべき例。→ Ch5 §5.5

**Collider / 合流点**
: 二つ以上の親を持つノード。**adjustment すると bias を生む**（selection bias の源）。→ Ch5 §5.4

**Confounder / 交絡変数**
: T と Y の共通原因。adjustment 必須。→ Ch5 §5.2

**do-calculus**
: Pearl の 3 rules による intervention distribution `P(Y|do(T=t))` の identifiability 判定 formal system。→ Ch5 §5.7

**Frontdoor criterion / フロントドア基準**
: Unobserved confounder があっても、treatment → mediator → outcome の完全 mediator が観測可能なら identify 可能。→ Ch5 §5.7

**Identifiability / 識別可能性**
: 因果効果が観測分布と DAG assumption から一意に求まる性質。unidentifiable なら Skill は L1 gate で fail-close。→ Ch4 §4.6.1

**M-bias / M バイアス**
: X が collider（親：U1 → X ← U2）で、U1/U2 が unobserved の場合、X の adjustment が bias を生む。→ Ch5 §5.5

**Mediator / 媒介変数**
: T の効果を Y に伝える中間変数。total effect と direct effect の区別に重要。→ Ch5 §5.4

**Positivity / 陽性性**
: すべての covariate 値で `0 < P(T=1|X) < 1` が成立する条件。violated だと ATE 推定不能。→ Ch7 §7.4

---

## D.2 効果推定量（Estimands）

**ATE (Average Treatment Effect) / 平均処置効果**
: 母集団全体の平均因果効果 `E[Y(1) - Y(0)]`。canonical estimand_type。→ Ch7 §7.2

**ATT (Average Treatment on the Treated) / 処置群における処置効果**
: 実際に処置を受けた個体の平均効果 `E[Y(1) - Y(0) | T=1]`。DiD の標準推定量。→ Ch7 §7.3

**CATE (Conditional Average Treatment Effect) / 条件付き平均処置効果**
: 共変量条件付きの効果 `τ(X) = E[Y(1) - Y(0) | X]`。DR-Learner / X-Learner 等で推定。→ Ch8 §8.2

**HTE (Heterogeneous Treatment Effect) / 異質処置効果**
: CATE の同義語。集団間で異なる効果を示す状況全般。→ Ch8 §8.1

**ITE (Individual Treatment Effect) / 個体処置効果**
: 個体レベルの効果 `Y_i(1) - Y_i(0)`。counterfactual の一方が欠測なため直接推定不能。→ Ch8 §8.6

**LATE (Local Average Treatment Effect) / 局所平均処置効果**
: Complier（instrument に反応する部分集団）における ATE。IV 推定の標準推定量。→ Ch7 §7.6

---

## D.3 特殊 identification 戦略

**DiD (Difference-in-Differences) / 差の差分法**
: 処置群と対照群の pre/post 差分を比較。**Parallel trend assumption** 必須。→ Ch7 §7.3

**Front-door adjustment / フロントドア調整**
: Unobserved confounder に対して、完全 mediator 経由で identify する手法。→ Ch5 §5.7

**Instrumental Variable (IV) / 操作変数**
: T と correlated だが Y に直接影響しない変数 Z。unobserved confounder を bypass。**Exclusion restriction** + **relevance** 必須。→ Ch7 §7.6

**Parallel trends assumption / 平行トレンド仮定**
: DiD の core assumption。処置群と対照群の time trend が処置がなければ同じ。placebo test で検証。→ Ch7 §7.3

**Propensity score / 傾向スコア**
: `e(X) = P(T=1|X)`。Matching / IPW / DR-Learner の中核。→ Ch7 §7.4

**RDD (Regression Discontinuity Design) / 回帰不連続デザイン**
: cutoff 前後の局所比較で ATE を identify。→ Ch7 §7.7

**SCM (Synthetic Control Method) / 合成対照法**
: 少数処置ユニットに対し、対照群の重み付き合成で counterfactual を作る。→ Ch7 §7.5

**Two-stage least squares (2SLS)**
: IV 推定の標準実装。First stage で T を Z で説明、second stage で予測値を Y の説明変数に。→ Ch7 §7.6

---

## D.4 反実仮想（Counterfactual）

**Counterfactual / 反実仮想**
: 「もし T=t だったら Y はどうだったか」を問う仮想結果 `Y(t)`。因果推論の中心概念。→ Ch3 §3.2

**Counterfactual scope gate**
: reasoning が counterfactual であることを明示する gate。エージェントが correlational な reasoning を counterfactual claim として提示するのを防ぐ。→ Ch4 §4.6.5

**do-operator / do-作用素**
: intervention を表す `do(T=t)`。conditioning `P(Y|T=t)` とは異なる。→ Ch3 §3.3

**g-formula**
: time-varying treatment に対する counterfactual expectation の recursive formula。longitudinal setting の canonical。→ Ch9 §9.4

**Potential outcome / 潜在結果**
: Rubin の counterfactual framework。個体 i の `Y_i(0), Y_i(1)` は同時に観測できない（fundamental problem of causal inference）。→ Ch3 §3.2

**SCM (Structural Causal Model) / 構造的因果モデル**
: DAG + structural equations で counterfactual を計算可能にする framework（Pearl）。→ Ch9 §9.3

**SUTVA (Stable Unit Treatment Value Assumption)**
: (1) no interference between units, (2) consistency（treatment version 一意）。violation 時は effect estimand が定義不能。→ Ch3 §3.6

---

## D.5 感度分析・refutation

**Data subset refuter**
: データの subset で estimate が変わらないかを検証する refutation test。→ Ch9 §9.6

**declared_required_tests**
: refutation_gate で pass 必須と事前登録した test の canonical list。post-hoc 追加禁止。→ Ch9 §9.7.1

**E-value**
: Observed effect を説明する unmeasured confounding の最小 strength。VanderWeele & Ding (2017)。→ Ch9 §9.5

**Manski bound / Manski 境界**
: assumption を最小化した partial identification bound。ATE の point identify なしで range を得る。→ Ch9 §9.5

**Placebo test / プラセボ検定**
: 実際には effect のない outcome / period で estimate が 0 になるか検証。→ Ch9 §9.6

**Random common cause refuter**
: DAG に random noise variable を追加して estimate の robustness を検証。→ Ch9 §9.6

**Refutation gate**
: `declared_required_tests` のすべての test が pass しないと downstream release を封じる canonical gate。`aggregate_status ∈ {pass, partial_diagnostic_only, fail}`。→ Ch9 §9.7.1

**Rosenbaum bounds / ローゼンバウム境界**
: Matching estimate に対する unobserved confounding sensitivity の formal bound。→ Ch9 §9.5

**Sensitivity analysis / 感度分析**
: assumption violation に対する estimate の robustness を定量化する分析全般。→ Ch9 §9.5

---

## D.6 DoE 用語（Vol-04 Part III）

**Blocking / ブロック化**
: 制御可能な nuisance factor を block として扱い、within-block randomization を実施。→ Ch10 §10.4

**Box-Behnken design**
: 3-level fractional design、center point + edge midpoint、CCD より少ない runs。→ Ch10 §10.7

**Central Composite Design (CCD)**
: Response surface method の canonical design、factorial + axial + center points。→ Ch10 §10.7, Ch11 §11.4

**Design of Experiments (DoE) / 実験計画法**
: 実験の割付・順序・繰返しを事前に systematic に設計する統計手法群。→ Ch10 §10.1

**Full factorial**
: 全水準組合せを実施する完全実施計画。因子数増加で runs 爆発。→ Ch10 §10.6

**Fractional factorial**
: 全組合せの部分を選ぶ実施計画。resolution で交絡構造を管理。→ Ch10 §10.6

**Latin Hypercube Sampling (LHS)**
: 各因子軸を N 区間に等分し、1 sample ずつ配置する space-filling design。Bayesian DoE の initial design 定番。→ Ch12 §12.3

**Resolution (III / IV / V) / 分解能**
: Fractional design での交絡パターン。V が main + 2-way interaction を分離可、III は main 効果同士は分離不可。→ Ch10 §10.6

**Response surface / 応答曲面**
: 目的変数 Y の因子空間での連続関数近似。CCD + quadratic model が典型。→ Ch11 §11.4

**Taguchi method / タグチ法**
: Signal-to-noise ratio を最大化する robust design 手法。inner/outer array で noise factor に頑健な条件探索。→ Ch11 §11.6

**assignment_log / 割付記録**
: 実際の実験順序・seed・design hash の canonical log。4-stage detection の基盤。→ Ch10 §10.5.3

**randomization seed pinning / 乱数シード固定**
: canonical library `numpy.random.default_rng` の seed を pre-register し、post-hoc 変更を fatal 化。→ Ch10 §10.5.3

---

## D.7 Bayesian DoE 用語（Vol-04 Ch12）

**Bayesian optimization / ベイズ最適化**
: 事前分布 + acquisition function（EI / UCB / PI）で sequential に optimum を探索。→ Ch12 §12.4

**Expected Improvement (EI) / 期待改善量**
: Acquisition function の一つ、`E[max(y_best - y, 0)]`。exploration + exploitation バランス。→ Ch12 §12.4

**Gaussian Process (GP) / ガウス過程**
: Response surface の non-parametric Bayesian modeling。kernel で smoothness を表現。→ Ch12 §12.3

**Prior data alignment / 事前分布−データ整合性**
: 事前分布と観測データの整合を prior_predictive_check で検証する canonical step。post-hoc に prior を change するのを禁じる。→ Ch12 §12.7.2

**Prior predictive check**
: Prior + likelihood から生成される予測分布が domain 常識に整合するかを事前検証。→ Ch12 §12.2.3

**Upper Confidence Bound (UCB)**
: `μ(x) + κ σ(x)`。exploration を強調する acquisition function。→ Ch12 §12.4

---

## D.8 承認・監査用語（Vol-04 Part I, Ch13, Ch14）

**Agent action log**
: エージェントの全 gate attempt を append-only で記録する canonical log。bypass 検出の基盤。→ Ch13 §13.3

**Audit manifest v1**
: L4 施設標準昇格 gate の pre-condition として実行する 19 checks の canonical container。→ Ch14 §14.4

**Authorization gates (L1 / L2 / L3 / L4)**
: canonical 4 層承認：`L1_dag_authorization` / `L2_variable_selection_authorization` / `L3_intervention_execution_authorization` / `L4_facility_standard_promotion`。→ Ch4 §4.6.1

**Canonical schema**
: Ch4-Ch15 で確定した Skill 契約の正規スキーマ。Ch15 §15.1.1 にクイックリファレンス。→ Ch4 §4.9

**Egress channel / エグレスチャネル**
: L3 承認前に intervention recommendation が漏出しうる通信経路（Slack / メール / API 等）。canonical `authorized_broadcast_targets` で宣言。→ Ch14 §14.3.6

**Evidence chain sha256**
: 17 fields（Ch13 §13.4.4）を RFC 8785 canonical JSON 化し SHA-256 で決定論的に計算した audit key。→ Ch13 §13.4.4

**Fail-close**
: canonical fatal 検知時に Skill 実行を停止し estimate release を封じる default policy。→ Ch4 §4.6.4

**Fatal**
: 承認 gate の bypass / silent modification を検出した際に throw する canonical error 名。`Ch{n}.{fatal_name}` 形式で参照。→ Ch4 §4.8

**Preregistration manifest**
: DAG / adjustment set / declared_required_tests / seed 等を **事前登録** した immutable artifact。→ Ch4 §4.5

**Provenance URI / SHA-256**
: 承認証跡 artifact への reference。verify handler で hash 一致を確認。→ Ch13 §13.4

**RFC 8785 canonical JSON**
: JSON Canonicalization Scheme。key sort + number normalize + escape 統一で byte-exact 再現可能な JSON representation。→ Ch13 §13.4.4

---

## D.9 参考文献・SoT 章早見表

| 用語カテゴリ | SoT 章 |
|:---|:---|
| DAG / adjustment | Ch5 |
| do-calculus / SCM | Ch5, Ch9 |
| ATE / ATT / IPW / matching | Ch7 |
| CATE / DR-Learner | Ch8 |
| Refutation / sensitivity | Ch9 |
| DoE 基礎 | Ch10 |
| Taguchi / response surface | Ch11 |
| Bayesian DoE | Ch12 |
| Provenance / evidence chain | Ch13 |
| Audit manifest / L4 gate | Ch14 |
| Canonical schema quick ref | Ch15 §15.1.1 |
| Skill テンプレート | 付録 A |
| MCP handler パターン | 付録 B |
| データセット | 付録 C |

---

## 章末チェックリスト

- [ ] 用語 1 つにつき 1 行定義 + SoT 章参照が付いている
- [ ] canonical schema 名は Ch15 §15.1.1 クイックリファレンスと一致
- [ ] fatal 名は `Ch{n}.{fatal_name}` 命名規約（Ch4 §4.8 S-2）に準拠
- [ ] 承認 gate 名は `L{n}_{name}` 完全形（Ch4 §4.6.1）
- [ ] 用語が Vol-05 / Vol-06 で redefine される場合は「Vol-05 で拡張」のマーク（該当なし、本 vol-04 完結）
