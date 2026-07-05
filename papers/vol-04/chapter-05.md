# 第5章 DAG と識別戦略 — Skill 化する前に決めること（SCM と反実仮想の骨格を含む）

> **本章の到達目標**
> - **DAG（Directed Acyclic Graph）の 5 つの基本要素**——confounder / mediator / collider / M-bias / butterfly bias——を材料実験の文脈で識別できる
> - **backdoor 基準・frontdoor 基準**によって、観測データから ATE を identifiable にする adjustment set を選べる
> - **IV（操作変数）・DiD（差分の差分）・RDD（回帰不連続）**の**骨格**（前提と識別戦略）を、ARIM の実例で語れる
> - **SCM（Structural Causal Model）と反実仮想の骨格**（Pearl の 3 階層——観測 / 介入 / 反実仮想）を材料実験の文脈で理解し、**Ch13b Phase 3** への橋渡しをつくる
> - **エージェントが DAG を提案する Skill と Human が承認する Skill を分離**する設計を書き下せる
> - **causal-learn による探索型 DAG** と **pgmpy によるベイジアンネット表現**を、それぞれ「どの Skill で / どの権限で」使うかを判断できる
>
> **本章で扱わないこと**
> - Skill 契約テンプレート（第4章）
> - 具体的 estimator の実装（第6-8章：Propensity / IPW / DR / CATE / DiD / IV / Synthetic Control）
> - refutation の実装（第9章）
> - DoE 設計行列の生成（第10-11章）
> - 逐次実験計画・BO acquisition（vol-05）

---

## 5.1 なぜ DAG を Skill 化の**前**に決めるのか

第2章 §2.3 で見たとおり、**DAG は「データが生成された物語」の視覚表現**です。ATE / CATE / DiD / IV いずれの推定においても、**DAG が変われば identifiable な estimand も、必要な adjustment set も、正しい estimator も変わります**。したがって：

> **DAG を pin してから Skill を書く**——これが vol-04 の一貫した設計原則です。

第4章 §4.4 で導入した「因果 provenance 3 レイヤ」の**L1（identification-level）と L2（identification assumptions）は、本質的にすべて DAG を土台に成立**しています。**Skill を書き始める前に、DAG と識別戦略を Human が確定させる**——これがなければ、`dag_authorization` ゲートも `variable_selection_authorization` ゲートも意味を持ちません。

**エージェントに DAG そのものを勝手に書かせるのは危険**です（第2章 §2.5 Pattern 1）。したがって本章では、**エージェントに何を任せて、Human が何を握るか**を DAG 操作のレベルで分離します。

### 本章の 4 部構成

| 部 | 節 | 主題 |
|---|---|---|
| 第1部 | 5.2, 5.3 | **DAG 記法**（confounder / mediator / collider / M-bias / butterfly bias）と **backdoor / frontdoor 基準** |
| 第2部 | 5.4 | **IV / DiD / RDD の骨格**（骨格のみ；実装は Ch6-7） |
| 第3部 | 5.5 | **SCM と反実仮想の骨格**（Pearl の 3 階層。Ch13b Phase 3 の土台） |
| 第4部 | 5.6, 5.7 | **DAG を提案する Skill と承認する Skill の分離**、**causal-learn / pgmpy の使い分け** |

---

## 5.2 DAG 記法の 5 要素（confounder / mediator / collider / M-bias / butterfly bias）

DAG の**ノード**は変数、**有向エッジ**は「因果の向き」を表します。**因果推論で最低限区別すべきパターンは 5 つ**です。

### 5.2.1 Confounder（交絡因子）

**定義**：処置 $T$ と結果 $Y$ の**両方に因果的な矢を持つ**変数 $C$。

```mermaid
graph LR
    C[C: 交絡] --> T[T: 処置]
    C --> Y[Y: 結果]
    T --> Y
```

- **未調整だと**：$T \to Y$ の効果推定にバイアスが乗る（**交絡バイアス**）
- **調整方法**：backdoor 基準を満たす集合に $C$ を含めて条件付ける（回帰・matching・IPW など）
- **ARIM 例**：装置差、オペレータ差、試料バッチ、環境温湿度

### 5.2.2 Mediator（媒介変数）

**定義**：処置 $T$ の**結果として**生じ、そこから $Y$ に影響する変数 $M$。

```mermaid
graph LR
    T[T: 処置] --> M[M: 媒介]
    M --> Y[Y: 結果]
    T --> Y
```

- **調整すべきでない**：$M$ を confounder のように調整すると、**直接効果しか見えなくなり総効果を過小評価**します（**over-adjustment bias**）
- **frontdoor 基準では逆に $M$ を使う**（§5.3.2）
- **ARIM 例**：熱処理条件が結晶構造を経て特性に影響する場合の「結晶構造」

### 5.2.3 Collider（合流因子）

**定義**：処置 $T$ と結果 $Y$（またはそれらに関連する変数）の**両方から矢を受ける**変数 $S$。

```mermaid
graph LR
    T[T: 処置] --> S[S: 合流]
    Y[Y: 結果] --> S
    T --> Y
```

- **調整すべきでない**：$S$ を条件付けると、**もともと独立だった $T$ と $Y$ の間に見せかけの相関**が生まれる（**collider bias / selection bias**）
- **ARIM 例**：「装置測定に成功したサンプルだけ」で分析する場合の "測定成功フラグ"（処置と結果の両方の影響を受ける）

### 5.2.4 M-bias

**定義**：$T$ の**前**に発生した変数 $Z$ が、**未観測交絡** $U_1, U_2$ を介して $T$ と $Y$ の共通原因になっている構造。

```mermaid
graph LR
    U1[U1: 未観測] --> Z[Z]
    U2[U2: 未観測] --> Z
    U1 --> T[T]
    U2 --> Y[Y]
    T --> Y
```

- **一見 confounder に見える $Z$ を調整すると**：$Z$ は $U_1, U_2$ の collider なので**逆にバイアスが増える**
- **教訓**：**「$T$ より前の変数だから調整して良い」は誤り**。DAG の構造を見なければ判断できない
- **ARIM 例**：試料の産地（過去情報）が装置選定と結果の両方に、**未観測の研究方針**を介して影響している場合

### 5.2.5 Butterfly bias（M-bias と confounding の同時併存）

**定義**：同じ変数 $Z$ が、**confounder の役割**と **collider の役割**を同時に持つ構造（M と confounding が重なった蝶型）。

```mermaid
graph LR
    U1[U1: 未観測] --> Z[Z]
    U2[U2: 未観測] --> Z
    Z --> T[T]
    Z --> Y[Y]
    U1 --> T
    U2 --> Y
    T --> Y
```

- **調整しても・しなくても**バイアスが残る難所
- **対処**：**IV や DiD など別の identification 戦略**へ切替、または**ドメイン知識で $U_1, U_2$ の proxy を追加観測**する
- **ARIM 例**：装置状態が processing 条件選択と最終特性の両方に影響し、装置状態は未観測の較正履歴（$U_1$）と経年劣化（$U_2$）を介する場合

> [!IMPORTANT]
> **5 パターンの見分けは、変数の名前ではなく DAG の局所構造で決まります**。同じ「温度」でも、confounder のこともあれば mediator のこともあり、collider のこともある——これが**第2章 §2.5 Pattern 1（DAG rewriting）が危険**な理由です。**DAG は Human が pin し、エージェントは提案までに留める**（§5.6）。

---

## 5.3 backdoor 基準と frontdoor 基準 — identifiability の骨格

DAG が pin されたら、次に「どの変数集合で条件付ければ ATE が identifiable になるか」を決めます。**Pearl の do-calculus** の基礎となる **2 つの基準**を骨格として押さえます。

### 5.3.1 backdoor 基準

**backdoor path**：$T$ から $Y$ への**因果的でない**（arrows into $T$ を含む）経路。

**backdoor 基準**：ある集合 $Z$ が以下を満たせば、$Z$ で条件付けることで $T \to Y$ の因果効果を identifiable にできる：
1. $Z$ は $T$ の子孫を含まない
2. $Z$ は $T$ から $Y$ への**すべての backdoor path**をブロックする

**推定式**（backdoor adjustment formula）：
$$P(Y \mid do(T = t)) = \sum_z P(Y \mid T = t, Z = z) \, P(Z = z)$$

**ARIM 実装イメージ**：
- 処置 $T$ = 熱処理温度、結果 $Y$ = 硬度
- backdoor path：$T \leftarrow \text{装置} \to \text{較正状態} \to Y$
- adjustment set：$Z = \{\text{装置}, \text{較正状態}\}$
- 実装：$Z$ を共変量として回帰（Ch6）、または propensity score にして IPW（Ch6）

### 5.3.2 frontdoor 基準

**backdoor が使えない**（未観測交絡 $U$ が $T$ と $Y$ の両方に影響）ときの救済策。**mediator $M$ を経由**して identification を試みます。

**frontdoor 基準**：mediator $M$ が以下を満たせば identifiable：
1. $M$ は $T$ から $Y$ へのすべての**directed path** を intercept する
2. $T$ から $M$ に backdoor path がない
3. $T$ で条件付ければ $M$ から $Y$ への backdoor がすべてブロックされる

**推定式**（frontdoor adjustment formula）：
$$P(Y \mid do(T = t)) = \sum_m P(M = m \mid T = t) \sum_{t'} P(Y \mid M = m, T = t') P(T = t')$$

**ARIM 実装イメージ**：
- 処置 $T$ = 前駆体組成、結果 $Y$ = 導電率、未観測 $U$ = 装置履歴
- mediator $M$ = 結晶粒径（$T$ の結果、$Y$ の原因、$U$ からは独立）
- $T \to M$ と $M \to Y$（$T$ で条件付き）をそれぞれ推定して composition

> [!NOTE]
> **frontdoor は「mediator が識別を救う唯一の場面」**であり、§5.2.2 の「mediator を confounder として調整するな」とは**逆方向の使い方**です。**Skill 契約では、mediator を "adjustment 用"（禁止）で使うのか "frontdoor 用"（許可）で使うのかを明示**する必要があります（第4章 Table 4.4 item 4）。

### 5.3.3 identifiability の判断表

| 状況 | 使う基準 | 必要な変数 | ARIM 典型例 |
|---|---|---|---|
| すべての交絡が観測可能 | **backdoor** | 交絡集合 $Z$ | 装置・較正状態が記録済み |
| 未観測交絡があり、mediator を経由できる | **frontdoor** | mediator $M$ | 前駆体 → 結晶構造 → 特性 |
| 未観測交絡があり、外生的な shifter がある | **IV**（§5.4.1） | 操作変数 $Z$ | 装置更新（外生ショック） |
| 未観測交絡があり、時間軸で before/after 比較できる | **DiD**（§5.4.2） | 処置群・対照群 × 時期 | プロトコル変更前後 |
| 処置が閾値ルールで決まる | **RDD**（§5.4.3） | running variable $X$、閾値 $c$ | 検査基準値による選別 |
| いずれも困難 | 準実験不可 → **DoE で介入** | ランダム化割当 | Ch10-11 の実験計画 |

---

## 5.4 IV / DiD / RDD の骨格

3 つの準実験手法は、いずれも「observational data で介入効果を identifiable にする」ための戦略です。**実装は Ch7**、本章では**骨格と DAG 表現**のみ扱います。

### 5.4.1 IV（Instrumental Variables、操作変数法）

**骨格**：**未観測交絡 $U$** があっても、**$T$ にだけ影響し $Y$ には直接影響しない**外生変数 $Z$（instrument）を使えば ATE が identifiable。

**3 つの仮定**：
1. **Relevance**：$Z \to T$ は非ゼロ
2. **Exclusion restriction**：$Z \to Y$ の直接効果はない（すべて $T$ 経由）
3. **Exchangeability**：$Z$ と $U$ は独立（$Z$ は "randomizer" のように振る舞う）

**DAG**：

```mermaid
graph LR
    Z[Z: 操作変数] --> T[T: 処置]
    U[U: 未観測交絡] --> T
    U --> Y[Y: 結果]
    T --> Y
```

**ARIM 例**：
- 処置 $T$ = 新プロトコル採用、結果 $Y$ = 生成物純度、未観測 $U$ = オペレータ熟練度
- 操作変数 $Z$ = **装置更新タイミング**（外生的にプロトコル切替を促した外部要因）
- 実装：2SLS（Two-Stage Least Squares）、linearmodels（Ch7）

### 5.4.2 DiD（Difference-in-Differences、差分の差分）

**骨格**：処置群と対照群の**時間トレンドが並行**（parallel trends assumption）なら、処置前後の差分の差分が ATT を identifiable。

**推定量**：
$$\text{ATT} = \{E[Y_{\text{treated}, \text{after}}] - E[Y_{\text{treated}, \text{before}}]\} - \{E[Y_{\text{control}, \text{after}}] - E[Y_{\text{control}, \text{before}}]\}$$

**ARIM 例**：
- 装置 A（処置群、2026 年に較正プロトコル刷新）と装置 B（対照群、変更なし）
- 2025 年（before）と 2026 年後半（after）の測定精度を比較
- **並行トレンド**の可視化と placebo test（Ch9）が必須

### 5.4.3 RDD（Regression Discontinuity Design、回帰不連続デザイン）

**骨格**：処置 $T$ が **running variable $X$ の閾値 $c$ で決定的に切り替わる**とき、$X = c$ 近傍で処置効果を identifiable。

**前提**：閾値近傍で $X$ を能動的に manipulate できない（continuity of potential outcomes）

**ARIM 例**：
- 検査基準：純度 $X < 99.5\%$ なら再処理（$T = 1$）、$X \geq 99.5\%$ ならそのまま（$T = 0$）
- $X = 99.5\%$ 近傍で「再処理された最上位群」と「されなかった最下位群」を比較 → 再処理効果を identifiable

### 5.4.4 3 手法の使い分け

| 手法 | 前提 | 使える ARIM シナリオ | 主要ライブラリ（Ch7） |
|---|---|---|---|
| **IV** | 外生的 shifter が存在 | 装置更新・規制変更・部材品切れ等の外生ショック | linearmodels, DoWhy |
| **DiD** | 並行トレンド | プロトコル変更前後の処置群 vs 対照群 | linearmodels, CausalPy |
| **RDD** | 閾値ルール、閾値近傍の連続性 | 検査基準・スペック基準による選別 | CausalPy, rdrobust |

> [!WARNING]
> **前提が壊れると identification 自体が破綻**します。**Skill 契約に `identification_strategy` を pin し、前提の assessment レポート（`identification_report_uri`）を artifact に含める**——これは第4章 §4.3 で導入した設計です。**エージェントが暗黙に IV → DiD に切り替えるのは fatal（Table 4.4 item 2）**。

---

## 5.5 SCM と反実仮想の骨格（Pearl の 3 階層）— Ch13b Phase 3 への橋渡し

第2章 §2.7 で「反実仮想（"もし処置していなかったら"）」に触れました。**SCM（Structural Causal Model）**は、反実仮想を**数学的に扱う枠組み**です。**Ch13b Phase 3** で SCM ベースの反実仮想シミュレーションを実装するため、ここでは**骨格**（形式的定義の最小限）を導入します。

### 5.5.1 SCM の定義

**SCM** は 3 つの組で定義される：
$$\mathcal{M} = \langle U, V, F \rangle$$

- $U$：**exogenous variables**（外生変数、モデル外の要因）
- $V$：**endogenous variables**（内生変数、モデル内の観測変数）
- $F$：$V$ の各要素 $V_i$ を、$U$ と $V \setminus \{V_i\}$ の関数で表す**構造方程式**の集合 $\{f_1, f_2, \ldots\}$

**ARIM 材料実験の SCM 骨格（例）**：
- $U = \{u_{\text{装置履歴}}, u_{\text{環境ノイズ}}\}$
- $V = \{T = \text{熱処理温度}, C = \text{装置}, M = \text{結晶粒径}, Y = \text{硬度}\}$
- $F$：
  - $C = f_C(u_{\text{装置履歴}})$
  - $T = f_T(C, u_{\text{環境ノイズ}})$
  - $M = f_M(T, C)$
  - $Y = f_Y(M, C, u_{\text{環境ノイズ}})$

**DAG は $F$ から自動的に生成**されます（親変数 → 子変数のエッジ）。

### 5.5.2 Pearl の因果階層（3 rungs of causation）

**SCM は 3 つの階層の問いに答えられる**：

| Rung | 問い | 表現 | ARIM 例 |
|---|---|---|---|
| **1. Association**（観測） | 「もし〜を観測したら？」 | $P(Y \mid T)$ | 高温処理サンプルの硬度分布 |
| **2. Intervention**（介入） | 「もし〜を意図的に設定したら？」 | $P(Y \mid do(T))$ | 温度を強制的に $T=800°C$ にしたら硬度は？ |
| **3. Counterfactual**（反実仮想） | 「もし過去の $T$ を変えていたら？」 | $P(Y_{T=t'} \mid T=t, Y=y)$ | このサンプル（実際は $T=600°C$ で硬度 $y$ だった）が、もし $T=800°C$ だったら硬度は？ |

**重要な区別**：
- **Rung 2 = ATE / CATE**：**集団平均**の反実仮想（Ch6-8）
- **Rung 3 = 個別反実仮想**：**個別サンプル**についての反事実、SCM が必要（Ch13b）
- **Rung 3 は Rung 2 より強い仮定を要求**（すべての $f_i$ と $U$ の分布の同定）

### 5.5.3 反実仮想の 3 ステップ計算

Pearl の**abduction–action–prediction**：

1. **Abduction**：観測値 $T = t, Y = y$ から exogenous $U$ の事後分布 $P(U \mid T=t, Y=y)$ を推定
2. **Action**：構造方程式で $T \leftarrow t'$ に強制置換（$T$ の親を切断）
3. **Prediction**：更新した SCM で $Y$ の分布 $P(Y_{T=t'} \mid T=t, Y=y)$ を計算

**Ch13b Phase 3 での実装**：
- SCM を PyMC / DoWhy で表現
- Abduction は事後分布サンプリング
- Action は intervention (`do` 演算)
- Prediction は forward simulation
- **その結果を `counterfactual_scope_gate` で判定**（Ch8 §8.5 の operational 実装、第4章 §4.5 の契約定義）

> [!IMPORTANT]
> **反実仮想は "external validity" の問題を最も強く受ける**——観測データにない領域（新規組成・未経験プロセス）についての反実仮想は、**構造方程式の外挿**に依存します。**第4章 §4.5 の `counterfactual_scope_gate` は、この外挿を Human review にゲートする**ためにあります。**Ch13b で SCM を扱う際、必ずこのゲートを通す**——これが vol-04 の運用ルールです。

### 5.5.4 SCM を書くための最小仕様（Ch13b 用テンプレート）

```yaml
# scm_specification.yaml (Ch13b で使用)
scm:
  exogenous_variables:
    - name: u_device_history
      distribution: normal(0, sigma_device)
    - name: u_environmental
      distribution: normal(0, sigma_env)
  endogenous_variables:
    - name: C_device
      parents: [u_device_history]
      structural_function: linear   # または nonlinear / gp / bnn
    - name: T_temperature
      parents: [C_device, u_environmental]
      structural_function: nonlinear
    - name: M_grain_size
      parents: [T_temperature, C_device]
      structural_function: gp   # ガウス過程（Ch11）
    - name: Y_hardness
      parents: [M_grain_size, C_device, u_environmental]
      structural_function: nonlinear
  identification_dag_ref: "arxiv:xxx/DAG_v3.mmd"
  identification_dag_sha256: "0xabc..."
```

> [!NOTE]
> **SCM の各 $f_i$ を "どのモデルで表現するか"**——線形回帰・GP・BNN——は**モデリング判断**であり、その選択自体が provenance に残ります。Ch13b では PyMC で $f_i$ をベイズモデルとして書き、posterior に基づき反実仮想サンプリングします。

---

## 5.6 DAG を提案する Skill と Human が承認する Skill の分離

第4章 §4.6 で導入した**3 層承認ゲート**の第1層 `dag_authorization` は、**DAG の変更・確定は Human 承認**でした。これを **Skill レベルで 2 種類に分離**します。

### 5.6.1 Skill 型 1：`dag_proposal_skill`（エージェント自律）

**目的**：観測データからの DAG **候補**を提案する
**権限**：エージェント自律実行可能
**成果物**：DAG 候補ファイル（`.mmd` / `.gml`）、根拠レポート

```yaml
skill_type: dag_proposal_skill
authorization_level: agent_autonomous
inputs:
  - dataset_uri
  - domain_prior_uri            # ドメイン知識制約（例：「装置 → 温度」の向き固定）
outputs:
  - dag_candidate_uri
  - dag_candidate_sha256
  - proposal_report_uri         # なぜこの DAG を提案したかの説明
  - search_algorithm_provenance # causal-learn の PC / GES / LiNGAM 等
prohibited_actions:
  - overwrite_existing_dag_of_record   # 記録済み DAG を書き換えない
```

### 5.6.2 Skill 型 2：`dag_approval_skill`（Human 承認、Skill 化された workflow）

**目的**：DAG 候補を審査し、`dag_of_record` に昇格させる
**権限**：Human 承認必須（`dag_authorization` ゲート）
**成果物**：正式版 DAG、承認証跡

```yaml
skill_type: dag_approval_skill
authorization_level: human_required
authorization_gate: dag_authorization
inputs:
  - dag_candidate_uri
  - dag_candidate_sha256
  - proposal_report_uri
  - reviewer_comments
outputs:
  - dag_of_record_uri
  - dag_of_record_sha256
  - approval_evidence:
      - reviewer_signatures
      - approval_timestamp
      - approved_dag_version
      - dissenting_opinions   # 少数意見も記録
approver: causal_review_board
approver_independence:
  conflict_policy: independent_reviewer_required_if_approver_is_data_producer
```

### 5.6.3 2 種類の分離が生む便益

| 便益 | 説明 |
|---|---|
| **監査可能性** | 「エージェントが提案したもの」と「Human が承認したもの」が別 artifact として残る |
| **反復可能性** | DAG 候補を複数出させて Human が選ぶ workflow が明確 |
| **責任分担** | 提案の質はエージェント（+ アルゴリズム）、採用の責任は Human |
| **失敗の localization** | 誤った DAG が採用された場合、原因は "提案" と "承認" のどちらか特定できる |

> [!TIP]
> 第2章 §2.5 Pattern 1（DAG rewriting）は、**提案 Skill と承認 Skill を分けずに 1 つの Skill で "改訂＋採用" まで行っていた**ことに由来します。**Skill 型を分ければ、そもそも "silent rewriting" は Skill 契約違反として検知**できます。

---

## 5.7 causal-learn と pgmpy の使い分け

DAG 提案 Skill と、承認済み DAG の運用 Skill——それぞれに適したライブラリが異なります。

### 5.7.1 causal-learn — 探索型 DAG 学習

**位置づけ**：観測データから DAG **候補**を統計的に探索するライブラリ

**主要アルゴリズム**：
- **PC アルゴリズム**：条件付き独立性テストで edge を削除、方向づけ
- **GES（Greedy Equivalence Search）**：BIC などのスコア関数で greedy 探索
- **LiNGAM**：線形・非ガウス性で方向を identifiable にする
- **FCI**：未観測交絡があっても部分的な方向づけ

**Skill 対応**：`dag_proposal_skill` の実装ライブラリ

**Skill 契約に含めるべき項目**：
```yaml
search_algorithm_provenance:
  algorithm: pc | ges | lingam | fci
  ci_test: fisherz | chisq | gsq   # PC の場合
  score_function: bic | bdeu       # GES の場合
  alpha: 0.05                       # 有意水準
  domain_prior_edges:               # 固定エッジ（Human 制約）
    - [device, temperature]
```

> [!WARNING]
> **causal-learn の出力は "候補" であって "真の DAG" ではありません**。統計的独立性は同値な DAG クラス（Markov equivalence class）までしか区別できない場合が多く、**方向づけには domain prior または介入実験が必要**です。**Skill 契約で "候補" と明示し、Human 承認ゲートを必ず通す**。

### 5.7.2 pgmpy — ベイジアンネットワーク表現と推論

**位置づけ**：**承認済み DAG** の上で確率推論（条件付き分布・介入・観測整合）を行うライブラリ

**主要機能**：
- ベイジアンネットワークとしての DAG 表現（`BayesianNetwork`）
- CPD（Conditional Probability Distribution）の学習と保持
- variable elimination・belief propagation による推論
- **`do` operator の実装**（介入分布の計算）

**Skill 対応**：`dag_of_record` を運用する Skill、Ch13b の SCM Skill

**Skill 契約に含めるべき項目**：
```yaml
runtime_representation:
  library: pgmpy
  network_uri: "s3://arim-causal/dag_of_record_v3.pkl"
  network_sha256: "0xdef..."
  cpd_source: learned_from_data | expert_specified | hybrid
```

### 5.7.3 使い分け早見表

| 目的 | ライブラリ | Skill 型 | 権限 |
|---|---|---|---|
| DAG **候補**を観測データから探索 | causal-learn | `dag_proposal_skill` | エージェント自律 |
| DAG の **domain prior** を pin する | pgmpy（BayesianNetwork） | `dag_of_record_skill` | Human 承認済み |
| 承認済み DAG で **条件付き / 介入分布**を計算 | pgmpy | 各 estimator Skill（Ch6-8） | エージェント自律（DAG 変更なし） |
| **SCM ベース反実仮想** を実行 | pgmpy + PyMC | Ch13b SCM Skill | エージェント自律 + scope gate |
| **数値推定** の estimator を実装 | DoWhy / EconML / CausalPy / linearmodels | Ch6-8 Skill | エージェント自律（DAG 変更なし） |

> [!NOTE]
> **causal-learn と pgmpy は "上流（DAG 提案）と下流（DAG 運用）" の分業**です。causal-learn を運用に使ったり、pgmpy で DAG を勝手に書き換えたりすることは**Skill 契約違反**（Table 4.4 item 1 に相当）。

---

## まとめ — 本章のチェックリスト

- [ ] **§5.2 DAG 5 要素**（confounder / mediator / collider / M-bias / butterfly bias）を自分のデータで指摘できる
- [ ] **§5.3 backdoor / frontdoor 基準**の違いと、mediator の "adjustment 禁止 vs frontdoor 用" 使い分けを理解した
- [ ] **§5.4 IV / DiD / RDD の骨格**（前提と DAG）を、ARIM 実例で 1 分で語れる
- [ ] **§5.5 Pearl の 3 階層**（観測 / 介入 / 反実仮想）を区別し、SCM が Rung 3 に必要なことを理解した
- [ ] **§5.5.3 反実仮想の abduction–action–prediction 3 ステップ**を書き下せる
- [ ] **§5.6 DAG 提案 Skill と DAG 承認 Skill の分離**を、自分の workflow に適用できる
- [ ] **§5.7 causal-learn と pgmpy の使い分け**を、Skill 型と権限で説明できる

---

## 章末演習

### 演習 5.1：自分のデータの DAG を描く

自分の ARIM データ（または演習用合成データ、付録 C）について、以下を実施：

- [ ] 変数を 5-10 個列挙し、それぞれを 5 パターン（confounder / mediator / collider / M-bias / butterfly bias）のいずれかに分類
- [ ] Mermaid 記法で DAG を描く
- [ ] backdoor 基準を満たす adjustment set を最低 1 つ特定
- [ ] frontdoor 基準が使える mediator が存在するか判定
- [ ] IV / DiD / RDD のいずれかが適用可能な場面があるか検討

**提出物**：`dag_v1.mmd` + adjustment set 一覧 + 根拠 500 字

### 演習 5.2：識別戦略の判断表

以下のシナリオそれぞれで、**どの identification 戦略を使うか**を根拠付きで答えてください：

1. 装置 A と装置 B で同じ試料バッチを測定、装置 A のみ較正プロトコルを更新した
2. 新しい前駆体組成が導電率に与える効果を、未観測の "原料ロット差" があるなかで推定したい
3. 検査で純度 99% 未満のサンプルだけ再処理される。再処理の効果を評価したい
4. すべての交絡因子が観測されているが、mediator を経由した効果と直接効果を分離したい

（解答は Ch7-9 で照合）

### 演習 5.3：SCM の設計（Ch13b の準備）

演習 5.1 の DAG を SCM として書き下してください：

- [ ] $U$（外生変数）を最低 2 つ列挙、各分布を仮定
- [ ] $V$（内生変数）と親変数を明記
- [ ] 各 $f_i$ に想定される model family（linear / GP / BNN 等）を記す
- [ ] Rung 2（介入）と Rung 3（個別反実仮想）で聞きたい問いを 1 つずつ書く
- [ ] **`counterfactual_scope_gate` の閾値（distance / density / variance）を仮設定**（第4章 §4.5 のテンプレート、Ch8 で operational 化）

### 演習 5.4：DAG 提案 Skill と DAG 承認 Skill の分離

自分の workflow に対して、§5.6.1-5.6.2 のテンプレートを埋めた **2 つの Skill 契約**を作成してください：

- [ ] `dag_proposal_skill.yaml`（causal-learn 前提）
- [ ] `dag_approval_skill.yaml`（`dag_authorization` gate、`approver_independence` policy 含む）

---

## 参考資料

### 内部 cross-reference

- 第2章：ARIM 特有の DAG 課題（装置差・オペレータ差・少データ）、Pattern 1（DAG rewriting）
- 第3章：causal-learn と pgmpy の位置づけ、権限マップ
- **第4章**：`dag_authorization` / `variable_selection_authorization` ゲート、`counterfactual_scope_gate` の契約定義、Table 4.4 の禁止事項
- 第6-8章：backdoor（Ch6）、frontdoor（Ch6-7）、IV/DiD（Ch7）、CATE と g-formula（Ch8）の実装
- 第9章：identification 前提の refutation テスト
- **第13章 (13b Phase 3)**：SCM ベース反実仮想シミュレーション、本章 §5.5 の実装先
- 付録 A：DAG 提案 Skill、SCM Skill、g-formula Skill テンプレート
- 付録 D：DAG 記法用語集（confounder / mediator / collider / M-bias / butterfly bias / positivity / SUTVA / SCM / do-calculus / backdoor / frontdoor / IV / DiD / RDD）

### 外部参考文献

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. — SCM と do-calculus の原典
- Pearl, J., & Mackenzie, D. (2018). *The Book of Why*. Basic Books. — 因果階層の入門
- Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference*. MIT Press. — SCM の数学的基礎
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. — backdoor / IV / g-methods の教科書
- causal-learn documentation: https://causal-learn.readthedocs.io/
- pgmpy documentation: https://pgmpy.org/
- DoWhy documentation: https://www.pywhy.org/dowhy/

---

**本章の位置づけ**：DAG と identification 戦略、SCM の骨格を pin することで、Ch6-8 の estimator 実装が「どの DAG の上で / どの identification 戦略で / どの外挿範囲で」動くかが明確になります。次章（Ch6）では、backdoor 基準を満たす adjustment set が確定した観測データに対し、**Propensity Score / IPW / DR** による ATE / ATT 推定を Skill 化していきます。
