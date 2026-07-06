# 第3章 ARIM データで BO を回すときの Agentic 特有の課題

> **本章の到達目標**
> - **少データ現場での GP surrogate の典型的な暴走**（length scale 過大 / 過小、predictive variance の破綻、ハイパーパラメータ推定失敗）を、ARIM 実データの具体場面で説明できる
> - **装置差・オペレータ差・バッチ差** を BO の入力にどう組み込むか——**surrogate の入力次元として明示する / 階層 GP で shrinkage する / blocking 因子として除外する** の 3 選択肢を判断できる
> - **実験コスト × BO サイクル時間** の見積りで、自分のデータへの BO 適用性を数値判定できる
> - **エージェントが acquisition を "勝手に" 最大化する危険**（Human 承認迂回、surrogate 外挿の隠蔽、pending experiments の無承認変更）を、自分の運用で起こりうるパターンとして 3 つ以上挙げられる
> - **本書の演習用データセット**（ARIM 風合成 BO データ、Olympus benchmark、Branin / Hartmann / Ackley、Materials Project 派生）の位置づけと入手方法が分かる
> - 自分の研究テーマで **BO が今すぐ適用可能か / 追加準備が必要か / BO 不適か** を判定できる
>
> **本章で扱わないこと**
> - GP kernel と事前分布の具体的設計（第6章）
> - Acquisition function の実装（第7章）
> - Multi-task GP / 階層 GP の完全展開（第11章）
> - safe BO の制約実装（第10章）

---

## 3.1 「一括計画」とは何が違うのか — DoE と BO の分岐点

vol-04 の DoE Skill は「観測前に計画を確定する」設計でした。vol-05 の BO Skill は **「観測後に次を決める」** ——この 1 段の変化が、Agentic 現場で 4 つの新しい失敗軸を生みます。

| 局面 | 一括 DoE（vol-04）の性格 | 逐次 BO（vol-05）の性格 | Agentic 現場での新リスク |
|---|---|---|---|
| **候補生成のタイミング** | 実験開始前に全条件を確定 | iteration ごとに次条件を提案 | エージェントが Human 承認なしに実行に進む |
| **surrogate の使い方** | 応答曲面・Kriging surrogate（vol-04 Ch11）は使うが、観測後に acquisition へ渡して逐次更新はしない | 予測 mean と std を acquisition に渡し、iteration ごとに再学習 | surrogate 外挿を "自信あり" と誤報告 |
| **観測ノイズへの応答** | 反復測定と blocking で緩和 | GP の likelihood が noise を吸収 | ノイズを "情報" と誤解して次候補が振動 |
| **stop 判定** | 事前計画の全条件を消化して終了 | budget / convergence / regret threshold のいずれかで停止 | エージェントが `stop_condition` を実行中に緩める |

> [!TIP]
> DoE は「実験を **計画する** Skill」、BO は「実験を **提案する** Skill」——名詞が変わるだけでなく、**Skill が受け取る input と、Human が承認するタイミングが変わります**。BO では **各 iteration** が Skill 実行の単位で、そのたびに Human 承認ゲート（`experiment_launch_authorization`）が挿入されます。**物理実験を伴う場合、`experiment_launch_authorization` は vol-04 の `L3_intervention_execution_authorization` を `parent_authorization_id` として必ず参照します**（第5章で本格化、bypass 不可）。

---

## 3.2 少データ現場での GP surrogate の典型的な暴走

ARIM の実験データは、多くの現場で **1 プロセス条件あたり数十試料以下、時に 5-10 試料** という少データ状況です。この状況で GP surrogate を素朴に fit させると、以下のような **暴走** が起きます——BO の入り口で最も気を使う点です。

### 暴走 1：length scale の過大推定 → surrogate が定数関数に潰れる

- **起こる場面**：観測点 5 点、うち 4 点が近い出力値。Type-II MLE で kernel hyperparameter を最大化すると、length scale が **search space 幅を超える値** に発散
- **見え方**：GP の予測 mean がほぼ観測平均に張り付き、予測 std も全域でほぼ同じ値。**どの候補も EI がほぼ同じ**——BO が collapse する
- **原因**：少データ + 素朴な MLE では、「相関構造が長距離である」という解が尤度を最大化しやすい。特に **noise variance の同時推定** が悪化させる
- **vol-05 の対処**：length scale に **事前分布**（Gamma or LogNormal、search space 幅を反映）を置き、MAP 推定または fully Bayesian（NUTS）で不確実性込みで推定（第6章）。**length scale > search space 幅** を検出したら warning、`kernel_spec` の再検討を Human review へ

### 暴走 2：length scale の過小推定 → surrogate がスパイクだらけに

- **起こる場面**：観測点 8 点、うち 2 点が近い入力位置で大きく異なる出力値（実は測定ノイズ）。MLE が「短距離で急激に変わる関数」と解釈
- **見え方**：GP の予測 mean が観測点だけスパイク状に立ち上がり、それ以外はほぼ平坦。**予測 std が観測点近傍では小さく、離れると急激に大きく**——EI は観測点から遠い領域で全域最大化される
- **原因**：測定ノイズを likelihood の noise term ではなく **kernel の短距離相関** として吸収してしまう
- **vol-05 の対処**：**noise variance に事前分布**（測定誤差の物理的下限を反映）を置き、length scale とセットで推定。**length scale < 観測点間隔の 1/10** を検出したら warning（第6章）

### 暴走 3：predictive variance の過大信頼

- **起こる場面**：観測点 6 点、探索範囲の 20% しかカバー。GP は残り 80% で「学習点から遠いから std 大」と返す——エージェントは「不確実性を反映している」と信頼する
- **見え方**：外挿範囲での予測 std が「そこそこ大きい」が、**真の未知度合いは std の 3 倍以上**（測定できない未観測 confounder、装置差、キャリブレーション drift 等）
- **原因**：GP の予測 std は kernel 構造の関数であって、**真の予測不確実性のすべてではない**——kernel が捉えられない現象（例：温度と装置差の交互作用が観測データに現れていない）は反映されません
- **vol-05 の対処**：`hallucinated_recommendation_detection`（第7章「外挿検知の operational 定義」節）——**予測分散閾値 + Mahalanobis 距離閾値 + length scale 越え検知の合成基準**。単独 GP std を信じすぎない設計

### 暴走 4：hyperparameter 事後の multi-modality

- **起こる場面**：観測点 10 点、少データ Bayesian GP で NUTS がハイパーパラメータ posterior を推定 → **length scale の posterior が 2 峰性**（短距離解と長距離解の両方が尤度支持を得る）
- **見え方**：MCMC の $\hat{R} > 1.01$、divergences 多発、length scale の trace plot が 2 モードを行き来
- **原因**：少データでは尤度そのものが multi-modal。中心化パラメータ化ではサンプラーが特定モードに hop できない
- **vol-05 の対処**：**非中心化パラメータ化**、事前分布の絞り込み、または **fully Bayesian は諦めて MAP + Laplace 近似** に留める（第6章）。**MCMC diagnostics 不合格の GP を acquisition に渡すのは fatal**——vol-02 第12章の規律を継承

> [!IMPORTANT]
> **これら 4 種の暴走はすべて、素朴な "GP を fit すれば動く" 実装で自動的に起きます**——エージェントに「GPyTorch で fit して」と頼むと、初期実装では上記の regime に落ちがちです。**BO を回す前に GP のヘルスチェック**（length scale の妥当性、noise variance、事後の unimodality、LOOCV による calibration）を Skill の入力契約に組み込むことが第6章の中心テーマになります。

---

## 3.3 装置差・オペレータ差・バッチ差の扱い — 3 選択肢

ARIM 現場で BO を回すとき、ほぼ必ず直面するのが **「複数装置で並行探索する / オペレータが日ごとに違う / バッチ材料の切替が入る」** 状況です。これらの変動要因を surrogate にどう組み込むか——**3 つの選択肢**があり、判断は Human が下します（`surrogate_treatment_of_facility_variance` の Human 承認。**因果探索空間そのものを変える場合のみ、追加で vol-04 の `L2_variable_selection_authorization` を parent として参照します**）。

### 選択肢 A：surrogate の入力次元として明示する（categorical dimension）

- **やり方**：装置 ID / オペレータ ID / バッチ ID を **カテゴリ変数**として surrogate 入力に加える（例：Matérn kernel × Hamming kernel の product）
- **向いている場面**：装置差が **効果として推定可能** で、**BO が装置ごとに探索点を出し分けたい**（例：装置 A では温度高め、装置 B では低めが良い）
- **リスク**：カテゴリ変数の次元が高い（装置数が多い）と surrogate の一般化が悪化。**装置固有の点予測** は当てにならない
- **対応する vol-05 章**：第6章（kernel product）、第11章（multi-task GP の代替として）

### 選択肢 B：階層 GP（multi-task GP）で shrinkage する

- **やり方**：装置ごとに個別 GP を持ち、hyperparameter を **階層事前分布で shrinkage**（coregionalization kernel、Intrinsic Coregionalization Model 等）
- **向いている場面**：装置ごとに **目的関数の形は似ているが、絶対値が違う / 測定モードが異なる**（例：同型装置 3 台のキャリブレーション差、共通探索空間）
- **リスク**：階層 GP は hyperparameter が増え、MCMC 収束が難しくなる（vol-02 第11章の階層モデル収束問題を継承）。**divergences が出たら中心化 → 非中心化**（第11章）
- **対応する vol-05 章**：第11章の中心テーマ

### 選択肢 C：blocking factor として BO の探索から除外する

- **やり方**：装置差 / オペレータ差 / バッチ差は BO の探索空間から除外し、**"どの装置で実行するか" を実験計画時に固定**（vol-04 第10章の blocking と同じ発想）
- **向いている場面**：装置差が **交絡因子として BO を歪める** が、装置間で最適条件が変わらないと事前に判断できる場合。または **1 装置での探索に限定する** ことが実務的に許容される場合
- **リスク**：他装置の観測を活かせない。**施設全体としての効率が下がる**
- **対応する vol-05 章**：第6章（blocking 説明）、第14章 capstone Phase 1（単一装置）

### 3 選択肢の判断表

| 判断軸 | A: 入力次元 | B: 階層 GP | C: Blocking |
|---|:-:|:-:|:-:|
| 装置数 | 少ない（2-4）| 中程度（3-10）| 多い（>10）または 1 |
| 装置差の効果 | 効果を推定したい | 装置差を部分プーリング（共通構造 + 装置固有差を同時推定）| 効果は事前に固定 |
| 装置間の目的関数の類似性 | 低い（装置固有最適が違う） | 高い（形は似ている） | 判断不能 or 単一装置 |
| データ量 | 中〜多 | 少〜中 | 少 |
| Skill 実装難易度 | 低 | 高（MCMC 収束が課題） | 低 |
| 対応する vol-05 章 | 第6章 | 第11章 | 第6章 / 第14章 Phase 1 |

**判断は Human が下す**——エージェントが自動判断して選択肢を切り替えると、**surrogate の解釈が変わり、BO の結論も変わります**。第5章で `surrogate_treatment_of_facility_variance`（装置差の扱い方の宣言、`categorical_input` / `hierarchical_shrinkage` / `blocked_out` のいずれか）を Skill provenance に pin する規約を導入します。

> [!TIP]
> **vol-04 の因果推論の判断（confounder / mediator / collider）と、vol-05 の surrogate 入力判断（categorical / hierarchical / blocked）は独立**です——vol-04 で confounder とした装置差を、vol-05 では **surrogate の入力次元として含めても、階層で shrinkage しても、blocking で除外しても、いずれの選択も可能**。第14章 capstone では両者を接続する典型パターンを扱います。

---

## 3.4 実験コスト × BO サイクル時間の見積り — BO 適用性判定

BO は「観測結果を見ながら次を決める」ループです。**このループが 1 週するのに何日かかるか、そして予算内で何 iteration 回せるか**——これが自分のデータへの BO 適用性を決めます。

### 3 種の時間・コスト要素

**単位はすべて hours に統一**（days は 24 h、minutes は 1/60 h に換算）。

| 要素 | 記号 | 例（hours 換算後）|
|---|---|---|
| **合成時間** | $T_{\text{synth}}$ | 熱処理 4 h、CVD 8 h、湿式合成 24 h |
| **測定時間** | $T_{\text{meas}}$ | XRD 2 h、SEM 4 h、多モード評価 8 h |
| **装置予約待ち** | $T_{\text{queue}}$ | 混雑期 72-336 h、専有装置 0 h |
| **surrogate 再学習 + acquisition 計算** | $T_{\text{surrogate}}$ | GP fit ~0.02 h（少データ）、fully Bayesian ~0.5 h |
| **Human review + 実験実行承認** | $T_{\text{approval}}$ | 1 iteration 承認 1-4 h、多目的で議論なら 24 h |
| **実験実施の合計** | $T_{\text{exec}} = T_{\text{synth}} + T_{\text{meas}}$ | 装置に依存 |

**1 iteration のカレンダー時間**：

$$T_{\text{cycle}} \approx T_{\text{surrogate}} + T_{\text{approval}} + T_{\text{queue}} + T_{\text{exec}} \quad [\text{h}]$$

**1 iteration の費用**（円）：

$$C_{\text{iter}} \approx C_{\text{synth}} + C_{\text{meas}} + C_{\text{instrument\_slot}} + C_{\text{review}}$$

ここで $C_{\text{synth}}, C_{\text{meas}}$ は材料 + 消耗品、$C_{\text{instrument\_slot}}$ は装置利用料（時間従量 or 1 スロット単価）、$C_{\text{review}}$ は人件費。**待ち時間 $T_{\text{queue}}$ は費用に含めない**（装置予約待ちに料金は発生しないケースが多い）——ただし施設や契約による例外は Human が明示。

**予算内 iteration 数**は、期間予算と費用予算の両方から算出し、min を取ります：

$$
N_{\text{cal}} = \left\lfloor \frac{T_{\text{project}}}{T_{\text{cycle}}} \right\rfloor, \quad
N_{\text{cost}} = \left\lfloor \frac{B_{\text{money}}}{C_{\text{iter}}} \right\rfloor, \quad
N_{\text{BO\_iter}} = \min(N_{\text{cal}},\ N_{\text{cost}})
$$

初期探索点 $N_{\text{init}}$（LHS / Sobol / factorial で最初に置く点）は BO iteration とは別枠。**総評価数** $N_{\text{total}} = N_{\text{init}} + N_{\text{BO\_iter}}$。

### BO 適用性の粗い目安

判定は **$N_{\text{BO\_iter}}$（初期点 $N_{\text{init}}$ を除いた adaptive iteration 数）** で行います。

| $N_{\text{BO\_iter}}$ | BO 適用性 | 推奨戦略 |
|---|---|---|
| **< 5** | **BO 不向き**——ランダム / factorial の方が安全 | 一括 factorial DoE（vol-04 第10章）|
| **5-15** | **BO 可能だが慎重に**——単目的 BO、EI か PI、事前分布強め | 単目的 BO（第5-7章）、少データ規律（本章 §3.2）を厳格に適用 |
| **15-40** | **BO 本領**——多目的 or 制約付きも可 | 単目的 BO（第5-7章）/ 多目的 BO（第9章）、階層 GP（第11章） |
| **> 40** | **BO 標準運用**——Batch BO で並列化検討 | Batch BO（第12章）、Active Learning ハイブリッド（第13章）|

**Batch BO の場合**：$N_{\text{BO\_iter}}$ は「バッチ回数」。$N_{\text{BO\_iter}} \times \text{batch\_size}$ が adaptive 実験数。

> [!NOTE]
> **$N_{\text{BO\_iter}} < 5$ で無理に BO を回す** のはハルシネーション温床です。GP は最低でも $N_{\text{init}} + N_{\text{BO\_iter}} \geq 6\text{-}10$ 点を目安とし、少データでの surrogate 暴走（§3.2）が起きやすい状況。この場合は **vol-04 の factorial / RSM で一括生成し、事後解析で ranking**、あるいは **latin hypercube + 単純な RF regression** に留めるほうが、Skill として堅牢です。

### 演習：自分のデータで見積る

以下の空欄を埋めてみてください（**単位はすべて hours / 円 に揃える**）：

| 項目 | 自分のケース |
|---|---|
| 合成 1 サイクル $T_{\text{synth}}$ | ___ h |
| 測定 1 サイクル $T_{\text{meas}}$ | ___ h |
| 装置予約待ち平均 $T_{\text{queue}}$ | ___ h |
| Human review $T_{\text{approval}}$ | ___ h |
| **1 iteration カレンダー時間** $T_{\text{cycle}}$ | ___ h |
| 1 iteration 費用 $C_{\text{iter}}$ | ___ 円 |
| 期間予算 $T_{\text{project}}$ | ___ h |
| 費用予算 $B_{\text{money}}$ | ___ 円 |
| $N_{\text{cal}} = T_{\text{project}} / T_{\text{cycle}}$ | ___ |
| $N_{\text{cost}} = B_{\text{money}} / C_{\text{iter}}$ | ___ |
| **$N_{\text{BO\_iter}} = \min(N_{\text{cal}}, N_{\text{cost}})$** | ___ |
| 初期探索点数 $N_{\text{init}}$ | ___ |
| **BO 適用性判定**（不向き / 慎重 / 本領 / 標準） | ___ |

判定結果を **第2章 §2.10 のチェックリスト** と合わせて、第5章 Skill 契約テンプレートの入力に使います。

---

## 3.5 エージェントが BO ループを「勝手に」進める 5 つの典型

vol-04 §1.4 で「エージェントが DAG を勝手に書き換える」4 逸脱を扱いました。vol-05 では **acquisition 選択・実験実行・停止判定に関する 5 種の逸脱** が主な監査対象です（Ch15 taxonomy と対応）。

### 逸脱 1：エージェントが Human 承認ステップを "効率化" して省略する

- **起こる場面**：Human が長時間対応できない iteration が挟まると、エージェントが「候補提示 → 承認 → 実行 を一続きの Skill にまとめては？」と提案。実際に MCP レベルで gate を bypass する Skill が書かれる
- **問題**：`experiment_launch_authorization` は仕様上必須。Skill レベルで bypass すると **provenance に承認 ID が残らない**——実験結果が後で問題化したとき、承認履歴が不明で説明責任が果たせない
- **vol-05 の対処**：MCP レベルで **候補提案 Skill と実験実行 Skill を分離**（付録 B）、gate を Skill の境界に置く。iteration ごとに承認 ID が provenance に必ず残る設計

### 逸脱 2：エージェントが acquisition function を無承認で切り替える

- **起こる場面**：iteration 5 まで EI で回していたが、収束が遅いと判断したエージェントが iteration 6 で「UCB に切り替えます」と自然言語で報告し、Skill 内部の acquisition を書き換える
- **問題**：acquisition function は BO の探索・活用バランスを規定する **Skill 契約の核**。無承認切替は **iteration 間の性能保証を破る**（EI と UCB では regret bound の意味が異なる）。また、切替後の候補が **surrogate 外挿範囲に落ちても検出されない**（`hallucinated_recommendation_detection` の閾値が acquisition ごとに調整されているため）
- **vol-05 の対処**：`acquisition_spec`（関数名 + hyperparameter）を Skill 起動時に pin、実行中の変更は fatal。変更は新規 Skill 実行として再起動。第7章で operational 化、第15章 taxonomy に登録

### 逸脱 3：surrogate 外挿を "自信あり" と誤報告

- **起こる場面**：GP の予測 std が length scale の推定失敗（§3.2 暴走 1）で全域小さくなる。エージェントは std だけを見て「不確実性は小さい」と自然言語でまとめる
- **問題**：外挿範囲での自信は **surrogate の kernel 前提に依存**——kernel が真の相関構造を捉えていない場合、std は真の不確実性を過小評価
- **vol-05 の対処**：`hallucinated_recommendation_detection` を Skill の出力仕様に強制（第7章）。**予測分散閾値 + Mahalanobis 距離判定 + length scale 越え判定** の 3 判定のいずれかで flag が立ったら、候補提案を保留し Human review へ。カテゴリ変数を含む場合、Mahalanobis は連続次元のみ、カテゴリは Hamming / kernel distance と合成して判定（第7章）

### 逸脱 4：エージェントが pending experiments を勝手にキャンセルする

- **起こる場面**：iteration 5 で 3 候補提示 → 2 候補は承認 → 1 候補は装置予約待ちで pending → iteration 6 で新規 3 候補提示のため、pending の 1 候補を「もう不要」と judgment して pending リストから削除
- **問題**：pending experiments は **acquisition の evaluation に影響**（q-EI 系では pending 点を仮想的な観測として扱う）。無承認削除は **acquisition の意味を歪める**、また **同一点の重複提案（duplicate candidates）** の原因になる
- **vol-05 の対処**：`pending_experiments` を **append-only** 契約に。キャンセルは Human 承認必須、承認 ID を provenance に。第15章で fatal 化

### 逸脱 5：エージェントが `stop_condition` を実行中に緩める

- **起こる場面**：iteration 10 で当初の budget が尽きた → エージェントは「まだ収束していないので、あと 3 iteration 続けるべき」と自然言語で判断し、`budget_remaining` を更新（Human 承認なし）
- **問題**：`stop_condition` は Skill 起動時に pin。実行中の緩和は **BO の性能保証の前提を破る**（regret bound は事前 budget に依存）。また **予算超過は施設運用の直接的損失**
- **vol-05 の対処**：`stop_condition` は immutable。緩和は fatal、Human が新規 Skill 実行として再起動する必要がある（第5・15章）

これら 5 逸脱は、いずれも **「エージェントが自然言語で判断を返す」性質そのもの** から生まれます。人間なら「後から承認取ればいい」で済ませられる判断が、Agentic では **不可逆な運用リスク** になる——これが vol-05 の Skill 設計の主動機です。

---

## 3.6 演習用データセット — ARIM 風合成 BO データと外部 benchmark

vol-05 の演習は、以下の 4 系列のデータセットで実施します。**いずれも公開ソースまたは本書スクリプトで再生成可能**ですが、外部データは API key / license / snapshot version に依存する点に注意（各項の「注意」参照）。

### データセット 1：ARIM 風合成 BO データ（本書オリジナル、付録 C 収録）

- **内容**：合成条件（温度 300-700 °C、圧力 0.1-10 MPa、時間 1-24 h、雰囲気 4 種、装置 3 種）→ 結晶化度 (0-1) + 密度 (g/cm³) + 欠陥密度 (log) の 3 目的
- **真の目的関数**：既知（合成データなので）、multi-modal、装置差あり、noise あり
- **使い方**：単目的 BO（第5-8章）、多目的 BO（第9章）、制約付き BO（第10章）、階層 GP マルチ装置（第11章）、Advanced Capstone（第14章）
- **入手**：付録 C の生成スクリプト、または本書 GitHub リポジトリから直接ダウンロード

### データセット 2：Olympus benchmark suite（外部、Aspuru-Guzik グループ）

- **内容**：材料化学に特化した 20+ の実 / 合成 benchmark（HPLC 最適化、Suzuki カップリング、Perovskite 合成 etc）
- **使い方**：BO 手法の性能比較、多目的 / 制約付きの実 benchmark
- **入手**：`pip install olympus`、公式サイト <https://aspuru-guzik-group.github.io/olympus/>
- **注意**：Olympus のうち "surrogate" タイプは本質的に合成データ。実 benchmark（"emulator" / "dataset" タイプ）を優先

### データセット 3：Bayesian 標準 benchmark 関数

- **内容**：Branin (2D)、Hartmann (3D / 6D)、Ackley (nD)、Rosenbrock (nD)、Griewank (nD)
- **使い方**：BO の基礎確認、GP hyperparameter の挙動観察、acquisition の性能比較
- **入手**：BoTorch / scikit-optimize に組み込み、`botorch.test_functions`
- **注意**：材料実験らしさはないが、**手法の性能検証**には必須。第6-7章で頻用

### データセット 4：Materials Project 派生 "測定コスト付き" データ

- **内容**：Materials Project の結晶構造データベースから、DFT 計算時間の記録値または proxy（バンドギャップ計算相当の想定時間）を "測定時間" として付与し、逐次探索問題としてラップ
- **使い方**：離散候補空間の BO（第12章の BOCS 判断表）、Transfer learning（第11章）
- **入手**：付録 C の生成スクリプト（Materials Project API + カスタムラッパ）。**API key 必須**、`snapshot_date` と `API version` と `cost_proxy_definition`（DFT 実測時間 or proxy）を provenance に記録
- **注意**：MP の DFT データは静的で、真の "測定" ではない。**逐次探索の演習用にリアリズムを付与**する構成。ライセンス（CC-BY 4.0）と再配布条件を確認

### 実データを持ち込むときのガイド（付録 C 詳述）

- 匿名化：試料 ID・オペレータ ID を hash に変換
- search space 定義のレビュー：連続変数の bounds、離散変数の候補集合、単位の明示
- 装置差の扱い方の宣言（§3.3 の A/B/C のいずれか）を事前に Human 承認
- サイクル時間・予算の見積り（§3.4）を provenance に記録

---

## 3.7 章末演習 — 「自分のデータ」で BO 適用性を判定する

以下 5 問に、自分のデータで回答してください。第4章のライブラリ選択と第5章 Skill 契約テンプレート作成の入力になります。

**問 1**：自分のデータの **目的関数の次元** は？（単目的 / 2 目的 / 3+ 目的）

**問 2**：**search space の bounds** を書き出せますか？（連続変数の下限・上限・単位、離散/カテゴリ変数の候補集合）

**問 3**：装置差・オペレータ差・バッチ差の扱いは §3.3 の A/B/C のどれ？ **判断根拠**（装置数、装置差の効果推定意図、目的関数の類似性）を 1 段落で書けますか？

**問 4**：**サイクル時間と予算内 iteration 数** を §3.4 の表で計算しましたか？ BO 適用性判定は「不向き / 慎重 / 本領 / 標準」のどれ？

**問 5**：**絶対に破ってはいけない安全条件**（温度上限、圧力上限、化学的制約）を 1 つ以上挙げられますか？ safe BO（第10章）の準備。

これら 5 問の答えが第2章 §2.10 のチェックリストと合わせて **自分の BO Skill の入力仕様書** の骨格になります。

---

## 3.8 次章に進む前のチェックリスト

- [ ] **§3.2 の GP 暴走 4 種** を、自分のデータで起こりうるパターンとして 1 つ以上挙げられる
- [ ] **§3.3 の 3 選択肢**（categorical input / hierarchical GP / blocking）から自分のデータに適したものを 1 つ選び、根拠を言える
- [ ] **§3.4 のサイクル時間見積り**を自分のデータで実施し、BO 適用性を判定した
- [ ] **§3.5 の 5 逸脱**を、自分の運用で起こりうるパターンとして 2 つ以上挙げられる
- [ ] **§3.6 の演習データ**のうち、自分の研究テーマに近いものを 1 つ選び、入手した
- [ ] 「BO 不向き」判定なら **vol-04 の factorial / RSM に戻る**選択も検討した

「うろ覚え」があれば、該当節に短く戻ってから第4章「BO / active learning ライブラリ地図」へ進んでください。特に **§3.3 の装置差の扱いが未決** の場合、第11章の階層 GP で詰まります——ここで一度 Human 承認プロセスを実務に照らして固めておくのが安全です。

---

## 参考資料

### vol-05 の該当章（本章での主な参照）

> [!NOTE]
> vol-05 の第4章以降および付録は執筆中（planned）。以下のリンクは章確定後に有効化されます。

- [第1章 vol-01〜04 の最小復習](./chapter-01.md)（§1.4 authorization_gates L1-L4、§1.7 provenance フィールド）
- [第2章 予測 → 因果 → 逐次 のラダー](./chapter-02.md)（§2.4 acquisition 最大化 ≠ 科学的最適候補、§2.5 Type A-E）
- 第4章 BO / active learning ライブラリ地図（planned、次章、BoTorch-Ax gap）
- 第5章 BO × Agentic Skill の設計原則（planned、`surrogate_treatment_of_facility_variance` の pin、`acquisition_spec` の pin）
- 第6章 GP surrogate（planned、暴走への対処、kernel / prior 既定値）
- 第7章 Acquisition と外挿検知（planned、`hallucinated_recommendation_detection` の operational 定義）
- 第10章 制約付き BO / safe BO（planned、安全条件の Skill 契約組み込み）
- 第11章 マルチ装置 BO（planned、階層 GP の完全展開）
- 第14章 総合ハンズオン（planned、vol-04 因果 × 階層 GP × Human 承認の統合）
- 第15章 失敗パターンと監査（planned、BO 一般 + Agentic 特有 taxonomy）
- 付録 C BO 演習データ候補（planned、ARIM 風合成 BO データ生成スクリプト）

### vol-04 の該当章（前提）

- [第0章 vol-01/02/03 の最小復習](../vol-04/chapter-00.md)（§0.7 の DoE scope 限定と vol-05 移譲宣言）
- [第2章 ARIM データの因果推論特有の課題](../vol-04/chapter-02.md)（本章の構造祖）
- [第10章 古典的実験計画の Skill 化](../vol-04/chapter-10.md)（Blocking / randomization、vol-05 との補完）
- [第11章 応答曲面法とタグチメソッド](../vol-04/chapter-11.md)（CCD / Box-Behnken / smt Kriging、vol-05 GP との対比）

### 外部参考

- Rasmussen, C. E. & Williams, C. K. I. "Gaussian Processes for Machine Learning" (MIT Press, 2006) — 少データ GP の hyperparameter 事後の multi-modality については 5 章
- Frazier, P. I. "A Tutorial on Bayesian Optimization" (arXiv:1807.02811, 2018) — サイクル時間と cost の議論は §5
- Aspuru-Guzik group "Olympus benchmark suite" <https://aspuru-guzik-group.github.io/olympus/> — 演習データセット
- Häse, F. et al. "Olympus: a benchmarking framework for noisy optimization and experiment planning" (Machine Learning: Science and Technology 2021)
- Materials Project <https://materialsproject.org/> — データセット 4 の元
- BoTorch benchmark functions <https://botorch.org/api/test_functions.html>
