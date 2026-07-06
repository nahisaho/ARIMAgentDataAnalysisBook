# 第3章 ARIM データに深層を持ち込むときの Agentic 特有の課題

> **本章の到達目標**
> - ARIM の実験データに深層学習を持ち込むときに現れる **5 つの Agentic 特有の課題**（**装置固有性 × fine-tune 判断・実験ノート連動・少データでの自動化・GPU 非決定性と重み provenance・fine-tune 特有のデータリークと augmentation 契約**）を、自分のデータに照らして分類できる
> - **点推定だけでは足りない理由**（不確かさへの再訪）を、Agentic な自律判断の停止条件の観点から説明できる
> - 本書で使う **演習用データセット**——主軸となる **ARIM 風合成階層データ（深層版）** と、対比のための公開ベンチマーク（NFFA-SEM / UHCS / MatBench の 1 タスク）——の入手方法・ライセンス・**モデル重みの再配布ルール**を理解する
>
> **本章で扱わないこと**
> - 具体的なモデル実装や PyTorch/JAX/HF の使い分け（第4章以降）
> - GPU determinism を担保する具体的な API 呼び出し（第5章）
> - 深層モデルの不確かさ推定手法の実装（第8-9章）
> - Foundation Model の事前学習そのもの（本書は fine-tune / feature extractor 利用に限定）

---

## 3.1 なぜ「ARIM × 深層 × Agentic」だけの章が必要か

vol-02 第3章では、ARIM データに現れる **5 つの統計的課題**（小サンプル・階層構造・反復測定・ロット差・物理制約）を扱いました。これらは深層学習に切り替えても消えるわけではなく、むしろ **深層 + Agentic の文脈で新しい形の落とし穴として再登場**します。

一般的な深層学習の教科書（ImageNet や自然言語処理を主戦場にしたもの）を ARIM データにそのまま持ち込むと、次のような事故が起きます。

- **「NFFA-SEM の分類で高精度だった CNN を自分の SEM に転移したら、装置差で大きく精度が落ちた」**（→ 装置固有性 × fine-tune 判断の欠落）
- **エージェントが「今日測ったばかりのデータ」を勝手に fine-tune に混ぜて、翌週の校正が壊れた**（→ Human 承認なしの学習ジョブ起動）
- **論文の「精度 95% の Transformer」を再現しようとしたら、seed も cuDNN 設定も分からず、再現できない**（→ GPU 非決定性と重み provenance の欠落）
- **augmentation で「回転・反転」を入れたら、結晶方位という物理的意味を持つデータを壊した**（→ augmentation 契約の欠落）
- **点推定 0.87 の分類確率を信じてエージェントが自律決定したが、実は out-of-distribution で自信過剰なだけだった**（→ 不確かさ推定なしの自律判断）

（上記は本書の議論を導入するための代表例で、具体的な数値は仮想例です。）

これらは **モデル選択の問題ではなく、"エージェントが深層モデルを扱うときにどこで人間の承認を挟むか" が設計されていない** 問題です。この章で 5 つの課題を洗い出しておくと、第5章以降のハンズオンで「なぜこの承認ゲートが必要か」が結び付きます。

---

## 3.2 課題 1：装置固有性 × fine-tune 判断

### 現場での典型

- 「同じ SEM でも装置 A と装置 B では画像の輝度分布・ノイズ特性が異なる」
- 「Raman 分光は装置ごとに波数校正が微妙にズレる」
- 「XRD は X 線源（Cu Kα / Mo Kα）で全ピーク位置が変わる」
- 「同じ研究室内でも、装置更新前後でモデルの精度が段落ちする」

### なぜ深層 × Agentic で問題になるか

- **事前学習モデルの領域ギャップ (domain gap)**：NFFA-SEM や ImageNet で事前学習した CNN は、輝度分布・解像度・視野が異なる自研究室 SEM に対して系統的にズレる
- **fine-tune するかどうかの判断が動的**：装置更新・光源交換・オペレータ変更のたびに再 fine-tune が必要か、既存重みで足りるかを判断しなければならない
- **エージェントに任せると危険**：「精度が下がったから自動で再学習しよう」とエージェントが勝手に fine-tune を回すと、**checkpoint 上書き**により過去の再現性が失われる

### vol-03 での主な対処

- **第5章 学習権限設計**：エージェントに与える 3 段階権限——推論のみ / 承認済み範囲内の fine-tune 可 / 事前承認済みワークフロー内の自律実行可
- **第8章 装置別 fine-tune 判断表**：「今日のバッチだけで再 fine-tune するか」の Human 承認ゲート、frozen feature extractor / full fine-tune / LoRA の判断表
- **第9章 不確かさゲート**：装置更新後に不確かさが閾値を超えたら自律決定を停止する契約

> [!TIP]
> 「装置固有性」は vol-02 でも階層構造の一つとして扱いましたが、深層学習では **事前学習重みが担う "共通表現"** と、**現場で fine-tune する "装置特化"** の 2 段階に分離できます。この分離こそが Agentic な学習権限設計の骨格になります（第4・第8章）。

---

## 3.3 課題 2：実験ノートとの連動

### 現場での典型

- 「試料 ID・測定日時・オペレータ・装置状態が実験ノートには書いてあるが、モデル訓練データには落ちていない」
- 「Grad-CAM で attribution が『画像端の情報』を見ていたが、それがサンプルホルダの映り込みだったと数週間気付かなかった」
- 「モデル予測が急に悪化したが、原因が装置校正か試料変質か分からない」

### なぜ深層 × Agentic で問題になるか

- **深層モデルは "画像だけ" を見る**：試料 ID や測定日時が学習に反映されないと、装置校正のドリフト・オペレータ差・試料変質を切り分けられない
- **エージェントが誤診断を強化する**：Human に判断を投げる際、「なぜその予測に至ったか」を試料 ID・測定条件と紐付けて説明できないと、Human が判断を承認するにも棄却するにも根拠がない
- **provenance が切れる**：訓練データの試料 ID が記録されていないと、後日「あの学習に混じった試料 X は実は損傷していた」と分かっても取り除けない

### vol-03 での主な対処

- **付録 A の provenance スキーマ**：`sample_id`, `measurement_datetime`, `operator_id`, `instrument_state_snapshot` を Skill の入力メタデータに含める
- **第6章 前処理契約**：試料 ID を訓練/検証/テスト分割の grouping key として使う（測定日時・オペレータでの stratified split も含む）
- **第11章 Human-in-the-loop 拡張**：Grad-CAM / integrated gradients を、試料 ID・測定条件のメタデータと**並置**して Human に提示する UX

---

## 3.4 課題 3：少データ現場での自動化 — 価値と危険

### 現場での典型

- 「新規物質の分光データが週 20-50 枚しか取れない」
- 「n=200 のラベル付き画像から CNN を学習させたい」
- 「装置時間の制約で、fine-tune 用の追加ラベルを月に 10 サンプルしか増やせない」

### なぜ深層 × Agentic で問題になるか

- **自動化の価値は大きい**：少データ × 少人手だからこそ、エージェントが前処理・特徴抽出・推論を自動で回せる価値は大きい
- **同時に危険も大きい**：少データほど「たまたま合った」モデルが生まれやすく、それをエージェントが自律的に使うと **誤った推論を高速に大量生産**する
- **Foundation Model が救世主にも罠にもなる**：MatBERT / CrystaLLM / MACE / UMA を凍結特徴抽出器として使えば少データでも表現学習の恩恵を受けられるが、**pretraining データとの領域ギャップを検知できないまま使う**と systematic error を運び込む

### vol-03 での主な対処

- **第8章 転移学習の判断表**：少データ × 自研究室装置では **まず frozen feature extractor → grouped CV / 外部検証で改善が確認できたら fine-tune** の優先順位
- **第9章 不確かさ + 停止ゲート**：エージェントの自律決定は、predictive entropy / deep ensemble variance / MC-Dropout variance が閾値以下のときのみ許可
- **第11章 attribution の sanity check**：Grad-CAM / integrated gradients が入力・パラメータのランダム化に対して頑健かを検査（`saliency map` の失敗パターン）[^adebayo]
- **第12章 Foundation Model 利用時の hallucination 対策**：pretraining データとの領域ギャップ検知、FM 出力に対する外部検証

[^adebayo]: Adebayo, J., et al. "Sanity checks for saliency maps." *NeurIPS 2018*. — 事前学習済みモデルの解釈可視化（saliency map）が、モデルパラメータをランダム化しても似た画像を返すことがあるという警告論文。深層モデルの attribution を「信じる」前に必要な最低限のチェック。

---

## 3.5 課題 4：GPU 非決定性と事前学習重みの provenance

### 現場での典型

- 「同じ seed で同じスクリプトを回したのに、GPU が違うと結果が違う」
- 「学会発表で使ったモデルの重みが、どのバージョンの Hugging Face Hub から落としたものか分からない」
- 「PyTorch のバージョン更新後、fine-tune の loss 曲線が変わった」

### なぜ深層 × Agentic で問題になるか

- **GPU 非決定性**：CUDA/cuDNN の一部演算（`atomicAdd`, `scatter`, cuDNN benchmark モード）は非決定的で、seed を固定しても結果が完全に一致しない
- **事前学習重みが provenance の穴**：`weights_uri` / `weights_sha256` / **重みのライセンス** / 事前学習に使われたデータのライセンスを記録しないと、後日「この重みは商用利用不可だった」「事前学習に自研究室のデータが混じっていた」といった事故が発覚したときに追跡できない
- **エージェントが重み更新を勝手に行う**：Hugging Face Hub の重みバージョンをエージェントが自動更新すると、**同じ Skill が翌週別の予測を返す**という非再現性が発生

### vol-03 での主な対処

- **付録 A の provenance 拡張**：`gpu_backend`, `cuda_version`, `cudnn_deterministic`, `cudnn_benchmark`, `torch_deterministic_algorithms`, `cublas_workspace_config`, `random_seed_per_worker`, `weights_uri`, `weights_sha256`, `weights_license`, `pretraining_data_license`, `tolerance` を Skill 契約に必須化
- **第5章 CI 上の CPU 学習**：フル学習は GPU で回すが、CI では CPU で小規模再現テストを走らせて **結果差分を追跡しやすい範囲** に収める
- **第5章 checkpoint 上書き承認ゲート**：エージェントが checkpoint を上書きする前に Human 承認、または新 checkpoint を別名保存する契約

> [!WARNING]
> `torch.backends.cudnn.deterministic=True` や `torch.use_deterministic_algorithms(True)` は、パフォーマンス低下や一部演算の未対応（RuntimeError）を伴います。「bit 単位の一致」ではなく **「差分が追跡できる範囲」** を目標にする現実的な妥協が必要です（第5章で具体化）。

---

## 3.6 課題 5：fine-tune 特有のデータリークと augmentation 契約

### 現場での典型

- 「vol-02 でスプリット時に注意した grouping key を、fine-tune では忘れていた」
- 「augmentation で回転を入れたら、結晶方位が意味を持つデータで物理的にありえない訓練になった」
- 「CLIP 系のモデルを使うとき、text と image が同じ論文由来のペアでリークしていた」

### なぜ深層 × Agentic で問題になるか

- **fine-tune 特有のデータリーク**：**事前学習に使われたデータ**が、fine-tune の評価データと重複していると、精度が実力より高く出る。特に公開データセット（arXiv 由来のテキストなど）で発生しやすい
- **augmentation の物理的妥当性**：**回転・反転・色調変換**は、ImageNet では意味を保つが、結晶方位や偏光を含む材料データでは物理的に破壊的
- **エージェントが augmentation を勝手に増強する**：精度向上を目的にエージェントが augmentation の強度を上げると、**物理的にありえない訓練データ**を生成し、結果として現場で外れるモデルが生まれる

### vol-03 での主な対処

- **第6章 augmentation 契約**：`train のみで使い、test/val には使わない` / `stratified split との整合` / **エージェントが augmentation の種類・強度を勝手に変更しない**契約
- **第6章 深層 anti-leakage split contract**：fine-tune 前の事前学習データとの重複禁止、CLIP-like モデルでの text-image leakage の検査手順
- **第6章 装置間差 augmentation の判断ゲート**：「装置差を augmentation で消してよいか」——**消してはいけない場面**（装置差を検出したい・オペレータ差を分析したい）と**消すべき場面**（外部検証で装置差を跨いだ汎化を狙う）を分ける判断表

---

## 3.7 なぜ点推定だけでは足りないか（不確かさへの再訪）

vol-02 第10章で **頻度論の限界** を扱いましたが、深層モデルではこの問題がさらに深刻化します。

### 深層モデルの softmax は信じられるか

- **softmax は確率のように見えるが calibration が壊れている**：深層モデルは典型的に **自信過剰**（overconfident）で、softmax=0.97 でも実際には 60% しか当たらないことがある[^guo]
- **Brier score / Expected Calibration Error (ECE)** で calibration を測る（第9章で実装）
- **reliability diagram** で見ると、「予測確率」と「実際の頻度」の乖離が可視化される

[^guo]: Guo, C., et al. "On calibration of modern neural networks." *ICML 2017*. — 現代の深層ネットワーク（ResNet 等）は古典的な浅いネットワークよりも自信過剰であるという古典的論文。calibration の議論の出発点。

### Agentic な自律判断には calibration + 停止ゲートが必須

エージェントが「softmax=0.97 だから自律決定してよい」と判断するには、**その 0.97 が実際に 97% の頻度で当たる**ことが担保されていなければなりません。担保されていない場合、エージェントは：

1. **予測確率のまま自律決定** → 誤診断を高速大量生産
2. **不確かさゲートで Human に投げる** → 遅くなるが安全

vol-03 は 2 番目の設計を採ります。停止ゲートの入力には、**予測確率**（softmax）そのものではなく、**calibrated predictive probability**（温度スケーリング等で較正した確率）に加えて、**不確かさ / OOD 指標**（deep ensemble variance、MC-Dropout variance、predictive entropy、conformal prediction interval）を組み合わせて使います（第8-9章）。

### 対処のロードマップ

| フェーズ | 手法 | 章 |
|---|---|---|
| 不確かさを可視化 | reliability diagram / Brier / ECE | 第9章 |
| 不確かさを推定 | deep ensemble / MC-Dropout / BNN | 第8-9章 |
| 不確かさで自律を止める | Agentic 停止ゲート契約 | 第9章 |
| PyMC/NumPyro との接続 | vol-02 第11-13章の Bayesian と対比 | 第10章 |

---

## 3.8 自分のデータを分類してみる（章末ワーク）

読者自身のデータについて、以下を書き出してみてください（第5章以降のハンズオンで、この分類が Skill 設計の入力になります）。

**Q1. 装置固有性はどのくらいか？**

- [ ] 装置は 1 台のみ、光源・検出器の変更なし（→ 装置差の心配は少ない）
- [ ] 同種装置が複数台、装置間差あり（→ grouped CV、装置別 fine-tune 判断が必要）
- [ ] 異なる装置種を混在（→ Foundation Model + 装置別 head の設計が必要）

**Q2. 実験ノートは学習データに反映できるか？**

- [ ] 試料 ID・測定日時・オペレータが電子データで取得できる（→ meta feature として活用可能）
- [ ] 実験ノートは紙／PDF、手作業で数百件程度なら統合できる（→ 半自動化）
- [ ] 実験ノートが失われている（→ 訓練データを絞る必要あり）

**Q3. ラベル付きデータの規模は？**

- [ ] 数十件（→ frozen feature extractor + Bayesian regression、fine-tune はほぼ不可）
- [ ] 数百〜千件（→ frozen feature extractor か LoRA、慎重な fine-tune）
- [ ] 数千件以上（→ full fine-tune 検討可、ただし装置別に分ける）

**Q4. GPU 環境は？**

- [ ] GPU なし／共用 GPU で数時間だけ（→ CPU 推論 + LoRA 中心、Foundation Model は frozen 中心）
- [ ] 単一 GPU（12-24 GB）で日常運用（→ 中規模 fine-tune 可能）
- [ ] マルチ GPU / A100 級（→ 本書想定を超える大規模 fine-tune も可能）

**Q5. Agentic な自律判断はどこまで許すか？**

- [ ] 推論のみ、学習ジョブは全て Human 承認（→ 保守的設計、第5章 権限レベル 1）
- [ ] 事前承認済み範囲内で fine-tune 可、checkpoint は追記のみ（→ 権限レベル 2）
- [ ] 事前承認済みワークフロー内で checkpoint 上書きも可（→ 権限レベル 3、本書はこの手前まで）

---

## 3.9 演習用データセット（主軸：ARIM 風合成、対比：公開ベンチマーク）

vol-03 は **ARIM 風合成階層データを主軸**とし、公開ベンチマークは対比・スケール確認の位置づけで使います（企画書 v0.2、方針 A5）。

### データセット 1：ARIM 風合成階層データ（深層版）— **主軸**

vol-02 第14章の合成階層データ（表形式・装置/ロット/研究室構造）を継承し、**深層入力**（スペクトル・画像・テキスト記述）を追加した本書提供予定のデータセット。

- **構造**：3 研究室 × 装置 2 台 × ロット 5 × 試料 20〜50、階層構造は vol-02 と共通
- **入力**：
  - 1D スペクトル（ラベル：物質分類 5 クラス）
  - 2D 画像（ラベル：形状分類 4 クラス）
  - 短いテキスト記述（オペレータメモ、fine-tune 用の合成テキスト）
- **サイズ**：訓練 6,000 / 検証 2,000 / テスト 2,000（画像・スペクトル共通の名寄せ ID あり）
- **ライセンス**：本書著者による CC-BY-4.0 予定、生成スクリプトも同時公開
- **ダウンロード**：付録 B（第7章までに公開）

> [!IMPORTANT]
> このデータは **合成であることが利点**です。装置差・ロット差・オペレータ差の "正解" が生成器側で分かっているので、モデルが階層構造を正しく捉えているかを検証できます。実データではこの検証ができないからこそ、練習用に合成を主軸に据えています。

### データセット 2：NFFA-EUROPE SEM（対比、公開）

- **内容**：ナノ構造の SEM 画像を 10 カテゴリに分類（Nanowires, Fibres, MEMS 等）
- **サイズ**：約 21,000 枚
- **ライセンス**：CC BY 4.0（配布元条件に従う）
- **URL**：https://b2share.eudat.eu/records/80df8606fcdb4b2bae1656f0dc6db8ba
- **本書での位置づけ**：第7章の 2D CNN 分類 Skill の**対比用**（合成データで検出できない現実データ特有のノイズを確認する）

### データセット 3：UHCS（Ultra-High Carbon Steel、対比、公開）

- **内容**：超高炭素鋼の顕微鏡画像（分類 4-7 クラス、標準ベンチマーク）
- **サイズ**：約 961 枚（NIST 公開データ）
- **ライセンス**：NIST 公開データ（多くの用途で自由利用可、詳細は配布元）
- **URL**：https://hdl.handle.net/11256/940
- **本書での位置づけ**：**少データ現場**の再現用対比（第8章 転移学習）

### データセット 4：MatBench の 1 タスク（対比、公開）

- **内容**：`matbench_perovskites`（DFT 形成エネルギー予測、18,928 件）など、表形式物性予測タスクの 1 つ
- **サイズ**：タスクにより数千〜約 2 万件（`matbench_perovskites` は 18,928 件）
- **ライセンス**：ベンチマーク配布パッケージ（`matminer` 等）のコードライセンスと、データセット本体（Materials Project 由来）のライセンスは別。**各タスクごとに公式で確認**すること
- **URL**：https://matbench.materialsproject.org/
- **本書での位置づけ**：**Tabular 深層モデル**（TabNet / FT-Transformer）を vol-02 の GBM と比較する対比用（第7章）

### ライセンス・引用ルール（深層特有の追加）

vol-02 のライセンスルール（データセット引用・出典明記）に加え、vol-03 では **モデル重みライセンス** を必ず記録します。

| 項目 | 必要な記録 | Skill provenance のフィールド |
|---|---|---|
| 事前学習データのライセンス | 商用利用の可否、再配布条件 | `pretraining_data_license` |
| モデル重みのライセンス | Apache-2.0 / MIT / CC-BY-NC / 独自 | `weights_license` |
| 重みの正確な入手元 | Hugging Face Hub の revision hash | `weights_uri` + `weights_sha256` |
| 引用要求 | 論文引用の要否と書式 | Skill の README / 出力レポート |

> [!WARNING]
> 事前学習データが不明な重みは、**商用利用・共同研究・特許出願で問題化する可能性**があります。特に arXiv 由来のテキストで事前学習された重みは、当該論文の著者が知らないまま含まれている場合があります（第12章 Foundation Model の議論で再訪）。

---

## 3.10 まとめ

本章で確認したこと：

- **ARIM × 深層 × Agentic** には、vol-02 の統計的課題とは別軸の 5 つの Agentic 特有課題がある：**装置固有性 × fine-tune 判断・実験ノート連動・少データでの自動化・GPU 非決定性と重み provenance・fine-tune 特有のデータリークと augmentation 契約**
- **点推定（softmax）は信じられない**。calibration + 不確かさ推定 + Agentic 停止ゲート の 3 点セットが本書の骨格
- **演習用データは ARIM 風合成階層データを主軸**とし、公開ベンチマーク（NFFA-SEM / UHCS / MatBench）は対比・スケール確認に使う
- **深層特有のライセンス**（モデル重み・事前学習データ）を Skill provenance に必須化する

次章（第4章）では、これらの課題を扱うための **PyTorch / JAX / Hugging Face の Agentic 使い分け**——特にエージェントが Jupyter MCP 経由でどのライブラリをどこまで自律的に叩けるかの**権限マップ**——を扱います。

---

## 参考資料

- vol-01 第6章「Human-in-the-loop」、第8章「データ分析用 Skill の設計原則」、付録 A「provenance の基本形」
- vol-02 第3章「ARIM データに現れる統計的課題」、第8章「CV とデータリーク」、第10章「不確かさ入門：頻度論の限界と Bayesian への橋」、第11章「PyMC 入門」、第12章「階層モデル：反復測定・ロット差・測定誤差」、第13章「MCMC の実務と限界：判断と修正」、第14章「総合ハンズオン（Advanced Capstone）：sklearn → PyMC → 階層モデル」
- PyTorch 決定性ガイド: https://pytorch.org/docs/stable/notes/randomness.html
- Hugging Face Hub ライセンスガイド: https://huggingface.co/docs/hub/repositories-licenses
- NFFA-EUROPE SEM: https://b2share.eudat.eu/records/80df8606fcdb4b2bae1656f0dc6db8ba
- UHCS: https://hdl.handle.net/11256/940
- MatBench: https://matbench.materialsproject.org/
