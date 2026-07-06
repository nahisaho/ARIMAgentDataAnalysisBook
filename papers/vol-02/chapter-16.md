# 第16章　組織展開と終章 — Skill 共有・監査ログ・次巻への道しるべ

> [!IMPORTANT]
> **本章の到達目標**
>
> - Ch4-14 で作った統計/ML Skill を **個人の道具**から **組織の共通言語**へ引き上げる 3 つの展開パターン（Skill 共有・専用 MCP 化・テンプレ配布）の判断基準を得る
> - **監査ログ**の最小要件（誰が・いつ・どの Skill バージョン・どの provenance で動かしたか）を Ch4-14 の provenance 拡張と接続する
> - vol-01（データ整備）と vol-02（統計/ML）の**到達点**を、6 データ型 × 統計/ML 手法の完成マップとして振り返る
> - **vol-03（深層 × エージェント）・vol-04（因果・実験計画）**への道しるべを、なぜ vol-02 の枠組みでは扱えなかったか（=前提となる別軸）とともに示す

## 本章で扱わないこと

- **具体的な MCP 実装**：Python SDK による MCP サーバ実装は vol-03 の付録候補（本章は「作るべきかどうか」の判断基準まで）
- **組織の権限管理・監査対応**：ISO/IEC 27001, SOC 2 等の適合作業は vol-04 以降
- **人事・評価制度**：Skill 作成者を評価する仕組みは組織固有
- **法務・契約**：外部共有時のライセンス設計は付録の参考リンクへ

---

## 16.1 個人 Skill から組織 Skill へ

Ch4-14 の Skill は基本的に「1 人の研究者が 1 テーマで作る」前提でした。組織展開とは、**同じ Skill を複数人が同じ結果で使える**状態にすることです。

以下 3 段階で成熟度が上がります。

| 段階 | 状態 | 典型的な利用者数 | 主な課題 |
|---|---|---|---|
| **L0: 個人 Skill** | 作った本人だけが使う | 1 | 再現性・provenance の自己管理 |
| **L1: チーム Skill** | 数人の同僚が同じ Skill を呼び出す | 2〜10 | バージョン統一、命名衝突、依存環境 |
| **L2: 組織 Skill** | 部門横断で使う。第三者が監査する | 10〜100+ | 監査ログ、責任分界、廃止手続き |

> [!TIP]
> **早すぎる L2 化は失敗しやすい**：L0 で作った直後に組織展開すると、Ch14 の失敗パターン（特にリーク・チェイン切れ）が発見前に伝播します。**L1 で最低 3 プロジェクト分の再利用実績**を経てから L2 を検討するのが安全です。

---

## 16.2 展開パターン A：Skill 共有

**もっとも軽量**な展開。Ch4-14 で作った Skill ディレクトリ（`SKILL.md` + `references/` + `assets/`）を Git リポジトリに置き、利用者に「clone して `.github/skills/` 配下に置いてください」と伝えるだけ。

### A-1. 適する条件

- 手法が**まだ流動的**（改訂頻度が月次以上）
- 利用者が Skill の中身（プロンプト・スクリプト）を**読んで理解する**ことを許容できる
- 監査要件が低い（社内実験・PoC）

### A-2. 最低限そろえるもの

| 要素 | 内容 |
|---|---|
| `README.md` | 目的・対象データ型・前提バージョン（Python, sklearn, PyMC, ArviZ, NumPyro） |
| `SKILL.md` の frontmatter | `version: X.Y.Z`（Ch14 §15.9 のバージョニング原則） |
| `CHANGELOG.md` | Major / Minor / Patch の変更履歴 |
| `known_issues.yaml` | Ch14 §15.9 で追加した schema |
| ライセンス | 内部利用範囲・再配布可否を明示 |

### A-3. 落とし穴

- **依存バージョン未固定**：sklearn 1.4 → 1.5 で挙動が変わる箇所がある。`requirements.txt` または `environment.yml` を Skill 同梱
- **サンプルデータの二重管理**：合成データを Skill 内に埋め込むと、Ch13 §14.5 の `synthetic_dataset.generator_script_sha256` が壊れる。**別リポジトリ or Git LFS に切り分け**
- **プロンプト依存の暗黙前提**：作者の頭にあった「常識」（例：`log10` を取る前提）が SKILL.md に書かれていない

---

## 16.3 展開パターン B：専用 MCP 化

Skill を **MCP サーバ（scikit-learn-mcp / pymc-mcp）**として実装し、ツールとして呼び出す形。エージェントは自然言語で `train_ridge_pipeline` `run_calibration_curve` `run_hierarchical_model` 等の関数を呼ぶ。

### B-1. 適する条件

- 手法が**安定**（Ch14 §15.9 でいう Major バージョンが 3 か月以上不変）
- API・入出力 schema・データ contract が**固定できている**
- 利用者に**中身を読ませたくない**（誤改変防止 / IP 保護 / 監査要件）
- **固定サーバ環境での再現可能性**が要る（同じコンテナ・依存・seed で **exact hash 一致は決定的アルゴリズムに限る**。MCMC・BLAS スレッド並列・GPU 系は Ch12 §13.7 の sd 正規化 tolerance で扱う）
- 監査ログを**サーバ側で一元管理**したい

### B-2. Skill vs MCP の設計上の違い

| 観点 | Skill（パターン A） | 専用 MCP（パターン B） |
|---|---|---|
| 実行主体 | エージェント（プロンプト解釈） | MCP サーバ（コード直接実行） |
| 再現性 | プロンプト解釈の揺れが残る | 固定コンテナ内で決定的アルゴリズムは exact 一致、確率的アルゴリズムは tolerance 内一致（Ch12 §13.7） |
| バージョン管理 | Git tag + SKILL.md frontmatter | MCP サーバの API バージョン + 内部モデル hash |
| 監査 | エージェントのログを事後突合 | サーバ側の access log + input/output hash |
| 変更コスト | 低（テキスト編集） | 中〜高（コード変更＋テスト） |
| 誤用リスク | 中（プロンプトずれで別解釈） | 低（型で縛られる） |

### B-3. 判断フローチャート（A / B / C 統合版）

以下の 5 段で A / B / C（および複合）を決めます。**上から順に**判定してください。

```
Skill を組織展開したい
├─ Q1. 同型の Skill を組織で複数本作る／教育・標準化目的か？
│   ├─ Yes → C（テンプレ配布）を採用。個別 Skill は Q2 以降で判定して A/B に昇格
│   └─ No ↓
├─ Q2. 手法・API・入出力 schema は安定しているか？（Major 3か月以上不変・data contract 固定）
│   ├─ No → A（Skill 共有）。安定後に Q3 で再判定
│   └─ Yes ↓
├─ Q3. 中央監査 / IP 保護 / 高リスク判断（品質・出荷・認証）のいずれかが必要か？
│   ├─ No → A で十分。C との併用（A+C）も検討
│   └─ Yes ↓
├─ Q4. サーバ運用リソース（実装・テスト・SLA・保守・監視）を確保できるか？
│   ├─ No → A + 定期監査で暫定運用。将来 B へ昇格を計画
│   └─ Yes ↓
└─ Q5. データ機密性・計算資源・モデル更新頻度が MCP 化と整合するか？
    ├─ No → A + 定期監査に留める
    └─ Yes → B（専用 MCP 化）
```

**判定軸まとめ**：
- Q1: 教育・標準化目的（C 判定）
- Q2: API/schema 安定性（A ↔ B 分岐の前提）
- Q3: 監査・IP・業務影響（A → B 昇格の必要条件）
- Q4: 実装・運用リソース（B の実装可能性）
- Q5: データ機密性・計算資源・更新頻度（B の運用整合性）

### B-4. 最小 MCP 化候補（vol-02 の Skill のうち MCP 化しやすい順）

| 優先度 | Skill | 理由 |
|---|---|---|
| 1 | **Ch12 §15.6 診断チェック** | 入出力が単純（idata → dict）、決定的、依存が薄い。`check_diagnostics()` そのものをライブラリ化して MCP 経由で呼ぶだけ |
| 2 | **Ch6 Pipeline + CV 評価** | 入出力が明確（X, y, cv_scheme → scores）、sklearn 前提で API 設計しやすい |
| 3 | **Ch10 校正曲線推定** | `sampler_config` / `backend_config` / `posterior_artifact` schema が固定できてから。tolerance は Ch12 §13.7 に従う |
| 4 | Ch11 階層モデル | 構造が多様で API 設計に工夫が要る。プロジェクト個別性が高い。C（テンプレ）が先 |
| 5 | Ch13 Capstone 統合 | provenance チェイン管理が複雑。MCP 化より orchestration 側の責務 |

> [!NOTE]
> **MCP は「銀の弾丸」ではない**：MCP 化すれば安全になるわけではありません。Ch14 のリーク・prior 暴走・多重比較未補正は、**MCP の内部実装で守られていなければ**サーバ経由でも同じように起きます。**パターン B の実装段階で Ch14 のゲートを組み込む**ことが必須です。

---

## 16.4 展開パターン C：テンプレ配布

Skill そのものではなく、**Skill を作るためのテンプレート**（`cookiecutter` 相当）を配布するパターン。

### C-1. 適する条件

- 「同じ問題ではないが**同じ型**」の Skill を組織で何本も作る（例：物性ごとに校正曲線 Skill を独立で作る）
- 標準化はしたいが**個別最適化の自由度**を残したい
- 教育目的（新人がテンプレを埋めれば形になる）

### C-2. テンプレに含めるもの

- Ch4-14 の推奨構造（`data_split` / `provenance` / `check_diagnostics()` / `known_issues.yaml`）
- 到達目標・扱わないことのスケルトン
- 参考文献の書式（Ch12 §13.10 のスタイル）
- **必ず埋めるべき TODO マーカー**（データ contract, applicability domain 等）

### C-3. パターン A/B との組み合わせ

多くの組織で最終形は **A + C**：テンプレでスキャフォールドし、成熟した Skill だけを A（共有）に昇格、さらに安定した数本を B（MCP）に昇格。

---

## 16.5 監査ログの最小要件

Ch4-14 で拡張してきた provenance を、**組織監査に耐える形**にまとめます。監査ログは 4 レイヤ（誰／何／どんな環境で／どの判断に）で構成します。

### 16.5.1 誰が・いつ・どのバージョンで動かしたか

以下 7 項目は監査の**最小共通項**です。

| フィールド | 例 | 由来 |
|---|---|---|
| `run_id` | UUIDv7 | 実行ごと一意 |
| `run_timestamp` | ISO 8601 | 実行時刻（UTC 推奨） |
| `operator` | ORCID or 組織 ID | 実行者。エージェント経由でも人間を紐付ける |
| `skill_id` | `capstone_l2_pymc_calibration` | Ch10 で導入した命名規約 |
| `skill_version` | `1.2.0` | Ch14 §15.9 の SemVer |
| `code_sha256` | Skill ディレクトリの hash | Skill 改変検知（ディレクトリ tree hash） |
| `git_commit_sha` | `abc1234…` | リポジトリの commit（`git rev-parse HEAD`） |

### 16.5.2 何を入力し・何を出力したか

Ch13 §14.6 の `verify_provenance_chain()` を組織監査に拡張します。

- `inputs.dataset_sha256` — 入力データの hash（Ch4 の data contract）
- `inputs.data_contract_hash` — Ch4 の data contract 定義の hash（schema 変更検知）
- `inputs.split_manifest_hash` — Ch7 の CV/split 定義の hash
- `inputs.upstream_run_ids` — 依存する過去 run の `run_id`（チェイン検証）
- `outputs.artifact_sha256` — 出力アーティファクト hash（Ch10 の `posterior_artifact` 相当）
- `outputs.diagnostics_summary` — Ch10-12 の MCMC 診断（divergences / R-hat / ESS / BFMI / treedepth）
- `outputs.checklist_results` — Ch12 §13.9 の 14 項目チェック個別結果（診断 10 + provenance 4）
- `outputs.certification_pass` — Ch13 §14.7 の全ゲート合否（true/false）
- `outputs.known_issues_snapshot_hash` — 実行時点の `known_issues.yaml` の hash
- `outputs.acknowledged_known_issue_ids` — 該当した既知 issue の ID 列と `acknowledged_by`（Ch14 §15.9 schema）

### 16.5.3 どんな環境で動かしたか（再現性の要）

Ch10 §11.8 の `sampler_config` / `backend_config` を組織監査に拡張します。

| フィールド | 例 |
|---|---|
| `environment.container_digest` | `sha256:...`（コンテナイメージ digest） |
| `environment.python_version` | `3.11.7` |
| `environment.package_versions` | sklearn / pymc / arviz / numpyro / jax の pin |
| `environment.backend` | `pymc-nuts` / `numpyro-nuts-cpu` / `numpyro-nuts-gpu` |
| `environment.hardware` | CPU モデル / GPU モデル / スレッド数 |
| `run_config.random_seeds` | chain seed 列 + init seed |
| `run_config.cv_scheme` | Ch7 の `cv_scheme`（`group_key` 含む） |
| `run_config.thresholds` | R-hat / ESS / divergences の判定閾値 |
| `run_config.checklist_version` | Ch12 §13.9 チェックリストのバージョン |
| `arim_context.instrument_id` | 装置 ID（Ch11 の instrument 階層と紐付く） |
| `arim_context.facility_id` | 実施拠点 ID |
| `arim_context.sample_id` / `lot_id` | 試料 / ロット ID（Ch11 の階層メタ） |
| `arim_context.measurement_protocol_id` | 測定プロトコル ID |
| `arim_context.unit_schema_version` | Ch4 の単位 schema バージョン |

### 16.5.4 どの決定を下したか

**Skill を回した結果、業務上どの判断につながったか**を残します。ここが抜けると監査は「動かした記録」だけで終わり、影響追跡ができません。

| フィールド | 例 |
|---|---|
| `decision.type` | `pilot_go_nogo` / `lot_release` / `paper_submission` / `internal_report` |
| `decision.value` | `go` / `hold` |
| `decision.rationale_link` | 判断根拠のドキュメント URI |
| `decision.reviewer` | レビュアー ORCID（研究者本人 ≠ レビュアーが原則） |

### 16.5.5 ログ自体の完全性

監査ログを**後から改変されない**仕組みも要件です。

| フィールド | 例 |
|---|---|
| `integrity.log_sha256` | このログエントリ全体の hash |
| `integrity.prev_log_sha256` | 直前ログの hash（hash chain / append-only store） |
| `integrity.signature` | 組織署名（PKI / HSM） |
| `access_policy_id` | 参照権限ポリシーの ID |
| `retention_policy_id` | §16.5.6 の保存期間ポリシーの ID |

> [!IMPORTANT]
> **「実行 → 判断」を切らない**：Ch14 のチェイン切れ（provenance 側）と同様、**実行と業務判断を紐付ける ID**（`decision` セクション）がないと、後日「あの判断は何を根拠にしたか」を追跡できなくなります。加えて、`integrity` セクションがないと**「改変されていない」保証**ができません。

### 16.5.6 保存期間と参照権限（例示レンジ）

以下は**例示レンジ**であり、実際は業界規制・契約・研究費規定・関連規格（ISO 9001, ISO/IEC 17025, GLP, GMP 等）で決まります。`retention_policy_id` で組織のポリシー本体にリンクさせる設計が推奨です。

| 種類 | 保存期間の例示レンジ | 参照権限 |
|---|---|---|
| 個人 PoC | 1 年 | 本人 + チームリーダー |
| 内部レポート | 3〜5 年 | 部門内 |
| 論文・特許 | 論文 5〜10 年 / 特許 存続期間+α | 共著者・法務 |
| 認証系（品質判定・製造判定） | 業界規定に従う（5〜10 年以上が多い） | 品質保証部門・監査 |

具体年数は組織のコンプライアンス方針・関連規制の適用範囲に従ってください。本書は**「保存する」こと自体の重要性**と、**組織ポリシーへの参照リンクを持つ設計**を示すに留めます。

---

## 16.6 廃止（deprecation）とロールバック

組織展開すると **Skill を消す・古くする**手続きが必要になります。Ch14 §15.9 のバージョニングと連動します。

### 16.6.1 廃止プロセス

1. `SKILL.md` frontmatter に `deprecated: true` と `deprecation_date: YYYY-MM-DD`
2. `CHANGELOG.md` に「代替 Skill 名」と「移行期間」を明記
3. 依存する下流 Skill（Ch13 の Layer 参照）に**警告**（`known_issues.yaml` に Ch14 §15.9 の schema で `category: deprecation`, `status: deprecated`, `severity: <影響度>`, `affected_versions`, `resolution: <代替 Skill と移行手順>` を追加）
4. 移行期間中は **並行運用**（新旧の出力を Ch12 §13.7 の tolerance で比較）
5. 移行期間終了後に**新規実行を禁止**（監査ログには残す）

### 16.6.2 ロールバック条件

- Ch14 で新規に検出された Blocking 級の failure が既存バージョンにも遡って影響する場合
- ロールバック時は`decision.type = rollback` として、影響を受けた過去 run に対する再評価を記録

---

## 16.7 vol-01 + vol-02 到達点マップ

vol-01（データ整備）と vol-02（統計/ML）の合算で、6 データ型 × 主要手法の**どこまで到達したか**を確認します。

| データ型 | vol-01 で整えたもの | vol-02 で扱えるようになったもの | vol-03 以降で扱うもの |
|---|---|---|---|
| **スペクトル型** | 単位・波数校正・欠損補間・メタ | PCA, PLS, ピーク回帰, 校正曲線 (PyMC), 装置間階層 | 深層特徴抽出、自己教師あり |
| **クロマトグラム・時系列** | 時刻同期・ベースライン | 分類、外れ試料検出、反応速度階層 | 動的因果・介入設計 |
| **画像・顕微鏡** | メタ・解像度・キャリブレーション | 特徴量後の回帰/分類、粒径階層 | CNN・Foundation Model の転移 |
| **回折・散乱** | 単位・角度校正 | パターンクラスタリング、格子定数事後分布 | Rietveld の end-to-end 微分 |
| **表形式・プロセス条件** | データ contract・単位統一 | 物性予測、感度分析、ロット/オペレータ階層 | ベイズ最適化、逐次実験計画 |
| **マルチモーダル統合** | ID キー・時刻同期 | 統合特徴量からの予測 | マルチモーダル生成・逆設計 |

> [!NOTE]
> **到達点の言い換え**：vol-02 完了時点で、**「点推定 + 標準化された CI/CrI + 階層構造 + 診断チェック + provenance」**が組織的な共通語彙になります。これは vol-03/04 で扱う深層学習・因果推論の**入力側の品質**を担保する土台です。

---

## 16.8 vol-03（深層 × エージェント）への道しるべ

vol-02 で扱わなかった深層学習は、**エージェント × 学習という別軸**が入るため独立巻としました。

### 16.8.1 なぜ vol-02 の枠で扱えなかったか

| vol-02 の前提 | 深層学習で崩れるもの |
|---|---|
| CPU で数分〜数十分の再現実行 | GPU 前提の学習時間（数時間〜数日） |
| 決定的アルゴリズムでの exact hash 一致 | GPU 非決定性（cuDNN の非決定的畳み込み、atomic add、TF32 等） |
| 特徴量エンジニアリングを人間が設計 | 表現学習が自動 → 解釈可能性の再定義 |
| Skill = プロンプト + 少量スクリプト | 学習済み重み（数百 MB〜）の配布・バージョニング |
| Bayesian workflow の MCMC 診断ゲート（Ch12 §13.9 の 14 項目） | 学習曲線・early stopping・LR schedule の判断論 |

### 16.8.2 vol-03 想定内容

- **PyTorch / JAX** の Skill 化：訓練・推論・チェックポイント管理
- **Hugging Face 等のモデルハブ連携 MCP**（既存があれば利用、なければ組織内実装）による事前学習モデル呼び出し
- **転移学習・fine-tuning**：材料分野の少データ環境での実践
- **不確実性のあるニューラルネット**：deep ensemble, MC-dropout, Bayesian Neural Nets
- **Foundation Model の材料応用**：MatBERT, CrystaLLM 等の位置づけ
- **provenance の GPU 拡張**：非決定性の許容 tolerance、reproducibility bundle（seed + backend + cuDNN flag）

### 16.8.3 vol-02 とのブリッジ

vol-03 の初章では、vol-02 の校正曲線・階層モデルを「事前学習モデルの上に載せる」ケースを扱う予定です。**深層特徴 + PyMC の階層**は、多くの材料実務で「両方いる」構成になります。

---

## 16.9 vol-04（因果・実験計画）への道しるべ

### 16.9.1 なぜ vol-02 の枠で扱えなかったか

vol-02 の Bayesian は**生成モデル**として構造仮定を表現できますが、**介入・識別仮定・DAG を明示していない**ため、得られた事後分布を**因果効果としては解釈しません**。以下は扱っていません。

- **介入**：条件を変えたときに何が起きるかの予測
- **交絡**：見えない変数の効果の分離
- **反実仮想**：「もし別の温度で実験していたら」の推論
- **実験計画**：次にどの試料を測るかの最適化

これらは因果推論（DoWhy / EconML）と実験計画（BoTorch / GPyOpt）の別体系を必要とします。

### 16.9.2 vol-04 想定内容

- **DAG による因果構造化**：材料プロセスの介入変数と観測変数
- **反実仮想推論**：X-learner, T-learner, IV 法
- **ベイズ最適化**：多目的（性能 × コスト）・制約付き（危険領域回避）
- **Active Learning**：Ch7 の CV スコアではなく「情報獲得」で次を選ぶ
- **実験倫理・研究計画**：pre-registration の実務

### 16.9.3 vol-02 とのブリッジ

vol-04 は「vol-02 の階層モデルを介入モデルに拡張する」章から始まる想定です。**ロット効果を潜在変数として扱う**（vol-02）から**ロット条件を介入変数として設計する**（vol-04）への飛躍を、vol-02 の provenance 拡張と接続します。

---

## 16.10 章末ワーク

### Workshop 1：あなたの Skill を A/B/C のどれで展開するか判定する

Ch4-13 で作った Skill を 1 本選び、§16.3 のフローチャートに従ってパターン A/B/C を判定してください。判定結果を以下の表で書き出します。

| 項目 | 記述 |
|---|---|
| Skill 名 | |
| 手法の安定度（Major 不変月数） | |
| 監査要件（低/中/高） | |
| 誤用時の業務影響 | |
| 判定 | A / B / C / A+C / A+B / … |
| 判定根拠（3 行） | |

### Workshop 2：監査ログの最小フィールドを YAML で書く

§16.5.1 の 7 項目（`run_id` / `run_timestamp` / `operator` / `skill_id` / `skill_version` / `code_sha256` / `git_commit_sha`）に §16.5.4 の `decision` セクションを加えて、あなたの Ch13 Capstone の Layer 1 実行に対して埋めてください。`operator` は仮の ORCID を、`code_sha256` は Skill ディレクトリの tree hash（例：`find skills/<skill_id> -type f | sort | xargs sha256sum | sha256sum` の先頭）で、`git_commit_sha` は `git rev-parse HEAD` で作成します。

### Workshop 3：廃止シナリオを 1 本設計する

Ch14 で「Blocking 級 failure が既存 Skill にも遡って影響する」場合を仮定し、§16.6 の 5 ステップに沿って**移行期間・並行運用の tolerance・rollback 条件**を書き出してください。

### Workshop 4：vol-03/04 で扱う予定のトピックを 1 つ選び、vol-02 の何が土台になるかを 5 行で説明

例：「vol-03 の Foundation Model 転移学習では、vol-02 Ch7 の CV 設計と Ch11 の階層構造が、少データ材料タスクの汎化評価と装置間差の吸収に土台として使える。」

---

## まとめ

- **展開パターン**：Skill 共有（A、軽量）／専用 MCP 化（B、堅牢）／テンプレ配布（C、標準化）。L0 → L1 → L2 の成熟度と一致させる
- **判断フロー**：安定度・監査要件・誤用影響・実装リソースの 4 軸で A/B/C を決める
- **監査ログ**：「誰が・いつ・どのバージョン」+「何を入力し・何を出力」+「どの判断につながったか」の 3 段で最小要件を満たす
- **廃止手続き**：`deprecated: true` → 並行運用 → 新規実行禁止 の 3 段階。バージョニング（Ch14 §15.9）と連動
- **vol-01 + vol-02 到達点**：6 データ型 × 統計/ML 手法のマップで、深層 (vol-03) と因果 (vol-04) の**入力側品質**が担保された
- **次巻への道しるべ**：vol-03（深層 × エージェント）は GPU 非決定性・重み配布・表現学習が新軸。vol-04（因果・実験計画）は介入・反実仮想・実験選択が新軸。どちらも vol-02 の Skill 資産の上に載る

vol-02 はここで一区切りです。**Ch4-14 で作った Skill 群と、Ch15 の展開パターン**が、あなたの組織で統計/ML 実務の共通言語になることを願っています。

---

## 参考資料

<a id="ref-15-1">[15-1]</a> Model Context Protocol 公式仕様: https://modelcontextprotocol.io/ - Anthropic, 参照 2026-07-05

<a id="ref-15-2">[15-2]</a> Sculley, D. et al. "Hidden Technical Debt in Machine Learning Systems." *NeurIPS 2015*. https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html - 参照 2026-07-05

<a id="ref-15-3">[15-3]</a> Semantic Versioning 2.0.0: https://semver.org/ - 参照 2026-07-05

<a id="ref-15-4">[15-4]</a> ORCID: Open Researcher and Contributor ID: https://orcid.org/ - 参照 2026-07-05

<a id="ref-15-5">[15-5]</a> Wilkinson, M. D. et al. "The FAIR Guiding Principles for scientific data management and stewardship." *Scientific Data*, 3, 160018 (2016). https://doi.org/10.1038/sdata.2016.18 - 参照 2026-07-05

<a id="ref-15-6">[15-6]</a> Pearl, J. and Mackenzie, D. *The Book of Why: The New Science of Cause and Effect*. Basic Books, 2018.（vol-04 の理論的背景）

<a id="ref-15-7">[15-7]</a> Frazier, P. I. "A Tutorial on Bayesian Optimization." arXiv:1807.02811 (2018). https://arxiv.org/abs/1807.02811 - vol-04 のベイズ最適化基礎

<a id="ref-15-8">[15-8]</a> Nosek, B. A. et al. "The preregistration revolution." *PNAS*, 115(11), 2600-2606 (2018). https://doi.org/10.1073/pnas.1708274114 - vol-04 の pre-registration

<a id="ref-15-9">[15-9]</a> W3C PROV Overview: https://www.w3.org/TR/prov-overview/ - 参照 2026-07-05（provenance モデルの標準）

<a id="ref-15-10">[15-10]</a> NIST AI Risk Management Framework (AI RMF 1.0): https://www.nist.gov/itl/ai-risk-management-framework - 参照 2026-07-05

<a id="ref-15-11">[15-11]</a> ISO/IEC 17025:2017 General requirements for the competence of testing and calibration laboratories: https://www.iso.org/standard/66912.html - 参照 2026-07-05（校正・試験ラボの品質要件）

<a id="ref-15-12">[15-12]</a> DoWhy: https://www.pywhy.org/dowhy/ - vol-04 の因果推論ライブラリ

<a id="ref-15-13">[15-13]</a> EconML: https://econml.azurewebsites.net/ - vol-04 の因果効果推定

<a id="ref-15-14">[15-14]</a> BoTorch: https://botorch.org/ - vol-04 のベイズ最適化
