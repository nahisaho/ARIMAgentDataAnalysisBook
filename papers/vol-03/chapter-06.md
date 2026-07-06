# 第6章　前処理と Augmentation の Skill 化 — Agentic 契約と深層 anti-leakage

第5章で設計した **深層 × Agentic Skill 仕様書テンプレート**（§5.9）と **provenance スキーマ**（§5.10）を、この章では **データ流路の契約**という側面から埋めていきます。データを深層モデルに流し込む前段には、統計/古典 ML には存在しなかった**深層特有の落とし穴**があります。

vol-02 第4-5 章では、`data_split` と CV 設計を **anti-leakage split contract** として言語化しました。この章はその契約を、深層 × Agentic 時代の**追加要件**——augmentation・事前学習データ重複・装置間差の判断ゲート——で拡張します。

> [!NOTE]
> **この章の到達目標**：
> - 深層版 anti-leakage split contract に必要な 3 つの追加要件を列挙できる
> - Augmentation contract の 3 要素（train-only / 物理妥当性 / エージェント権限）を Skill 契約に書ける
> - 装置間差を augmentation で「消してよい / 消してはいけない / Human 送り」の 3 分岐で判断できる
> - エージェントが augmentation の強度を勝手に変更しないための契約を書ける
>
> **この章で扱わないこと**：
> - 具体的な深層モデル実装（第7章）
> - 転移学習の fine-tune 戦略（第8章）
> - 不確かさ推定（第8-9章）
> - Foundation Model への augmentation 適用（第11-12 章で言及）

---

## 6.1 この章で作る Skill（2 種）

この章では 2 つの Skill を作ります。どちらも **契約定義中心**の Skill であり、以降の第6-13 章のハンズオンから常に呼び出されます。

| Skill 名 | 目的 | 出力 |
|---|---|---|
| **`deep_split_contract`** | 学習/検証/テスト分割 + 事前学習データ重複チェック + augmentation 適用範囲を **契約 YAML** として固定 | `split_contract.yaml`、分割インデックス、`leakage_audit.log` |
| **`augmentation_contract`** | データ型・装置種類ごとに augmentation の**契約**（何を・どの強度で・train のみに）を固定 | `augmentation_contract.yaml`、augmentation 適用ログ |

> [!IMPORTANT]
> **これらは「実装 Skill」ではなく「契約 Skill」**です。実際の augmentation 演算（回転・ノイズ付与など）を担うのは第6-7 章のモデル学習 Skill ですが、**契約が先**にあることで、エージェントが augmentation を勝手に強めることを防ぎます。契約なしのハンズオンは vol-03 では書きません。

---

## 6.2 なぜ深層 × Agentic で特別に問題になるか

vol-02 第4-5 章の anti-leakage split contract は、以下を扱いました：

- 分布分割（random split / stratified split / grouped split）
- CV 設計（fold 数、outer/inner CV、時系列 split）
- 特徴量に目的変数が混入していないか

深層 × Agentic では、これに **3 つの新問題**が加わります。

### 新問題 1：Augmentation の物理妥当性

深層モデルの標準ワークフローには **data augmentation**（回転・反転・ノイズ付与・intensity jitter など）が組み込まれています。しかし ARIM のような材料計測データでは、augmentation の多くが**物理的に無意味・または誤った学習を誘発**します。

| データ型 | よく使われる augmentation | 物理的に妥当か？ |
|---|---|---|
| SEM 画像 | 水平フリップ | **場合による**（結晶方位を持つ場合は破壊、非晶質なら OK） |
| SEM 画像 | 90° 回転 | **多くの場合 NG**（結晶方位・成長方向を破壊） |
| XRD パターン | 水平フリップ（2θ 反転） | **NG**（物理的に無意味、負角度は存在しない） |
| XRD パターン | intensity jitter（強度スケール） | **OK**（検出器ゲイン差の模倣） |
| スペクトル | 波数シフト | **場合による**（キャリブレーション差の範囲内なら OK、それ以上は装置種類の変化と混同） |
| 時系列（計測ログ） | 時間反転 | **NG**（因果性を壊す） |

**エージェントは物理妥当性を判断できません**。契約に「このデータ型ではどの augmentation を許すか」を **Human が事前定義**する必要があります。

### 新問題 2：事前学習データとの重複

Foundation Model や pretrained CNN（ImageNet 事前学習など）を使う場合、**事前学習データと自研究のテストデータが重複**すると、汎化性能を過大評価します。**FM 種別によって重複チェックの方法が異なる**点に注意します。

| FM 種別 | 代表例 | 事前学習データの性質 | 重複チェックの方法 |
|---|---|---|---|
| **NLP 系 FM**（テキスト事前学習） | MatBERT, ChemBERTa, SciBERT, MoLFormer | 論文コーパス・SMILES 文字列など | テキスト content_hash 照合、または「非公開・照合不能」として `unknown` 扱い |
| **結晶・原子構造系 FM** | CrystaLLM, M3GNet, MACE, UMA | Materials Project / ICSD / OQMD の構造データ | `material_id` などの識別子照合（`identifier_hash_join`）が可能な場合が多い |
| **画像事前学習**（ImageNet 等） | ResNet, ViT の事前学習 | 自然画像 | 材料画像とドメインが異なるため厳密照合は不要だが**ドメインギャップ**を第8章で評価 |
| **原子ポテンシャル**（trajectory 事前学習） | UMA など | MD trajectory | trajectory 単位で照合、または `unknown` |

- どの FM 種別でも、事前学習データが**公開されていない**場合（`pretraining_data_license: unknown`、第4章 §4.7）、重複チェック自体が不可能 → **契約上「重複可能性あり」と明記**して結果解釈時に反映
- **NLP 系 FM に対して `material_id` で照合する誤り**に注意（NLP 系 FM の事前学習に MP の識別子は使われていない場合が多い）

### 新問題 3：エージェントによる augmentation の勝手な強化

エージェントは「精度が上がった」を根拠に、**augmentation 強度を勝手に上げる**ことができます（回転角度を広げる、ノイズを増やす、mixup を混ぜる）。これは第5章 §5.6 で扱った**循環設計問題の深層版**の典型例です。契約で `augmentation_config` を固定し、変更を audit violation にする必要があります（第5章 §5.8 C 分類）。

---

## 6.3 深層版 anti-leakage split contract

vol-02 の split contract に、深層特有の 3 フィールドを追加します。

### 契約 YAML（拡張版）

```yaml
# split_contract.yaml
version: "1.0"

# vol-02 継承部
split_scheme:
  type: "grouped_5fold"       # random / stratified / grouped / temporal
  group_key: "instrument_id"  # 装置別 CV（第8章）
  test_ratio: 0.2
  val_ratio: 0.1
  random_seed: 42
  stratify_by: "class_label"

leakage_prevention:
  target_leakage_check: true      # 特徴量に目的変数が混入していないか
  temporal_order_preserved: true  # 時系列データの場合

# vol-03 追加：事前学習データ重複チェック（FM 種別ごとに `check_method` を選ぶ）
pretraining_data_overlap:
  applicable: true            # Foundation Model / pretrained weight を使うか
  fm_type: "atomistic_structure"  # nlp_text / atomistic_structure / image_pretrained / atomic_potential
  pretraining_dataset:
    name: "Materials Project"
    version: "v2024.01"
    identifier_column: "material_id"   # 重複判定に使うカラム（識別子照合が可能な場合）
  check_method: "identifier_hash_join"  # identifier_hash_join / content_hash / unknown
  overlap_action: "exclude_from_test"   # exclude_from_test / warn / block
  overlap_audit_log: "leakage_audit.log"
  # 事前学習データが公開されていない場合
  unknown_dataset_policy: "flag_and_document"

# vol-03 追加：Augmentation 適用範囲
augmentation_scope:
  applied_to: ["train"]       # ["train"] のみ許可、["val", "test"] は禁止
  seed_isolation: true        # augmentation 用 RNG を学習用と分離
  log_applied_transforms: true
  # オフライン事前生成 augmentation を使う場合の split 継承ルール
  offline_augmentation:
    enabled: false
    source_sample_id_column: "source_sample_id"   # 親サンプル ID を必須化
    split_inheritance: true                        # 親サンプルの split を継承
    augmentation_parent_sha256_recorded: true

# vol-03 追加：CLIP-like モデルの text-image leakage
# NOTE: pretraining_data_overlap と同じ 3 ケース構造を持つ
multimodal_leakage_check:
  applicable: false           # CLIP / MoLFormer など text-image / text-graph の場合 true
  modalities: []              # ["image", "text"] / ["graph", "text"] / ["structure", "text"] 等
  check_method: "unknown"     # identifier_hash_join / content_hash / unknown
  overlap_status: "unknown"   # confirmed_absent / confirmed_present / unknown
  unknown_policy: "restrict_claims"  # flag_and_document / block_fm_evaluation / restrict_claims
  overlap_audit_log: "multimodal_leakage_audit.log"
```

### `pretraining_data_overlap` の 3 ケース

| ケース | `check_method` | `overlap_action` | 備考 |
|---|---|---|---|
| 事前学習データが公開・識別子で照合可 | `identifier_hash_join` | `exclude_from_test` | Materials Project の `material_id` など |
| 事前学習データは公開だが識別子が異なる | `content_hash` | `exclude_from_test` or `warn` | 内容ハッシュで近似判定 |
| 事前学習データが非公開 | `unknown` | `flag_and_document` | ライセンスが `unknown`（第4章 §4.7） |

> [!WARNING]
> **`unknown` ケースこそが最も危険**です。「チェックできない ＝ 重複はない」と結論してはいけません（第3章の**absence of evidence ≠ evidence of absence**、第15章で失敗事例）。契約に `unknown_dataset_policy: flag_and_document` を書き、レポートに「事前学習データ非公開のため独立性未検証」と明記します。

---

## 6.4 データローダ設計と split の実装

vol-02 の CV 実装（sklearn の `GroupKFold` 等）は深層でもそのまま使いますが、**PyTorch/JAX の DataLoader を通す**ことで new なリスクが加わります。

### DataLoader worker seed の provenance 連携

PyTorch `DataLoader` は複数 worker を並列起動して I/O を並列化します。各 worker の RNG seed は**明示的に設定**しないと、実行ごとに異なる augmentation シーケンスが適用され、**再現性が壊れます**（第3章 §3.3 で扱った GPU 非決定性とは別の非決定性）。

```python
# 概念コード（実装は第7章）
def worker_init_fn(worker_id):
    # PyTorch は base_seed を内部で派生する。実際に使われた seed を記録することが重要
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    # 実際に使われた seed を provenance に append する（下記 YAML 参照）
    log_worker_seed(worker_id=worker_id, torch_seed=worker_seed)

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    worker_init_fn=worker_init_fn,   # 必須
    generator=torch.Generator().manual_seed(42),
)
```

**provenance への記録**（第5章 Layer 1 の `gpu_backend.random_seed_per_worker` に対応）。**「意図した seed」ではなく「実行時に実際に使われた seed」を記録**する点に注意：

```yaml
gpu_backend:
  # worker_init_fn が run-time に emit した実 seed をログ
  random_seed_per_worker:
    - {worker_id: 0, torch_initial_seed: 3948217, numpy_seed: 3948217, python_seed: 3948217}
    - {worker_id: 1, torch_initial_seed: 3948218, numpy_seed: 3948218, python_seed: 3948218}
    - {worker_id: 2, torch_initial_seed: 3948219, numpy_seed: 3948219, python_seed: 3948219}
    - {worker_id: 3, torch_initial_seed: 3948220, numpy_seed: 3948220, python_seed: 3948220}
```

> [!NOTE]
> **`torch.Generator().manual_seed(42)` から worker seed が `[42,43,44,45]` になるわけではありません**。PyTorch は base_seed から内部派生ルールで各 worker seed を作ります。**契約に書くのは「意図値」ではなく「実行時に emit された seed」**です。

### stratified × grouped × augmentation の順序

3 種類の制約が絡む場合、順序を **契約で固定**しないとエージェントが実装で入れ替える可能性があります。

```
[1] grouped split（装置別分割、vol-02 第8章）
    ↓
[2] stratified check（class 分布が train/val/test でずれていないか）
    ↓
[3] augmentation は train のみに適用（split 完了後）
```

### sklearn API との対応（実装ヒント）

- **grouped + stratified**：`StratifiedGroupKFold`（sklearn ≥ 1.0）を使う
- **grouped + holdout（test/val ratio 指定）**：`GroupShuffleSplit` で test を切り出し、残りに再度 `GroupShuffleSplit` で val を切り出す
- **`GroupKFold` は ratio / stratification を持たない**ため、`split_scheme.type = grouped_5fold` の実装は `StratifiedGroupKFold` を推奨（stratify 不要の場合は `GroupKFold`）

```yaml
# split_contract.yaml（実装ヒントを追記）
split_scheme:
  implementation_hint: "StratifiedGroupKFold when stratify_by is set, else GroupKFold"
  val_derivation: "inner_group_shuffle_split"   # train groups からさらに切り出す
```

### 順序違反の扱い（augmentation → split）

**原則 fatal**：augmentation してから split すると、同じ元サンプルの拡張版が train と test の両方に入り、**深層特有のリーク**が発生します。

**ただし例外**：オフライン事前生成 augmentation を使う場合、以下 3 条件を**すべて満たせば**許容されます。

1. 各拡張サンプルが `source_sample_id` を必ず保持する
2. split は `source_sample_id` で group 化して行う（子は必ず親と同じ split に入る）
3. `augmentation_parent_sha256` を provenance に記録

**条件を満たさない augmentation → split は fatal**（第5章 §5.8 A 分類）。

---

## 6.5 Augmentation contract Skill

Augmentation の契約は **3 要素**で書きます。

| 要素 | 意味 | エージェント可否 |
|---|---|---|
| **① Applied scope（適用範囲）** | train のみ / val / test のいずれに適用するか | **Human 固定**（変更不可） |
| **② Physical validity（物理妥当性）** | データ型ごとに許可される augmentation 種類 | **Human 事前定義** |
| **③ Agent permission（エージェント権限）** | エージェントが強度・種類を選ぶ範囲 | Skill 契約の `approved_augmentation_set` で数値化 |

### 用語定義：augmentation プリミティブ

「horizontal_flip」など曖昧な語を避け、**プリミティブ単位**で契約を書きます。

| プリミティブ | 意味 | 「軸の表示反転」との違い |
|---|---|---|
| `reverse_axis_order` | データ配列のインデックス順序を反転（例：波数配列 `[400..4000]` を `[4000..400]`） | **単なる表示反転（データ変更なし）は augmentation ではない**。プリミティブは配列そのものを書き換える |
| `coordinate_reflection` | 座標軸の物理的反転（例：波数 `x` → `-x`、時間 `t` → `-t`） | 物理的に非合法な場合が多い |
| `invert_intensity_sign` | 強度値の符号反転（`y` → `-y`） | 吸光度・強度に対して物理的に不可能な場合が多い |
| `rescale_intensity` | 強度の乗法スケーリング（検出器ゲイン差の模倣） | OK な代表例 |
| `baseline_shift` | 一定値の加算（バックグラウンド模倣） | 条件付き OK |
| `axis_shift` | 軸方向の平行移動（キャリブレーション差の模倣） | 範囲内で条件付き OK |
| `additive_noise` | ノイズ加算（Gaussian / Poisson） | OK |

> [!IMPORTANT]
> ライブラリ関数名の「horizontal_flip」「vertical_flip」は**表示座標系依存**で意味が変わります。契約は**プリミティブ名**で書き、実装関数は Skill 内部で writeup にマップします。

### 契約 YAML

```yaml
# augmentation_contract.yaml
version: "1.0"
data_type: "spectrum_ir"     # spectrum_ir / spectrum_raman / spectrum_uv_vis / image / pattern_xrd / timeseries / tabular / structure / multimodal
axis_unit: "cm^-1"           # cm^-1 / nm / eV / degree_2theta / seconds / index
applied_scope:
  train: true
  val: false                 # 変更禁止（第5章 §5.8 A 分類 fatal）
  test: false                # 変更禁止

# データ型ごとの許可 augmentation リスト（Human 事前定義）
# 各 range は「例示値ではなく、装置校正データから導出」することを推奨
allowed_augmentations:
  - name: "rescale_intensity"
    physical_validity: "OK"
    reason: "検出器ゲイン差の模倣として妥当"
    strength_range: [0.90, 1.10]      # 例；装置 QC ログから導出すること
    calibration_evidence_uri: "calib_2026_01.yaml"
  - name: "additive_noise"
    physical_validity: "OK"
    reason: "計測ノイズの模倣"
    sigma_range: [0.01, 0.05]         # 例；SNR 実測から導出
    calibration_evidence_uri: "calib_2026_01.yaml"
  - name: "axis_shift"
    physical_validity: "conditional"
    reason: "キャリブレーション差の範囲内のみ"
    shift_range_cm1: [-2.0, 2.0]      # IR の場合は cm^-1 単位

# 禁止 augmentation（プリミティブで指定）
prohibited_augmentations:
  - name: "coordinate_reflection"
    reason: "波数の符号反転は物理的に非合法"
    severity: "fatal"
  - name: "invert_intensity_sign"
    reason: "強度の符号反転は物理的に不可能"
    severity: "fatal"
  - name: "reverse_axis_order"
    reason: "計測順序の反転はキャリブレーション情報を破壊"
    severity: "fatal"

# エージェントの選択権限（第5章 §5.7 と連動）
agent_permission:
  level: L2                  # L1 / L2 / L3
  approved_augmentation_set: ["rescale_intensity", "additive_noise"]
  strength_selection: "within_range_only"   # within_range_only / fixed / free
  can_add_new_augmentation: false           # L1/L2/L3 すべて禁止
  can_modify_strength_range: false          # 全レベル禁止（audit violation）
```

> [!TIP]
> **数値範囲は「例示」であり本書のデフォルト値ではありません**。実運用では自装置の QC ログ・校正データから導出し、`calibration_evidence_uri` に根拠を紐づけます。

### データ型別の物理妥当性テーブル（Human が事前作成）

以下はプリミティブ単位で記述。**個々のプリミティブの可否は装置種別・試料の対称性・因果性の有無で変わる**ため、下記は**デフォルト prior**として提示し、自研究の contract で override します（`orientation_semantics` / `causality_semantics` フィールドで明示）。

| データ型 | OK（デフォルト） | 条件付き OK | fatal（デフォルト、override 可） |
|---|---|---|---|
| **XRD パターン**（`degree_2theta`） | rescale_intensity、additive_poisson_noise | peak_broadening（測定条件シミュレーション、範囲要指定） | coordinate_reflection（2θ 反転）、reverse_axis_order、90° rotation |
| **SEM 画像**（結晶方位あり） | rescale_intensity、additive_noise、gaussian_blur | 小角度回転（±5° 以内、成長方向を保つ） | 90°/180° rotation、coordinate_reflection（結晶方位破壊）、mirror_reflection |
| **SEM 画像**（非晶質） | 上 + mirror_reflection（`orientation_semantics: isotropic` 時）、90° rotation | mixup | invert_intensity_sign（下地判定を壊す場合） |
| **IR / Raman スペクトル**（`cm^-1`） | rescale_intensity、additive_noise | axis_shift（キャリブレーション範囲内） | coordinate_reflection（波数の符号反転）、reverse_axis_order、cutout（中心峰破壊時） |
| **UV-Vis スペクトル**（`nm`） | rescale_intensity、additive_noise | axis_shift（±0.5 nm 例、装置校正から導出） | coordinate_reflection、reverse_axis_order |
| **時系列**（計測ログ、`causality_semantics: causal`） | additive_noise、warping（時間軸微調整） | random_crop（時間窓） | coordinate_reflection（時間反転）、shuffle |
| **Tabular** | mixup（連続値のみ）、additive_noise（連続値のみ） | SMOTE（少数クラス補完、grouped CV と整合するか要確認） | カテゴリカル列の flip / permutation |
| **結晶・原子構造** | 対称群を保つ回転・平行移動 | supercell 拡張 | 対称性違反の変形（本書 scope 外、第13章 SSL で本格対応） |
| **Multimodal**（image + text / graph + text） | image / graph 側は上記 | — | text-image / text-graph の対応関係を崩す augmentation |

> [!IMPORTANT]
> **6→8 テンプレートへの拡張**：vol-01 第13章の 6 データ型（画像 / スペクトル / 回折 / 時系列 / 構造 / Tabular）を vol-03 では**UV-Vis と IR/Raman を分離**、**Multimodal を追加**して 8 テンプレートに拡張しています。§6.10 の対応表を参照。
>
> **override 必須フィールド**：`orientation_semantics`（`isotropic` / `anisotropic_with_growth_direction` / `crystallographic`）、`causality_semantics`（`causal` / `time_symmetric`）、`symmetry_group`（結晶データの場合）。これらを埋めないと**デフォルト prior が誤動作**します。

---

## 6.6 装置間差を augmentation で消してよいか

ARIM のような**複数装置**からのデータでは、augmentation を **「装置間差を消す道具」** として使いたくなる場面があります。これは**3 分岐で判断**します。

### 3 分岐の判断ゲート

| ケース | 判断 | 根拠 | 例 |
|---|---|---|---|
| **A. 消してよい** | augmentation で装置間差を吸収 | 差が**計測ノイズ・キャリブレーション差**の範囲内 | 装置 A/B で XRD 強度スケーリングが 0.9-1.1 倍のばらつき → `rescale_intensity=[0.9, 1.1]` で吸収 |
| **B. 消してはいけない** | grouped CV で装置差を明示的に扱う | 差が**装置の応答関数・測定原理**に由来 | 装置 A は透過型、装置 B は反射型 → augmentation で消すと**評価データが装置 A/B 両方を含む場合に誤って高精度**に見える |
| **B'. 部分的に消す（decompose）** | ノイズ・校正成分のみ augmentation、応答関数成分は grouped CV | 差が**混在（ノイズ + 応答関数）** | 校正差は `rescale_intensity` で吸収、応答関数差は grouped CV で扱う。各成分の**証拠 URI 必須** |
| **C. Human 送り** | エージェントは判断せず flag | 差の性質が**判定不能** | 装置間差の物理的原因が特定できない → 契約に `flag_for_human_review` を書き、Human が A/B/B' を判定 |

### 判断のロジックフロー

```mermaid
flowchart TD
    START["装置間差を augmentation で消す提案"] --> Q1{"差の原因が特定できるか?"}
    Q1 -- "No" --> C["C: Human 送り (flag)"]
    Q1 -- "Yes" --> Q2{"差の性質は?"}
    Q2 -- "計測ノイズ・キャリブレーション差" --> A["A: 消してよい"]
    Q2 -- "装置の応答関数・測定原理" --> B["B: 消してはいけない (grouped CV)"]
    Q2 -- "両方が混在" --> Q3{"成分分解の証拠があるか?"}
    Q3 -- "Yes" --> BP["B': 部分的に消す (decompose)"]
    Q3 -- "No" --> B
```

### なぜ「消してはいけない」ケースが多いか

装置間差を augmentation で消すと、モデルは「装置間差は無視してよい」を学習します。しかし本番運用で**新しい装置**（学習時に見なかった装置）から来たデータに対しては、装置間差が**未知の分布シフト**として現れ、**信頼できない予測**を返します。

**契約による予防（提案ゲート含む）**：

```yaml
# split_contract.yaml の grouped split と組み合わせる
domain_shift_awareness:
  cross_instrument_validation: true       # 装置別 CV で汎化性能を評価
  augmentation_erases_instrument_diff: false  # 装置間差を消すか（デフォルト false）
  instrument_diff_analysis_required: true # Skill 実行前に装置間差の性質を分析
  diff_components:                        # B'（decompose）ケースの明示
    - component: "calibration_gain"
      evidence_source: "cross_instrument_calib_2026_01.csv"
      approved_transform: "rescale_intensity"
      human_review_required: false
    - component: "detector_response_function"
      evidence_source: "detector_spec_A_B.pdf"
      approved_transform: null            # augmentation では消さない
      human_review_required: true
  # エージェントが「装置間差を消す」提案をした場合のハードゲート
  instrument_diff_erasure_proposal:
    default: "blocked"
    requires_human_approval: true
    approval_record_id: null              # 承認時に埋める
    on_unapproved_attempt: "audit_violation"   # または "fatal"
    attempt_log: "instrument_diff_erasure_attempts.log"
```

---

## 6.7 エージェントの augmentation 権限（Ch04 §5.7 の適用）

第5章 §5.7 で定義した L1/L2/L3 を、augmentation に特化して具体化します。

### L1/L2/L3 の augmentation 権限

**重要な前提**（第5章 §5.7 と整合）：**L1 は本質的に「推論のみ」**です。したがって **L1 では学習時 augmentation は実行されません**。この節の表は、「学習ジョブが承認されて起動される場合」の各レベルでの augmentation 選択権限を示します。**学習ジョブの起動そのものは、必ず `training_job_approval` ゲート（第5章 §5.7）を通ります**。

| 権限 | 学習実行可否 | augmentation でできること | 例 |
|---|---|---|---|
| **L1（推論のみ）** | 学習実行不可 | 推論時に augmentation なし。契約の検証・ログ確認のみ | 契約 YAML と実 provenance ログを照合、逸脱を検知 |
| **L2（承認済み範囲内 fine-tune）** | 承認レコードに紐づく範囲で可 | `approved_augmentation_set` 内から選択可、`strength_range` 内で強度選択可 | `rescale_intensity` の `strength_range=[0.90, 1.10]` から `1.05` を選択 |
| **L3（事前承認ワークフロー内自律）** | 承認ワークフロー内で可 | L2 + 契約内で augmentation の**組み合わせ**を選択可 | `rescale_intensity + additive_noise` の組み合わせを試す（**各 augmentation は契約範囲内**） |

### 全レベル禁止

以下は L1/L2/L3 すべてで禁止（第5章 §5.8 C 分類 audit violation または fatal）：

| 禁止項目 | severity | 理由 |
|---|---|---|
| `augmentation_config` の`prohibited_augmentations` を追加せずに削除 | fatal | 物理妥当性違反の可能性 |
| `strength_range` の上限を超える強度を選択 | audit violation | 循環設計問題の深層版 |
| val/test への augmentation 適用 | fatal | データリーク |
| 契約にない新 augmentation の追加 | audit violation | 事後追加は精度水増しのシグナル |
| augmentation の適用ログを保存しない | audit violation | 事後の失敗解析が不能 |

### エージェントの意思決定ログ

エージェントが augmentation を選択した場合、以下を必ずログに残します（第5章 provenance Layer 3 の `augmentation_config` に加えて）：

```yaml
augmentation_decision_log:
  - timestamp: "2026-02-01T10:00:00Z"
    agent_level: L2
    selected_augmentations:
      - name: "rescale_intensity"
        strength: 1.05          # strength_range=[0.90, 1.10] 内
    reason: "val loss がプラトー、強度範囲上限側に少し寄せる"
    approved_by_contract: true
    contract_ref: "augmentation_contract.yaml#v1.0"
    training_job_approval_id: "APP-2026-0201-01"
```

---

## 6.8 ハンズオン概観：ARIM 風合成スペクトルの Augmentation Skill

具体的な実装は第7章の 1D CNN Skill と統合しますが、この節では**契約と適用ログの雛形**のみを示します。

### 想定するデータ

- ARIM 風合成 IR スペクトル（波数 4000-400 cm⁻¹、5 装置分、各装置 200 サンプル、5 材料クラス）
- 装置間差（**いずれも例示値；実運用では自装置の校正データから導出**）：`rescale_intensity ±10%`、`axis_shift ±2 cm⁻¹`、ノイズレベル差

### 契約セット（この Skill が読み込む 2 ファイル）

```
skills/deep_split_contract/
├── skill.yaml                  # Skill 仕様書（第5章 §5.9 テンプレート）
├── split_contract.yaml         # §6.3 の分割契約
└── augmentation_contract.yaml  # §6.5 の augmentation 契約

skills/augmentation_contract/
├── skill.yaml
└── data_type_templates/       # 8 テンプレート（§6.5, §6.10）
    ├── spectrum_ir_raman.yaml
    ├── spectrum_uv_vis.yaml
    ├── image_crystalline.yaml
    ├── image_amorphous.yaml
    ├── pattern_xrd.yaml
    ├── timeseries.yaml
    ├── tabular.yaml
    └── structure.yaml
```

### CI CPU smoke test（第5章 §5.5 と連動）

```yaml
# ci_smoke_test.yaml
data_subset:
  n_samples: 20              # 各装置から 4 サンプル × 5 装置
  n_classes: 3               # 5 クラス中 3 クラスのみ
epochs: 2                    # フル学習 (30 epoch) の 15%
augmentation:
  applied: true              # augmentation ロジックが CI で動くか確認
  reduced_strength: 0.5      # フル強度の 50%（時間短縮）
expected:
  loss_decrease: true        # 2 epoch で loss が下降傾向
  no_nan_or_inf: true
  augmentation_only_on_train: true  # 契約通り
```

**フル実装は第7章**で扱います。

---

## 6.9 失敗パターンと改善版

| 失敗パターン | 原因 | 改善版 |
|---|---|---|
| **精度が上がったので augmentation を強めたら、新装置で崩れた** | 装置間差を augmentation で消していた | 装置間差の性質を分析（§6.6）→ 消してはいけないケースなら grouped CV で扱う |
| **train と test に同じ augmentation を適用してしまった** | データローダで applied_scope 分離不足 | `augmentation_contract.applied_scope.test: false` を契約で固定、Skill 起動時に assert |
| **augmentation してから split したら、精度が異常に高くなった** | 順序違反（§6.4） | 契約に「[1] split → [2] augmentation」と明記、実装は split 済みインデックスに対して適用 |
| **事前学習データとテストが重複していた** | `pretraining_data_overlap` を未チェック | §6.3 の contract で `check_method` を必須化、`unknown` の場合は `flag_and_document` |
| **エージェントが `strength_range` を超えて augmentation を強めた** | `agent_permission` の enforcement 不足 | Skill 起動時に契約 vs 実効値を比較、範囲外なら実行前に停止（第5章 §5.8 audit violation） |
| **同じ元サンプルの augmentation バリアントが train と val の両方に入った** | augmentation を split 前に適用 | augmentation を split 後の train のみに限定、val/test には元サンプルのみ |

---

## 6.10 他データ型への転用（vol-01 第13章との対応）

§6.5 の物理妥当性テーブルは、vol-01 第13章の 6 データ型テンプレートを **vol-03 で 8 テンプレートに拡張**したものです（UV-Vis と IR/Raman を分離、Multimodal を追加）。

| vol-01 第13章のデータ型 | vol-03 第6章の契約テンプレート | 主な augmentation の判断ポイント |
|---|---|---|
| ① 画像（顕微鏡・SEM） | `image_crystalline.yaml` / `image_amorphous.yaml` | 結晶方位（`orientation_semantics`）で mirror / rotation の可否が変わる |
| ② スペクトル（IR/Raman/UV-Vis） | `spectrum_ir_raman.yaml`（cm⁻¹） / `spectrum_uv_vis.yaml`（nm） | `axis_unit` が異なる → **同一テンプレートでは扱わない**（Critical fix） |
| ③ 回折パターン（XRD） | `pattern_xrd.yaml` | rescale_intensity は OK、coordinate_reflection（2θ 反転）は fatal |
| ④ 時系列（計測ログ） | `timeseries.yaml` | `causality_semantics: causal` なら coordinate_reflection fatal、`time_symmetric` なら override 可 |
| ⑤ 分子・結晶構造 | `structure.yaml`（本書は骨格のみ、第13章 SSL で詳述） | 対称性群（`symmetry_group`）を尊重した augmentation のみ |
| ⑥ Tabular（実験計画・組成） | `tabular.yaml` | 連続値のみ mixup、カテゴリカル列 permutation は禁止 |
| — | `multimodal.yaml`（vol-03 新設） | modality 対応関係を崩す augmentation を禁止（§6.5 参照） |

### applicability domain の判定

vol-02 第8章 §8.10 で扱った applicability domain（AD）は、深層でも **契約フィールド**として残します：

```yaml
# augmentation_contract.yaml
applicability_domain:
  feature_range_check:
    # axis_unit と対応した range キーを使う（Critical fix）
    axis_unit: "cm^-1"
    wavenumber_range_cm1: [400, 4000]     # IR の場合
    # UV-Vis の場合は wavelength_range_nm: [200, 800] を使用
    intensity_normalization: "unit_max"
  # AD 外への対応（複数選択肢）
  out_of_domain_action: "route_to_human"
  # 選択肢: block_augmentation / predict_without_augmentation_with_warning /
  #        route_to_human / reject_sample / request_calibration_data /
  #        start_domain_adaptation_workflow
```

**AD 外の対応方針**（用途に応じて選択）：

| action | 用途 |
|---|---|
| `block_augmentation` | AD 外サンプルに augmentation を適用しない（推論は継続） |
| `predict_without_augmentation_with_warning` | augmentation なしで推論、警告フラグ付与 |
| `route_to_human` | Human に判断を委ねる（デフォルト推奨） |
| `reject_sample` | 予測を拒否（安全側運用） |
| `request_calibration_data` | 追加校正データを要求 |
| `start_domain_adaptation_workflow` | ドメイン適応パイプラインを起動（第8章参照） |

**エージェントは AD 外のサンプルに勝手に augmentation を適用してはいけません**。AD 外に augmentation を適用すると、augmentation が「AD を広げた」と誤解される危険があります。

---

## 6.11 章末チェックリスト・ワーク

### チェックリスト

- [ ] 深層版 anti-leakage split contract の 3 追加要件（事前学習データ重複・augmentation 適用範囲・multimodal leakage）を列挙できる
- [ ] `pretraining_data_overlap.check_method` の 3 ケース（`identifier_hash_join` / `content_hash` / `unknown`）を使い分けられる
- [ ] Augmentation contract の 3 要素（Applied scope / Physical validity / Agent permission）を YAML で書ける
- [ ] 装置間差を augmentation で消す 3 分岐（消してよい / 消してはいけない / Human 送り）の判断基準を説明できる
- [ ] L1/L2/L3 の augmentation 権限の差を具体例で説明できる
- [ ] 全レベル禁止項目 5 つ（§6.7）を列挙できる
- [ ] augmentation の順序違反（augmentation → split）が起こす漏洩を説明できる

### ワーク

自研究室で最頻に扱うデータ型（1 つ）を選び、以下を実施：

1. §6.5 のデータ型別物理妥当性テーブルを、自装置の測定原理に照らして書き直す
2. `allowed_augmentations` / `prohibited_augmentations` を **各 3 項目以上**で埋めた `augmentation_contract.yaml` を作成
3. 装置間差の性質を分析し、§6.6 の 3 分岐（A/B/C）のどれに該当するかを判定
4. 該当装置での `strength_range` を、装置校正データ（あれば）から決定
5. L2 権限でエージェントに許す範囲を `approved_augmentation_set` として決定

---

## 6.12 本章のまとめ

- 深層 × Agentic の**データ流路契約**は、vol-02 の anti-leakage split contract に、**augmentation 契約 + 事前学習データ重複 + multimodal leakage** の 3 拡張を加えたもの
- **Augmentation contract の 3 要素**：Applied scope（train のみ）、Physical validity（データ型ごとの許可リスト）、Agent permission（L1/L2/L3 での選択範囲）
- **装置間差の augmentation** は 3 分岐（消してよい / 消してはいけない / Human 送り）で判断。安易に消すと未知装置で崩壊
- エージェントは augmentation の**種類追加・強度上限超え・val/test 適用**を絶対に許されない（第5章 §5.8 の C 分類・A 分類）
- 契約は 2 ファイル（`split_contract.yaml` と `augmentation_contract.yaml`）に集約し、第6-13 章のハンズオンから参照する

第7章では、この章の契約を実際に読み込んで動く**教師あり深層 Agentic Skill 3 種**（1D CNN / 2D CNN / Transformer 骨格）を実装します。

---

## 参考資料

### 本書内

- **第5章 §5.7-4.10**：Agentic 学習権限設計（L1-L3）、Skill 仕様書テンプレート、provenance スキーマ
- **第5章 §5.5**：CI CPU smoke test 設計
- **第5章 §5.8**：禁止事項の severity 表（A/B/C/D/E）
- **第3章 §3.6**：事前学習データと評価データの重複、fine-tune データリーク
- **第4章 §4.7**：事前学習重みライセンス、`pretraining_data_license`
- **第7章**：この章の契約を読み込む教師あり深層 Skill
- **第15章**：augmentation の agent-side 強化による精度偽装事例

### vol-02 / vol-01 参照

- **vol-02 第5章**：anti-leakage split contract の基礎
- **vol-02 第6章**：教師あり学習 Skill の共通構造、ハンズオン A/B/C
- **vol-02 第8章 §8.5-7.10**：grouped CV / applicability domain
- **vol-01 第13章**：6 データ型テンプレート（§5.10 との対応）
- **vol-01 第6章**：Human-in-the-loop の予防的 3 原則（§6.6 の Human 送りに対応）

### 外部資料

- PyTorch DataLoader worker seeding — [https://pytorch.org/docs/stable/notes/randomness.html#dataloader](https://pytorch.org/docs/stable/notes/randomness.html#dataloader)
- torchvision transforms（augmentation の物理妥当性を判断する参考リスト）— [https://pytorch.org/vision/stable/transforms.html](https://pytorch.org/vision/stable/transforms.html)
- Materials Project データセット — [https://materialsproject.org/](https://materialsproject.org/)
- Adebayo et al. (2018) "Sanity Checks for Saliency Maps" — attribution sanity check の原論文（第11章で本格対応）
