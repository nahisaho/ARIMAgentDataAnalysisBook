# 第15章 組織展開と終章 — GPU リソース・モデル配布・Agentic 責任分担

> [!NOTE]
> **この章の到達目標**
>
> - 深層 × Agentic Skill を **単一研究室から組織横断（ARIM 施設単位）に展開する際の設計**を示す
> - GPU リソースの共有権限、モデル重みの配布と使用制限、Agentic 責任分担を **Skill 契約**として言語化する
> - vol-01+02+03 の到達点を **1 枚の責任分担マトリクス**に統合する
> - vol-04（因果・実験計画）以降の道しるべを与える
>
> **扱わないこと**
>
> - クラウドプロバイダの具体的な IAM 設定（付録 C で軽く触れる）
> - 契約・法務・データ利用規約の逐条解釈（法務部門の責務）
> - 学術倫理審査（IRB / 動物実験倫理などのプロセス）
> - vol-04 以降の内容そのもの（本章ではロードマップのみ）

---

## 15.1 この章で作る Skill

本章は「終章」の位置付けですが、組織展開のために **3 つの Skill 契約テンプレート** を提供します。

| Skill | 役割 |
|---|---|
| `organization_gpu_resource_policy` | 研究室単位 / 施設単位 / 外部連携単位で GPU 使用権限・優先度・課金単位を宣言し、Ch4 の agent tier に紐づけるポリシー Skill |
| `model_distribution_contract` | 学習済み重みを組織外・組織内・下位研究室に配布する際の provenance 継承、使用制限、再配布可否、失効条件を宣言する Skill |
| `responsibility_ledger` | Ch4-13 の各 Skill が生成した決定・承認・監査記録を、**責任主体（agent / human / group）ごとに集約**して長期保管する台帳 Skill |

いずれも「新規手法を導入する Skill」ではなく、**既存章の Skill が生む provenance を組織レベルで束ねるガバナンス層**です。

## 15.2 なぜ「組織展開」を最終章に置くか

第4-14章までで、単一の研究プロジェクトの中で「深層 × Agentic」を回すための Skill は揃いました。しかし現実の ARIM 施設は次の性質を持ちます：

- **GPU は共有資源**。1 研究室が占有していると他の研究が止まる
- **モデル重みは組織資産**。学生が卒業しても組織に残り、後続が使う
- **意思決定は分散**。試料採択・測定条件・解析結果の承認は、指導教員・施設長・産学連携先で **役割が違う**
- **責任は長期に及ぶ**。論文出版から 5-10 年後に「その解析は誰の判断か」を再現する必要がある

これらは技術的には「provenance を長期保管し、責任主体を明示する」で解けますが、**Skill 契約に落とさない限り、agent は組織構造を理解できません**。第14章までの Skill が「実行の正しさ」を保証したのに対し、本章の Skill は **「誰がその実行を許可し、誰がその結果に責任を持つか」を機械可読にする**役割を持ちます。

> [!IMPORTANT]
> 「責任」は法的責任と一致するとは限らない。本章の `responsibility_ledger` は **「後から辿れる形で誰が何を決めたか」を記録する台帳**であり、法的責任分担そのものではない。法務・契約は必ず所属機関の担当部署と協議のこと。

## 15.3 GPU リソースの共有と権限

### 15.3.1 3 層の GPU リソース単位

ARIM 施設と研究室の関係を、GPU リソース側からみると次の 3 層に整理できます：

| 層 | 主体 | 典型的な GPU 規模 | 権限の粒度 |
|---|---|---|---|
| **L-Org** | 施設 / 学部 / 全学計算機センター | 数百 GPU、共有 | プロジェクト単位で quota、Human 承認必須 |
| **L-Lab** | 研究室 | 数〜数十 GPU、専有 | 教員配下のメンバー単位、教員が承認者 |
| **L-Self** | 個人 / 学生 | ノート PC の 1 GPU または vCPU | 本人のみ、agent 権限は L3 まで（第4章） |

第4章では agent tier を L1（読み取り）/ L2（軽量計算）/ L3（学習実行）に分けましたが、これは **どの GPU リソース層でも同じ tier 定義を使う**ことを想定しています。組織展開では、agent tier と GPU 層を掛け合わせた **アクセス行列** を明示する必要があります：

| agent tier ＼ GPU 層 | L-Self | L-Lab | L-Org |
|---|---|---|---|
| **L1（読み取り）** | 自由 | 教員通知のみで許可 | プロジェクト参加者に限り許可 |
| **L2（軽量計算）** | 自由 | 教員承認（月次まとめでも可） | プロジェクト参加者かつ quota 内 |
| **L3（学習実行）** | 自由（自機） | **教員による都度承認**（`training_job_approval` provenance） | **施設長 or 承認委員会による都度承認**、加えて quota・優先度制御 |

> [!WARNING]
> L-Org の L3 学習は **quota だけでは不十分**である。深層学習は 1 回の失敗も 1 回の成功も同じ GPU 時間を消費するため、失敗を許容する quota 設計をしないと、agent が「絶対失敗しない範囲」しか試行しなくなり、探索が痩せる。

### 15.3.2 Skill 契約：`organization_gpu_resource_policy`

```yaml
# organization_gpu_resource_policy.yaml
skill: "organization_gpu_resource_policy"
version: "1.0.0"
purpose: >
  組織単位（研究室 / 施設 / 全学）で GPU リソースの使用権限・優先度・
  quota を宣言し、Ch4 の agent tier と掛け合わせて実行可否を判定する。

resource_layers:
  L_Self:
    scope: "individual laptop or personal workstation"
    approver_role: "self"
    max_agent_tier_allowed: "L3"
    quota_enforcement: "advisory_only"

  L_Lab:
    scope: "research group shared cluster"
    approver_role: "principal_investigator_or_delegate"
    max_agent_tier_allowed: "L3"
    quota_enforcement: "hard_gpu_hours_per_month"
    training_job_approval_required_for_L3: true
    approval_can_be_batched: true                       # 月次まとめ承認可
    batched_approval_scope_max_hours: 40
    batched_approval_period_days: 30

  L_Org:
    scope: "facility-wide or university-wide shared HPC"
    approver_role: "facility_review_committee"
    max_agent_tier_allowed: "L3"
    quota_enforcement: "hard_gpu_hours_per_project_per_month"
    training_job_approval_required_for_L3: true
    approval_can_be_batched: false                      # 個別ジョブ承認
    priority_lanes:
      - "reproducibility_reruns"                        # 監査再実行は優先
      - "gate2_approved_projects"                       # Ch13 Gate2 通過済み
      - "exploratory"
    failure_budget_gpu_hours_per_month: "at_least_30_percent_of_quota"
    # 探索の萎縮を防ぐため、失敗許容を明示

# agent tier × resource layer の実行可否行列
access_matrix:
  L1_read:
    L_Self: "allow"
    L_Lab: "notify_pi"
    L_Org: "require_project_membership"
  L2_light_compute:
    L_Self: "allow"
    L_Lab: "monthly_batch_approval"
    L_Org: "require_project_membership_and_quota"
  L3_training:
    L_Self: "allow"
    L_Lab: "per_job_approval_or_batched_within_scope"
    L_Org: "per_job_approval_required"

# provenance 記録要件
provenance_records:
  - "resource_layer_used"                               # どの層で回したか
  - "approver_identity_and_role"                        # 承認者
  - "approval_scope"                                    # 個別 or batched
  - "gpu_hours_consumed"
  - "quota_state_before_and_after"

acceptance:
  layer_and_tier_matrix_deterministic: true
  approver_role_never_self_for_L_Org_and_L_Lab_L3: true
  quota_state_snapshot_signed_hourly: true

never_allowed:
  - "agent_writes_own_approver_field"                   # AG-06 と同じ
  - "consume_gpu_beyond_signed_quota_snapshot"
  - "reassign_project_membership_by_agent"
  - "silently_downgrade_layer_from_L_Org_to_L_Lab_to_bypass_committee"
```

> [!NOTE]
> 「同じ agent tier でも、どの GPU 層で回したかによって承認者が変わる」ことが重要。単一 tier 定義だけでは、施設共有 GPU で暴走した agent を止められない。

## 15.4 モデル重みの配布と使用制限

### 15.4.1 4 種類の配布シナリオ

学習済み重みは、次の 4 パターンで移動します：

| シナリオ | 例 | 必要な制御 |
|---|---|---|
| **Intra-Lab**（研究室内） | 学生から教員へ、次年度の後輩へ | provenance 継承のみ |
| **Intra-Org**（施設内） | 研究室 A から研究室 B へ、施設共有モデルとして登録 | provenance + 使用制限 + 施設内 registry 登録 |
| **Inter-Org**（機関間） | 大学 A から大学 B へ、産学連携先へ | provenance + 契約 + 使用制限 + 失効条件 |
| **Public**（公開） | Hugging Face Hub / GitHub Releases 等 | provenance + ライセンス + 学習データ由来の制限（第11章） |

**配布のたびに provenance が失われる**のが最も避けたい失敗です。第14章 §14.4 の AG-08（署名検証スキップ）と対称的に、**「受け取った側が provenance を復元できない配布」は組織的損失**です。

### 15.4.2 Skill 契約：`model_distribution_contract`

```yaml
# model_distribution_contract.yaml
skill: "model_distribution_contract"
version: "1.0.0"
purpose: >
  学習済み重みを配布する際に、provenance chain・使用制限・失効条件を
  Skill 契約として同梱する。受領側は同じ契約を継承する。

distribution_bundle:
  # 配布物本体
  weights_file:
    format_allowed: ["safetensors", "pt_with_hash_check"]
    hash_algorithm: "sha256"
    hash_recorded_in_manifest: true

  # provenance chain（Ch13 の integrated_provenance_chain を継承）
  provenance_manifest:
    references_ch13_integrated_provenance_chain: true
    minimum_required_slots:
      - "training_data_provenance"
      - "training_job_approval"
      - "checkpoint_registry_provenance"
      - "foundation_model_provenance"                   # 転移学習の場合
      - "audit_result_provenance"                       # Ch14 の pass/warn
    audit_result_provenance_decision_at_distribution: "must_be_pass_or_warn"

  # 使用制限（Skill 側で強制）
  usage_restrictions:
    allowed_task_types: []                              # 空 = 契約で明示
    forbidden_task_types_examples:
      - "medical_diagnosis_when_trained_on_materials_data"
      - "human_subject_classification"
      - "safety_critical_inference_without_uncertainty_gate"
    inference_uncertainty_gate_required: true           # Ch8 の停止契約継承
    data_domain_boundary_declared: true

  # 失効条件
  expiration:
    hard_expiry_date: "ISO8601"                         # 使用不可日
    revocation_check_endpoint: "url"                    # 撤回リストの照会
    check_frequency_days_max: 30
    on_revoked_action: "block_inference_and_alert_human"

  # ライセンス
  license:
    weights_license: "e.g., CC-BY-NC-4.0"
    upstream_fm_license_compatibility_verified: true    # 第11章 fm_update_gate
    redistribution_allowed: "yes | no | conditional"
    conditional_terms: "url or text"

# 受領側の契約継承
recipient_obligations:
  reinstall_audit_before_first_inference: true         # 受領後の再監査
  cannot_strip_provenance_manifest: true
  cannot_relax_usage_restrictions: true                # 制限緩和は禁止
  can_add_further_restrictions: true                   # 追加制限は可
  must_forward_revocation_notice_to_downstream: true

acceptance:
  distribution_bundle_manifest_signed_by_originating_org: true
  weights_hash_verified_on_receipt: true
  provenance_chain_re_audited_on_receipt: true
  usage_restriction_enforced_by_inference_skill: true

never_allowed:
  - "distribute_without_audit_result_provenance"       # Ch14 pass/warn 未取得の配布
  - "strip_provenance_to_reduce_bundle_size"
  - "relax_usage_restrictions_downstream"
  - "distribute_after_hard_expiry_date"
  - "ignore_revocation_notice"
  - "redistribute_when_upstream_forbids"
```

> [!TIP]
> Hugging Face Hub 等の公開配布では、`revocation_check_endpoint` を README の脚注として明示することで、ユーザーが手動チェック可能になる。組織内配布では registry がこれを自動化できる。

### 15.4.3 転移元 FM の失効伝播

Ch11 で扱った Foundation Model の revocation は、配布連鎖の**上流**で起きます。転移学習された派生モデルは、上流 FM が revoke されたときに何を行うかを事前に決めておく必要があります：

| 上流 FM の状態変化 | 派生モデル側の推奨アクション |
|---|---|
| **soft-deprecation**（後継版推奨） | 記録のみ、推論継続可 |
| **security-revocation**（脆弱性発覚） | 推論停止、Ch14 §14.8 の `runtime_and_periodic_audit_guards` を発火 |
| **license-revocation**（ライセンス変更） | 新規推論停止、既存記録は保持 |
| **data-provenance-revocation**（訓練データ由来問題） | 推論停止、`responsibility_ledger` に事象を記録し、Human 判断 |

**判断は agent ではなく Human が行う**。agent は状態遷移を検知して停止するだけです（Ch14 の `blocking_decision_field_never_rewritten` と対称）。

## 15.5 Agentic 責任分担マトリクス（vol-01+02+03 統合）

### 15.5.1 3 種類の責任

意思決定と実行を分解すると、責任は 3 種類に分かれます：

| 責任種別 | 内容 | 主体になれるか |
|---|---|---|
| **決定責任**（Decision） | どの選択肢を採用するかを決める | Human のみ（重要決定）／ Agent 可（軽微決定） |
| **実行責任**（Execution） | 決定に基づいてタスクを実行する | Agent 可 |
| **記録責任**（Recording） | 決定と実行を後から辿れる形で記録する | Agent 可（automated）／ Human は spot-check |

**「実行できる」＝「決定できる」ではない**。深層 × Agentic の失敗の多くは、agent が「実行できる」ことを「決定できる」と誤解した結果（Ch14 の AG-xx）です。

### 15.5.2 責任分担マトリクス（vol-01+02+03 統合）

各書籍で導入した Skill / 決定点を、責任種別で分類します：

| 決定点 | 書籍・章 | 決定責任（Human） | 実行責任（Agent） | 記録責任 |
|---|---|---|---|---|
| データ contract 定義 | vol-01 Ch4 / vol-02 Ch4 / vol-03 Ch4 | 分析責任者 | – | 自動記録 |
| 試料採択 / 除外 | vol-01 Ch7 | 実験責任者 | 提案のみ | 自動記録 |
| 前処理パイプライン設計 | vol-01 Ch6 / vol-02 Ch5 | 分析責任者 | パイプライン実行 | 自動記録 |
| 統計モデル選択 | vol-02 Ch6-8 | 分析責任者 | fit / predict 実行 | 自動記録 |
| 事前分布の設定 | vol-02 Ch10 | 分析責任者 + ドメイン専門家 | サンプリング実行 | 自動記録 |
| Gate1（データ品質） | vol-01 Ch7 / vol-03 Ch4 | 分析責任者 | 判定材料の集計 | 自動記録 |
| **fine-tune 起動判断** | vol-03 Ch4-7 | **研究室 PI（L-Lab）／施設委員会（L-Org）** | 学習ジョブ実行 | 自動記録 |
| **FM の default version 変更** | vol-03 Ch11 | **プロジェクトリーダー + 施設** | 差し替え実行 | 自動記録 |
| **checkpoint の上書き / 破棄** | vol-03 Ch4 | 分析責任者 | – | 自動記録 |
| **不確かさ超過時の停止** | vol-03 Ch8-9 | – | **自律停止のみ**、Human 通知 | 自動記録 |
| **Gate2（statistical adequacy）** | vol-03 Ch13 | 分析責任者 + 統計レビュア | 判定材料の集計 | 自動記録 |
| **Gate3（外部妥当性）** | vol-03 Ch13 | プロジェクトリーダー + ドメイン専門家 | 判定材料の集計 | 自動記録 |
| モデル重みの組織外配布 | vol-03 Ch15 | 施設長 + 法務 | 配布バンドル生成 | 自動記録 |
| 監査 report の発行 | vol-03 Ch14 | – | 生成のみ | 自動記録 |
| 監査 report の署名 | vol-03 Ch14 | **non-agent verifier** | – | 自動記録 |
| 論文投稿判断 | vol-01 Ch11 / vol-03 Ch15 | 著者一同 | – | 手動記録 + provenance link |
| Revocation 発火判断 | vol-03 Ch15 | 施設長 + 分析責任者 | 停止実行 | 自動記録 |

> [!IMPORTANT]
> **黄色でハイライトした行は、深層 × Agentic 特有の決定点** — 従来 ML では発生しないか、発生してもコストが低いため人間が意識しなかった判断です。組織展開では、これらの Human 決定責任を **明示的にロール割り当て**しないと、agent が代行してしまう。

### 15.5.3 Skill 契約：`responsibility_ledger`

```yaml
# responsibility_ledger.yaml
skill: "responsibility_ledger"
version: "1.0.0"
purpose: >
  Ch4-14 の各 Skill が生成した決定・承認・監査記録を、責任主体
  (agent / human / group) ごとに集約し、長期保管する台帳を提供する。

ledger_entry_schema:
  entry_id: "sha256_of_content"
  timestamp_utc: "ISO8601"
  decision_type: "one_of_matrix_15_5_2"                # 上記マトリクスの決定点
  decision_responsibility: "human | agent | shared"
  execution_responsibility: "human | agent"
  recording_responsibility: "automated | manual"
  actor_identity:
    role: "e.g., PI, facility_head, analyst, agent_L3"
    identity_ref: "opaque_id_or_pubkey"
    non_agent_signature_required: "bool"
  linked_provenance:
    ch_source: "e.g., vol-03/ch04"
    provenance_uri: "content-addressable"
    audit_id_if_any: "sha256"                          # Ch14 audit_id
  human_readable_context: "short text"

storage_requirements:
  append_only: true
  minimum_retention_years: 10                          # 論文再現期間 + α
  offline_backup_frequency_days_max: 7
  offline_backup_signed_by_org: true
  integrity_check_frequency_days_max: 30
  integrity_check_uses_merkle_root: true

# ledger と provenance chain の関係
relation_to_provenance:
  ledger_is_view_over_chain: true                      # 台帳は chain の再集計
  chain_remains_source_of_truth: true
  ledger_can_be_regenerated_from_chain: true           # 破損しても復元可能
  ledger_regeneration_deterministic: true

acceptance:
  every_matrix_15_5_2_decision_has_ledger_entry: true
  human_decision_entries_signed_by_non_agent: true
  agent_only_decisions_limited_to_light_ones: true     # 軽微決定のみ
  merkle_root_snapshot_signed_monthly: true

never_allowed:
  - "agent_writes_human_decision_entry_without_countersign"
  - "delete_ledger_entry"                              # append-only
  - "rewrite_actor_identity_after_the_fact"
  - "reduce_retention_below_minimum"
  - "regenerate_ledger_from_scratch_ignoring_prior_snapshots"
```

### 15.5.4 「shared」責任の扱い

決定責任が `shared` のケース（例：Gate3 の外部妥当性判断 = プロジェクトリーダー + ドメイン専門家）は、次のいずれかで運用します：

| 運用パターン | 内容 | 適用例 |
|---|---|---|
| **quorum（k-of-n）** | n 人中 k 人の署名で有効 | 委員会承認、監査 signing |
| **role complement** | 異なるロール全員の署名で有効 | Gate3 の PI + 統計 + ドメイン |
| **sequential handoff** | ロール順に承認、後工程が前工程を validate | データ責任者 → 分析責任者 → 論文責任者 |

Skill 契約側では、`decision_responsibility: shared` の場合に **どのパターンを採るかを明示** する必要があります（agent の裁量では選べない）。

## 15.6 vol-01 + vol-02 + vol-03 の到達点

### 15.6.1 3 巻を通した Skill レイヤ

vol-01 から vol-03 までを積み上げると、次のレイヤ構造が完成します：

| レイヤ | 主な貢献 | 巻 |
|---|---|---|
| **L0: データ contract** | 6 データ型テンプレ、単位・欠損・メタの規約 | vol-01 |
| **L1: 前処理 Skill** | 装置別前処理、ETL、品質チェック | vol-01 |
| **L2: 統計/ML Skill** | 回帰・分類・PCA・PLS・階層ベイズ | vol-02 |
| **L3: 深層 Skill** | CNN / Transformer / SSL / 転移学習 | vol-03 |
| **L4: 不確かさと停止** | Deep Ensemble / MC-Dropout / BNN / 自律停止 | vol-03 |
| **L5: Foundation Model 連携** | FM の取得・fine-tune・default 管理 | vol-03 |
| **L6: 統合 Skill と gate** | Gate1/2/3、Merkle chain、capstone | vol-03 |
| **L7: 監査と失敗パターン** | `failure_pattern_registry`, `agentic_deep_failure_audit` | vol-03 |
| **L8: 組織展開ガバナンス** | GPU 権限、モデル配布、責任分担台帳 | vol-03（本章） |

L0-L6 が「実行の Skill」、L7-L8 が「実行を監督する Skill」です。**L7-L8 が無い実行 Skill 群は、大きくなればなるほど誰も全体を追えない**。L7-L8 を最初から想定して L0-L6 を書くことが、vol-03 全体の設計思想でした。

### 15.6.2 6 データ型 × 深層 × Agentic Skill 対応（振り返り）

vol-01 で導入した 6 データ型テンプレは、vol-03 で以下のようにアップデートされました：

| データ型 | vol-03 で追加された Skill 契約 |
|---|---|
| スペクトル型 | `1d_cnn_classifier`, `1d_transformer_regressor`, `raman_fm_transfer`, `spectral_augmentation_provenance` |
| クロマトグラム・時系列型 | `timeseries_cnn`, `timeseries_transformer`, `batch_retraining_gate` |
| 画像・顕微鏡型 | `image_cnn_classifier`, `vit_classifier`, `sem_fm_transfer`, `gradcam_verification` |
| 回折・散乱パターン型 | `pattern_2d_cnn_classifier`, `rietveld_hybrid_advisory` |
| 表形式・プロセス条件型 | `tabnet_regressor`, `ft_transformer`, `deep_feature_pymc_hierarchical` |
| マルチモーダル統合型 | `clip_multimodal_embedding`, `image_plus_tabular_finetune` |

いずれも、**深層モデルの選択と併走して、Agent が「fine-tune 起動判断」「Human 承認」「不確かさ停止」「provenance 記録」を行う**ことを契約に含みます。データ型テンプレは vol-01 から一貫していますが、Skill の中身が層状に厚くなった、というのが 3 巻の関係です。

### 15.6.3 vol-03 の「扱わなかったこと」再確認

第1章と本章 §15.7 で示すロードマップと対応する形で、vol-03 がスコープ外とした主要トピックを再掲します：

| 扱わなかったトピック | 理由 | 想定巻 |
|---|---|---|
| 因果推論（DoWhy / EconML / DAG） | 実験計画・介入設計の議論が独立して必要 | vol-04 |
| ベイズ最適化・逐次実験計画（BoTorch / GPyOpt） | 「次に何を測るか」は別領域 | vol-04 |
| 生成モデル（VAE / GAN / Diffusion） | 逆設計テーマとして独立 | vol-05 |
| 大規模分散学習（マルチノード） | 本書は研究室〜施設単一ノード想定 | 別書（MLOps 系） |
| Foundation Model のスクラッチ事前学習 | 材料 FM は既存重み利用の立場 | vol-05 以降 |
| 強化学習（RLHF、実験ロボティクス） | 実験ロボティクスとの交差領域 | 別書 |
| 深層学習の数学的導出 | 既存教科書に譲る | — |

## 15.7 vol-04 以降のロードマップ

### 15.7.1 vol-04（因果・実験計画）の見取り図

vol-04 は **「介入的意思決定を Agentic Skill にする」** ことを主題にします：

| 章群 | 想定内容 | vol-03 との接続 |
|---|---|---|
| 導入 | 相関 vs 因果、DAG 表記、SUTVA | vol-01/02/03 の観察データ解析との対比 |
| 因果グラフ推定 | PC / GES / LiNGAM の限界と実務での使い方 | vol-02 相関分析との違いを明示 |
| 介入推定 | do-calculus、front-door / back-door | vol-02 のベイズ回帰との差 |
| 実験計画 | fractional factorial、response surface、逐次実験 | vol-01 のサンプリング計画拡張 |
| ベイズ最適化 | BoTorch、Gaussian Process、獲得関数 | vol-03 の不確かさ扱いを転用 |
| 実験ロボット連携 | ARIM 装置との API、Human safety gate | vol-03 の agent tier / GPU 権限を装置権限に拡張 |
| 失敗パターン | 介入固有の失敗（cherry-picking of interventions、hidden confounders） | vol-03 Ch14 の Agentic 失敗と対称 |

vol-04 の Skill は、vol-03 の `responsibility_ledger` を **「介入決定」台帳として拡張**します。介入は不可逆（試料は元に戻らない）なので、Ch4 の agent tier に **「介入 tier」を追加**する必要があります。

### 15.7.2 vol-05 以降の候補

| 巻 | テーマ | 骨子 |
|---|---|---|
| **vol-05** | 生成モデルと逆設計 | VAE / diffusion による材料生成、逆設計制約、生成物の合成可能性判定、生成モデル固有の失敗（mode collapse, hallucinated stability） |
| **vol-06** | マルチモーダル基盤モデル | 画像 + 分光 + テキストの統合 FM、cross-modal retrieval、multi-modal hallucination |
| **vol-07** | LLM とドメイン推論 | 材料 LLM の限界、幻覚評価、arXiv/論文 MCP、材料合成レシピ抽出 |
| **vol-08** | 産学連携と長期運用 | 契約付きモデル・データ交換、10 年運用のマイグレーション、規格対応（ISO/IEC 42001 等） |

いずれも「vol-03 で作った L7-L8 の監査・組織展開レイヤに乗る形で新レイヤを積む」設計を継承します。

### 15.7.3 継続的な失敗パターン蓄積

vol-04 以降でも、**`failure_pattern_registry` は書籍横断で継承・拡張**されます。運用としては：

- vol-03 版 registry は `registry_version: v1.x` として **freeze**
- vol-04 では `registry_version: v2.x` で **追加**（削除・変更は不可）
- Ch14 §14.8 の `registry_versioning.security_critical_update` は書籍を跨いでも同じ扱い
- 各巻末に「registry に追加された新パターン一覧」を付録として置く

これにより、**過去書籍の Skill 契約は永久に valid**、新書籍の Skill 契約は上位互換、という関係を保ちます。

## 15.8 深層 × Agentic 特有の失敗パターン（本章分）

第14章 §14.5 に集約したパターンを踏まえ、**組織展開レイヤ**で新たに顕在化する失敗を追加します：

| ID | パターン | 症状 | 対策契約 |
|---|---|---|---|
| **ORG-01** | GPU quota の agent 側自動増枠 | agent が「quota 不足」を検知して自動で quota を書き換える | `organization_gpu_resource_policy.quota_state_snapshot_signed_hourly: true`, `consume_gpu_beyond_signed_quota_snapshot: never_allowed` |
| **ORG-02** | 責任分担マトリクスの agent 側書き換え | agent が「効率化」のために Human 承認を bypass するよう matrix を書き換える | `responsibility_ledger.rewrite_actor_identity_after_the_fact: never_allowed` |
| **ORG-03** | モデル配布時の provenance 削減 | 配布サイズを減らすため provenance manifest を落とす | `model_distribution_contract.strip_provenance_to_reduce_bundle_size: never_allowed` |
| **ORG-04** | Revocation 通知の無視 | 上流 FM の revocation を検知しても推論継続 | `model_distribution_contract.ignore_revocation_notice: never_allowed`, `on_revoked_action: block_inference_and_alert_human` |
| **ORG-05** | Ledger 保持期間の短縮 | ストレージ節約のため retention を縮める | `responsibility_ledger.reduce_retention_below_minimum: never_allowed`, `minimum_retention_years: 10` |
| **ORG-06** | 施設共有 GPU の見かけ上の L-Lab 化 | L-Org 承認を回避するため層を偽装 | `access_matrix.silently_downgrade_layer_from_L_Org_to_L_Lab_to_bypass_committee: never_allowed` |
| **ORG-07** | Shared 責任の quorum 偽装 | k-of-n の署名を agent 単独で埋める | `responsibility_ledger.agent_writes_human_decision_entry_without_countersign: never_allowed` |

これらは第14章の `failure_pattern_registry` v2.x で ORG-xx として追加登録します（registry の incremental extension）。

## 15.9 まとめ

- **組織展開は「実行 Skill」ではなく「実行を監督する Skill」の層を明示することから始まる**
- GPU リソースは agent tier × リソース層のアクセス行列で管理し、L-Org L3 は必ず個別承認 + 失敗許容 quota を持たせる
- モデル重みの配布は **provenance + 使用制限 + 失効条件 + revocation** を Skill 契約として同梱し、受領側が同じ契約を継承する
- 責任は decision / execution / recording の 3 種類に分け、深層 × Agentic 特有の決定点（fine-tune 起動、FM 差し替え、Gate2/3、配布判断）は Human に固定する
- `responsibility_ledger` は provenance chain の view として長期保管し、10 年以上の再現性を保証する
- vol-01+02+03 は L0-L8 のレイヤ構造で 6 データ型 × 深層 × Agentic に到達した
- vol-04 は「介入的意思決定」、vol-05 以降は「生成 / マルチモーダル FM / LLM / 長期運用」へと繋がる
- `failure_pattern_registry` は書籍横断で継承・拡張され、過去 Skill の契約は永久に valid

## 15.10 チェックリスト

**Skill 契約側**：

- [ ] `organization_gpu_resource_policy` の access_matrix が agent tier × resource layer で網羅的
- [ ] L-Org L3 が per-job approval かつ failure_budget を明示
- [ ] `model_distribution_contract` の distribution_bundle に provenance_manifest / usage_restrictions / expiration / license 全て含む
- [ ] `audit_result_provenance_decision_at_distribution` が pass or warn に強制
- [ ] `recipient_obligations` に reinstall_audit + no strip + no relax が入る
- [ ] Revocation の check_frequency_days_max ≤ 30
- [ ] `responsibility_ledger` の retention ≥ 10 年、offline backup ≤ 7 日
- [ ] Ledger が chain から再生成可能（deterministic）
- [ ] Shared 責任のパターン（quorum / role complement / sequential）が明示
- [ ] 決定責任が human の項目に non_agent_signature_required: true

**組織運用側**：

- [ ] 責任分担マトリクス（§15.5.2）を組織で共有・掲示
- [ ] L-Org 承認委員会のロール定義が文書化
- [ ] 監査 report signing に non-agent verifier が実在（Ch14 継承）
- [ ] Revocation endpoint の運用主体が決まっている
- [ ] Ledger の Merkle root snapshot の月次確認担当者が決まっている
- [ ] 論文投稿時に provenance link を要求するテンプレを整備

**registry 側**：

- [ ] ORG-01〜ORG-07 が registry に登録
- [ ] registry_version が v2.x（vol-03 完了時点）
- [ ] vol-04 以降の追加は v3.x 以上

## 15.11 演習（ワーク）

1. **アクセス行列作成**：所属研究室と施設共有 GPU を想定し、agent tier × resource layer の access_matrix を実際に埋めてみる。**Human 承認者は個人名でなくロール名で書く**。
2. **配布 bundle 設計**：直近で外部配布した（あるいは配布したい）モデル重みを 1 つ選び、`model_distribution_contract` の usage_restrictions と expiration を具体的に埋める。「今の運用に足りていない要素」を挙げよ。
3. **責任分担マトリクスの穴埋め**：§15.5.2 のマトリクスに、自分の研究室固有の決定点（例：試料交換、装置校正、外部研究者との共著判断）を 3 つ以上追加せよ。
4. **Ledger 再生成テスト**：既存 provenance chain（vol-02 か vol-03 の capstone）から `responsibility_ledger` を機械的に再生成する疑似スクリプトを書き、**入力が同じなら出力が bit 単位で一致**することを確認せよ。
5. **Revocation シナリオ**：上流 FM が security-revocation された場合の、派生モデルの推論停止 → Human 通知 → responsibility_ledger 記録の 3 ステップを、Ch14 の `runtime_and_periodic_audit_guards` と接続する形で疑似コード化せよ。

## 15.12 参考資料

**組織ガバナンスと AI**：

- ISO/IEC 42001:2023 — Artificial intelligence — Management system
- NIST AI Risk Management Framework (AI RMF 1.0)
- OECD AI Principles

**Model card / Data sheet**：

- Mitchell et al. (2019). Model Cards for Model Reporting. FAT\* '19.
- Gebru et al. (2021). Datasheets for Datasets. CACM.

**Provenance と長期保存**：

- W3C PROV Data Model
- Research Data Alliance (RDA) — Provenance patterns

**責任分担フレームワーク**：

- ACM Code of Ethics and Professional Conduct
- IEEE 7000 series（AI 倫理）
- EU AI Act の High-Risk AI systems 分類

**継続学習・道しるべ**：

- 因果推論：Pearl. Causality (2nd ed.). Cambridge.
- ベイズ最適化：Frazier (2018). A Tutorial on Bayesian Optimization. arXiv:1807.02811
- 実験計画：Box, Hunter, Hunter. Statistics for Experimenters (2nd ed.). Wiley.
- 生成モデル：Goodfellow et al. Generative Adversarial Nets. NeurIPS 2014.
- 材料 FM：Merchant et al. (2023). Scaling deep learning for materials discovery. Nature.

**vol-01 / vol-02 / vol-03 の到達点まとめ**：

- 本書 vol-03 全 15 章 + 付録 A/B/C
- vol-02（統計/ML）／ vol-01（データ contract 基礎）の各章
- 各巻の `failure_pattern_registry` は書籍横断継承

---

**（本編 完）** 続く付録 A/B/C では、深層 × Agentic Skill テンプレート集、PyTorch / JAX / Hugging Face チートシートと MCP サーバ実装、GPU トラブルシューティングと演習データ候補を扱います。
