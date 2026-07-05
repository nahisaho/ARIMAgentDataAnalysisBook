# 第7章 介入効果の推定を Skill 化する（後半：DiD / IV / Synthetic Control）

> **本章の到達目標**
> - **backdoor が使えない**（未観測交絡が残る）観測データに対して、**準実験手法** DiD / IV / Synthetic Control を Skill 化できる
> - **DiD の parallel trends assumption**、**IV の 3 仮定（relevance / exclusion / exchangeability）**、**Synthetic Control の donor pool assumption** を Skill 契約項目として書き下せる
> - **linearmodels** による DiD / IV の実装と **CausalPy** による Synthetic Control の実装を、Skill としてどこまでエージェントに委ねてよいかを判断できる
> - **「バッチ変更前後」「装置更新前後」などの ARIM 実例**を、DiD / IV に対応させて Skill 化できる
> - **エージェントが IV 候補を提案する場合の妥当性判定**を、`variable_selection_authorization` のサブフローとして設計できる
>
> **本章で扱わないこと**
> - backdoor が使える場面での ATE / ATT 推定（第6章）
> - DAG 記法・IV / DiD / RDD の骨格（第5章 §5.4）
> - CATE / heterogeneous effect（第8章）
> - refutation・感度分析（第9章）
> - DoE（第10-12章）

---

## 7.1 なぜ準実験手法が必要か（backdoor の限界）

第6章の Propensity / IPW / DR / DML は、**すべての交絡が観測されている**（backdoor が使える）という前提で成立します。しかし ARIM の現場では：

- **オペレータ熟練度**は記録されないことが多い
- **装置内部の較正履歴**は部分的にしか残らない
- **試薬ロットの微妙な違い**は測定困難
- **研究者の "勘"** による処置選択

——**未観測交絡 $U$ が残る**のが常です。第5章 §5.4 で導入した **準実験 3 手法**（DiD / IV / Synthetic Control）は、こうした場面での identification 戦略でした。本章ではこれらを **Skill として運用可能な形**に落とします。

### 本章の 5 部構成

| 部 | 節 | 主題 |
|---|---|---|
| 第1部 | 7.2 | **DiD の Skill 化**（parallel trends + linearmodels 実装） |
| 第2部 | 7.3 | **IV の Skill 化**（3 仮定の operational 判定 + 2SLS with linearmodels） |
| 第3部 | 7.4 | **Synthetic Control の Skill 化**（CausalPy + donor pool governance） |
| 第4部 | 7.5 | **エージェントによる IV 候補提案**の Skill 分離（`iv_proposal_skill` と `iv_approval_skill`） |
| 第5部 | 7.6 | **識別戦略比較表**——backdoor vs DiD vs IV vs SC の使い分け early warning |

---

## 7.2 DiD（Difference-in-Differences）— Skill 化

### 7.2.1 骨格の再確認（第5章 §5.4.2 より）

**parallel trends assumption**：処置がなかった仮想シナリオで、処置群と対照群の $Y$ の**時間トレンドが並行**。

**推定量**（2 期間 × 2 群の基本形）：
$$\widehat{\text{ATT}}_{\text{DiD}} = \bigl(\bar{Y}_{\text{treated, after}} - \bar{Y}_{\text{treated, before}}\bigr) - \bigl(\bar{Y}_{\text{control, after}} - \bar{Y}_{\text{control, before}}\bigr)$$

**回帰形**（unit + time 固定効果、共変量調整あり）：
$$Y_{it} = \alpha_i + \lambda_t + \tau \cdot D_{it} + \boldsymbol{X}_{it}^\top \boldsymbol{\beta} + \varepsilon_{it}$$

ここで $D_{it}$ は「unit $i$ が時点 $t$ で処置されたか」の指示変数、$\tau$ が推定対象の ATT。

> [!IMPORTANT]
> **DiD が identify する estimand は ATT**（Average Treatment effect on the Treated）——処置群における処置効果です。**ATE ではありません**。Skill 契約で `estimand_type: att` を必須明示。

### 7.2.2 parallel trends assumption の operational 判定

**理論的仮定** → **経験的 assessment** への落とし込み：

| 判定項目 | 方法 | 契約項目 |
|---|---|---|
| **pre-trend 検定** | 処置前の複数期間で処置群 vs 対照群のトレンド差を可視化 + 検定 | `pre_trend_test_uri` |
| **placebo test** | 処置がまだ発生していない過去時点を "仮想処置" として推定 → 0 に近いことを確認 | `placebo_test_uri`（Ch9 感度分析と共通） |
| **時変共変量調整** | parallel trends が絶対でなく "共変量条件付き" なら明示 | `conditional_parallel_trends: true` + 共変量列挙 |

> [!WARNING]
> **parallel trends assumption は理論的仮定であり、経験的に "満たされている" と言えるのは pre-trend + placebo が pass した場合のみ**（第0章の `assessed` レベル）。**`checked`（完全経験的裏付け）にはなりません**——処置後の反実仮想トレンドは観測不能だからです。Skill 契約：`parallel_trends: {status: assessed, evidence_uri: <string>}`。

### 7.2.3 Skill 化のポイント（linearmodels 実装）

```yaml
estimator: did
estimand_type: att
did_config:
  library: linearmodels | causalpy
  library_version: <string>
  design:
    unit_fixed_effect: true
    time_fixed_effect: true
    cluster_se: unit                    # クラスタロバスト SE、unit レベル
    covariates: [batch_id, calibration_state]   # 時変共変量（任意）
    treatment_indicator: D_it            # 処置 dummy 列名
  pre_trend_assessment:
    pre_periods: [t-3, t-2, t-1]
    test_method: joint_f_test | event_study
    pre_trend_test_uri: <string>
    p_value_threshold: 0.10              # pre-trend F 検定の下限
  placebo_test_uri: <string>              # Ch9 と共通
identification_validity:
  strategy: did
  parallel_trends:
    status: assessed                      # checked にはならない
    evidence_uri: <string>
    conditional_covariates: [...]
  dag_of_record_uri: <string>
  dag_of_record_sha256: <string>
prohibited_actions:
  - silently_switch_fixed_effect_structure       # fatal
  - drop_pre_periods_without_review              # fatal（pre-trend 検定の質を下げる）
  - relabel_att_as_ate                           # fatal（estimand mislabeling）
  - use_pre_trend_p_value_below_threshold_without_review  # fatal
  - change_cluster_se_level_silently             # fatal
```

### 7.2.4 ARIM ケース：装置較正プロトコル刷新

**設定**：
- 装置 A（処置群）：2026 Q2 に較正プロトコル刷新（$D_{A,t \geq Q2} = 1$）
- 装置 B（対照群）：変更なし（$D_{B,t} = 0$）
- outcome $Y$：測定再現性 CV（%）
- 期間：2025 Q1 〜 2026 Q4（8 四半期）

**Skill 実行フロー**：
1. **pre-trend assessment**：2025 Q1 〜 2026 Q1（処置前 5 四半期）で装置 A vs B のトレンド差を event study 回帰、p > 0.10 を確認
2. **DiD 回帰**：unit + time FE、cluster SE を装置レベル
3. **placebo test**：仮想処置を 2025 Q4（実際は変更なし時点）に置いて 0 に近いか確認
4. **artifact 出力**：`att_estimate`, `ci`, `pre_trend_test_uri`, `placebo_test_uri`

### 7.2.5 DiD の拡張（本章の scope 外）

- **Staggered DiD**（処置タイミングが unit ごとに異なる）：Callaway-Sant'Anna / Sun-Abraham 推定量——**heterogeneous treatment effect が絡む場合の bias 修正**が必要（`differencesindifferences` / `did` パッケージ、または Ch8 の CATE と組み合わせ）
- **Synthetic DiD**（SDID）：Arkhangelsky et al. (2021)——本章では触れず、Ch7 SC 節との接続を軽く言及

---

## 7.3 IV（Instrumental Variables）— Skill 化

### 7.3.1 3 仮定の operational 判定

第5章 §5.4.1 で列挙した 3 仮定を、**Skill が判定可能な形**にします：

| 仮定 | 経験的判定 | 契約項目 | 破綻時の対応 |
|---|---|---|---|
| **Relevance**：$Z \to T$ 強度 | first-stage F 統計量（rule of thumb: F > 10） | `first_stage_f_stat`, `first_stage_f_threshold: 10` | weak instrument → fail-close |
| **Exclusion restriction**：$Z \not\to Y$ 直接 | 経験的検定は困難（理論的正当化が主）、over-identification test（$Z$ が複数なら Sargan-Hansen） | `exclusion_justification_uri`, `overid_test_uri` | 理論的正当化不足 → Human review 必須 |
| **Exchangeability**：$Z \perp U$ | 観測共変量とのバランス、DAG での正当化 | `iv_balance_report_uri` | バランス崩壊 → 追加 covariate 制御 or fail-close |

> [!WARNING]
> **Exclusion restriction は経験的にはほぼ検証不可能**——`assessed` にすら達しない場合が多く、**Human review + DAG レベルの正当化**が事実上必要です。**IV の妥当性は Skill 単独では確定できない**——`iv_approval_skill`（§7.5）で Human ゲートを通す設計にします。

### 7.3.2 estimand の明示（LATE / 2SLS / ATE）

第5章 §5.4.1 で強調した通り：

| 前提 | identify される estimand |
|---|---|
| Imbens-Angrist（monotonicity のみ） | **LATE（compliers での平均処置効果）** |
| 線形構造 + 効果均質性 | **ATE と一致**（線形 2SLS estimand） |
| 連続 IV + additional assumptions | **weighted average of local effects** |

**Skill 契約**：`estimand_type: late | 2sls_linear | ate`（`ate` は追加仮定を明記した場合のみ）。

### 7.3.3 Skill 化のポイント（linearmodels 実装）

```yaml
estimator: iv
estimand_type: late | 2sls_linear | ate    # ate は追加仮定明示時のみ
iv_config:
  library: linearmodels | dowhy
  library_version: <string>
  method: 2sls | limited_information_ml | gmm
  instruments: [Z1, Z2, ...]                # 複数可、over-id テスト対象
  endogenous_regressors: [T]
  exogenous_regressors: [X1, X2, ...]        # exclusion restriction を尊重する共変量
  cluster_se: <string>                       # クラスタ変数
  first_stage_f_stat: <float>                # Skill 実行後に出力
  first_stage_f_threshold: 10
  overid_test:
    method: sargan | hansen_j
    p_value_threshold: 0.10                  # 下回れば instrument invalidity 疑い
    overid_test_uri: <string>
  exclusion_justification_uri: <string>       # 理論的正当化文書（Human 生成）
identification_validity:
  strategy: iv
  iv_assumptions:
    relevance: {status: assessed, evidence: first_stage_f_stat}
    exclusion: {status: declared, evidence_uri: exclusion_justification_uri, human_reviewed: true}
    exchangeability: {status: assessed, evidence_uri: iv_balance_report_uri}
  dag_of_record_uri: <string>
  dag_of_record_sha256: <string>
prohibited_actions:
  - proceed_with_first_stage_f_below_threshold      # fatal（weak instrument）
  - proceed_without_exclusion_justification         # fatal（Human review 未実施）
  - silently_add_instruments                        # fatal（instrument 集合の変更は §7.5 承認）
  - relabel_late_as_ate                             # fatal（estimand mislabeling）
  - drop_overid_test_when_multiple_instruments      # fatal
```

### 7.3.4 ARIM ケース：装置更新を IV とする

**設定**：
- 処置 $T$ = 新プロトコル採用（0/1）、outcome $Y$ = 生成物純度
- 未観測交絡 $U$ = オペレータ熟練度、ラボの研究方針
- **候補 instrument $Z$ = 装置更新タイミング**（外生的スケジュール、$Z$ が起きた四半期以降は新装置が使える）

**judgment**：
- **Relevance**：装置更新後にプロトコル採用率が上がる → first-stage F を確認（例：F = 25）
- **Exclusion**：装置更新自体は純度に直接影響しない（すべて新プロトコル経由）→ 理論的正当化必要
  - 反例：新装置自体が測定精度を上げる ⇒ exclusion 破綻 → **この IV は不採用**
- **Exchangeability**：装置更新スケジュールは施設管理側で決定、ユーザーのオペレータ熟練度と独立 → DAG で正当化

**identify される**のは「装置更新に反応してプロトコル切替した装置群での LATE」——集団全体の ATE ではない。

---

## 7.4 Synthetic Control — Skill 化

### 7.4.1 骨格

**Synthetic Control (SC; Abadie & Gardeazabal, 2003)**：**処置を受けた 1 unit** に対して、**未処置 unit（donor pool）の凸結合**で "反実仮想の処置群" を合成し、処置後の $Y$ を比較。

**推定量**：
$$\widehat{Y}_{\text{treated}, t}^{(0)} = \sum_{j \in \text{donor pool}} w_j Y_{j, t}, \quad w_j \geq 0, \sum_j w_j = 1$$

$w_j$ は**処置前期間**で $Y_{\text{treated}}$ と合成 unit の $Y$ が最も近くなるように**最適化**。処置後の $Y_{\text{treated}, t} - \widehat{Y}_{\text{treated}, t}^{(0)}$ が処置効果。

### 7.4.2 donor pool の governance

**SC の質は donor pool の設計**で決まります。エージェントに勝手に選ばせるのは危険：

| 判定項目 | 内容 | 契約項目 |
|---|---|---|
| **donor 候補の除外基準** | 処置群と類似した処置を受けた unit を donor から除外 | `donor_exclusion_rules_uri` |
| **pre-treatment fit** | 処置前期間の $Y$ trajectory の RMSE | `pre_treatment_rmse` + threshold |
| **weight sparsity** | 過度に少数 donor に集中していないか | `effective_donor_count_min` |
| **placebo units** | donor unit を "仮想処置" として同じ手続き → 効果分布を比較 | `placebo_distribution_uri` |

### 7.4.3 Skill 化のポイント（CausalPy 実装）

```yaml
estimator: synthetic_control
estimand_type: itt_single_unit             # 1 unit の intent-to-treat 相当
sc_config:
  library: causalpy | synthdid
  library_version: <string>
  method: original_abadie | augmented_sc | synthetic_did
  treated_unit: <string>
  donor_pool_uri: <string>                  # 承認済み donor リスト
  donor_pool_sha256: <string>
  donor_exclusion_rules_uri: <string>       # 何が除外理由か文書化
  optimization:
    objective: pre_period_rmse
    constraints:
      non_negativity: true
      simplex_sum: true
    solver: quadprog | slsqp
  pre_treatment_fit:
    pre_treatment_rmse: <float>
    rmse_threshold: <float>                 # データ scale に応じ設定
    fit_plot_uri: <string>
  weight_sparsity:
    effective_donor_count_min: 3
    weights_uri: <string>
  placebo_analysis:
    method: leave_one_out | in_space | in_time
    placebo_distribution_uri: <string>
    treated_effect_percentile_threshold: 0.90  # 実効果が placebo 分布の上位 10% にあるべき
identification_validity:
  strategy: synthetic_control
  donor_pool_assumption:
    status: assessed
    evidence_uri: <string>
  no_spillover_assumption:
    status: declared
    evidence_uri: <string>
prohibited_actions:
  - add_donor_units_after_fitting                    # fatal（cherry-picking）
  - remove_donor_units_to_improve_fit                # fatal（overfitting）
  - use_donor_that_experienced_similar_treatment     # fatal（SUTVA / no spillover 違反）
  - report_effect_without_placebo_analysis           # fatal
  - proceed_with_pre_treatment_rmse_above_threshold  # fatal
```

### 7.4.4 ARIM ケース：単一装置の較正プロトコル刷新

**設定**：装置 A のみ較正刷新、他装置（B, C, D, E）は変更なし → SC で装置 A の "刷新なし" 反実仮想を合成。

- **donor pool**：装置 B, C, D, E（**それぞれ独自の較正刷新をしていないことを事前確認**）
- **pre-treatment fit**：2024-2025 の CV(%) trajectory で RMSE < 0.05
- **weight**：例、$w_B = 0.4, w_C = 0.3, w_D = 0.2, w_E = 0.1$
- **placebo (in-space)**：装置 B, C, D, E それぞれを "仮想 treated" に、他 3 台を donor にして同じ手続き → 効果分布を得る

---

## 7.5 エージェントによる IV / donor pool 提案の Skill 分離

第5章 §5.6 で **DAG proposal vs approval** の Skill 分離を導入しました。**IV 候補と donor pool の提案**も同じ設計原則で分離します。

### 7.5.1 `iv_proposal_skill`（エージェント自律）

**目的**：候補 instrument を挙げ、first-stage F と exchangeability バランスの初期評価を出す。

```yaml
skill_type: iv_proposal_skill
authorization_level: agent_autonomous
inputs:
  - dataset_uri
  - dataset_sha256
  - dag_of_record_uri
  - dag_of_record_sha256
  - candidate_instruments: [Z1, Z2, ...]     # ドメインエキスパートが列挙した候補
outputs:
  - iv_candidate_report_uri
  - first_stage_f_stat_per_candidate
  - balance_diagnostics_per_candidate_uri
  - exclusion_restriction_theoretical_argument_uri   # AI が下書き、Human が確定
  - proposal_bundle_sha256
prohibited_actions:
  - use_iv_candidate_not_in_input_list               # fatal（範囲外の変数を勝手に IV 候補にしない）
  - promote_candidate_to_final_iv_without_approval   # fatal
  - modify_dag_of_record                             # fatal
```

### 7.5.2 `iv_approval_skill`（Human 承認、`variable_selection_authorization`）

**目的**：候補を審査し、正式な instrument に採用する。

```yaml
skill_type: iv_approval_skill
authorization_level: human_required
authorization_gate: variable_selection_authorization
inputs:
  - iv_candidate_report_uri
  - proposal_bundle_sha256
  - exclusion_restriction_theoretical_argument_uri
  - reviewer_comments_uri
outputs:
  - approved_instruments: [Z1]
  - iv_approval_evidence:
      - reviewer_signatures
      - approval_timestamp
      - dissenting_opinions_uri
approver: causal_review_board
approver_independence:
  conflict_policy: independent_reviewer_required_if_approver_is_data_producer
prohibited_actions:
  - approve_without_reviewer_signatures            # fatal
  - approve_iv_with_first_stage_f_below_threshold  # fatal
  - approve_iv_without_exclusion_justification     # fatal
```

### 7.5.3 donor pool にも同じ設計

**Synthetic Control の donor pool 選定**も 2 Skill に分離：
- `donor_pool_proposal_skill`（エージェント自律：候補列挙 + pre-fit RMSE 評価）
- `donor_pool_approval_skill`（Human 承認：**類似処置を受けた unit の除外**、SUTVA / no spillover 判定）

---

## 7.6 識別戦略比較表 — backdoor vs DiD vs IV vs Synthetic Control

**Skill 実行前**に、以下の early warning matrix でどの戦略が使えるかを判定します。

| 状況 | backdoor | DiD | IV | Synthetic Control |
|---|---|---|---|---|
| **すべての交絡が観測可能** | ✅ 第一選択 | 過剰 | 不要 | 不要 |
| **未観測交絡あり、時間軸で before/after 比較可能** | ❌ | ✅ | 検討可 | ✅ 単一 unit |
| **未観測交絡あり、外生的 shifter がある** | ❌ | ❌ | ✅ | ❌ |
| **未観測交絡あり、閾値ルール** | ❌ | ❌ | ❌ | **RDD**（Ch5、実装は本書では扱わない） |
| **処置 unit が 1 つ、pre-treatment データ長い** | ❌ | 弱 | ❌ | ✅ 第一選択 |
| **処置タイミングが unit ごとに異なる** | 弱 | staggered DiD | 検討可 | staggered SC |
| **short panel、大量 unit** | ✅ | ✅ | 検討可 | 弱 |
| **large panel、少数 unit** | 弱 | 弱 | 弱 | ✅ |

| 手法 | 主要 estimand | 主要ライブラリ | 実装難易度 |
|---|---|---|---|
| backdoor（Ch6） | ATE / ATT | DoWhy, EconML, sklearn | 中 |
| DiD | ATT | linearmodels, causalpy | 中 |
| IV | LATE（or 2SLS estimand） | linearmodels, DoWhy | 高（exclusion 正当化） |
| Synthetic Control | ITT (single unit) | causalpy, synthdid | 高（donor 選定） |

> [!IMPORTANT]
> **識別戦略を silent に切り替えることは fatal**（第4章 Table 4.4 item 2）。エージェントが「backdoor で positivity 違反だから IV に切り替える」等の判断は、**必ず `variable_selection_authorization` および `dag_authorization` を通す**——第5章 §5.6 の DAG amendment protocol と整合させる必要があります。

---

## まとめ — 本章のチェックリスト

- [ ] **§7.2 DiD の parallel trends assumption**——`assessed` レベルまでしか達しない理由と、pre-trend / placebo test の役割を説明できる
- [ ] **§7.2.3 DiD Skill 契約**——fixed effect 構造 / cluster SE / pre_trend p-value threshold を書き下せる
- [ ] **§7.3 IV 3 仮定の operational 判定**——特に exclusion restriction が経験的にほぼ検証不可能な理由と、Human review 必須性を理解した
- [ ] **§7.3.2 IV の estimand**——LATE / 2SLS estimand / ATE の使い分けを説明できる
- [ ] **§7.4 Synthetic Control の donor pool governance**——placebo analysis と類似処置 unit 除外の必然性を理解した
- [ ] **§7.5 IV / donor pool の proposal vs approval Skill 分離**——第5章 §5.6 の設計原則が同じ構造で拡張できることを理解した
- [ ] **§7.6 識別戦略比較表**——自分の状況で backdoor / DiD / IV / SC のどれが第一選択かを判断できる

---

## 章末演習

### 演習 7.1：DiD Skill 契約を書く

自分の ARIM データ（または合成データ）で、以下の想定シナリオに対する DiD Skill 契約を §7.2.3 テンプレートで完成させてください：

**シナリオ**：装置 A で 2026 Q2 に較正プロトコル変更、装置 B は変更なし。2024 Q1 〜 2026 Q4 のデータ。outcome = 硬度測定 CV(%)。

- [ ] fixed effect 構造（unit + time）
- [ ] cluster SE の level
- [ ] pre-trend 検定の method + p-value threshold
- [ ] placebo test の設計（in-time / in-space）
- [ ] `parallel_trends.status: assessed` として evidence を artifact 化

### 演習 7.2：IV 候補の妥当性判定

以下の IV 候補について、3 仮定それぞれの**判定と根拠**を答えてください：

1. **候補 $Z$ = 施設の年間予算**（処置 $T$ = 高価な新プロトコル採用、$Y$ = 生成物特性）
2. **候補 $Z$ = 装置更新スケジュール**（処置 $T$ = 新プロトコル採用、$Y$ = 純度）
3. **候補 $Z$ = 部材メーカーの生産停止**（処置 $T$ = 代替部材使用、$Y$ = 特性）
4. **候補 $Z$ = 隣接研究室での成功事例**（処置 $T$ = 手法採用、$Y$ = 収率）

各候補に対し：
- Relevance の期待（first-stage F の予想）
- Exclusion restriction の理論的正当化 or 破綻シナリオ
- Exchangeability の確認方法

### 演習 7.3：Synthetic Control の donor pool 設計

処置 unit：装置 A（2026 Q1 に較正プロトコル刷新）
候補 donor：装置 B, C, D, E, F の 5 台

- [ ] 各 donor が過去 3 年以内に類似の較正刷新を受けていないか調査
- [ ] pre-treatment fit を評価する期間（2023-2025 の 12 四半期）を指定
- [ ] `pre_treatment_rmse` の閾値を設定（scale に応じ）
- [ ] in-space placebo の設計（donor を仮想 treated に）
- [ ] weight sparsity の下限（effective donor ≥ 3）

### 演習 7.4：識別戦略の選択

以下のシナリオそれぞれで、**backdoor / DiD / IV / SC のどれ**を第一選択にするか、演習 5.2 での回答と整合させて答えてください：

1. すべての交絡（装置・較正・バッチ）が記録済、$n = 3000$
2. 2 装置のみ、5 年 pre / 3 年 post の long panel、1 装置だけプロトコル変更
3. 未観測交絡（オペレータ熟練度）あり、外生的な部材切替タイミング利用可
4. 未観測交絡あり、処置群 / 対照群の time trend が過去 5 年並行
5. 処置閾値ルール（純度基準）で切り替わる（→ 実装は本書では扱わない）

---

## 参考資料

### 内部 cross-reference

- 第0章：4 識別仮定、declared / assessed / checked
- 第2章：ARIM 特有の未観測交絡（オペレータ・較正履歴）
- 第3章：linearmodels、causalpy、DoWhy の位置づけ
- **第4章**：Skill 契約テンプレート、`variable_selection_authorization` / `dag_authorization`、Table 4.4 禁止事項
- **第5章**：§5.4 IV / DiD / RDD 骨格、§5.6 DAG proposal / approval skill 分離（本章 §7.5 の設計テンプレート）
- 第6章：Propensity / IPW / DR / DML、re-fit ポリシー、`estimator_contract_change_gate`
- 第8章：CATE + heterogeneous treatment（本章 estimand の $X$ 依存版）
- 第9章：refutation / placebo test / robustness（本章の pre-trend / placebo と共通）
- **第13章 (13a)**：本章 + Ch5, Ch6, Ch8 を統合した Capstone
- 付録 A：DiD Skill、IV Skill、Synthetic Control Skill テンプレート
- 付録 D：DiD / IV / Synthetic Control / parallel trends / donor pool の用語

### 外部参考文献

- Abadie, A., & Gardeazabal, J. (2003). The economic costs of conflict: A case study of the Basque Country. *American Economic Review*, 93(1), 113-132.
- Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and methodological aspects. *Journal of Economic Literature*, 59(2), 391-425.
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
- Arkhangelsky, D., et al. (2021). Synthetic difference-in-differences. *American Economic Review*, 111(12), 4088-4118.
- linearmodels documentation: https://bashtage.github.io/linearmodels/
- CausalPy documentation: https://causalpy.readthedocs.io/

---

**本章の位置づけ**：backdoor が使えない場面での ATE / ATT / LATE / ITT の推定 Skill が揃いました。次章（Ch8）では、**「集団平均ではなく個別ごとの効果」CATE / ITE** を Skill 化——DR スコアを $X$ に対する関数として扱い、`counterfactual_scope_gate` の operational 実装（Mahalanobis 距離 + CATE 予測分散）を導入します。
