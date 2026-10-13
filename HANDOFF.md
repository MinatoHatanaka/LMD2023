# 引き継ぎ書：LMD2023 の批判的再現研究

- 作成日：2026-10-14（ローカル環境の WSL2・7.9GB RAM から、研究室サーバーへ移行するため）
- 読み手：研究室サーバーで作業を引き継ぐ人と、そこで起動する Claude Code セッション

> **Claude へ：** このファイルを最初に全部読んでから作業してください。詳しい根拠は `results/REPRO_REPORT.md`（再現結果）と `results/ISSUES_AND_HYPOTHESES.md`（問題点の仮説）にあります。ユーザーへの報告は日本語で行ってください。実験の範囲（軽い版にするか、完全版にするか）はユーザーが決めます（§8）。

---

## 1. 研究の目的
- **対象論文：** Smiliotopoulos, Kambourakis, Barbatsalou, *On the detection of lateral movement through supervised machine learning and an open-source tool to create turnkey datasets from Sysmon logs*, IJIS 22:1893–1919 (2023)。以下「LMD2023」と呼びます。前身の研究は Appl. Sci. 2022（LMD-2022）です。PDF は `docs/` にあります。
- **目的：** LMD2023 の問題点を検証すること（批判的な再現研究）。論文の数値を再現すること自体が目的ではありません。再現は、検証の基準点として行いました。
- **これまでの経緯：**
  1. 論文と公開されているデータ・コードを調べました。
  2. 後続研究を調べ、著者グループの後続論文（2024・2025・2026年）がどれも問題を解決していないことを確認しました。
  3. 再現実験を行いました（**完了**）。
  4. 第三者による検証を整理し、問題点の仮説を立てました（**完了**）。
  5. 次は、仮説 H1・H2・H5 の検証実験です（**未着手、§7**）。

## 2. ここまでの主な結果
### 2.1 再現（`results/REPRO_REPORT.md`）
- **LR・SGDC・LinearSVC・NB・KNN・CatBoost は、著者の行列ファイルでほぼ再現できました。** AUC の差は2pt以内、Accuracy の差は0.3pt以内です。
- **論文の AUC は、予測確率ではなく予測ラベルから計算されています。** Precision / Recall / F1 は、モデルによって weighted 平均と macro 平均が混在しています。
- **木系モデルは、論文の表10の設定では論文値に届きません。** 論文で最良の ET は weighted F1 97.62（論文99.41）でした。枝刈り（`ccp_alpha`）を外すと99.04まで近づきます。
- **DNN は再現できませんでした。** 終わったのは MLP・AE・CNN の3つで、CNN は学習が発散しました。RNN と LSTM は CPU では時間がかかりすぎるため中止しました。

### 2.2 問題点の仮説（`results/ISSUES_AND_HYPOTHESES.md`）
| ID | 仮説 | 今わかっている根拠 |
|---|---|---|
| H1 | ラベルがほぼホストで決まる | ホスト名だけで正解率97.6%（著者の行列ファイル）。第三者の検証でも、学習に含めなかったホストでは AUC 約0.50 |
| H2 | ラベルがほぼ収集時期で決まる | 公開版は時刻のしきい値2本で正解率99.998%。著者の行列ファイルは月の値だけで正解率97.2% |
| H3 | EoRS の大部分は Splunk Attack Range の公開ログ由来 | ホスト名・日付・EventRecordID の並びが一致。EoRS の64%は Splunk 転送エージェントのプロセス |
| H4 | ラベルの誤り | mimikatz などの攻撃ツールがラベル0。実行ファイルのパスの大文字・小文字でラベルが変わる |
| H5 | 行の大量複製で、ランダム分割では暗記で正解できる | 著者の行列ファイルの95%が重複行で、テスト行の96.5%が学習データにもある |
| H6〜H9 | 汎化しない／評価指標の問題／データの版の不整合／ハイパーパラメータの不一致 | 詳しくは `ISSUES_AND_HYPOTHESES.md` |

## 3. 重要な前提と、これまでの判断
1. **データは2つの経路（トラック）で扱っています**（論文で使われたデータそのものは入手できません）。
   - **`authors`：** 著者の行列ファイル `full-csv(Evaluated Titles).csv`（2023-01、90特徴、1,777,930行）。著者の特徴量重要度スクリプトがこのファイルを読み込んでいて、DNN の入力次元90とも一致するため、**論文の ML 実験で使われたデータに最も近い**と判断しました。
     - EoHT が55,547件で、論文の30,507件と合いません。
     - ラベル0・1・2の順に連続したブロックで並んでいて、MinMax 変換はファイルごとに別々に行われています。
     - ホスト列は `CompSTA1`〜`CompSTA8` に匿名化されていて、`CompSTA3` という列名が2つあります（読み込むと `CompSTA3.1` になります）。
   - **`public15`：** 公開版の生データ「LMD-2023 [1.75M Elements][Labelled]checked.csv」（2024-03 の改訂版）から、論文と著者スクリプトの手順で15特徴を作り直したもの（1,723,300行・75特徴）。ホスト名と時刻の生の値が残っているので、検証実験ではこちらが扱いやすいです。
     - 公開版の時刻は付け直されていて（2022-10〜2023-12）、論文で使われた時刻（2020〜2023）とは異なります。
2. **評価指標：** 論文と比べるときは、予測ラベルから計算した AUC（`auc_hard_macro`）を使い、それと並べて、予測確率から計算した AUC（`auc_score_macro`）、weighted F1、macro F1、クラス別 F1 を必ず報告します。**検証実験の主な指標は、macro F1、予測確率からの AUC、クラス別 F1** にしてください。正常クラスが92%を占めるので、Accuracy と weighted F1 は性能を高く見せてしまいます。
3. **分割方法：** 論文に書かれた fold の件数（1,314,668 / 438,223）は、1回の75/25分割の件数と一致します。そのため、論文の再現にはホールドアウト（`train_test_split(test_size=0.25, stratify=y, random_state=42)`）を使いました。10分割交差検証との差は小さいことを確認済みです。
4. **ハイパーパラメータ：** 論文の表10に従っています。実装できなかった部分の対応は `src/lmd/models.py` の `DEVIATIONS` に記録しています（LR の l1_ratio を0.5、SGDC の max_iter を1000、CatBoost の損失を MultiClass など）。
5. **公開版の重複判定：** 浮動小数点の値は float64 のまま比べてください。float32 にすると、別の行が同一とみなされてしまいます。

## 4. リポジトリの構成
```
docs/                          論文の PDF（2022・2023）
src/
  lmd/data.py                  load_matrix(name)、test_in_train_rate()（行のハッシュでテスト行と学習行の重複率を計算）
  lmd/models.py                表10のモデル定義と DEVIATIONS（ET-noccp・DT-noccp は感度分析用）
  lmd/metrics.py               evaluate()：AUC（予測ラベル・予測確率の両方）、P/R/F1（weighted・macro・クラス別）、混同行列
  lmd/dnn.py                   表12の DNN 定義
  load_fullcsv.py              著者の行列 CSV → data/processed/fullcsv.parquet
  convert_raw.py               公開版の CSV → Parquet（全列を文字列のまま、ストリーミングで変換）
  preprocess_public.py         公開版のラベル付き Parquet → public15_raw / public15_matrix
  audit_raw.py                 公開版の生データの監査（ホスト×ラベル、時刻、公開ルールでの再ラベル付け、複製）→ results/audit/raw_audit.json
  run_repro.py                 浅層モデルの実験（--dataset authors|public15、--protocol holdout|cv10、--models …、--folds …）
  run_dnn.py                   DNN（ホールドアウトのみ）
  summarize.py                 results/repro/** を集計 → results/repro_summary.{csv,md}
results/
  REPRO_REPORT.md              再現結果のレポート（論文値との比較）
  ISSUES_AND_HYPOTHESES.md     第三者による検証の整理と仮説 H1〜H9
  repro/<dataset>/<protocol>/  モデル×fold ごとの結果 JSON（すでにあるファイルは実行時にスキップ）
  audit/raw_audit.json         公開版の生データの監査結果
  logs/                        実行ログ（Git の管理対象外）
HANDOFF.md                     このファイル
```
`data/` は Git の管理対象外です（§5 で入手します）。`catboost_info/` は CatBoost が自動で作る出力で、誤ってコミットされています。`.gitignore` に追加することを勧めます。

## 5. サーバーでの環境構築
### 5.1 Python 環境
```bash
git clone https://github.com/MinatoHatanaka/LMD2023.git && cd LMD2023
uv sync            # Python 3.12。uv.lock で sklearn 1.9.1、LightGBM 4.7.0、CatBoost 1.2.10、pandas 3.0.6、TF 2.21 を固定
```
GPU があるサーバーで DNN を再開する場合は、`tensorflow-cpu` を GPU 版の TensorFlow に入れ替えてください。

### 5.2 データ（方法A：手元のファイルをコピーする。速くて確実）
ローカル環境から、次の2つをサーバーへ転送します（合計約63MB）。
```bash
rsync -av data/processed/ <server>:LMD2023/data/processed/
rsync -av data/extracted/rbpolicy/ <server>:LMD2023/data/extracted/rbpolicy/
```
- `data/processed/` には5つの Parquet があり、`run_repro.py`、`preprocess_public.py`、`audit_raw.py` はこれだけで動きます。
- `audit_raw.py` は `data/extracted/rbpolicy/RBPolicy/*.txt` も使います。

### 5.3 データ（方法B：一から取り直す）
入手元は著者の Google Drive です：https://drive.google.com/drive/folders/1PkJiGpD0Kn1rV8GC9m2b_eWvqQTSa1eH

| ファイル | Drive の ID | sha256 |
|---|---|---|
| full-csv_EvaluatedTitles.rar | 1A3Aa3gygV9KpdwVA0VAugvU0-pxrfTjU | 308dfa623f269d2b7643cc9050ff5076afc379351b83688cfcafc275fa1464dd |
| LMD-2023.rar（345MB） | 1ApiVzhAYv1ARRoCzDOIR6UhTjqn7lKh6 | 89aeea485c471b013cb10a2407bd49d62684a22b4278d30fe977705423eb25d9 |
| LMD-2022.rar | 1n4kM5xzvwPPozHsaXR_cfoR0L9ezFJo_ | 775aa09a1986281f56faf1a4745575a1ba96343081b542c57ccc6320d4ea6afa |

```bash
mkdir -p data/raw/drive data/raw/etcexp data/extracted/{fullcsv,lmd2023,rbpolicy} data/processed
cd data/raw/drive
for p in 1A3Aa3gygV9KpdwVA0VAugvU0-pxrfTjU:full-csv_EvaluatedTitles.rar 1ApiVzhAYv1ARRoCzDOIR6UhTjqn7lKh6:LMD-2023.rar; do
  curl -sSL -o "${p#*:}" "https://drive.usercontent.google.com/download?id=${p%%:*}&export=download&confirm=t"; done
sha256sum *.rar   # 上の表の値と照合する
cd ../etcexp && git clone --depth 1 https://github.com/ChristosSmiliotopoulos/evtx_To_CSV_ExportTool.git
cd ../../..
unrar x -o+ data/raw/drive/full-csv_EvaluatedTitles.rar data/extracted/fullcsv/
unrar x -o+ data/raw/etcexp/evtx_To_CSV_ExportTool/RBPolicy.rar data/extracted/rbpolicy/
unrar x -o+ data/raw/drive/LMD-2023.rar data/extracted/lmd2023/
unrar x -o+ "data/extracted/lmd2023/LMD-2023 [1.75M Elements].rar" data/extracted/lmd2023/
D="data/extracted/lmd2023/LMD-2023 [1.75M Elements]/LMD-2023 [1.75M Elements] Checked"
uv run python src/load_fullcsv.py
uv run python -I src/convert_raw.py "$D/Labelled LMD-2023/LMD-2023 [1.75M Elements][Labelled]checked.csv" data/processed/lmd2023_1.75M_labelled.parquet
uv run python src/preprocess_public.py
```
- `LMD-2023.rar` の中には、1.75M・1.87M・2.3M の3つの版が入っています。第三者の分析の一部は2.3M版を使っています。
- 解凍すると約4.3GB になります。
- 取得したファイルは信頼できないデータとして扱い、`-I` を付けた Python で読み込んでください。

### 5.4 動作確認
```bash
uv run python src/summarize.py   # 既存の結果が集計され、REPRO_REPORT.md の数値と一致すれば OK
```

## 6. 注意点
- **メモリ：** ローカル環境（7.9GB）では、何度もメモリ不足で強制終了しました。`run_repro.py` は、重複率を計算した後に float64 の行列を解放するようにしてあります。Bagging-CB と LinearSVC はメモリを多く使います。サーバーでも、**1プロセスにつき1モデル**で実行するのが安全です。ローカル環境では次のように回しました。
  ```bash
  for m in LSVC KNN DT NB LGBM CB SGDC RF Bagging-DT ET; do
    uv run python src/run_repro.py --dataset authors --protocol cv10 --models $m >> results/logs/cv10_authors.log 2>&1 || echo "FAILED $m" >> results/logs/cv10_authors.log
  done
  ```
- **時間がかかるモデル（ローカル環境の実測値、1fold あたり）：**
  - LR（solver が saga）：約1時間
  - Bagging-ET：約45分、Bagging-RF：約25分
  - ET：約5分
  - KNN：予測に約20分
  - DNN-RNN：1エポック約1時間
  - その他（NB・DT・LightGBM・CatBoost・LinearSVC・SGDC・RF）：数秒〜1.5分
- **途中からの再開：** `run_repro.py` は、すでにある結果 JSON をスキップします。止まった場合は、同じコマンドをもう一度実行すれば続きから再開します。
- **著者の行列ファイルの癖：** 日付・時・分の列は、ファイルごとに MinMax 変換されています（例：日付の値が42種類あり、本来の最大31種類を超える）。時期のブロックを作るときは、年・月・週の One-Hot 列を使ってください。
- **公開版の UtcTime 列**は表計算ソフトで壊れています（例：「30:32.4」）。時刻は SystemTime（`public15_raw.t`）を使ってください。

## 7. 次の作業：H1・H2・H5 の検証実験
### 7.1 共通の設定
- **データ：** `authors`（論文に最も近い）と `public15`（ホスト名と時刻の生の値がある）の両方。
- **モデル：** NB、LinearSVC、DT、RF、LightGBM、CatBoost（いずれも表10の設定）。ET は主要な条件だけで回します。KNN と LR は時間がかかるので除外します（必要なら §8 でユーザーに確認）。
- **指標：** `lmd.metrics.evaluate()` をそのまま使います。主な指標は macro F1、予測確率からの AUC、クラス別 F1 です。
- **出力：** `results/verify/<dataset>/<experiment>/<featureset>/<model>_fold<k>.json`（`run_repro.py` と同じ形式）。集計用のスクリプトは新しく作ってください。

### 7.2 特徴量セット（除外するグループ）
| セット | 内容 | authors の列 | public15 の列 |
|---|---|---|---|
| F0 全部 | 論文と同じ15特徴 | 全90列 | 全75列 |
| F1 時刻なし | SystemTime から作った7特徴を除く | `SysTime*` | `SystemTime_*` |
| F2 ホストなし | Computer を除く | `CompSTA*` | `Computer_*` |
| F3 ID系なし | EventRecordID・ExecutionProcessID・ProcessId を除く | `EventRecID`、`ExecProcessID`、`ProcessId` | `EventRecordID`、`Execution_ProcessID`、`ProcessId` |
| F4 最小構成 | F1〜F3 を全部除く（EventID・Initiated・SourceIsIpv6・DestinationPortName だけ） | 残りの列 | 残りの列 |

**ベースライン：** ホスト名だけ、時刻だけ（authors は年・月・週、public15 は時刻 `t`）の多数決ルールで分類した結果。参考値として、これまでの監査ではホスト名だけで正解率97.64%・macro F1 0.884（authors）、時刻のしきい値で正解率99.998%（public15）でした。

### 7.3 分割方法
| 実験 | 分割 | 確かめる仮説 |
|---|---|---|
| E1 ランダム分割 | 層化5分割（`random_state=42`） | 基準（論文と同じ条件） |
| E2 ホストを1台ずつ外す | ホストごとのグループで分割（authors は `CompSTA*` の One-Hot からホスト ID を復元、public15 は `public15_raw.Computer`）。テスト側に含まれないクラスがあるので、クラス別の指標と「攻撃か正常か」の2値 AUC も出す | H1 |
| E3 時期のブロックで分割 | public15 は `t` の日単位、authors は年×月×週の単位でグループ分割（5分割）。時系列順の分割も参考に出す | H2 |
| E4 重複除去 | (a) 重複行を除いて E1 を実施（authors は89,295行）、(b) 同じ特徴ベクトルが学習とテストに分かれないようにグループ分割 | H5 |

`public15` では、ノートPCの4,537件のイベントが時刻だけ変えて約333回ずつ複製されています。F1（時刻なし）にすると、これらが完全な重複になるので、E4 は「F1 と組み合わせた場合」も評価してください。

### 7.4 結果の見方（反証の条件）
- **H1 の反証：** E2 で、ホストを除いた特徴量でも macro F1 や AUC が高いままなら、H1 は否定されます。
- **H2 の反証：** E3 と F1 の組み合わせで高い性能が保たれるなら、H2 は否定されます。
- **H5 の反証：** E4 で、性能が E1 とほぼ変わらなければ、H5 は否定されます。
- まとめ方は `REPRO_REPORT.md` と同じように、表と所見を書きます。出力先は `results/VERIFY_H1H2H5.md` を想定しています。

### 7.5 時間の見積もり（ローカル環境の実測値から）
- 実装：2〜3時間
- 計算：完全版で約7〜8時間（直列実行の場合）。軽い版（3分割、ET なし、authors のみ）なら約2〜3時間です。
- サーバーでは、メモリに余裕があれば、データ×特徴量セットの組み合わせごとにプロセスを並列で走らせると短縮できます。

## 8. ユーザーに確認が必要なこと
1. **軽い版にするか、完全版にするか**（§7.5）。ET・KNN・LR を含めるかどうか。
2. GPU があるなら、DNN（RNN・LSTM）の再現を再開するかどうか。現状では優先度は低いと考えています。
3. H1・H2・H5 の次に進む順番。提案は H4（ラベルを付け直して影響を測る）→ H3（Splunk の `attack_data` のログと突き合わせる。Git LFS の大きなファイルのダウンロードが必要）→ H6（別のデータセットへの転移）です。

## 9. 参考資料
- **論文：** doi:10.1007/s10207-023-00725-8（IJIS 2023）、doi:10.3390/app12157746（Appl. Sci. 2022）
- **著者のコード：**
  - https://github.com/ChristosSmiliotopoulos/evtx_To_CSV_ExportTool （`labelling.py`、`dataPreProcessing.py`、`dfToOHEMinMax.py`、`concatScript.py`、`featureImportance(PeX).py`、`RBPolicy.rar`、`initializationFile/initializationFile.xml`）
  - https://github.com/ChristosSmiliotopoulos/Python_Evtx_Analyzer
- **著者グループの後続研究：**
  - IJIS 2026（doi:10.1007/s10207-025-01182-1）
  - Computers & Security 2025（doi:10.1016/j.cose.2024.104190）
  - 博士論文（doi:10.12681/eadd/59397）
  - いずれも問題を解決していません。
- **第三者による検証：** `ISSUES_AND_HYPOTHESES.md` の §2 を参照（itamarit77、FocusPA、Roy の NCI 修士論文、dannybroyd、FETA、Splunk の security_content と attack_data）。
- **評価方法の参考：** ANSSI「On fair and realistic performance evaluations for graph-based LM detectors」（arXiv 2607.29390、https://github.com/cl-anssi/LMDEval）
