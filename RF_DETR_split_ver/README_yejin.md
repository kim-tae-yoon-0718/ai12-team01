# README_yejin — 박예진 작업 내역 안내

masked pool 병합 실험(task2-masked / task3 / task4)과 최종 WBF 앙상블 추론까지의 작업물을
`RF_DETR_split_ver`(저장소 공용 코드) 안으로 병합했습니다. 노트북에서 반복 정의되던 코드는
공용 모듈로 분리했고, 데이터 생성 계열 노트북 5종만 원본 그대로 남겨뒀습니다.

관련 PR: #13(corrections 스냅샷 복원) → #15(공용 모듈 추가) → #17(노트북 이동+모듈 적용, #15 기반 스택).

---

## 1. 폴더 구성

```
RF_DETR_split_ver/
├── README_yejin.md            # 이 문서
├── dataset.py, model.py, train.py, utils.py, visualize.py, colab_setup.py  # 저장소 원본 (일부 확장)
├── folds.py, pools.py, ensemble.py, yolo.py   # 이번 작업으로 추가한 공용 모듈
├── colab_notebooks/           # Colab 실행
│   ├── extract_pills.ipynb            # (원본 그대로 - RF_DETR_split_ver 미사용)
│   ├── Masking Generator.ipynb        # (원본 그대로 - RF_DETR_split_ver 미사용)
│   ├── task1_synthesis.ipynb          # (원본 그대로 - RF_DETR_split_ver 미사용)
│   ├── task2_synthesis.ipynb          # (원본 그대로 - RF_DETR_split_ver 미사용)
│   ├── ts2_leakage_check.ipynb        # (원본 그대로 - RF_DETR_split_ver 미사용)
│   └── task4_yolov8_5fold_masked_colab.ipynb   # 공용 모듈 적용 (fold0 + 전체 fold 실행/추론 포함)
└── kaggle_notebooks/          # Kaggle 실행 - 전부 공용 모듈 적용
    ├── task0_baseline_kaggle.ipynb
    ├── task1_train56_boost_kaggle.ipynb
    ├── task2_test18_boost_kaggle.ipynb
    ├── task2_test18_boost_masked_kaggle.ipynb
    ├── task3_fulldata_large_masked_kaggle.ipynb
    ├── task4_yolov8_5fold_masked_kaggle.ipynb
    └── ensemble_wbf_inference_kaggle.ipynb
```

**데이터 생성 노트북 5종은 이번 정리에서 의도적으로 제외**했습니다 (원본 그대로 유지, 모듈화하지
않음). 이에 따라 그 5종만 쓰던 `pill_extraction.py`/`synthesis.py` 모듈은 만들었다가 삭제했습니다
(커밋 히스토리에 생성/삭제 둘 다 남아 있음 — 필요해지면 그 커밋에서 복원 가능).

## 2. 전체 실험 흐름

```
[데이터 생성 - Colab, 원본 그대로]
  extract_pills ──(SAM 알약 누끼)──> task1_synthesis / task2_synthesis ──> 합성 pool
  Masking Generator ──(74종 외 알약 마스킹)──> masked pool (dataset-74-masked)
  ts2_leakage_check ──(TS/TL 조합 간 이미지 중복 점검)

[학습 - Kaggle/Colab, 공용 모듈 적용]
  task0 (baseline 56종) → task1 (+합성 pool1) → task2 (+합성 pool2, 74종)
  → task2-masked (+masked pool) ──> fold_split_masked.json export (분할의 원본)
  → task3 (동일 데이터 full-data, RF-DETR Large)
  → task4 (동일 데이터, YOLOv8m: Colab이 fold0(및 전체 재현) / Kaggle이 fold1~4 분담)

[추론/제출 - Kaggle]
  ensemble_wbf_inference: 위 체크포인트들을 Input으로 모아 WBF 앙상블 → 제출 CSV
```

## 3. 노트북별 상세

### 3-1. `kaggle_notebooks/` (학습·추론)

| 파일 | 목적 | 입력 | 산출물 |
|---|---|---|---|
| `task0_baseline_kaggle` | 원본 train 56종 5-fold baseline (RF-DETR medium) | competition 데이터 | fold별 체크포인트, 제출 CSV |
| `task1_train56_boost_kaggle` | train 56종 + 합성 pool1 보강 학습 | + `task1_synthesized` | 〃 |
| `task2_test18_boost_kaggle` | test 전용 18종 + 합성 pool2로 74종 학습 | + `task2_synthesized` | 〃 |
| `task2_test18_boost_masked_kaggle` | 위 + **masked pool 병합** 5-fold. **`fold_split_masked.json`을 export하는 분할의 원본** | + `dataset-74-masked` | fold별 체크포인트 `medium_task2_syn74_masked_fold{i}_best.pth`, fold 분할 json, 제출 CSV |
| `task3_fulldata_large_masked_kaggle` | task2-masked와 동일 데이터를 **split 없이 전량 학습** (RF-DETR Large, 15ep, rfdetr==1.8.3 고정) | task2-masked와 동일 | `large_task3_full74_masked_ep{N}_last.pth`, 단일모델 제출 CSV |
| `task4_yolov8_5fold_masked_kaggle` | YOLOv8m **fold1~4 분담** 학습 (cls gain 1.5), `fold_split_masked.json` 로드 | task2-masked와 동일 + fold json | `yolov8m_task4_syn74_masked_fold{i}_best.pt` |
| `ensemble_wbf_inference_kaggle` | 체크포인트 Input → **WBF 앙상블** 추론·제출 (그룹 자동 발견, 그룹별 weight, 파일명에 조합 반영) | `test_images` + `task2_synthesized`(라벨 매핑) + 체크포인트들 | `submission_wbf_{그룹+weight}.csv` |

corrections는 5개 노트북(task2/task2-masked/task3/task4-colab/task4-kaggle) 모두
**하드코딩을 걷어내고 저장소 `corrections.json`을 직접 참조**하도록 바꿨습니다 (PR #13에서
그 파일 자체를 이 실험들의 스냅샷과 일치시켜 뒀으므로 안전). task0/task1은 원래부터 파일 참조
방식이라 변경 없음.

### 3-2. `colab_notebooks/`

| 파일 | 목적 | 입력 | 산출물 |
|---|---|---|---|
| `extract_pills` | SAM으로 알약 누끼 추출 (클래스별 폴더 저장) | `train_56_45_merged_coco.zip`의 aihub_45_fill 이미지 | `pills/` 누끼 이미지 |
| `task1_synthesis` | 누끼+배경 합성으로 X종(희소 클래스) 균형화 pool 생성 | `pills/`, `backgrounds/` | 합성 pool1 (images + COCO json) |
| `task2_synthesis` | N종(test 전용 18종) 누끼 추출 + X종/N종 균형 합성 pool 생성 | AIHub 원천, `pills/` | 합성 pool2 (`task2_synthesized`) |
| `Masking Generator` | AIHub TS/TL 조합(1, 3~8번 zip)에서 74종 외 알약을 마스킹한 실사 이미지 생성 | AIHub TS/TL zip | **masked pool** (`dataset-74-masked` / `dataset_74_masked.zip`) |
| `ts2_leakage_check` | TS/TL 2번 조합 이미지가 다른 조합(1, 3~8번)에 중복 포함되는지 누수 점검 | AIHub TS/TL zip | 점검 결과 (데이터 산출 없음) |
| `task4_yolov8_5fold_masked_colab` | YOLOv8m 학습 (Drive 백업 이어하기) + test 추론/WBF/시각화까지 포함. masked pool은 Drive의 zip을 로컬 해제해 사용 | task2-masked와 동일 (Drive 경로) | `yolov8m_task4_syn74_masked_fold{i}_best.pt` (Drive) |

## 4. 실험 공통 규칙 (재현 시 반드시 유지)

- **corrections 스냅샷**: task2 계열 실험의 기준 스냅샷이 저장소 `RF_DETR_split_ver/corrections.json`
  자체입니다 (PR #13). 이후 팀이 이 파일을 다시 검수/수정하면 이 실험들의 재현 조건도 함께 바뀌니
  주의하세요.
- **고정 fold 분할**: `fold_split_masked.json`(task2-masked가 export)을 task3/task4가 로드합니다.
  StratifiedGroupKFold 재계산은 sklearn/numpy 버전이 다르면 다른 분할이 나올 수 있어,
  세션·계정·플랫폼(Colab↔Kaggle)이 달라도 fold 구성을 동일하게 유지하기 위한 장치입니다
  (fold-matched WBF의 전제). **버그 수정**: task2-masked 노트북이 분할을 계산만 하고 이 json을
  export하지 않던 누락이 있었는데, 노트북 정리(PR #17) 중 발견해 `folds.export_fold_split` 호출을
  추가했습니다.
- **masked pool 리네임**: 파일명이 원본 train과 동일 체계(K코드+촬영조건)라 전체 파일에 `msk_` 접두어를
  붙여 병합합니다 (이미지 캐시 덮어쓰기·corrections 오적용 방지). fold 그룹 키 계산 시에는 접두어를
  벗겨 원본과 같은 구성 그룹으로 묶습니다 (train/valid 누수 방지).
- **데이터 제외 목록**: `syn_00505.png`, `syn_00102.png` (task2-masked 시각화 검수에서 제외 확정).
  fold json이 이 제외 적용 후 파일 집합 기준이므로 목록을 바꾸면 분할-데이터 일치가 깨집니다.
- **버전 고정**: task3는 `rfdetr==1.8.3` 고정 (1.8대의 체크포인트 파일명 체계를 소스에서 확인한 버전 —
  버전이 바뀌면 산출물 파일명이 달라져 백업 로직이 깨질 수 있음).

## 5. 외부 데이터 리소스

| 리소스 | 위치 | 생성/출처 |
|---|---|---|
| competition 데이터 (`train_images`/`train_annotations`/`test_images`) | Kaggle Input / Drive | 대회 제공 |
| `task2_synthesized` (합성 pool2) | Kaggle Dataset / Drive | `task2_synthesis.ipynb` |
| `dataset-74-masked` (masked pool) | Kaggle Dataset / Drive `dataset_74_masked.zip` | `Masking Generator.ipynb` |
| `fold_split_masked.json` | Kaggle Dataset / Drive | `task2_test18_boost_masked_kaggle.ipynb` export |
| 체크포인트 | 각 노트북 커밋 Output / Drive | 3-1 표의 파일명 규칙 참고 |

## 6. `RF_DETR_split_ver` 모듈 구성 (이번 작업으로 추가/확장된 부분)

기존 6개 파일(`dataset.py`/`model.py`/`train.py`/`utils.py`/`visualize.py`/`colab_setup.py`)은
저장소 공용 코드로 그대로 유지했고, 그중 2개를 확장 + 신규 4개를 추가했습니다.

| 파일 | 상태 | 추가된 함수 | 주 사용처 |
|---|---|---|---|
| `dataset.py` | 확장 | `find_input_dir` | Kaggle 노트북 전반 (find_data_root의 Kaggle용 짝) |
| `visualize.py` | 확장 | `show_gt_class_crops`, `show_pred_class_crops` | GT/예측 클래스별 crop 시각화 |
| `folds.py` | 신규 | `make_folds_masked`, `export_fold_split`, `load_fold_split`, `assert_no_group_leak`, `summarize_fold_distribution`, `print_fold_warnings` | fold 분할 계산/고정 공유/검증 (task2-masked가 export, task3/task4가 load) |
| `pools.py` | 신규 | `load_pool_annotations`, `load_masked_annotations`, `merge_pool`, `merge_masked_pool`, `apply_exclusions`, `build_cat2label_74` | 합성/masked pool 병합, 74종 라벨 매핑 |
| `ensemble.py` | 신규 | `filter_valid_labels`, `fuse_predictions_wbf`, `fuse_merged_wbf`, `collect_predictions_ensemble_yolo`, `make_submission`, `extract_image_id` | WBF 융합, YOLO 앙상블 수집, 제출 CSV |
| `yolo.py` | 신규 | `get_yolo_model`, `build_yolo_fold`, `train_fold_yolo`, `report_fold_result_yolo`, `run_folds_yolo`, `summarize_kfold_results_yolo`, `summarize_per_class_yolo` | YOLOv8 변환/학습/평가/집계 |

각 함수의 상세 동작·안전장치·설계 이유는 해당 파일의 docstring에 있습니다. 통합 과정에서 노트북마다
조금씩 달랐던 "변형"들은 AST 레벨로 원본과 대조해 로직이 그대로인지 검증한 뒤 상위 호환 버전으로
합쳤습니다 (예: `show_pred_class_crops`의 "전부 표시"/"상위 N개 표시" 모드를 `max_per_class` 파라미터
하나로 통합).

**의도적으로 바꾼 부분 2곳**: `build_yolo_fold`가 `label2cat`을 (기존의 암묵적 전역변수 대신) 명시
파라미터로 받도록 함. `show_pred_class_crops`를 위에서 말한 두 변형의 상위 호환으로 통합.

### 6-1. 기존 6개 파일의 불용 함수 (실행 노트북 어디에서도 사용하지 않음 — 별도 정리 후보)

노트북 13종 전체의 import·호출 + 모듈 내부 상호참조를 전수 분석한 결과입니다 (이번 작업 대상이 아니라
별도 PR로 처리 권장).

| 구분 | 함수 | 비고 |
|---|---|---|
| 직접 불용 | `utils.summarize_missing_classes` | 미검출 클래스 요약 — 사용 이력 없음 |
| 직접 불용 | `utils.save_class_crops` | 클래스별 crop 저장 — 사용 이력 없음 |
| 직접 불용 | `visualize.visualize_errors` | 래퍼 함수 (내부 구현인 `visualize_errors_from_data`는 사용 중이므로 유지) |
| 직접 불용 | `visualize.save_ensemble_gallery` | 앙상블 갤러리 저장 — 노트북은 로컬 시각화 함수 사용 |
| 직접 불용 | `colab_setup.prepare_data` / `restore_data` | 초기 Colab 오케스트레이션 래퍼 — 현재는 하위 함수를 직접 호출 |
| 연쇄 불용 | `dataset.build_fold_dataset` / `restore_dataset` / `archive_dataset` | 위 colab_setup 래퍼들만 참조 → 래퍼가 불용이면 함께 불용 |

**⚠ 삭제 시 주의**: `train.load_config`는 노트북에서 안 쓰지만 **`python train.py` CLI(`__main__`)가
사용**하므로 불용이 아닙니다. 다른 팀원이 개인 스크립트에서 위 함수들을 쓰고 있다면 유지해야 합니다.

## 7. 외부 패키지

| 패키지 | 용도 | 비고 |
|---|---|---|
| `rfdetr[train]` | RF-DETR 학습·추론 | task3는 `==1.8.3` 고정 (4절 참고) |
| `ultralytics>=8.3` | YOLOv8 학습·추론, COCO→YOLO 변환(`convert_coco`) | task4, 앙상블 |
| `ensemble-boxes` | WBF(Weighted Box Fusion) | 5-fold/멀티모델 융합 |
| `torchmetrics` | mAP 계산 | `visualize.py`가 사용 |
| `segment-anything` (SAM) | 알약 누끼 추출 | 데이터 생성 노트북 (원본 그대로, 모듈화 대상 아님) |
