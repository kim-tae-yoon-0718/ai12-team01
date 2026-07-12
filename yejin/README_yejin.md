# README_yejin — 박예진 작업 폴더 안내

팀원별 폴더에 각자 작업물을 모은 뒤 **최종적으로 해체하여 팀 공용 코드로 병합**할 예정이므로,
병합 담당자가 참고할 수 있도록 이 폴더의 파일 구성, 각 파일이 사용하는 데이터·모듈·패키지,
그리고 저장소 공용 코드(`RF_DETR_split_ver`)와의 관계를 정리합니다.

---

## 1. 폴더 구성

```
yejin/
├── README_yejin.md            # 이 문서
├── colab_notebooks/           # Colab 실행 - 데이터 생성 + YOLO fold0 학습
│   ├── extract_pills.ipynb
│   ├── Masking Generator.ipynb
│   ├── task1_synthesis.ipynb
│   ├── task2_synthesis.ipynb
│   ├── ts2_leakage_check.ipynb
│   └── task4_yolov8_5fold_masked_colab.ipynb
└── kaggle_notebooks/          # Kaggle 실행 - RF-DETR/YOLO 학습 + 앙상블 추론
    ├── task0_baseline_kaggle.ipynb
    ├── task1_train56_boost_kaggle.ipynb
    ├── task2_test18_boost_kaggle.ipynb
    ├── task2_test18_boost_masked_kaggle.ipynb
    ├── task3_fulldata_large_masked_kaggle.ipynb
    ├── task4_yolov8_5fold_masked_kaggle.ipynb
    └── ensemble_wbf_inference_kaggle.ipynb
```

## 2. 전체 실험 흐름

```
[데이터 생성 - Colab]
  extract_pills ──(SAM 알약 누끼)──> task1_synthesis / task2_synthesis ──> 합성 pool
  Masking Generator ──(74종 외 알약 마스킹)──> masked pool (dataset-74-masked)
  ts2_leakage_check ──(TS/TL 조합 간 이미지 중복 점검)

[학습 - Kaggle/Colab]
  task0 (baseline 56종) → task1 (+합성 pool1) → task2 (+합성 pool2, 74종)
  → task2-masked (+masked pool)  ──> fold_split_masked.json export (분할의 원본)
  → task3 (동일 데이터 full-data, RF-DETR Large)
  → task4 (동일 데이터, YOLOv8m: Colab이 fold0 / Kaggle이 fold1~4 분담)

[추론/제출 - Kaggle]
  ensemble_wbf_inference: 위 체크포인트들을 Input으로 모아 WBF 앙상블 → 제출 CSV
```

## 3. 노트북별 상세

### 3-1. `kaggle_notebooks/` (학습·추론)

| 파일 | 목적 | 입력 | 산출물 |
|---|---|---|---|
| `task0_baseline_kaggle` | 원본 train 56종 5-fold baseline (RF-DETR medium) | competition 데이터 | fold별 체크포인트, 제출 CSV |
| `task1_train56_boost_kaggle` | train 56종 + 합성 pool1 보강 학습 | + `task1_synthesized` | 〃 |
| `task2_test18_boost_kaggle` | test 전용 18종 + 합성 pool2로 74종 학습. **corrections 하드코딩 스냅샷의 기준 노트북** | + `task2_synthesized` | 〃 |
| `task2_test18_boost_masked_kaggle` | 위 + **masked pool 병합** 5-fold. **`fold_split_masked.json`을 export하는 분할의 원본** | + `dataset-74-masked` | fold별 체크포인트 `medium_task2_syn74_masked_fold{i}_best.pth`, fold 분할 json, 제출 CSV |
| `task3_fulldata_large_masked_kaggle` | task2-masked와 동일 데이터를 **split 없이 전량 학습** (RF-DETR Large, 15ep, rfdetr==1.8.3 고정) | task2-masked와 동일 | `large_task3_full74_masked_ep{N}_last.pth`, 단일모델 제출 CSV |
| `task4_yolov8_5fold_masked_kaggle` | YOLOv8m **fold1~4 분담** 학습 (cls gain 1.5), `fold_split_masked.json` 로드 | task2-masked와 동일 + fold json | `yolov8m_task4_syn74_masked_fold{i}_best.pt` |
| `ensemble_wbf_inference_kaggle` | 체크포인트 Input → **WBF 앙상블** 추론·제출 (그룹 자동 발견, 그룹별 weight, 파일명에 조합 반영) | `test_images` + `task2_synthesized`(라벨 매핑) + 체크포인트들 | `submission_wbf_{그룹+weight}.csv` |

### 3-2. `colab_notebooks/` (데이터 생성 + YOLO fold0)

| 파일 | 목적 | 입력 | 산출물 |
|---|---|---|---|
| `extract_pills` | SAM으로 알약 누끼 추출 (클래스별 폴더 저장) | `train_56_45_merged_coco.zip`의 aihub_45_fill 이미지 | `pills/` 누끼 이미지 |
| `task1_synthesis` | 누끼+배경 합성으로 X종(희소 클래스) 균형화 pool 생성 | `pills/`, `backgrounds/` | 합성 pool1 (images + COCO json) |
| `task2_synthesis` | N종(test 전용 18종) 누끼 추출 + X종/N종 균형 합성 pool 생성 | AIHub 원천, `pills/` | 합성 pool2 (`task2_synthesized`) |
| `Masking Generator` | AIHub TS/TL 조합(1, 3~8번 zip)에서 74종 외 알약을 마스킹한 실사 이미지 생성 | AIHub TS/TL zip | **masked pool** (`dataset-74-masked` / `dataset_74_masked.zip`) |
| `ts2_leakage_check` | TS/TL 2번 조합 이미지가 다른 조합(1, 3~8번)에 중복 포함되는지 누수 점검 | AIHub TS/TL zip | 점검 결과 (데이터 산출 없음) |
| `task4_yolov8_5fold_masked_colab` | YOLOv8m **fold0 담당** 학습 (Kaggle판과 동일 설정, Drive 백업 이어하기). masked pool은 Drive의 zip을 로컬 해제해 사용 | task2-masked와 동일 (Drive 경로) | `yolov8m_task4_syn74_masked_fold0_best.pt` (Drive) |

## 4. 실험 공통 규칙 (병합/재현 시 반드시 유지)

- **corrections 스냅샷**: task2 계열 노트북에 하드코딩된 corrections dict가 실험의 기준입니다.
  루트 `RF_DETR_split_ver/corrections.json`을 이 스냅샷과 일치시키는 수정이 별도 PR(#13)로 올라가 있습니다.
- **고정 fold 분할**: `fold_split_masked.json`(task2-masked가 export)을 task3/task4가 로드합니다.
  StratifiedGroupKFold 재계산은 sklearn/numpy 버전이 다르면 다른 분할이 나올 수 있어,
  세션·계정·플랫폼(Colab↔Kaggle)이 달라도 fold 구성을 동일하게 유지하기 위한 장치입니다 (fold-matched WBF의 전제).
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

## 6. `RF_DETR_split_ver` (저장소 루트 공용 코드) 사용 현황

**배경**: 이 폴더는 저장소 공용 코드로 등록되어 있지만, 실질적으로는 제가 단독으로 사용해온
모듈입니다. yejin/ 하위로 옮기는 것도 검토했으나, 다른 브랜치·PR과 노트북들이 루트 경로
(`RF_DETR_split_ver/`)를 import 경로로 참조하고 있어 **꼬임 방지를 위해 루트에 그대로 유지**했습니다.
팀 코드 병합 시 이 폴더의 귀속(공용 유지 여부)을 함께 결정해 주세요.

### 6-1. 노트북들이 사용하는 함수 (병합 시 유지 필요)

| 모듈 | 사용 함수 | 주 사용처 |
|---|---|---|
| `dataset.py` | `load_raw_annotations`, `apply_corrections`, `build_category_mapping`, `build_coco`, `make_folds`, `cache_images`, `write_fold_dirs`, `save_label_map`, `compute_label_counts`, `find_data_root`, `check_data_paths` | 학습 노트북 전반 (find/check는 Colab task4만) |
| `model.py` | `get_rfdetr_model` | RF-DETR 학습·추론 전체, 앙상블 |
| `train.py` | `run_kfold`, `train_fold` | 5-fold 학습 노트북 |
| `utils.py` | `report_fold_result`, `summarize_kfold_results`, `summarize_per_class`, `show_error_gallery`, `read_metrics_csv`, `plot_history` | fold 리포팅, task3 학습곡선 |
| `visualize.py` | `collect_predictions_ensemble`, `crop_predictions_by_class`, `_cluster_same_class_boxes` | test 추론·시각화, 앙상블 |
| `colab_setup.py` | `mount_drive` | Colab task4 |

(내부 전용으로 위 함수들이 의존하는 것: `dataset.load_label_map`, `visualize.collect_predictions_from_coco`
/`evaluate_from_data`/`visualize_errors_from_data` 및 `_` 헬퍼들 — 직접 import하지 않아도 유지 필요)

### 6-2. 불용 모듈 (실행 노트북 어디에서도 사용하지 않음 — 병합 시 정리 후보)

yejin의 실행 노트북 13종 전체의 import·호출 + 모듈 내부 상호참조를 전수 분석한 결과입니다.

| 구분 | 함수 | 비고 |
|---|---|---|
| 직접 불용 | `utils.summarize_missing_classes` | 미검출 클래스 요약 — 사용 이력 없음 |
| 직접 불용 | `utils.save_class_crops` | 클래스별 crop 저장 — 사용 이력 없음 |
| 직접 불용 | `visualize.visualize_errors` | 래퍼 함수 (내부 구현인 `visualize_errors_from_data`는 사용 중이므로 유지) |
| 직접 불용 | `visualize.save_ensemble_gallery` | 앙상블 갤러리 저장 — 노트북은 로컬 시각화 함수 사용 |
| 직접 불용 | `colab_setup.prepare_data` / `restore_data` | 초기 Colab 오케스트레이션 래퍼 — 현재는 하위 함수를 직접 호출 |
| 연쇄 불용 | `dataset.build_fold_dataset` / `restore_dataset` / `archive_dataset` | 위 colab_setup 래퍼들만 참조 → 래퍼가 불용이면 함께 불용 |

**⚠ 삭제 시 주의**:
- `train.load_config`는 노트북에서 안 쓰지만 **`python train.py` CLI(`__main__`)가 사용**하므로 불용이 아닙니다
  (`config.yaml`도 이 CLI 전용).
- 위 목록은 "제 노트북 기준" 판정입니다. 다른 팀원이 개인 스크립트에서 쓰고 있다면 유지해야 합니다.
- 데이터 생성 계열 노트북 5종(extract_pills, Masking Generator, task1/2_synthesis, ts2_leakage_check)은
  `RF_DETR_split_ver`를 아예 사용하지 않는 독립 실행형입니다.

## 7. 외부 패키지

| 패키지 | 용도 | 비고 |
|---|---|---|
| `rfdetr[train]` | RF-DETR 학습·추론 | task3는 `==1.8.3` 고정 (4절 참고) |
| `ultralytics>=8.3` | YOLOv8 학습·추론, COCO→YOLO 변환(`convert_coco`) | task4, 앙상블 |
| `ensemble-boxes` | WBF(Weighted Box Fusion) | 5-fold/멀티모델 융합 |
| `torchmetrics` | mAP 계산 | `visualize.py`가 사용 |
| `segment-anything` (SAM) | 알약 누끼 추출 | 데이터 생성 노트북 |
