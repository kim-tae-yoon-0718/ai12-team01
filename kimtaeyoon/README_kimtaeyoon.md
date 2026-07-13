# kimtaeyoon (김태윤) 작업 노트북

RF-DETR 71클래스(기존 56종 + 데이터 있는 15종; id 공간은 74) 파이프라인 관련 **실험·분석·추론·시각화** 노트북 모음.
메인 모델 학습 코드는 `RF_DETR_origin/06_71cls_5fold_train.py` 에 있음.

## 원본 노트북 메모
56종, 56종+11종으로 확인한 캐글 점수는 0.98, 0.985입니다.

아직 56종+18종+56종에 대한 추가 데이터로 캐글 점수는 확인하지 않았습니다.

해당 5폴드 셋엔 예진님이 변경 요청하신 부분은 반영되어 있지 않은 상태입니다. (카테고리 ID 변경 요청 미반영)

## 파일
| 파일 | 내용 |
|---|---|
| 01_diagnose_combo_boxes | 조합코드로 원본 이미지+박스 표시 (진단) |
| 02_verify_kcode_dlidx | K코드 vs dl_idx 일치 검증 표 |
| 03_masking_result_view | 마스킹 결과 500장 시각화 확인 |
| 04_problem_image_detect | 문제 이미지 검출 |
| 05_wbf_infer_submission | WBF 앙상블 추론 → 제출 CSV 생성 |
| 06_class_thumbs_crop | 56종 대표 알약 크롭 |
| 07_valid_error_analysis_oof | valid 오류 분석 (OOF) |
| 08_suspect_visualize | 의심 항목 GT vs 예측 시각화 |
| 09_misclass_compare | 분류오류 대조 크롭 |
| 10_masking_redesign | (데이터준비) 조합단위 마스킹 재설계 |
| 11_build_71cls_5fold_dataset | (데이터준비) COCO 변환 + 5-fold 병합 + zip |

> 각 노트북은 앞부분에 설치/마운트/데이터 복원 셀을 포함해 단독 실행 가능하게 구성함.
