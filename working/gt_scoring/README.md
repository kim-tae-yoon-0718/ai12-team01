# GT Scoring

이 폴더는 테스트셋의 프로젝트 내부 정답 채점 전용 폴더다.

## 채점 기준

반드시 `ground_truth/test_ground_truth.csv`를 기준으로 **대회 지표인
`mAP@0.75:0.95`**를 채점한다. IoU 0.75, 0.80, 0.85, 0.90, 0.95의 AP를
평균한 단일 점수다. 이 GT는
0.99 dedup 정답 후보와 RF-DETR Large 결과를 위치별로 비교하고, 불일치
이미지를 직접 검수하여 대회 어노테이션의 클래스 오기재와 중복을 교정한
결과다. 확인된 오류 범위에서는 대회 제공 라벨보다 정확하다.

- 이미지: 842장
- bbox: 3,229개
- 클래스: 74종
- 클래스 교정: 67건
- N72/N73/N74: 27/30/8개

이 GT는 채점과 오류 분석에만 사용한다. 학습, validation, checkpoint 선택,
submission 생성에는 사용하지 않는다.

## 실행

프로젝트 루트에서 다음처럼 실행한다.

```bash
python working/gt_scoring/score.py \
  --submission working/submissions/your_submission.csv \
  --out-dir working/gt_scoring/results/your_run
```

## 구성

- `ground_truth/`: GT CSV, COCO JSON, 74종 매핑 및 교정 기록
- `results/`: 모델별 GT 채점 결과
- `score.py`: 정식 GT 경로를 고정한 간단한 실행 진입점
- `score_submission_against_gt.py`: GT 전용 COCO/IoU 평가기

채점기는 실제 제출 채점처럼 표준 출력에 최종 점수 한 줄만 표시하고,
결과 폴더에는 `score.txt`와 `score.json`만 저장한다. 현재 YOLO11m 100
epoch 5-fold 결과는 `results/yolo11m_local_100ep_5fold/`에 보관한다.
