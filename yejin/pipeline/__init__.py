# yejin/pipeline
"""박예진 실험 노트북 공통 파이프라인 모듈.

colab_notebooks/kaggle_notebooks의 실험 노트북들에서 반복 정의되던 로컬 함수들을
기능 단위로 모듈화한 패키지입니다. 노트북에서는 저장소 clone 후 아래처럼 사용합니다.

    sys.path.insert(0, os.path.join(REPO_DIR, 'yejin'))
    from pipeline import cloud, pools, folds, wbf, viz          # 필요 모듈만
    from pipeline.corrections import save_corrections_snapshot

모듈 구성
    cloud        실행 환경(Kaggle/Colab) 감지 + 입력 경로/파일 재귀 탐색
    corrections  annotation 수정 내역(corrections) 스냅샷 하드코딩 + 파일 저장
    pools        합성 pool / masked pool 로드·검증·병합, 74종 라벨 매핑
    folds        고정 fold 분할(json) 저장/로드, 그룹 누수 점검, fold 분포 요약
    wbf          WBF 융합, 예측 라벨 정제, 제출 CSV 생성
    viz          GT/예측 클래스별 crop 갤러리
    yolo         YOLO 5-fold 학습·COCO->YOLO 변환·앙상블 예측 수집

의존성 주의: 각 모듈은 자신이 필요한 외부 패키지만 import하므로 (예: yolo -> ultralytics,
wbf -> ensemble-boxes), 노트북에서 사용할 모듈에 맞는 패키지만 설치하면 됩니다.
무거운 의존성을 피하기 위해 패키지 차원의 일괄 import는 하지 않습니다.
"""
