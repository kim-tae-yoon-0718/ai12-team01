# rf-detr/colab_setup.py
"""
Colab 환경 전용 진입 스크립트.
dataset.py/model.py/train.py는 순수 로직만 담고, Google Drive 마운트처럼
Colab 런타임에 종속된 부분과 그 오케스트레이션만 이 파일에 모아둠.

사전 준비 (노트북 셀에서 직접 실행, 코드로 감싸지 않음):
    !pip install -q "rfdetr[train,loggers]"

사용 예 (노트북 셀 - 학습):
    from colab_setup import mount_drive
    from train import load_config, run_kfold

    mount_drive()
    config = load_config('config.yaml')
    run_result = run_kfold(config)

사용 예 (다음 셀 - fold별 요약, 재학습 없이 재실행 가능):
    from utils import summarize_kfold_results
    summarize_kfold_results(run_result['fold_metrics'], config['model']['tag'])

사용 예 (다음 셀 - 클래스별 mAP 집계 DataFrame):
    from dataset import compute_label_counts
    from utils import summarize_per_class

    label_counts = compute_label_counts(config['data']['dataset_dir'])
    df = summarize_per_class(run_result['fold_metrics'], run_result['label_to_category_id'], label_counts)
    df  # 노트북 셀 마지막 줄에 두면 자동으로 표로 표시됨
"""
def mount_drive():
    """Google Drive를 /content/drive에 마운트합니다."""
    from google.colab import drive
    drive.mount('/content/drive')

