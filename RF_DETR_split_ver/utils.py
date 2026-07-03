# rf-detr/utils.py
"""
학습 곡선(history) 구성, fold별/5-fold 리포팅 자동화.
"""
import os
import glob
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from model import get_rfdetr_model
from visualize import collect_predictions_from_coco, evaluate_from_data, visualize_errors_from_data


def read_metrics_csv(output_dir, use_ema=False):
    """
    RF-DETR가 학습 중 자동으로 남기는 {output_dir}/metrics.csv를 읽어
    plot_history()가 기대하는 형태의 history dict로 변환합니다.

    확인된 실제 컬럼(rf-detr 소스 tests/training/test_metrics_csv.py 기준):
      train/loss, train/lr, val/loss, val/mAP_50, val/mAP_50_95, val/mAP_75, val/mAR
      (use_ema=True면 val/ema_mAP_50 등 EMA 버전도 존재)

    Args:
        output_dir (str): model.train(output_dir=...)에 넘긴 경로
        use_ema (bool): EMA 가중치 기준 지표를 쓸지 여부

    Returns:
        dict: {'train_loss', 'val_map', 'val_map_50', 'val_map_75'}
    """
    df = pd.read_csv(os.path.join(output_dir, 'metrics.csv'))
    prefix = 'ema_' if use_ema else ''

    def per_epoch(col):
        # 같은 epoch에 여러 행(step)이 있고 컬럼별 로깅 시점이 달라 NaN이 섞여 있으므로 epoch 단위로 묶어서 각 epoch의 마지막 유효값만 취합니다.
        return df.groupby('epoch')[col].last().dropna().tolist()

    return {
        'train_loss': per_epoch('train/loss'),
        'val_map': per_epoch(f'val/{prefix}mAP_50_95'),
        'val_map_50': per_epoch(f'val/{prefix}mAP_50'),
        'val_map_75': per_epoch(f'val/{prefix}mAP_75'),
    }


def plot_history(history, title='Training History', save_path=None):
    """
    read_metrics_csv()가 만든 history dict를 그대로 넣어 학습 곡선을 시각화합니다.

    Args:
        history (dict): {'train_loss': [...], 'val_map': [...], 'val_map_50': [...], 'val_map_75': [...]}
        title (str): 그래프 제목
        save_path (str): 저장 경로 (None이면 화면에 표시만)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], marker='o', markersize=3, color='steelblue')
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    axes[1].plot(history['val_map'], marker='o', markersize=3, color='coral', label='mAP@0.5:0.95')
    axes[1].plot(history['val_map_50'], marker='o', markersize=3, color='seagreen', label='mAP@0.5')
    axes[1].plot(history['val_map_75'], marker='o', markersize=3, color='purple', label='mAP@0.75')

    best_epoch = int(np.argmax(history['val_map_75'])) + 1
    axes[1].axvline(x=best_epoch - 1, color='gray', linestyle='--', label=f'Best epoch {best_epoch}')
    axes[1].set_title('Validation mAP')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mAP')
    axes[1].legend(fontsize=7)
    axes[1].grid(True)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def show_error_gallery(vis_dir, ncols=4, figsize_per_image=4, start=0, limit=None):
    """
    visualize_errors()/save_ensemble_gallery() 등이 vis_dir에 저장해둔 이미지들을
    한 셀에서 grid(subplot) 형태로 한 번에 확인합니다. (이미지 재계산 없이 저장된 PNG만 읽음)
    이름은 "error"지만 폴더 안 PNG를 grid로 보여주는 범용 함수라 오답 이미지가 아닌
    다른 갤러리(예: 앙상블 예측 결과)에도 그대로 씁니다.

    이미지가 많은 폴더(예: test 전체)는 한 번에 다 그리면 무겁기 때문에,
    start/limit으로 페이지 단위로 나눠 볼 수 있습니다.

    Args:
        vis_dir (str): 이미지가 저장된 폴더
        ncols (int): 한 줄에 표시할 이미지 수
        figsize_per_image (float): 이미지 한 장당 figure 크기(inch) 기준
        start (int): 파일명 정렬 기준으로 몇 번째부터 볼지 (페이지네이션용)
        limit (int): 한 번에 표시할 이미지 수 (None이면 start부터 전부)
    """
    paths = sorted(glob.glob(os.path.join(vis_dir, '*.png')))
    if not paths:
        print(f'{vis_dir}에 이미지 없음')
        return

    total = len(paths)
    paths = paths[start:start + limit] if limit else paths[start:]
    if not paths:
        print(f'start={start}가 전체 {total}장 범위를 벗어남')
        return
    print(f'{start}~{start + len(paths) - 1} / 총 {total}장')

    nrows = math.ceil(len(paths) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_image * ncols, figsize_per_image * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, path in zip(axes, paths):
        ax.imshow(plt.imread(path))
        ax.set_title(os.path.basename(path), fontsize=7)
        ax.axis('off')
    for ax in axes[len(paths):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def report_fold_result(fold_idx, checkpoint_path, model_variant, dataset_dir,
                        label_to_category_id, vis_dir, score_threshold=0.5, iou_threshold=0.5):
    """
    fold 하나에 대해 mAP 계산(클래스별 포함) + 오답 이미지 시각화를 자동으로 수행합니다.
    train.run_kfold()의 fold 루프 안에서 각 fold 학습 직후 호출됩니다.
    클래스별 AP는 fold마다 콘솔에 출력하지 않고 반환값에만 담아둡니다 
    — 전체 fold가 끝난 뒤 summarize_per_class()로 한 번에 집계해서 보는 쪽으로 통일했습니다.

    collect_predictions_from_coco()로 추론을 1회만 수행하고, 그 결과(pred_data)를
    mAP 계산(evaluate_from_data)과 오답 시각화(visualize_errors_from_data) 양쪽에
    재사용해서 추론 중복을 피합니다.

    collect_predictions_from_coco()는 score_threshold=0.0으로 모든 예측을 걸러내지 않고 가져오는데, 
    여기엔 RF-DETR의 배경/no-object 클래스(0)나 학습에 쓰인 
    실제 카테고리 범위를 벗어난 내부 예약 클래스까지 섞여 나올 수 있습니다.
    이런 라벨은 label_to_category_id에 없으므로, mAP/오답 시각화에 넘기기 전에 걸러냅니다.

    Args:
        fold_idx (int): fold 번호 (0-indexed)
        checkpoint_path (str): train_fold()가 반환한 best 체크포인트 경로
        model_variant (str): 'small' | 'medium' 등
        dataset_dir (str): fold 디렉토리들의 루트
        label_to_category_id (dict): 모델 라벨 -> 원본 category_id 매핑
        vis_dir (str): 오답 이미지 저장 폴더
        score_threshold (float): 오답 판정 시 예측 최소 confidence
        iou_threshold (float): GT-예측 매칭 IoU 기준

    Returns:
        dict: evaluate_from_data()의 반환값 (map, map_50, map_75_95 등 - 5-fold 집계용)
    """
    valid_dir = os.path.join(dataset_dir, f'fold{fold_idx}', 'valid')
    coco_json_path = os.path.join(valid_dir, '_annotations.coco.json')

    model = get_rfdetr_model(model_variant, checkpoint_path=checkpoint_path)

    pred_data = collect_predictions_from_coco(model, coco_json_path, valid_dir, score_threshold=0.0)

    valid_labels = set(label_to_category_id.keys())
    for d in pred_data:
        keep = torch.tensor([lbl.item() in valid_labels for lbl in d['pred_labels']], dtype=torch.bool)
        d['pred_boxes'] = d['pred_boxes'][keep]
        d['pred_labels'] = d['pred_labels'][keep]
        d['pred_scores'] = d['pred_scores'][keep]

    metrics = evaluate_from_data(pred_data)

    visualize_errors_from_data(pred_data, label_to_category_id, vis_dir,
                                score_threshold=score_threshold, iou_threshold=iou_threshold,
                                file_prefix=f'fold{fold_idx}_error')

    return metrics


def summarize_kfold_results(fold_metrics, model_tag):
    """
    fold별 evaluate_from_data() 결과 리스트를 받아 mAP@0.75:0.95 평균±표준편차를
    출력합니다. 

    Args:
        fold_metrics (list): report_fold_result()가 fold마다 반환한 dict 리스트
        model_tag (str): config['model']['tag'] - src의 model_name 자리에 해당

    Returns:
        dict: {'map': (mean, std), 'map_50': (mean, std), 'map_75_95': (mean, std)}
    """
    def agg(key):
        vals = [m[key] for m in fold_metrics]
        return float(np.mean(vals)), float(np.std(vals))

    map_mean, map_std = agg('map')
    map50_mean, map50_std = agg('map_50')
    strict_mean, strict_std = agg('map_75_95')

    print(f"\n{'='*50}\n{model_tag} 최종 결과 ({len(fold_metrics)}-fold 평균)\n"
          f"mAP@0.75:0.95: {strict_mean:.4f} ± {strict_std:.4f}\n{'='*50}")

    return {'map': (map_mean, map_std), 'map_50': (map50_mean, map50_std), 'map_75_95': (strict_mean, strict_std)}


def summarize_per_class(fold_metrics, label_to_category_id, label_counts):
    """
    fold별 evaluate_from_data() 결과(classes/map_per_class)를 폴드 전체에 걸쳐 집계해,
    클래스별 mAP를 mean_AP 내림차순 DataFrame으로 정리합니다.

    Args:
        fold_metrics (list): report_fold_result()가 fold마다 반환한 dict 리스트
        label_to_category_id (dict): 모델 라벨 -> 원본 category_id 매핑
        label_counts (dict): {label: 전체 데이터 등장 횟수} (dataset.compute_label_counts() 참고)

    Returns:
        pd.DataFrame: columns = [label, category_id, total_count, mean_AP, std_AP, valid_folds],
                       mean_AP 내림차순 정렬
    """
    per_class = [
        {
            (cls.item() if hasattr(cls, 'item') else int(cls)):
                (ap.item() if hasattr(ap, 'item') else float(ap))
            for cls, ap in zip(m['classes'], m['map_per_class'])
        }
        for m in fold_metrics
    ]

    all_labels = sorted(set(label for d in per_class for label in d))
    rows = []
    for label in all_labels:
        aps = [d.get(label, -1) for d in per_class]
        valid = [ap for ap in aps if ap >= 0]   # -1(해당 fold에 GT 없음) 제외
        rows.append({
            'label': label,
            'category_id': label_to_category_id.get(label, '?'),
            'total_count': label_counts.get(label, 0),
            'mean_AP': round(np.mean(valid), 4) if valid else -1,
            'std_AP': round(np.std(valid), 4) if valid else 0,
            'valid_folds': len(valid),
        })

    return pd.DataFrame(rows).sort_values('mean_AP', ascending=False)


def summarize_missing_classes(pred_data, label_to_category_id, score_threshold=0.5):
    """
    collect_predictions_ensemble()로 모은 test 예측 중 confidence >= score_threshold인
    것만 모아, label_to_category_id에 등록된 학습 클래스 중 test 전체에서 단 한 번도
    예측되지 않은 클래스를 찾습니다.

    주의: pred_count=0이라고 그 클래스가 test에 실제로 없다는 확정적 근거는 아닙니다.
    모델이 전부 놓쳤을 가능성도 있으니, 이 표는 육안 재확인 대상을 추리는 용도로만 쓰세요.

    Args:
        pred_data: collect_predictions_ensemble()의 반환값
        label_to_category_id (dict): 모델 라벨 -> 원본 category_id 매핑
        score_threshold (float): 집계에 포함할 예측의 최소 confidence

    Returns:
        pd.DataFrame: columns = [label, category_id, pred_count], pred_count 오름차순
                       (0인 행이 위쪽에 옴 - 육안 재확인 우선순위)
    """
    counts = defaultdict(int)
    for d in pred_data:
        keep = d['pred_scores'] >= score_threshold
        for lbl in d['pred_labels'][keep].tolist():
            counts[lbl] += 1

    rows = [
        {'label': label, 'category_id': cat_id, 'pred_count': counts.get(label, 0)}
        for label, cat_id in sorted(label_to_category_id.items())
    ]
    return pd.DataFrame(rows).sort_values('pred_count')


def save_class_crops(by_label, label_to_category_id, save_dir):
    """
    visualize.crop_predictions_by_class()의 결과를 라벨(예측 클래스)별 하위 폴더에
    저장합니다. save_dir/label{label:02d}_cat{category_id}/ 안에 그 클래스로 예측된
    crop들이 모이므로, 학습 클래스 대조표(PNG)와 나란히 놓고 클래스별로 훑어보기 좋습니다.

    Args:
        by_label: visualize.crop_predictions_by_class()의 반환값
        label_to_category_id (dict): 모델 라벨 -> 원본 category_id 매핑
        save_dir (str): 저장 루트 폴더

    Returns:
        dict: {label: 저장된 폴더 경로} (예측이 하나도 없던 라벨은 포함 안 됨)
    """
    class_dirs = {}
    for label, items in sorted(by_label.items()):
        cat_id = label_to_category_id.get(label, 'unknown')
        label_dir = os.path.join(save_dir, f'label{label:02d}_cat{cat_id}')
        os.makedirs(label_dir, exist_ok=True)

        for idx, item in enumerate(items):
            stem = os.path.splitext(item['file_name'])[0]
            agree = item.get('agree_count', 1)
            save_path = os.path.join(
                label_dir,
                f'{idx:03d}_{stem}_agree{agree}_fold{item["fold_idx"]}_{item["score"]:.2f}.png')
            plt.imsave(save_path, item['crop'])

        class_dirs[label] = label_dir

    print(f'{len(class_dirs)}개 클래스 폴더 생성 -> {save_dir}')
    return class_dirs
