# rf-detr/visualize.py
"""
RF-DETR 예측 수집, mAP 계산, 오답 시각화.
model.predict()가 supervision.Detections(.xyxy/.confidence/.class_id)를 반환한다는 점은 rf-detr 소스(rfdetr/detr.py의 'from supervision import Detections' 타입힌트)로 확인하였습니다.
"""
import os
import json
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision.ops import box_iou
from torchmetrics.detection import MeanAveragePrecision


def _xywh_to_xyxy(bbox):
    """COCO [x,y,w,h] -> [x1,y1,x2,y2] (dataset.py의 build_coco가 COCO 포맷으로 저장하므로 필요)."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def _draw_box(ax, box, label_idx, label_to_category_id, color, prefix, score=None):
    """src/visualize.py의 _draw_box()와 동일 - 모델 무관 순수 시각화 로직."""
    x1, y1, x2, y2 = box
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor=color, facecolor='none'
    )
    ax.add_patch(rect)

    cat_id = label_to_category_id.get(label_idx, '?')
    text = f'{prefix}: {cat_id}'
    if score is not None:
        text += f' ({score:.2f})'

    ax.text(x1, y1 - 4, text, color=color, fontsize=7,
            bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))


def _is_error(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_threshold):
    """src/visualize.py의 _is_error()와 동일 - GT 누락 또는 예측 오탐 여부 판정."""
    if len(gt_boxes) == 0:
        return len(pred_boxes) > 0

    if len(pred_boxes) == 0:
        return True

    iou = box_iou(gt_boxes, pred_boxes)   # (num_gt, num_pred)

    matched_pred = set()
    for gt_idx in range(len(gt_boxes)):
        best_iou, best_pred_idx = iou[gt_idx].max(0)
        best_pred_idx = best_pred_idx.item()

        if (best_iou >= iou_threshold
                and pred_labels[best_pred_idx] == gt_labels[gt_idx]
                and best_pred_idx not in matched_pred):
            matched_pred.add(best_pred_idx)
        else:
            return True

    return len(matched_pred) < len(pred_boxes)


def collect_predictions_from_coco(model, coco_json_path, image_dir, score_threshold=0.0):
    """
    추론을 1회만 돌리고 결과를 캐싱해서 이후 여러 threshold로 재시각화 가능하게 합니다.
    DataLoader 대신 dataset.py가 만든 fold의 _annotations.coco.json + 이미지 폴더를 직접 읽어서 추론합니다.

    Args:
        model: get_rfdetr_model()으로 만든 RF-DETR 모델 (학습된 가중치 로드된 상태)
        coco_json_path (str): fold의 '_annotations.coco.json' 경로
        image_dir (str): 위 json과 같은 폴더의 이미지 경로
        score_threshold (float): model.predict()에 전달할 최소 confidence

    Returns:
        list of dicts: [{'image': np.ndarray(RGB), 'gt_boxes': Tensor(xyxy), 'gt_labels': Tensor,
                          'pred_boxes': Tensor(xyxy), 'pred_labels': Tensor, 'pred_scores': Tensor}, ...]
    """
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    anns_by_image = defaultdict(list)
    for ann in coco['annotations']:
        anns_by_image[ann['image_id']].append(ann)

    all_data = []
    for img_info in coco['images']:
        img_path = os.path.join(image_dir, img_info['file_name'])
        image = np.array(Image.open(img_path).convert('RGB'))

        anns = anns_by_image[img_info['id']]
        if anns:
            gt_boxes = torch.tensor([_xywh_to_xyxy(a['bbox']) for a in anns], dtype=torch.float32)
            gt_labels = torch.tensor([a['category_id'] for a in anns], dtype=torch.int64)
        else:
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
            gt_labels = torch.zeros((0,), dtype=torch.int64)

        detections = model.predict(image, threshold=score_threshold)
        pred_boxes = torch.tensor(np.array(detections.xyxy), dtype=torch.float32)
        pred_scores = torch.tensor(np.array(detections.confidence), dtype=torch.float32)
        pred_labels = torch.tensor(np.array(detections.class_id), dtype=torch.int64)

        all_data.append({
            'image': image,
            'gt_boxes': gt_boxes,
            'gt_labels': gt_labels,
            'pred_boxes': pred_boxes,
            'pred_labels': pred_labels,
            'pred_scores': pred_scores,
        })

    return all_data


def evaluate_from_data(all_data, device='cpu'):
    """
    collect_predictions_from_coco()로 모아둔 예측/정답 데이터로 mAP를 계산합니다
    (모델 재추론 없음). RF-DETR의 metrics.csv와 무관하게 직접 torchmetrics로 계산하므로, mAP@0.75:0.95(5개 IoU 지점 평균)까지 원본과 완전히 동일하게 정확히 나옵니다.

    Args:
        all_data: collect_predictions_from_coco()의 반환값
        device (str): 'cuda' or 'cpu'

    Returns:
        dict: {'map', 'map_50', 'map_per_class', 'classes', 'map_75_95'}
    """
    metric_standard = MeanAveragePrecision(class_metrics=True)
    # IoU threshold 0.75~0.95, 0.05 간격 (COCO 기본 step과 동일한 방식) - 엄격한 기준
    metric_strict = MeanAveragePrecision(iou_thresholds=[0.75, 0.80, 0.85, 0.90, 0.95])

    for data in all_data:
        metric_targets = [{
            'boxes': data['gt_boxes'].to(device),
            'labels': data['gt_labels'].to(device),
        }]
        preds = [{
            'boxes': data['pred_boxes'].to(device),
            'labels': data['pred_labels'].to(device),
            'scores': data['pred_scores'].to(device),
        }]
        metric_standard.update(preds, metric_targets)
        metric_strict.update(preds, metric_targets)

    result_standard = metric_standard.compute()
    result_strict = metric_strict.compute()

    return {
        'map': result_standard['map'].item(),
        'map_50': result_standard['map_50'].item(),
        'map_per_class': result_standard.get('map_per_class'),
        'classes': result_standard.get('classes'),
        'map_75_95': result_strict['map'].item(),
    }


def visualize_errors_from_data(all_data, label_to_category_id, save_dir,
                                score_threshold=0.5, iou_threshold=0.5, file_prefix='error'):
    """
    RF-DETR은 이미지를 정규화된 텐서가 아니라 원본 RGB 배열(0~255)로 다루므로
    mean/std 역정규화 단계만 제거합니다. (나머지 오차 판정/시각화 로직은 동일).

    Args:
        all_data: collect_predictions_from_coco()의 반환값
        label_to_category_id (dict): 모델 라벨 -> 원본 category_id 매핑
        save_dir (str): 오답 이미지 저장 폴더
        score_threshold (float): 예측 최소 confidence
        iou_threshold (float): GT-예측 매칭 IoU 기준
        file_prefix (str): 저장 파일명 접두어 (fold/sanity check 등 산출물 구분용)

    Returns:
        int: 저장된 오답 이미지 수
    """
    os.makedirs(save_dir, exist_ok=True)
    error_count = 0

    for idx, data in enumerate(all_data):
        gt_boxes = data['gt_boxes']
        gt_labels = data['gt_labels']
        pred_boxes = data['pred_boxes']
        pred_labels = data['pred_labels']
        pred_scores = data['pred_scores']

        keep = pred_scores >= score_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        if not _is_error(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_threshold):
            continue

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(data['image'])

        for box, label in zip(gt_boxes, gt_labels):
            _draw_box(ax, box, label.item(), label_to_category_id, color='lime', prefix='GT')

        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            _draw_box(ax, box, label.item(), label_to_category_id,
                      color='red', prefix='Pred', score=score.item())

        ax.set_title(f'Error image {idx}  (score_thr={score_threshold}, iou_thr={iou_threshold})',
                     fontsize=10)
        ax.axis('off')

        save_path = os.path.join(save_dir, f'{file_prefix}_{error_count:04d}_{idx:04d}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        error_count += 1

    print(f'Saved {error_count} error images -> {save_dir}')
    return error_count


def visualize_errors(model, coco_json_path, image_dir, label_to_category_id, save_dir,
                      score_threshold=0.5, iou_threshold=0.5, file_prefix='error'):
    """
    collect_predictions_from_coco() + visualize_errors_from_data()를 한 번에 실행합니다.

    mAP 계산까지 같은 예측 데이터로 처리해서 추론 중복을 피하고 싶다면, 이 래퍼 대신 collect_predictions_from_coco()를 직접 호출해 evaluate_from_data()/
    visualize_errors_from_data()에 나눠 넘기는 쪽이 더 효율적입니다.

    Returns:
        int: 저장된 오답 이미지 수
    """
    all_data = collect_predictions_from_coco(model, coco_json_path, image_dir, score_threshold=0.0)
    return visualize_errors_from_data(all_data, label_to_category_id, save_dir,
                                       score_threshold=score_threshold, iou_threshold=iou_threshold,
                                       file_prefix=file_prefix)


def collect_predictions_ensemble(models, image_dir, score_threshold=0.5,
                                  extensions=('.png', '.jpg', '.jpeg')):
    """
    annotation이 없는 이미지 폴더(예: test_images)에 대해 여러 모델(fold별 체크포인트)의
    예측을 모아 이미지별로 병합합니다. GT가 없어 mAP 계산 대상이 아니라, 클래스가 무엇인지
    육안으로 판단하기 위한 탐색용입니다. 모델마다 놓치는 클래스가 다를 수 있어, 한 모델이라도
    잡아낸 예측은 전부 모읍니다(합집합 방식 앙상블).

    Args:
        models (list): get_rfdetr_model()으로 만든 모델 리스트 (fold별 체크포인트로 구성)
        image_dir (str): annotation 없이 이미지만 있는 폴더
        score_threshold (float): 각 모델 predict()에 넘길 최소 confidence
        extensions (tuple): 이미지로 취급할 확장자

    Returns:
        list of dicts: [{'file_name', 'image', 'pred_boxes', 'pred_labels', 'pred_scores', 'pred_fold'}, ...]
        pred_fold: 각 예측을 만든 모델의 models 리스트 내 인덱스 (어느 fold가 잡아냈는지 구분용)
    """
    file_names = sorted(fn for fn in os.listdir(image_dir) if fn.lower().endswith(extensions))

    all_data = []
    for file_name in file_names:
        image = np.array(Image.open(os.path.join(image_dir, file_name)).convert('RGB'))

        boxes_list, labels_list, scores_list, fold_list = [], [], [], []
        for fold_idx, model in enumerate(models):
            detections = model.predict(image, threshold=score_threshold)
            n = len(detections.xyxy)
            if n == 0:
                continue
            boxes_list.append(np.array(detections.xyxy))
            labels_list.append(np.array(detections.class_id))
            scores_list.append(np.array(detections.confidence))
            fold_list.extend([fold_idx] * n)

        if boxes_list:
            pred_boxes = torch.tensor(np.concatenate(boxes_list), dtype=torch.float32)
            pred_labels = torch.tensor(np.concatenate(labels_list), dtype=torch.int64)
            pred_scores = torch.tensor(np.concatenate(scores_list), dtype=torch.float32)
        else:
            pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
            pred_labels = torch.zeros((0,), dtype=torch.int64)
            pred_scores = torch.zeros((0,), dtype=torch.float32)

        all_data.append({
            'file_name': file_name,
            'image': image,
            'pred_boxes': pred_boxes,
            'pred_labels': pred_labels,
            'pred_scores': pred_scores,
            'pred_fold': torch.tensor(fold_list, dtype=torch.int64),
        })

    return all_data


def save_ensemble_gallery(pred_data, label_to_category_id, save_dir, file_prefix='test'):
    """
    collect_predictions_ensemble() 결과에 예측 박스(카테고리 id + confidence + 어느 fold가
    잡았는지)를 그려서 save_dir에 저장합니다. GT가 없어 오답 판정은 하지 않고 예측을 전부
    그대로 그립니다 - 학습 클래스 대조표(PNG)와 육안으로 비교하기 위한 용도입니다.

    Args:
        pred_data: collect_predictions_ensemble()의 반환값
        label_to_category_id (dict): 모델 라벨 -> 원본 category_id 매핑
        save_dir (str): 저장 폴더
        file_prefix (str): 저장 파일명 접두어

    Returns:
        int: 저장된 이미지 수
    """
    os.makedirs(save_dir, exist_ok=True)

    for idx, data in enumerate(pred_data):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(data['image'])

        for box, label, score, fold_idx in zip(data['pred_boxes'], data['pred_labels'],
                                                 data['pred_scores'], data['pred_fold']):
            _draw_box(ax, box, label.item(), label_to_category_id,
                      color='red', prefix=f'fold{fold_idx.item()}', score=score.item())

        ax.set_title(data['file_name'], fontsize=9)
        ax.axis('off')

        stem = os.path.splitext(data['file_name'])[0]
        save_path = os.path.join(save_dir, f'{file_prefix}_{idx:04d}_{stem}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close(fig)

    print(f'Saved {len(pred_data)} images -> {save_dir}')
    return len(pred_data)


def _cluster_same_class_boxes(boxes, labels, scores, folds, iou_threshold):
    """
    같은 라벨끼리 IoU >= iou_threshold로 겹치는 박스들을 한 그룹(=같은 알약을 여러 fold가
    예측한 것으로 간주)으로 묶어, 그룹당 confidence가 가장 높은 박스 하나만 대표로 남깁니다.
    (NMS와 유사하지만, 몇 개 fold가 동의했는지(agree_count)를 같이 기록한다는 점이 다름)

    Returns:
        list of dicts: [{'box', 'label', 'score', 'fold_idx', 'agree_count'}, ...]
    """
    order = torch.argsort(scores, descending=True)
    used = torch.zeros(len(boxes), dtype=torch.bool)
    entries = []

    for i in order.tolist():
        if used[i]:
            continue
        same_label = labels == labels[i]
        ious = box_iou(boxes[i:i + 1], boxes)[0]
        group = same_label & (ious >= iou_threshold) & (~used)
        used |= group

        entries.append({
            'box': boxes[i], 'label': labels[i], 'score': scores[i],
            'fold_idx': folds[i].item(), 'agree_count': int(group.sum().item()),
        })
    return entries


def crop_predictions_by_class(pred_data, score_threshold=0.5, padding=10,
                               iou_threshold=0.5, dedup=True):
    """
    collect_predictions_ensemble() 결과에서 confidence >= score_threshold인 예측 박스를
    예측 라벨(클래스)별로 잘라(crop) 모읍니다. 이미지 전체가 아니라 박스 영역만 잘라서,
    "클래스 하나당 대표 이미지 1개"로 정리된 학습 클래스 대조표와 같은 단위로 비교할 수
    있게 합니다.

    dedup=True(기본값)면, 같은 이미지 안에서 같은 클래스로 겹치는(IoU >= iou_threshold)
    여러 fold의 예측을 하나로 합쳐 크롭 1장만 남깁니다 (여러 fold가 같은 알약에 동의한
    경우 중복 저장 방지). 몇 개 fold가 동의했는지는 'agree_count'로 남아서, dedup 이후에도
    "5개 fold가 다 동의한 확실한 예측"인지 신호를 잃지 않습니다.

    Args:
        pred_data: collect_predictions_ensemble()의 반환값
        score_threshold (float): 모을 예측의 최소 confidence
        padding (int): 박스 주변 여백(px) - 잘랐을 때 알약이 너무 꽉 차 보이지 않게
        iou_threshold (float): dedup 시 "같은 알약"으로 볼 IoU 기준
        dedup (bool): False면 기존처럼 fold별 예측을 전부 따로 남김 (agree_count는 항상 1)

    Returns:
        dict: {label: [{'crop': np.ndarray, 'file_name', 'score', 'fold_idx', 'agree_count'}, ...]}
    """
    by_label = defaultdict(list)
    for d in pred_data:
        h, w = d['image'].shape[:2]
        boxes, labels = d['pred_boxes'], d['pred_labels']
        scores, folds = d['pred_scores'], d['pred_fold']

        keep = scores >= score_threshold
        boxes, labels, scores, folds = boxes[keep], labels[keep], scores[keep], folds[keep]
        if len(boxes) == 0:
            continue

        if dedup:
            entries = _cluster_same_class_boxes(boxes, labels, scores, folds, iou_threshold)
        else:
            entries = [
                {'box': boxes[i], 'label': labels[i], 'score': scores[i],
                 'fold_idx': folds[i].item(), 'agree_count': 1}
                for i in range(len(boxes))
            ]

        for e in entries:
            x1, y1, x2, y2 = e['box'].tolist()
            x1 = max(0, int(x1) - padding)
            y1 = max(0, int(y1) - padding)
            x2 = min(w, int(x2) + padding)
            y2 = min(h, int(y2) + padding)
            by_label[e['label'].item()].append({
                'crop': d['image'][y1:y2, x1:x2],
                'file_name': d['file_name'],
                'score': e['score'].item(),
                'fold_idx': e['fold_idx'],
                'agree_count': e['agree_count'],
            })
    return by_label
