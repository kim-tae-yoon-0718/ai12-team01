# yejin/pipeline/wbf.py
"""WBF(Weighted Box Fusion) 융합 + 예측 라벨 정제 + 제출 CSV 생성.

합집합 그대로 제출하면 같은 알약이 모델 수만큼 중복되어 mAP에서 FP로 깎입니다.
저장소의 _cluster_same_class_boxes(대표 박스 1개 선택)와 달리, WBF는 겹치는 박스들의
좌표를 confidence 가중 평균해 더 정교한 박스를 만듭니다 (외부 패키지 ensemble-boxes 사용).
conf_type='avg'(기본값) 기준, 일부 모델만 잡은 박스는 score가 (동의 모델 수/전체 모델 수)
비율로 낮아져 "몇 개 모델이 동의했는지" 신호가 score에 자연스럽게 반영됩니다.
"""
import os
import re

import numpy as np
import pandas as pd
import torch
from ensemble_boxes import weighted_boxes_fusion


def filter_valid_labels(pred_data, valid_labels):
    """RF-DETR 내부 예약 라벨(배경 0 등, label_map에 없는 라벨)을 in-place로 제거합니다.

    저장소 report_fold_result()가 valid 평가 때 하는 것과 동일한 정제 과정입니다.

    Args:
        pred_data: collect_predictions_ensemble 계열의 반환값
        valid_labels (set): 유효 라벨 집합 (예: set(label2cat))
    """
    for d in pred_data:
        keep = torch.tensor([int(l) in valid_labels for l in d['pred_labels']], dtype=torch.bool)
        for k in ('pred_boxes', 'pred_labels', 'pred_scores', 'pred_fold'):
            d[k] = d[k][keep]
    print('라벨 정제 후 예측 박스 수:', sum(len(d['pred_boxes']) for d in pred_data))


def fuse_predictions_wbf(pred_data, n_models, iou_thr=0.55, skip_box_thr=0.05):
    """collect_predictions_ensemble 계열 결과(단일 그룹)를 이미지 단위 WBF로 융합합니다.

    Args:
        pred_data: 반환 dict에 pred_fold(모델 구분)가 있는 예측 리스트
        n_models (int): 앙상블 모델(fold) 수
        iou_thr (float): 같은 객체로 간주할 IoU
        skip_box_thr (float): 융합 전 무시할 최소 score

    Returns:
        list of dicts: [{'file_name', 'image', 'pred_boxes'(xyxy), 'pred_labels', 'pred_scores'}, ...]
    """
    fused = []
    for d in pred_data:
        h, w = d['image'].shape[:2]
        scale = np.array([w, h, w, h], dtype=np.float32)

        if len(d['pred_boxes']) == 0:   # 예측이 하나도 없는 이미지
            fused.append({'file_name': d['file_name'], 'image': d['image'],
                          'pred_boxes': torch.zeros((0, 4), dtype=torch.float32),
                          'pred_labels': torch.zeros((0,), dtype=torch.int64),
                          'pred_scores': torch.zeros((0,), dtype=torch.float32)})
            continue

        # WBF는 모델별 리스트 + 0~1 정규화 좌표를 기대합니다.
        boxes_list, scores_list, labels_list = [], [], []
        for fi in range(n_models):
            m = (d['pred_fold'] == fi).numpy()
            b = d['pred_boxes'].numpy()[m] / scale
            boxes_list.append(np.clip(b, 0.0, 1.0).tolist())
            scores_list.append(d['pred_scores'].numpy()[m].tolist())
            labels_list.append(d['pred_labels'].numpy()[m].tolist())

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=iou_thr, skip_box_thr=skip_box_thr)

        fused.append({
            'file_name': d['file_name'],
            'image': d['image'],
            'pred_boxes': torch.tensor(np.asarray(boxes) * scale, dtype=torch.float32),
            'pred_labels': torch.tensor(labels, dtype=torch.int64),
            'pred_scores': torch.tensor(scores, dtype=torch.float32),
        })
    return fused


def fuse_merged_wbf(merged, n_models, weights=None, iou_thr=0.55, skip_box_thr=0.05):
    """여러 모델 그룹을 전역 모델 인덱스로 합친 merged 구조를 이미지 단위 WBF로 융합합니다.

    앙상블 추론 노트북용: RF-DETR 5-fold + Large + YOLO 5-fold처럼 이종 그룹의 예측을
    한꺼번에 융합할 때 사용합니다. 모델 1개 = 1표(x weight)이므로 그룹별 모델 수가 다르면
    weights로 발언권을 조절하세요.

    Args:
        merged (dict): file_name -> {'image', 'by_model': {전역 모델 idx: (boxes, labels, scores)}}
        n_models (int): 전역 모델 총수
        weights (list): 전역 모델 인덱스 순서의 WBF 가중치 (None이면 균등)

    Returns:
        list of dicts: fuse_predictions_wbf()와 동일 스키마
    """
    fused = []
    for fn in sorted(merged):
        m = merged[fn]
        h, w = m['image'].shape[:2]
        scale = np.array([w, h, w, h], dtype=np.float32)

        boxes_list, scores_list, labels_list = [], [], []
        for mi in range(n_models):
            b, l, s = m['by_model'].get(mi, (np.zeros((0, 4)), np.zeros(0), np.zeros(0)))
            boxes_list.append(np.clip(b / scale, 0.0, 1.0).tolist())
            scores_list.append(np.asarray(s, dtype=float).tolist())
            labels_list.append(np.asarray(l, dtype=int).tolist())

        if not any(len(b) for b in boxes_list):
            fused.append({'file_name': fn, 'image': m['image'],
                          'pred_boxes': torch.zeros((0, 4), dtype=torch.float32),
                          'pred_labels': torch.zeros((0,), dtype=torch.int64),
                          'pred_scores': torch.zeros((0,), dtype=torch.float32)})
            continue

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, weights=weights,
            iou_thr=iou_thr, skip_box_thr=skip_box_thr)

        fused.append({
            'file_name': fn, 'image': m['image'],
            'pred_boxes': torch.tensor(np.asarray(boxes) * scale, dtype=torch.float32),
            'pred_labels': torch.tensor(labels, dtype=torch.int64),
            'pred_scores': torch.tensor(scores, dtype=torch.float32),
        })
    return fused


def extract_image_id(file_name):
    """파일명에서 숫자를 추출해 image_id로 사용합니다.

    숫자 블록이 2개 이상이면 어떤 규칙인지 판단할 수 없으므로 일부러 에러를 내어
    확인을 요구합니다. (그 경우 test 파일명 규칙에 맞게 이 함수만 수정하면 됩니다)
    """
    stem = os.path.splitext(file_name)[0]
    digits = re.findall(r'\d+', stem)
    assert len(digits) == 1, f'파일명 숫자 규칙 확인 필요: {file_name} -> {digits}'
    return int(digits[0])


def make_submission(fused_data, label2cat, score_thr, out_path):
    """융합 예측을 제출 포맷 DataFrame으로 만들어 CSV 저장합니다.

    요구 포맷: annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
    - image_id: 이미지 "파일명의 숫자" / category_id: 원본 category_id (label2cat 역매핑)
    - annotation_id: 행마다 고유한 임의 값 (1부터 증가)
    - bbox: xyxy(내부 표현) -> xywh(COCO)로 변환
    """
    rows, ann_id, n_empty = [], 1, 0
    for d in fused_data:
        image_id = extract_image_id(d['file_name'])
        keep = d['pred_scores'] >= score_thr
        if int(keep.sum()) == 0:
            n_empty += 1
        for box, label, score in zip(d['pred_boxes'][keep], d['pred_labels'][keep],
                                     d['pred_scores'][keep]):
            x1, y1, x2, y2 = box.tolist()
            rows.append({
                'annotation_id': ann_id,
                'image_id': image_id,
                'category_id': label2cat[int(label)],
                'bbox_x': round(x1, 2), 'bbox_y': round(y1, 2),
                'bbox_w': round(x2 - x1, 2), 'bbox_h': round(y2 - y1, 2),
                'score': round(float(score), 4),
            })
            ann_id += 1
    df = pd.DataFrame(rows, columns=['annotation_id', 'image_id', 'category_id',
                                     'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score'])
    df.to_csv(out_path, index=False)
    print(f'저장 완료: {out_path}')
    print(f'총 {len(df)}행 / 이미지 {len(fused_data)}장 (예측 0건 이미지: {n_empty}장)')
    if n_empty:
        print('⚠ 예측이 하나도 없는 이미지가 있습니다. score_thr를 낮추거나 해당 이미지를 육안 확인하세요.')
    return df
