# yejin/pipeline/viz.py
"""GT/예측 클래스별 crop 갤러리.

저장소에는 "예측" crop 유틸(crop_predictions_by_class)만 있고 GT bbox crop용 함수가 없어
로컬로 정의해 쓰던 것을 모듈화했습니다. 예측 쪽은 WBF 융합 이후에는 fold 개념이 사라져
저장소 함수(pred_fold 요구)를 쓸 수 없으므로 융합 결과 전용 뷰어를 함께 제공합니다.

⚠ Colab/Kaggle 기본 matplotlib에는 한글 폰트가 없으므로, plot에 렌더링되는 텍스트는
  영어로만 표기합니다.
"""
import glob
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def build_image_path_index(*img_dirs, ext='*.png'):
    """여러 이미지 폴더를 재귀 검색해 file_name -> 실제 경로 인덱스를 만듭니다.

    train/pool/masked-스테이징처럼 이미지가 여러 폴더에 흩어져 있을 때 시각화 함수에
    넘길 단일 인덱스를 구성하는 용도입니다. 같은 파일명이 여러 폴더에 있으면 나중
    폴더가 우선합니다 (호출 순서로 제어).
    """
    index = {}
    for d in img_dirs:
        index.update({os.path.basename(p): p
                      for p in glob.glob(os.path.join(d, '**', ext), recursive=True)})
    print('이미지 경로 인덱스:', len(index), '개 파일')
    return index


def show_gt_class_crops(boxes_by_image, cats_by_image, ids_by_image, img_path_index,
                        ncols=6, pad=8, classes=None):
    """클래스(원본 category_id)별 GT bbox crop을 '전부' grid로 표시합니다.

    합성/masked 이미지의 bbox도 함께 표시되므로 원본과 섞였을 때의 품질(크기/배경/겹침)을
    함께 점검하세요. 각 crop 제목: 1줄째 = 파일명(전체), 2줄째 = annotation_id
    - train 박스: 원본 annotation JSON의 id (corrections의 add_boxes로 추가된 박스는 None)
    - pool 박스: pool JSON의 id

    Args:
        boxes_by_image / cats_by_image: 병합 완료된 annotation dict (COCO xywh, 원본 id)
        ids_by_image (dict): file_name -> [annotation_id, ...] (boxes와 순서 동기화 상태)
        img_path_index (dict): file_name -> 이미지 실제 경로 (build_image_path_index 참고)
        ncols (int): 한 줄에 표시할 crop 수 (행 수는 박스 수에 따라 자동 결정)
        pad (int): crop 시 bbox 주변 여백(px)
        classes (list): 지정하면 해당 category_id들만 표시 (None이면 전체)
    """
    by_cat = defaultdict(list)   # category_id -> [(file_name, bbox, ann_id), ...]
    for fn, cats in cats_by_image.items():
        ids = ids_by_image.get(fn)
        if not ids or len(ids) != len(boxes_by_image[fn]):   # 동기화가 깨졌으면 표시만 생략 (방어적)
            ids = [None] * len(boxes_by_image[fn])
        for c, b, aid in zip(cats, boxes_by_image[fn], ids):
            by_cat[c].append((fn, b, aid))

    targets = sorted(by_cat) if classes is None else [c for c in classes if c in by_cat]
    for c in targets:
        items = by_cat[c]
        # 같은 이미지를 bbox마다 다시 읽지 않도록, 클래스 단위로 파일당 1회만 로드
        img_cache = {fn: np.array(Image.open(img_path_index[fn]).convert('RGB'))
                     for fn in {fn for fn, _, _ in items}}

        nrows = math.ceil(len(items) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.8 * nrows))
        axes = np.atleast_1d(axes).reshape(-1)
        for ax in axes:
            ax.axis('off')
        for ax, (fn, b, aid) in zip(axes, items):
            img = img_cache[fn]
            h, w = img.shape[:2]
            x, y, bw, bh = [int(v) for v in b]
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
            ax.imshow(img[y1:y2, x1:x2])
            ax.set_title(f'{fn}\nann_id={aid}', fontsize=5)
        fig.suptitle(f'category_id={c}  (total {len(items)} boxes)', fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.97])   # suptitle과 첫 행이 겹치지 않게 상단 여백 확보
        plt.show()


def show_pred_class_crops(fused_data, label2cat, score_thr=0.3, max_per_class=None,
                          ncols=6, pad=8, classes=None):
    """융합(또는 단일 모델) 예측을 클래스별 crop grid로 표시합니다 (score 내림차순).

    각 crop 제목: 1줄째 = 파일명, 2줄째 = confidence score.
    제출 기준과 동일한 score_thr로 점검하는 것을 권장합니다.

    Args:
        fused_data: fuse_predictions_wbf()/fuse_merged_wbf()의 반환값
            (또는 같은 스키마의 단일 모델 예측)
        label2cat (dict): 모델 라벨 -> 원본 category_id
        score_thr (float): 표시할 예측의 최소 confidence
        max_per_class (int): 클래스당 표시할 crop 수 상한 (None이면 전부 표시)
        ncols (int): 한 줄에 표시할 crop 수
        pad (int): crop 여백(px)
        classes (list): 지정하면 해당 '라벨' 번호들만 표시 (None이면 전체)
    """
    by_label = defaultdict(list)
    for d in fused_data:
        h, w = d['image'].shape[:2]
        keep = d['pred_scores'] >= score_thr
        for box, label, score in zip(d['pred_boxes'][keep], d['pred_labels'][keep],
                                     d['pred_scores'][keep]):
            x1, y1, x2, y2 = box.tolist()
            x1, y1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
            x2, y2 = min(w, int(x2) + pad), min(h, int(y2) + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            by_label[int(label)].append((d['image'][y1:y2, x1:x2], float(score), d['file_name']))

    print(f'score >= {score_thr} 기준, 예측이 존재하는 클래스: {len(by_label)}개')
    targets = sorted(by_label) if classes is None else [l for l in classes if l in by_label]
    for label in targets:
        items = sorted(by_label[label], key=lambda t: -t[1])
        total = len(items)
        if max_per_class is not None:
            items = items[:max_per_class]

        nrows = math.ceil(len(items) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.8 * nrows))
        axes = np.atleast_1d(axes).reshape(-1)
        for ax in axes:
            ax.axis('off')
        for ax, (crop, score, fn) in zip(axes, items):
            ax.imshow(crop)
            ax.set_title(f'{fn}\nscore={score:.2f}', fontsize=5)
        fig.suptitle(f'label={label} / category_id={label2cat[label]}'
                     f'  (score>={score_thr}: {total} preds)', fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
