# yejin/pipeline/yolo.py
"""YOLO(v8/11) 5-fold 학습·COCO->YOLO 포맷 변환·앙상블 예측 수집.

저장소 train.py(train_fold/run_kfold)/model.py(get_rfdetr_model)/visualize.py
(collect_predictions_ensemble)의 패턴을 Ultralytics YOLO용으로 재정의한 것입니다.
(라이브러리가 달라 저장소 함수를 그대로 쓸 수 없어 별도 정의 - 저장소 무수정 원칙)

Ultralytics의 COCO 포맷 지원 확인 결과:
객체 검출 학습은 COCO json을 직접 쓰지 않고 images/+labels/(YOLO txt) 구조 + data.yaml을
요구합니다. 공식 변환 유틸 `ultralytics.data.converter.convert_coco()`가 COCO json -> YOLO
txt 변환을 대신해주므로 직접 파싱하지 않고 그대로 활용합니다 (build_yolo_fold 참고).

loss 가중치: Ultralytics 내장 하이퍼파라미터로 loss 성분별 가중치를 조절합니다 (코드 무수정).
  total = box_gain*box_loss + cls_gain*cls_loss + dfl_gain*dfl_loss   (v8DetectionLoss)
  기본값 box=7.5, cls=0.5, dfl=1.5 (box 위주). cls를 올리면 분류 오류에 더 큰 패널티가
  걸려 WBF 앙상블에서 classification 정확도에 기여하는 체크포인트를 만들 수 있습니다.
"""
import glob
import os
import shutil

import numpy as np
import torch
import yaml
from PIL import Image

from ultralytics import YOLO
from ultralytics.data.converter import convert_coco   # COCO json -> YOLO txt 공식 변환 유틸

# 저장소 model.py의 RF-DETR variant 매핑과 같은 패턴 (yolov8 계열)
YOLO_VARIANTS = {
    'nano': 'yolov8n.pt', 'small': 'yolov8s.pt', 'medium': 'yolov8m.pt',
    'large': 'yolov8l.pt', 'xlarge': 'yolov8x.pt',
}


def get_yolo_model(variant='medium', checkpoint_path=None, variants=None):
    """YOLO 모델을 생성합니다. checkpoint_path가 주어지면 그 가중치로 로드합니다.

    ultralytics 패키지(8.x)는 yolov8*/yolo11* 가중치를 모두 지원하므로,
    다른 세대를 쓰려면 variants 인자로 매핑을 교체하면 됩니다.
    """
    table = variants or YOLO_VARIANTS
    weights = checkpoint_path or table.get(variant.lower())
    if weights is None:
        raise ValueError(f"알 수 없는 YOLO variant: {variant} (지원: {list(table)})")
    return YOLO(weights)


def build_yolo_fold(fold_idx, coco_dataset_dir, yolo_dataset_dir, label2cat):
    """fold{fold_idx}의 COCO 포맷(train/valid)을 YOLO 포맷(images/labels)으로 변환합니다.

    - cls91to80=False: class_index = category_id - 1을 그대로 사용 (COCO 80/91클래스
      리매핑을 켜지 않음 - 우리 데이터는 원래 COCO 80종이 아니므로 반드시 꺼야 함)
    - 저장소 build_coco()가 넣는 더미 배경 카테고리(id=0, name='pill')에는 박스가 하나도
      없으므로, 별도 제거 없이 실제 N종이 정확히 YOLO class index 0~N-1로 매핑됩니다.
    - convert_coco()는 라벨 txt만 생성하고 이미지는 복사하지 않으므로, 이미지는 심볼릭
      링크로 연결해 디스크 중복을 피합니다.

    Args:
        fold_idx (int): fold 번호
        coco_dataset_dir (str): write_fold_dirs()의 output_dir (COCO 포맷 fold 루트)
        yolo_dataset_dir (str): YOLO 포맷 fold를 생성할 루트
        label2cat (dict): 라벨(1~N) -> 원본 category_id (data.yaml names 작성용)

    Returns:
        str: 이 fold의 data.yaml 경로
    """
    fold_root = os.path.join(yolo_dataset_dir, f'fold{fold_idx}')

    for split in ('train', 'valid'):
        coco_split_dir = os.path.join(coco_dataset_dir, f'fold{fold_idx}', split)
        img_dst = os.path.join(fold_root, split, 'images')
        lbl_dst = os.path.join(fold_root, split, 'labels')
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        # 1) 이미지: 복사 대신 심볼릭 링크 (COCO 포맷 fold 디렉토리와 디스크 중복 방지)
        for src in glob.glob(os.path.join(coco_split_dir, '*.png')):
            link = os.path.join(img_dst, os.path.basename(src))
            if not os.path.exists(link):
                os.symlink(os.path.abspath(src), link)

        # 2) 라벨: convert_coco()로 COCO json -> YOLO txt 변환 후 이동
        #    labels_dir은 *.json을 비재귀 탐색하므로, json 1개만 있는 split 폴더를 그대로 넘기면 됩니다.
        tmp_convert_dir = os.path.join(yolo_dataset_dir, '_convert_tmp', f'fold{fold_idx}_{split}')
        shutil.rmtree(tmp_convert_dir, ignore_errors=True)
        convert_coco(labels_dir=coco_split_dir, save_dir=tmp_convert_dir,
                     use_segments=False, use_keypoints=False, cls91to80=False)

        for txt_path in glob.glob(os.path.join(tmp_convert_dir, 'labels', '*', '*.txt')):
            shutil.move(txt_path, os.path.join(lbl_dst, os.path.basename(txt_path)))
        shutil.rmtree(tmp_convert_dir, ignore_errors=True)

        n_imgs = len(glob.glob(os.path.join(img_dst, '*.png')))
        n_lbls = len(glob.glob(os.path.join(lbl_dst, '*.txt')))
        print(f'fold{fold_idx}/{split}: 이미지 {n_imgs} / 라벨 {n_lbls}')

    # 3) data.yaml: names는 YOLO class index(0-based) -> 원본 category_id 문자열
    #    (class_index = category_id - 1 이므로, 라벨(1~N)과 index+1이 정확히 대응)
    names = {i: str(label2cat[i + 1]) for i in range(len(label2cat))}
    yaml_path = os.path.join(fold_root, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump({
            'path': os.path.abspath(fold_root),
            'train': 'train/images',
            'val': 'valid/images',
            'names': names,
        }, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def train_fold_yolo(fold_idx, fold_yaml, model_variant, model_tag, train_cfg,
                    local_output_dir, backup_dir):
    """fold 하나를 학습하고 best 체크포인트를 backup_dir에 복사합니다. 백업이 이미 있으면 건너뜁니다.

    - backup_dir에 {tag}_fold{i}_best.pt가 있으면 학습을 건너뜁니다 (이어하기).
    - loss 가중치는 train_cfg의 box_gain/cls_gain/dfl_gain으로 전달합니다 (모듈 docstring 참고).
    - Ultralytics는 학습 종료 시 results.csv/results.png(학습 곡선)를 자체 생성하므로 그대로 백업합니다.

    Args:
        train_cfg (dict): epochs/imgsz/batch/patience/seed/box_gain/cls_gain/dfl_gain

    Returns:
        str or None: 백업된 체크포인트 경로 (백업 실패 시 None)
    """
    exp = f'{model_tag}_fold{fold_idx}'
    dst = os.path.join(backup_dir, f'{exp}_best.pt')

    if os.path.exists(dst):
        print(f'[fold {fold_idx}] 백업 존재 → 건너뜀')
        return dst

    print(f"\n{'='*50}\n[fold {fold_idx}] 학습 시작\n{'='*50}")
    model = get_yolo_model(model_variant)
    model.train(
        data=fold_yaml,
        epochs=train_cfg['epochs'],
        imgsz=train_cfg['imgsz'],
        batch=train_cfg['batch'],
        patience=train_cfg['patience'],
        seed=train_cfg['seed'],
        box=train_cfg['box_gain'],
        cls=train_cfg['cls_gain'],   # 기본 0.5보다 크게 주면 분류 오류 패널티 강화 (WBF 앙상블용)
        dfl=train_cfg['dfl_gain'],
        project=local_output_dir,
        name=exp,
        exist_ok=True,
        plots=True,
    )

    os.makedirs(backup_dir, exist_ok=True)
    run_dir = os.path.join(local_output_dir, exp)
    src = os.path.join(run_dir, 'weights', 'best.pt')
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f'[fold {fold_idx}] best 백업 → {dst}')
    else:
        dst = None
        print(f'[fold {fold_idx}] best.pt 없음 — 백업 실패')

    for fn in ('results.csv', 'results.png'):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(backup_dir, f'{exp}_{fn}'))

    del model
    torch.cuda.empty_cache()
    return dst


def report_fold_result_yolo(fold_idx, checkpoint_path, fold_yaml):
    """fold 하나의 valid mAP를 계산합니다. 학습을 건너뛴 fold도 체크포인트를 다시 로드해 평가합니다.

    Returns:
        dict: {'map'(mAP@0.5:0.95), 'map_50', 'map_per_class'(class index 0~nc-1 정렬 배열), 'names'}
    """
    model = get_yolo_model(checkpoint_path=checkpoint_path)
    metrics = model.val(data=fold_yaml, split='val', plots=False, verbose=False)
    result = {
        'map': float(metrics.box.map),
        'map_50': float(metrics.box.map50),
        'map_per_class': np.asarray(metrics.box.maps, dtype=float),
        'names': metrics.names,
    }
    del model
    torch.cuda.empty_cache()
    return result


def run_folds_yolo(config, fold_yaml_paths, fold_indices=None, max_folds=None):
    """지정한 fold들을 학습+리포팅합니다 (저장소 run_kfold의 YOLO 버전).

    fold_indices를 주면 그 fold들만(플랫폼 분담용: 예 Colab이 [0], Kaggle이 [1,2,3,4]),
    없으면 0..max_folds-1(기본: n_splits) 순서로 돕니다. backup_dir에 best.pt가 있는
    fold는 자동으로 건너뜁니다.

    Returns:
        dict: {'checkpoints': {fold: 경로}, 'fold_metrics': {fold: report_fold_result_yolo 결과}}
    """
    print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    if fold_indices is None:
        n = max_folds if max_folds is not None else config['data']['n_splits']
        fold_indices = list(range(n))

    checkpoints, fold_metrics = {}, {}
    for fi in fold_indices:
        dst = train_fold_yolo(
            fold_idx=fi,
            fold_yaml=fold_yaml_paths[fi],
            model_variant=config['model']['variant'],
            model_tag=config['model']['tag'],
            train_cfg=config['train'],
            local_output_dir=config['output']['local_output_dir'],
            backup_dir=config['output']['backup_dir'],
        )
        checkpoints[fi] = dst
        if dst is None:
            print(f'[fold {fi}] 체크포인트 없음 — 리포팅 생략')
            continue
        metrics = report_fold_result_yolo(fi, dst, fold_yaml_paths[fi])
        fold_metrics[fi] = metrics
        print(f"[fold {fi}] 완료 | mAP@0.5:0.95: {metrics['map']:.4f} | mAP@0.5: {metrics['map_50']:.4f}")

    print(f'\n▶ fold {list(fold_indices)} 완료')
    return {'checkpoints': checkpoints, 'fold_metrics': fold_metrics}


def summarize_kfold_results_yolo(fold_metrics, tag):
    """fold별 mAP를 받아 평균±표준편차를 출력합니다 (저장소 summarize_kfold_results의 YOLO 버전).

    Args:
        fold_metrics (dict or list): report_fold_result_yolo() 결과들
            ({fold: metrics} dict 또는 리스트 모두 허용)
        tag (str): 실험 태그 (출력 제목)
    """
    vals = list(fold_metrics.values()) if isinstance(fold_metrics, dict) else list(fold_metrics)
    map_vals = [m['map'] for m in vals]
    map50_vals = [m['map_50'] for m in vals]
    map_mean, map_std = float(np.mean(map_vals)), float(np.std(map_vals))
    map50_mean, map50_std = float(np.mean(map50_vals)), float(np.std(map50_vals))
    print(f"\n{'='*50}\n{tag} 최종 결과 ({len(vals)}-fold 평균)\n"
          f"mAP@0.5:0.95: {map_mean:.4f} ± {map_std:.4f}\n"
          f"mAP@0.5: {map50_mean:.4f} ± {map50_std:.4f}\n{'='*50}")
    return {'map': (map_mean, map_std), 'map_50': (map50_mean, map50_std)}


def summarize_per_class_yolo(fold_metrics, label2cat, label_counts, valid_pivot):
    """클래스(라벨)별 mAP@0.5:0.95를 fold 평균으로 집계합니다 (summarize_per_class의 YOLO 버전).

    valid_pivot(라벨 x fold의 valid 박스 수)을 이용해, 그 fold의 valid에 해당 라벨 인스턴스가
    하나도 없었던 경우는 집계에서 제외합니다 (그 fold의 AP가 0으로 나와도 실제 성능이 아니라
    '평가 대상 없음'이기 때문).

    Args:
        fold_metrics (dict or list): report_fold_result_yolo() 결과들
        label2cat (dict): 라벨 -> 원본 category_id
        label_counts (dict): 라벨별 전체 박스 수 (저장소 compute_label_counts 결과)
        valid_pivot (pd.DataFrame): folds.summarize_fold_distribution()의 두 번째 반환값

    Returns:
        pd.DataFrame: mean_AP 내림차순 클래스별 집계표
    """
    import pandas as pd

    items = (sorted(fold_metrics.items()) if isinstance(fold_metrics, dict)
             else list(enumerate(fold_metrics)))
    rows = []
    for label in sorted(label2cat):
        aps = []
        for fi, m in items:
            if valid_pivot.loc[label, f'fold{fi}_valid'] == 0:
                continue
            aps.append(m['map_per_class'][label - 1])   # class index = label - 1
        rows.append({
            'label': label,
            'category_id': label2cat[label],
            'total_count': label_counts.get(label, 0),
            'mean_AP': round(float(np.mean(aps)), 4) if aps else -1,
            'std_AP': round(float(np.std(aps)), 4) if aps else 0,
            'valid_folds': len(aps),
        })
    return pd.DataFrame(rows).sort_values('mean_AP', ascending=False)


def collect_predictions_ensemble_yolo(checkpoints, image_dir, conf_thr=0.05,
                                      extensions=('.png', '.jpg', '.jpeg')):
    """YOLO 체크포인트 리스트로 test 폴더를 추론해 예측을 이미지별로 병합합니다 (합집합).

    저장소 collect_predictions_ensemble()과 동일한 반환 스키마입니다
    (rfdetr의 supervision.Detections 대신 ultralytics Results.boxes를 읽는 부분만 다름).
    ⚠ YOLO의 cls는 0-indexed(0~N-1)라서, cat2label/label2cat 라벨 체계(1~N)에 맞추기 위해
      +1 해서 저장합니다.

    Returns:
        list of dicts: [{'file_name', 'image', 'pred_boxes'(xyxy), 'pred_labels',
                         'pred_scores', 'pred_fold'}, ...]
    """
    models = [YOLO(p) for p in checkpoints]
    file_names = sorted(fn for fn in os.listdir(image_dir) if fn.lower().endswith(extensions))

    all_data = []
    for file_name in file_names:
        img_path = os.path.join(image_dir, file_name)
        image = np.array(Image.open(img_path).convert('RGB'))

        boxes_list, labels_list, scores_list, fold_list = [], [], [], []
        for fold_idx, model in enumerate(models):
            r = model.predict(img_path, conf=conf_thr, verbose=False)[0]
            n = len(r.boxes)
            if n == 0:
                continue
            boxes_list.append(r.boxes.xyxy.cpu().numpy())
            labels_list.append(r.boxes.cls.cpu().numpy().astype(int) + 1)   # 0-idx -> 라벨(1~N)
            scores_list.append(r.boxes.conf.cpu().numpy())
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
            'file_name': file_name, 'image': image,
            'pred_boxes': pred_boxes, 'pred_labels': pred_labels, 'pred_scores': pred_scores,
            'pred_fold': torch.tensor(fold_list, dtype=torch.int64),
        })

    del models
    torch.cuda.empty_cache()
    return all_data
