# yejin/pipeline/folds.py
"""fold 분할: masked 접두어 인지 그룹화 / 고정 분할(json) 저장·로드 / 누수 점검 / 분포 요약.

배경 - 왜 저장소 make_folds()를 그대로 쓰지 않는가:
1. 저장소 make_folds()는 그룹 키를 "파일명의 '_0_2' 앞부분"(K코드 구성)으로 계산하는데,
   masked pool은 'msk_' 접두어 때문에 같은 구성의 원본 train 이미지와 "다른 그룹"으로
   계산되어 같은 구성(위도/촬영조건만 다른 이미지)이 train/valid에 갈라지는 누수가 생깁니다.
   -> make_folds_masked()는 그룹 키에서 접두어만 벗겨 계산합니다 (나머지 로직 동일).
2. StratifiedGroupKFold가 "동일 입력 + 동일 seed"에서 같은 분할을 내놓는 것은
   sklearn/numpy 버전이 같을 때만 보장됩니다. 여러 계정/세션/플랫폼(Colab-Kaggle)에서
   fold-matched 앙상블을 하려면 분할 결과 자체를 json으로 고정해 공유해야 합니다.
   -> export_fold_split() / load_fold_split()
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

MASKED_PREFIX = 'msk_'


def group_key(fn, prefix=MASKED_PREFIX):
    """파일명 -> StratifiedGroupKFold 그룹 키 (구성 코드).

    masked 접두어를 벗긴 뒤 '_0_2' 앞부분(K코드 구성)을 취합니다.
    합성 이미지(syn_*.png)는 '_0_2' 패턴이 없어 이미지 1장 = 그룹 1개로 취급됩니다.
    """
    if fn.startswith(prefix):
        fn = fn[len(prefix):]
    return fn.split('_0_2')[0]


def make_folds_masked(file_names, boxes_by_image, cats_by_image, cat2label,
                      n_splits, seed, prefix=MASKED_PREFIX):
    """저장소 make_folds와 동일하되, 그룹 키에서 masked 접두어를 제거해 원본과 같은 구성 그룹으로 묶습니다.

    - group: 구성 코드('_0_2' 앞부분, masked는 접두어 제거 후)
      -> 같은 구성의 위도/촬영조건 변형 + masked 버전이 전부 같은 fold 쪽에 배치됨
    - 층화 기준: 이미지 내 "가장 희소한 클래스" -> 희소 클래스가 fold에 고르게 분산

    Returns:
        list: [(train_idx, val_idx), ...] (file_names 기준 인덱스, 길이 n_splits)
    """
    from sklearn.model_selection import StratifiedGroupKFold

    cls_freq = defaultdict(int)
    for cs in cats_by_image.values():
        for c in cs:
            cls_freq[c] += 1
    groups = np.array([group_key(fn, prefix) for fn in file_names])
    strat = np.array([cat2label[min(cats_by_image[fn], key=lambda c: cls_freq[c])]
                      for fn in file_names])
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(file_names, strat, groups))


def export_fold_split(folds, file_names, path):
    """fold 분할 결과를 json으로 저장합니다 (계정/세션/플랫폼 간 분할 고정 공유용).

    저장 형식: {"fold0": {"train": [...파일명], "valid": [...]}, ...}
    """
    payload = {
        f'fold{fi}': {
            'train': [file_names[i] for i in tr],
            'valid': [file_names[i] for i in va],
        }
        for fi, (tr, va) in enumerate(folds)
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    print('fold 분할 저장:', path)
    return path


def load_fold_split(path, file_names, n_splits):
    """export_fold_split()이 저장한 고정 분할을 로드해 (train_idx, val_idx) 리스트로 복원합니다.

    안전장치: 저장된 분할이 이번 세션의 병합 데이터와 완전히 일치하는지 양방향 검증합니다.
    (pool/masked 파일 목록·제외 목록이 export 시점과 다르면 여기서 바로 에러로 잡혀,
     조용히 다른 데이터로 학습되는 사고를 막습니다)

    Returns:
        list: [(train_idx, val_idx), ...] (np.ndarray 인덱스, make_folds와 동일 형태)
    """
    with open(path, encoding='utf-8') as f:
        fixed = json.load(f)
    name_to_idx = {fn: i for i, fn in enumerate(file_names)}
    folds = []
    for fi in range(n_splits):
        tr_names = fixed[f'fold{fi}']['train']
        va_names = fixed[f'fold{fi}']['valid']
        only_in_split = (set(tr_names) | set(va_names)) - set(file_names)
        only_in_data = set(file_names) - (set(tr_names) | set(va_names))
        assert not only_in_split and not only_in_data, (
            f'fold{fi} 분할-데이터 불일치\n'
            f'  분할에만 있는 파일 예: {sorted(only_in_split)[:5]}\n'
            f'  데이터에만 있는 파일 예: {sorted(only_in_data)[:5]}')
        folds.append((np.array([name_to_idx[fn] for fn in tr_names]),
                      np.array([name_to_idx[fn] for fn in va_names])))
    print('고정 fold 분할 로드 완료 (재계산 없음):', path)
    return folds


def assert_no_group_leak(folds, file_names, prefix=MASKED_PREFIX):
    """같은 그룹(접두어 제거 기준 구성 코드)이 train/valid 양쪽에 있으면 assert로 중단합니다."""
    for fi, (tr, va) in enumerate(folds):
        leak = ({group_key(file_names[i], prefix) for i in tr}
                & {group_key(file_names[i], prefix) for i in va})
        assert not leak, f'fold {fi} 그룹 누수: {sorted(leak)[:5]}'
    print('그룹 누수 없음 (masked/원본 동일 구성은 항상 같은 fold 쪽)')


def summarize_fold_distribution(folds, file_names, cats_by_image, cat2label):
    """fold별 train/valid 이미지·박스 수와 클래스 커버리지를 점검합니다.

    train에서 통째로 빠진 클래스가 있는 fold는 그 클래스를 전혀 학습하지 못하므로
    반환된 summary의 train_missing_labels를 반드시 확인하세요.

    Returns:
        summary (pd.DataFrame): fold별 요약 (누락 라벨 목록 포함)
        valid_pivot (pd.DataFrame): 라벨 x fold 형태의 valid 박스 수 표
            (YOLO 클래스별 mAP 집계에서 '평가 대상 없음' fold를 걸러낼 때 재사용)
    """
    all_labels = set(cat2label.values())
    rows, val_pivot = [], {}
    for fi, (tr, va) in enumerate(folds):
        def label_box_counts(idxs):
            cnt = defaultdict(int)
            for i in idxs:
                for c in cats_by_image[file_names[i]]:
                    cnt[cat2label[c]] += 1
            return cnt
        tr_cnt, va_cnt = label_box_counts(tr), label_box_counts(va)
        rows.append({
            'fold': fi,
            'train_imgs': len(tr), 'valid_imgs': len(va),
            'train_boxes': sum(tr_cnt.values()), 'valid_boxes': sum(va_cnt.values()),
            'train_missing_labels': sorted(all_labels - set(tr_cnt)),
            'valid_missing_labels': sorted(all_labels - set(va_cnt)),
        })
        val_pivot[f'fold{fi}_valid'] = va_cnt
    summary = pd.DataFrame(rows)
    valid_pivot = pd.DataFrame(val_pivot).reindex(sorted(all_labels)).fillna(0).astype(int)
    valid_pivot.index.name = 'label'
    return summary, valid_pivot


def print_fold_warnings(fold_summary):
    """summarize_fold_distribution() 결과에서 누락 라벨 경고를 출력합니다."""
    for _, r in fold_summary.iterrows():
        if r['train_missing_labels']:
            print(f"⚠ fold {r['fold']}: train에 없는 라벨 {r['train_missing_labels']}")
        if r['valid_missing_labels']:
            print(f"(참고) fold {r['fold']}: valid에 없는 라벨 {r['valid_missing_labels']}")
