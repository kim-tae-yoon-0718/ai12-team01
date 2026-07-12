# yejin/pipeline/pools.py
"""합성 pool / masked pool 로드·검증·병합 + 74종 라벨 매핑.

두 pool의 annotation 체계가 서로 달라 로더를 분리했습니다.
- 합성 pool(task2_synthesized): categories가 "라벨 네임스페이스"(id 1~74, name=원본 category_id).
  name 필드로 원본 id 공간으로 되돌린 뒤 병합해야 저장소 파이프라인(build_coco 등)과 호환됩니다.
- masked pool(dataset-74-masked): categories의 id가 "원본 category_id 그대로" -> name 매핑 불필요.
  단, 파일명이 원본 train과 동일 체계(K코드+촬영조건)라 충돌할 수 있어 전체 파일을
  접두어('msk_')로 리네임한 사본을 스테이징 폴더에 만들어 병합합니다.
  (cache_images()가 파일명 기준으로 한 폴더에 복사하므로 리네임 없이는 이미지가 덮어써지고,
   corrections(원본 파일명이 키)가 masked 사본에 잘못 적용되는 문제도 함께 방지)
"""
import glob
import json
import os
import shutil
from collections import defaultdict

MASKED_PREFIX = 'msk_'


def load_pool_annotations(pool_ann_path):
    """합성 pool COCO json을 원본 category_id 공간으로 되돌려 로드합니다.

    annotation id도 함께 수집합니다 (시각화에서 ann_id 표시용. pool JSON 자체의 id라서
    train의 원본 annotation id와 번호가 겹칠 수 있으나, 표시 용도로만 쓰므로 무방).

    Args:
        pool_ann_path (str): 합성 pool의 _annotations.coco.json 경로

    Returns:
        (boxes, cats, ids, meta, coco):
            boxes/cats/ids (dict): file_name -> [bbox]/[원본 category_id]/[ann_id]
            meta (dict): file_name -> (width, height)
            coco (dict): 원본 COCO dict (categories가 74종 매핑의 신뢰 소스)
    """
    with open(pool_ann_path, encoding='utf-8') as f:
        coco = json.load(f)
    # 라벨 -> 원본 category_id (name 필드가 원본 id 문자열, id 0은 RF-DETR용 더미 'pill')
    label2cat_pool = {c['id']: int(c['name']) for c in coco['categories'] if c['id'] != 0}
    fn_by_img_id = {im['id']: im['file_name'] for im in coco['images']}
    p_boxes, p_cats, p_ids, p_meta = defaultdict(list), defaultdict(list), defaultdict(list), {}
    for im in coco['images']:
        p_meta[im['file_name']] = (im['width'], im['height'])
    for a in coco['annotations']:
        fn = fn_by_img_id[a['image_id']]
        p_boxes[fn].append([float(v) for v in a['bbox']])
        p_cats[fn].append(label2cat_pool[a['category_id']])
        p_ids[fn].append(a.get('id'))
    return p_boxes, p_cats, p_ids, p_meta, coco


def load_masked_annotations(ann_path):
    """masked pool COCO json을 로드합니다. category_id를 원본 id로 그대로 사용합니다.

    Returns:
        (boxes, cats, ids, meta, coco): load_pool_annotations()와 동일 구조
    """
    with open(ann_path, encoding='utf-8') as f:
        coco = json.load(f)
    fn_by_img_id = {im['id']: im['file_name'] for im in coco['images']}
    m_boxes, m_cats, m_ids, m_meta = defaultdict(list), defaultdict(list), defaultdict(list), {}
    for im in coco['images']:
        m_meta[im['file_name']] = (im['width'], im['height'])
    for a in coco['annotations']:
        fn = fn_by_img_id[a['image_id']]
        m_boxes[fn].append([float(v) for v in a['bbox']])
        m_cats[fn].append(int(a['category_id']))
        m_ids[fn].append(a.get('id'))
    return m_boxes, m_cats, m_ids, m_meta, coco


def merge_pool(boxes_by_image, cats_by_image, ids_by_image, img_meta,
               pool_boxes, pool_cats, pool_ids, pool_meta, pool_name='pool'):
    """로드된 pool annotation을 병합 대상 dict들에 in-place로 합칩니다.

    안전장치:
    1. 파일명 충돌 시 assert 중단 (syn_*.png vs K-*.png라 합성 pool은 충돌 없어야 정상)
    2. 박스 0개 이미지는 제외 - 학습에 기여하지 못하고 make_folds의 층화 기준
       (이미지 내 최희소 클래스) 계산에서 에러가 나기 때문

    Returns:
        int: 실제 병합된 이미지 수
    """
    overlap = set(pool_meta) & set(boxes_by_image)
    assert not overlap, f'{pool_name} 파일명 충돌: {sorted(overlap)[:5]}'

    empty = [fn for fn in pool_meta if not pool_boxes.get(fn)]
    if empty:
        print(f'박스 0개인 {pool_name} 이미지 {len(empty)}장 제외:', empty[:5])

    n = 0
    for fn in pool_meta:
        if pool_boxes.get(fn):
            boxes_by_image[fn] = pool_boxes[fn]
            cats_by_image[fn] = pool_cats[fn]
            ids_by_image[fn] = pool_ids[fn]   # 시각화(ann_id 표시)에서 사용
            img_meta[fn] = pool_meta[fn]
            n += 1
    print(f"{pool_name} {n}장 병합 | 병합 후: 이미지 {len(boxes_by_image)}장"
          f" / 박스 {sum(len(v) for v in boxes_by_image.values())}개"
          f" / 클래스 {len({c for cs in cats_by_image.values() for c in cs})}종")
    return n


def merge_masked_pool(boxes_by_image, cats_by_image, ids_by_image, img_meta,
                      masked_img_dir, masked_ann_path, stage_dir,
                      cats_allowed, prefix=MASKED_PREFIX):
    """masked pool을 접두어 리네임 후 병합합니다 (이미지 사본은 stage_dir에 생성).

    안전장치:
    1. masked pool의 category_id가 허용 집합(74종 매핑의 원본 id 공간)에 전부 포함되는지 assert
       -> 라벨 공간(1~74) json을 잘못 넣은 경우 등 포맷 착오를 학습 전에 잡습니다.
    2. annotation에는 있는데 실제 이미지 파일이 없는 항목 assert
    3. 박스 0개 이미지 제외 + 리네임 후에도 파일명이 충돌하면 assert

    Args:
        masked_img_dir (str): masked pool images/ 폴더
        masked_ann_path (str): masked pool _annotations.coco.json 경로
        stage_dir (str): 리네임 사본을 만들 스테이징 폴더 (cache_images에 그대로 전달 가능)
        cats_allowed (set): 허용 category_id 집합 (예: pool_coco categories의 name 집합)
        prefix (str): 리네임 접두어 (기본 'msk_' - folds.group_key와 짝을 맞출 것)

    Returns:
        (int, dict): 병합된 이미지 수, masked COCO dict
    """
    os.makedirs(stage_dir, exist_ok=True)
    m_boxes, m_cats, m_ids, m_meta, m_coco = load_masked_annotations(masked_ann_path)
    print(f"masked pool: 이미지 {len(m_meta)}장 / 박스 {sum(len(v) for v in m_boxes.values())}개"
          f" / 클래스 {len({c for cs in m_cats.values() for c in cs})}종")

    unknown = sorted({c for cs in m_cats.values() for c in cs} - set(cats_allowed))
    assert not unknown, f'허용 매핑에 없는 masked pool category_id: {unknown} (원본 id/라벨 공간 착오 확인)'

    img_src = {os.path.basename(p): p
               for p in glob.glob(os.path.join(masked_img_dir, '**', '*.png'), recursive=True)}
    missing = sorted(set(m_meta) - set(img_src))
    assert not missing, f'annotation에는 있는데 이미지가 없는 파일: {missing[:5]}'

    empty = [fn for fn in m_meta if not m_boxes.get(fn)]
    if empty:
        print(f'박스 0개인 masked 이미지 {len(empty)}장 제외:', empty[:5])

    n = 0
    for fn in m_meta:
        if not m_boxes.get(fn):
            continue
        new_fn = prefix + fn
        assert new_fn not in boxes_by_image, f'리네임 후에도 파일명 충돌: {new_fn}'
        dst = os.path.join(stage_dir, new_fn)
        if not os.path.exists(dst):
            shutil.copy(img_src[fn], dst)
        boxes_by_image[new_fn] = m_boxes[fn]
        cats_by_image[new_fn] = m_cats[fn]
        ids_by_image[new_fn] = m_ids[fn]
        img_meta[new_fn] = m_meta[fn]
        n += 1

    print(f'masked pool {n}장 리네임 병합 완료 (스테이징: {stage_dir})')
    print(f"병합 후: 이미지 {len(boxes_by_image)}장 / 박스 {sum(len(v) for v in boxes_by_image.values())}개"
          f" / 클래스 {len({c for cs in cats_by_image.values() for c in cs})}종")
    return n, m_coco


def apply_exclusions(exclude_files, boxes_by_image, cats_by_image, ids_by_image, img_meta):
    """불량 이미지 파일명 목록을 병합 데이터에서 in-place로 제외합니다.

    ⚠ 반드시 라벨 매핑/fold 분할 "이전"에 실행해야 하며, 고정 fold 분할(json)을 쓰는
      노트북에서는 json export 시점과 동일한 목록이어야 분할-데이터 일치 검증을 통과합니다.
      masked pool 이미지는 접두어('msk_')가 붙은 리네임 후 파일명으로 적으세요.
    """
    for fn in exclude_files:
        if fn in boxes_by_image:
            for d in (boxes_by_image, cats_by_image, ids_by_image, img_meta):
                d.pop(fn, None)
            print('이미지 제외:', fn)
        else:
            print('⚠ 목록에 없는 파일명(오타 확인):', fn)
    print(f"제외 처리 후: 이미지 {len(boxes_by_image)}장"
          f" / 박스 {sum(len(v) for v in boxes_by_image.values())}개"
          f" / 클래스 {len({c for cs in cats_by_image.values() for c in cs})}종")


def build_cat2label_74(pool_coco, train_cats, cats_by_image=None):
    """합성 pool JSON의 categories를 신뢰 소스로 74종 라벨 매핑을 만듭니다.

    요구 체계: train 56종 -> 라벨 1~56, test 전용 18종 -> 라벨 57~74.
    ⚠ 저장소 build_category_mapping()을 74종에 그대로 쓰면 "등장 클래스 전체 오름차순"이라
      test 18종이 1~56 사이에 끼어들어 요구 체계가 깨지므로 pool JSON 기반 매핑을 사용합니다.

    검증:
    1. train 클래스들이 정확히 1~len(train_cats)에 오름차순 매핑되는가
       -> Task 0/1/2와 라벨 네임스페이스 호환 보장
    2. (cats_by_image 제공 시) 병합 데이터의 모든 클래스가 매핑에 존재하는가

    Args:
        pool_coco (dict): load_pool_annotations()가 반환한 COCO dict
        train_cats (list): 원본 train에 등장한 category_id 정렬 리스트 (56종)
        cats_by_image (dict): 병합 완료된 cats dict (검증 2용, 생략 가능)

    Returns:
        (cat2label, label2cat, all_cats): 원본 id<->라벨(1~74) 매핑과
            build_coco()에 넘길 카테고리 목록(라벨 오름차순)
    """
    cat2label = {int(c['name']): c['id'] for c in pool_coco['categories'] if c['id'] != 0}
    label2cat = {v: k for k, v in cat2label.items()}
    all_cats = [label2cat[l] for l in sorted(label2cat)]

    for i, c in enumerate(sorted(train_cats), start=1):
        assert cat2label.get(c) == i, f'train 클래스 {c}가 라벨 {cat2label.get(c)}에 매핑됨 (기대: {i})'

    if cats_by_image is not None:
        merged_cats = {c for cs in cats_by_image.values() for c in cs}
        missing = sorted(merged_cats - set(cat2label))
        assert not missing, f'매핑에 없는 클래스 존재: {missing}'

    test_only = sorted(set(cat2label.values()) - set(range(1, len(train_cats) + 1)))
    print(f'전체 {len(cat2label)}종 매핑 | train {len(train_cats)}종 -> 1~{len(train_cats)} 확인'
          f' | test 전용 라벨: {test_only}')
    return cat2label, label2cat, all_cats
