# rf-detr/synthesis.py
"""
누끼(pill_extraction.py 산출물) + 배경 이미지로 합성 학습 데이터를 생성합니다.

task1_synthesis.ipynb(train 희소 클래스 균형화)와 task2_synthesis.ipynb(test 전용 18종
추가 균형화)가 공유하는 로직입니다. 클래스별 목표 장수(needed)를 받아 배경에 알약을
겹치지 않게 배치·회전·리사이즈·그림자 합성한 뒤 COCO annotation을 생성합니다.

파이프라인: build_pill_pool(누끼 인덱싱) -> create_cycling_pool(순환 샘플링 준비)
-> run_synthesis(반복 호출: fill_slots로 이번 이미지에 넣을 알약 결정 -> synthesize_one으로
실제 합성) -> _annotations.coco.json 저장.
"""
import json
import os
import random

import cv2
import numpy as np

# ── 배치 클러스터 (레이아웃 슬롯 키 -> (중심x, 중심y, 표준편차x, 표준편차y)) ──
CLUSTER = {
    "top_left"   : (230, 330, 30, 40), "top_center" : (465, 330, 30, 40),
    "top_right"  : (700, 330, 35, 40), "bot_left"   : (245, 890, 35, 60),
    "bot_center" : (465, 1000, 30, 40),"bot_right"  : (715, 840, 40, 50),
}
LAYOUT_3 = [["top_left", "top_right", "bot_center"], ["top_center", "bot_left", "bot_right"]]
LAYOUT_4 = [["top_left", "top_right", "bot_left", "bot_right"]]

# ── 크기/변형 파라미터 ──
MAX_PILL_W = int(976 * 0.40)   # 390px
MAX_PILL_H = int(1280 * 0.40)  # 512px
RESIZE_JITTER = 0.10
ROTATE_RANGE = 15
SCALE_FACTOR = 0.85            # 배치 재시도 2단계에서 축소 비율
MAX_POS_RETRIES = 5
OVERLAP_MARGIN = 10

# ── 그림자 파라미터 ──
SHADOW_OFFSET = (12, 12)
SHADOW_BLUR = 18
SHADOW_OPACITY = 0.45


def weighted_sample_class(needed):
    """남은 목표 장수(needed: {category: 남은 수})에 비례해 클래스 1개를 뽑습니다."""
    cats = list(needed.keys())
    ws = [needed[c] for c in cats]
    return str(np.random.choice(cats, p=[w / sum(ws) for w in ws]))


def sample_position(key):
    """CLUSTER의 정규분포 파라미터로 슬롯 키(key)의 배치 좌표를 샘플링합니다."""
    cx, cy, sx, sy = CLUSTER[key]
    x = int(np.clip(np.random.normal(cx, sx), cx - 2 * sx, cx + 2 * sx))
    y = int(np.clip(np.random.normal(cy, sy), cy - 2 * sy, cy + 2 * sy))
    return x, y


def sample_cycling(cyc, cat):
    """순환 풀(cyc)에서 클래스(cat)의 다음 누끼 경로를 하나 꺼냅니다 (소진 시 재셔플)."""
    if isinstance(cyc, dict) and 'paths' in cyc:
        e = cyc  # 이미 entry가 넘어온 경우
    else:
        e = cyc[cat]  # 전체 pool이 넘어온 경우
    p = e['paths'][e['idx']]
    e['idx'] += 1
    if e['idx'] >= len(e['paths']):
        e['idx'] = 0
        random.shuffle(e['paths'])
    return p


def get_pill_actual_size(rgba):
    """RGBA 누끼의 alpha>10 픽셀 기준 실제 너비/높이를 계산합니다 (여백 제외)."""
    ys, xs = np.where(rgba[:, :, 3] > 10)
    if len(xs) == 0:
        return rgba.shape[1], rgba.shape[0]
    return int(xs.max() - xs.min()), int(ys.max() - ys.min())


def clip_pill_size(rgba, max_w=MAX_PILL_W, max_h=MAX_PILL_H):
    """알약 실제 크기가 상한(max_w/max_h)을 넘으면 비율을 유지하며 축소합니다."""
    curr_w, curr_h = get_pill_actual_size(rgba)
    if curr_w <= max_w and curr_h <= max_h:
        return rgba
    scale = min(max_w / curr_w, max_h / curr_h)
    nw = max(1, int(rgba.shape[1] * scale))
    nh = max(1, int(rgba.shape[0] * scale))
    return cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_LINEAR)


def resize_n_pill(rgba, tw, th, jitter=RESIZE_JITTER):
    """target(tw, th) 크기에 맞춰 리사이즈하되 jitter만큼 랜덤 변동을 줍니다 (N종 크기 정규화용)."""
    cw, ch = get_pill_actual_size(rgba)
    if cw == 0 or ch == 0:
        return rgba
    scale = min(tw / cw, th / ch) * np.random.uniform(1 - jitter, 1 + jitter)
    nw = max(1, int(rgba.shape[1] * scale))
    nh = max(1, int(rgba.shape[0] * scale))
    return cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_LINEAR)


def rotate_pill(rgba, angle):
    """RGBA 누끼를 angle(도)만큼 회전하고, 잘리지 않도록 캔버스를 확장합니다."""
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    return cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def any_overlap(new_bbox, placed):
    """new_bbox가 이미 배치된 박스(placed) 중 하나와라도 겹치는지 확인합니다."""
    return any(boxes_overlap(new_bbox, b) for b in placed)


def boxes_overlap(b1, b2, margin=OVERLAP_MARGIN):
    """두 xywh 박스가 margin을 감안해 겹치는지 판정합니다."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    return not (x1 + w1 + margin < x2 or x2 + w2 + margin < x1
                or y1 + h1 + margin < y2 or y2 + h2 + margin < y1)


def make_shadow(rgba, bg_shape, px, py):
    """알약 alpha 마스크를 오프셋+블러해 배경에 곱할 그림자 맵(0~SHADOW_OPACITY)을 만듭니다."""
    H, W = bg_shape[:2]
    ph, pw = rgba.shape[:2]
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    dx, dy = SHADOW_OFFSET
    smap = np.zeros((H, W), np.float32)
    sx, sy = px + dx, py + dy
    sx1, sy1 = max(0, sx), max(0, sy)
    sx2, sy2 = min(W, sx + pw), min(H, sy + ph)
    ax1, ay1 = sx1 - sx, sy1 - sy
    ax2, ay2 = ax1 + (sx2 - sx1), ay1 + (sy2 - sy1)
    if sx2 > sx1 and sy2 > sy1:
        smap[sy1:sy2, sx1:sx2] = alpha[ay1:ay2, ax1:ax2]
    k = SHADOW_BLUR * 2 + 1
    smap = cv2.GaussianBlur(smap, (k, k), SHADOW_BLUR / 3)
    return smap * SHADOW_OPACITY


def paste_pill(bg_float, rgba, cx, cy, bbox_margin=10):
    """알약 RGBA를 배경(bg_float, float32)의 (cx, cy) 중심에 알파 블렌딩으로 붙입니다.

    Returns:
        (result, bbox): 합성된 배경과 COCO xywh bbox (alpha>10 실측 영역 + margin, 캔버스 밖이면 None)
    """
    H, W = bg_float.shape[:2]
    ph, pw = rgba.shape[:2]
    px, py = cx - pw // 2, cy - ph // 2
    px1, py1 = max(0, px), max(0, py)
    px2, py2 = min(W, px + pw), min(H, py + ph)
    ax1, ay1 = px1 - px, py1 - py
    ax2, ay2 = ax1 + (px2 - px1), ay1 + (py2 - py1)
    if px2 <= px1 or py2 <= py1:
        return bg_float, None
    crop = rgba[ay1:ay2, ax1:ax2]
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR).astype(np.float32)
    result = bg_float.copy()
    result[py1:py2, px1:px2] = rgb * alpha + result[py1:py2, px1:px2] * (1 - alpha)
    ys, xs = np.where(rgba[:, :, 3] > 10)
    if len(xs) > 0:
        bbox = [int(xs.min()) + px - bbox_margin,
                int(ys.min()) + py - bbox_margin,
                int(xs.max()) - int(xs.min()) + bbox_margin * 2,
                int(ys.max()) - int(ys.min()) + bbox_margin * 2]
    else:
        bbox = [px, py, pw, ph]
    return result, bbox


def build_pill_pool(pills_dir):
    """pills_dir/class_<category>/*.png 구조를 읽어 {category: [누끼 경로, ...]} 풀을 만듭니다."""
    pool = {}
    for cls_dir in sorted(os.listdir(pills_dir)):
        if not cls_dir.startswith("class_"):
            continue
        cat = cls_dir[len("class_"):]
        paths = sorted([os.path.join(pills_dir, cls_dir, f)
                        for f in os.listdir(os.path.join(pills_dir, cls_dir))
                        if f.endswith(".png")])
        if paths:
            pool[cat] = paths
    print(f"pool: {len(pool)}개 클래스, 총 {sum(len(v) for v in pool.values())}개")
    return pool


def create_cycling_pool(pool):
    """build_pill_pool() 결과를 셔플된 순환 큐({category: {'paths', 'idx'}})로 변환합니다."""
    cyc = {}
    for cat, paths in pool.items():
        s = paths.copy()
        random.shuffle(s)
        cyc[cat] = {'paths': s, 'idx': 0}
    return cyc


def fill_slots(layout, combined_needed, all_pools, all_cyc, n_cats):
    """레이아웃(슬롯 키 리스트)의 각 슬롯에 넣을 (키, 누끼경로, 클래스, N종여부)를 결정합니다.

    남은 목표(combined_needed)가 있으면 가중 샘플링, 소진되면 아직 안 쓴 클래스 중 무작위.
    """
    used = set()
    slots = []
    for key in layout:
        avail = {k: v for k, v in combined_needed.items() if k not in used}
        if avail:
            cat = weighted_sample_class(avail)
        else:
            unused = [k for k in all_pools if k not in used]
            if not unused:
                break
            cat = random.choice(unused)
        used.add(cat)
        is_n = cat in n_cats
        path = sample_cycling(all_cyc[cat], cat) if cat in all_cyc else random.choice(all_pools[cat])
        slots.append((key, path, cat, is_n))
    return slots


def analyze_x_pill_sizes(pills_dir):
    """X종(원본 56종) 누끼들의 실제 크기 통계(mean/median)를 계산합니다 (N종 크기 정규화 기준값)."""
    widths, heights = [], []
    for cls_dir in os.listdir(pills_dir):
        if not cls_dir.startswith("class_"):
            continue
        for fn in os.listdir(os.path.join(pills_dir, cls_dir)):
            if not fn.endswith(".png"):
                continue
            rgba = cv2.imread(os.path.join(pills_dir, cls_dir, fn), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.shape[2] != 4:
                continue
            w, h = get_pill_actual_size(rgba)
            widths.append(w)
            heights.append(h)
    stats = {'median_w': int(np.median(widths)), 'median_h': int(np.median(heights)),
             'mean_w': int(np.mean(widths)), 'mean_h': int(np.mean(heights))}
    print(f"X종 크기 분석 (n={len(widths)})")
    print(f"  너비: mean={stats['mean_w']}  median={stats['median_w']}")
    print(f"  높이: mean={stats['mean_h']}  median={stats['median_h']}")
    return stats


def synthesize_one(bg_path, slots, cat2label, x_size_stats=None):
    """배경 1장에 slots(fill_slots 결과)의 알약들을 순서대로 합성하고 COCO annotation을 만듭니다.

    N종(x_size_stats 제공 + is_n=True)은 X종 중앙값 크기로 먼저 정규화한 뒤 공통 처리합니다.
    배치 실패 시(2단계 축소 재시도까지 실패) 해당 슬롯은 건너뜁니다.
    """
    bg = cv2.imread(bg_path)
    canvas = bg.astype(np.float32)
    ann_list = []
    placed = []
    for key, pill_path, cat, is_n in slots:
        rgba = cv2.imread(pill_path, cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim < 3 or rgba.shape[2] != 4:
            continue
        if is_n and x_size_stats:
            rgba = resize_n_pill(rgba, x_size_stats['median_w'], x_size_stats['median_h'])
        rgba = clip_pill_size(rgba)  # 상한선 초과 시 축소
        rgba = rotate_pill(rgba, np.random.uniform(-ROTATE_RANGE, ROTATE_RANGE))
        done = False
        cur = rgba.copy()
        for phase in range(2):
            if phase == 1:
                h, w = cur.shape[:2]
                cur = cv2.resize(cur, (max(1, int(w * SCALE_FACTOR)), max(1, int(h * SCALE_FACTOR))),
                                 interpolation=cv2.INTER_LINEAR)
            for _ in range(MAX_POS_RETRIES):
                cx, cy = sample_position(key)
                ph, pw = cur.shape[:2]
                est = [cx - pw // 2, cy - ph // 2, pw, ph]
                if not any_overlap(est, placed):
                    px, py = cx - pw // 2, cy - ph // 2
                    shadow = make_shadow(cur, canvas.shape, px, py)
                    canvas = canvas * (1 - shadow[:, :, np.newaxis])
                    canvas, bbox = paste_pill(canvas, cur, cx, cy)
                    if bbox is not None:
                        placed.append(bbox)
                        ann_list.append({"category_id": cat2label[cat],
                                         "bbox": [float(v) for v in bbox],
                                         "area": float(bbox[2] * bbox[3]), "iscrowd": 0})
                    done = True
                    break
            if done:
                break
    return canvas.astype(np.uint8), ann_list


def run_synthesis(combined_needed, all_pools, all_cyc, bg_paths, n_cats,
                  cat2label, x_size_stats, out_img, out_ann, max_images, seed):
    """combined_needed(클래스별 남은 목표 장수)가 소진되거나 max_images에 닿을 때까지 합성을 반복합니다.

    이미지마다 레이아웃(3종/4종)을 무작위 선택 -> fill_slots로 배치안 결정 -> synthesize_one으로
    실제 합성. 완료 후 COCO json(_annotations.coco.json)을 out_ann에 저장합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_ann, exist_ok=True)
    imgs, anns = [], []
    img_id, ann_id = 1, 1
    for i in range(max_images):
        if not combined_needed:
            print(f"\n[완료] 모든 클래스 목표 달성 ({i}장)")
            break
        layout = random.choice(LAYOUT_3 * 2 + LAYOUT_4)
        slots = fill_slots(layout, combined_needed, all_pools, all_cyc, n_cats)
        if not slots:
            continue
        try:
            img, ann_list = synthesize_one(random.choice(bg_paths), slots, cat2label, x_size_stats)
        except Exception as e:
            print(f"\n[ERROR]: {e}")
            continue
        name = f"syn_{img_id:05d}.png"
        cv2.imwrite(os.path.join(out_img, name), img)
        H, W = img.shape[:2]
        imgs.append({"id": img_id, "file_name": name, "width": W, "height": H})
        for ann in ann_list:
            ann["id"] = ann_id
            ann["image_id"] = img_id
            anns.append(ann)
            ann_id += 1
        for _, _, cat, _ in slots:
            if cat in combined_needed:
                combined_needed[cat] -= 1
                if combined_needed[cat] <= 0:
                    del combined_needed[cat]
        img_id += 1
        if i % 10 == 0:
            print(f"[{i+1}/{max_images}] 남은: {len(combined_needed)}개", end="\r")
    if combined_needed:
        print(f"\n[경고] 미달성 {len(combined_needed)}개")
    print(f"\n합성 완료: {img_id-1}장")
    cats = [{"id": 0, "name": "pill", "supercategory": "none"}]
    cats += [{"id": v, "name": k, "supercategory": "pill"} for k, v in sorted(cat2label.items(), key=lambda x: x[1])]
    coco = {"images": imgs, "annotations": anns, "categories": cats}
    p = os.path.join(out_ann, "_annotations.coco.json")
    with open(p, "w") as f:
        json.dump(coco, f, ensure_ascii=False)
    print(f"저장: {p}")
    return coco
