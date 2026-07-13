# rf-detr/pill_extraction.py
"""
SAM(Segment Anything Model)으로 알약 누끼(투명 배경 RGBA crop)를 추출합니다.

extract_pills.ipynb(원본 56종)와 task2_synthesis.ipynb(test 전용 18종 추가 추출)가
공유하는 로직입니다. synthesis.py는 이 모듈이 만든 누끼 결과물(pills/ 폴더)을
입력으로 사용할 뿐, 이 모듈의 함수를 직접 호출하지는 않습니다 (역할 분리:
"누끼 추출"과 "합성 배치"는 노트북에서 순차적으로 실행되는 별개 단계).
"""
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

SAM_CHECKPOINT = "/content/sam_vit_h_4b8939.pth"   # Colab 기본 경로 - 다르면 load_sam(checkpoint=...)로 지정
BLEND_FEATHER = 3   # crop_pill_rgba의 마스크 경계 블러 강도(px)


def load_sam(checkpoint=SAM_CHECKPOINT, model_type="vit_h", device="cuda"):
    """SAM 체크포인트를 로드해 SamPredictor를 반환합니다."""
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    print(f"SAM 로드 완료 | {model_type} | {device}")
    return SamPredictor(sam)


def segment_pill(predictor, image_bgr, bbox_xywh):
    """bbox 안의 알약을 SAM으로 세그멘테이션해 최고 score 마스크를 반환합니다."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    x, y, w, h = [float(v) for v in bbox_xywh]
    box = np.array([x, y, x + w, y + h])
    masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
    return masks[int(np.argmax(scores))]


def crop_pill_rgba(image_bgr, mask, bbox_xywh, padding=5):
    """세그멘테이션 마스크로 알약을 투명 배경 RGBA로 crop합니다 (경계 페더링 포함)."""
    H, W = image_bgr.shape[:2]
    x, y, w, h = [int(v) for v in bbox_xywh]
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(W, x + w + padding), min(H, y + h + padding)
    crop_bgr = image_bgr[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2].astype(np.uint8) * 255
    k = BLEND_FEATHER * 2 + 1
    crop_mask = cv2.GaussianBlur(crop_mask, (k, k), BLEND_FEATHER)
    rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = crop_mask
    return rgba
