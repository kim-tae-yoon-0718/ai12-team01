"""
data_pipeline.py
─────────────────────────────────────────────────────────────
Data Engineer 담당 모듈
"""

import os
import glob
import json
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

SEED = 42
IMG_SIZE_DEFAULT = 512
PROCESSED_DIR = "processed"
CLASS_MAPPING_FILENAME = "class_mapping.json"
BBOX_STATS_FILENAME = "bbox_stats.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def find_dirs(data_root):
    def find_one(*keywords):
        for kw in keywords:
            candidates = glob.glob(os.path.join(data_root, "**", f"*{kw}*"), recursive=True)
            dirs = [c for c in candidates if os.path.isdir(c)]
            if dirs:
                return sorted(dirs, key=len)[0]
        return None

    train_img_dir = find_one("train_images", "train_image")
    train_ann_dir = find_one("train_annotations", "train_annotation")
    test_img_dir = find_one("test_images", "test_image")
    return train_img_dir, train_ann_dir, test_img_dir


def is_valid_annotation(ann):
    bbox = ann.get("bbox", None)
    if bbox is None or len(bbox) == 0:
        return False
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    return True


def merge_coco_annotations(json_paths, verbose=True):
    all_images, all_annotations = [], []
    categories = {}
    n_dropped = 0

    for jp in json_paths:
        try:
            with open(jp, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        images = data.get("images", [])
        if isinstance(images, dict):
            images = [images]
        all_images.extend(images)

        for ann in data.get("annotations", []):
            if is_valid_annotation(ann):
                all_annotations.append(ann)
            else:
                n_dropped += 1

        for cat in data.get("categories", []):
            categories[cat["id"]] = cat.get("name", str(cat["id"]))

    seen_ids = set()
    dedup_images = []
    for img in all_images:
        if img["id"] not in seen_ids:
            dedup_images.append(img)
            seen_ids.add(img["id"])

    if verbose:
        print(f"JSON 파일 {len(json_paths):,}개 병합 완료")
        print(f"  이미지 수        : {len(dedup_images):,}")
        print(f"  유효 어노테이션   : {len(all_annotations):,}")
        print(f"  제외된 어노테이션 : {n_dropped:,}  (bbox 누락/형식 오류)")
        print(f"  클래스 수        : {len(categories):,}")

    return {"images": dedup_images, "annotations": all_annotations, "categories": categories}


def build_class_mapping(categories: dict, save_path=None):
    sorted_ids = sorted(categories.keys())
    mapping = {
        "cat_id_to_idx": {cid: i + 1 for i, cid in enumerate(sorted_ids)},
        "idx_to_name": {i + 1: categories[cid] for i, cid in enumerate(sorted_ids)},
    }
    mapping["idx_to_name"][0] = "background"

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    return mapping


def load_class_mapping(path):
    with open(path, encoding="utf-8") as f:
        mapping = json.load(f)
    mapping["cat_id_to_idx"] = {int(k): v for k, v in mapping["cat_id_to_idx"].items()}
    mapping["idx_to_name"] = {int(k): v for k, v in mapping["idx_to_name"].items()}
    return mapping


def train_val_split(images, val_ratio=0.15, seed=SEED):
    rng = random.Random(seed)
    shuffled = images.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def get_train_transform(img_size=IMG_SIZE_DEFAULT):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussNoise(p=0.1),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["labels"], min_visibility=0.3, clip=True),
    )


def get_val_transform(img_size=IMG_SIZE_DEFAULT):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["labels"], min_visibility=0.3, clip=True),
    )


class PillDataset(Dataset):
    def __init__(self, images, img_to_anns, image_dir, cat_id_to_idx, transform=None):
        self.images = images
        self.img_to_anns = img_to_anns
        self.image_dir = image_dir
        self.cat_id_to_idx = cat_id_to_idx
        self.transform = transform
        self._path_cache = {}

    def __len__(self):
        return len(self.images)

    def _resolve_path(self, file_name):
        if file_name in self._path_cache:
            return self._path_cache[file_name]
        direct = os.path.join(self.image_dir, file_name)
        if os.path.exists(direct):
            resolved = direct
        else:
            matches = glob.glob(os.path.join(self.image_dir, "**", os.path.basename(file_name)), recursive=True)
            resolved = matches[0] if matches else direct
        self._path_cache[file_name] = resolved
        return resolved

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = self._resolve_path(img_info["file_name"])
        image = np.array(Image.open(img_path).convert("RGB"))

        anns = self.img_to_anns.get(img_info["id"], [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([float(x), float(y), float(w), float(h)])
            labels.append(self.cat_id_to_idx[ann["category_id"]])

        if self.transform:
            out = self.transform(image=image, bboxes=boxes, labels=labels)
            image = out["image"]
            boxes = out["bboxes"]
            labels = out["labels"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        boxes_xyxy = [[x, y, x + w, y + h] for x, y, w, h in boxes]

        if len(boxes_xyxy) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([img_info["id"]]),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def compute_bbox_stats(annotations, save_path=None):
    widths = np.array([a["bbox"][2] for a in annotations], dtype=np.float32)
    heights = np.array([a["bbox"][3] for a in annotations], dtype=np.float32)
    areas = widths * heights
    stats = {
        "width_mean": float(widths.mean()), "width_std": float(widths.std()),
        "height_mean": float(heights.mean()), "height_std": float(heights.std()),
        "area_mean": float(areas.mean()),
        "area_p10": float(np.percentile(areas, 10)),
        "area_p50": float(np.percentile(areas, 50)),
        "area_p90": float(np.percentile(areas, 90)),
        "aspect_ratio_mean": float((widths / np.clip(heights, 1e-6, None)).mean()),
    }
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def get_dataloaders(
    data_root,
    batch_size=4,
    img_size=IMG_SIZE_DEFAULT,
    val_ratio=0.15,
    seed=SEED,
    num_workers=0,
    processed_dir=PROCESSED_DIR,
):
    train_img_dir, train_ann_dir, _ = find_dirs(data_root)
    if train_img_dir is None or train_ann_dir is None:
        raise FileNotFoundError(
            "train_images / train_annotations 폴더를 찾지 못했습니다. "
            "data_root 경로를 확인하거나 find_dirs() 결과를 직접 지정해 주세요."
        )

    json_paths = glob.glob(os.path.join(train_ann_dir, "**", "*.json"), recursive=True)
    merged = merge_coco_annotations(json_paths)

    os.makedirs(processed_dir, exist_ok=True)
    mapping = build_class_mapping(
        merged["categories"], save_path=os.path.join(processed_dir, CLASS_MAPPING_FILENAME)
    )
    compute_bbox_stats(merged["annotations"], save_path=os.path.join(processed_dir, BBOX_STATS_FILENAME))

    img_to_anns = {}
    for ann in merged["annotations"]:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    train_imgs, val_imgs = train_val_split(merged["images"], val_ratio=val_ratio, seed=seed)

    train_set = PillDataset(
        train_imgs, img_to_anns, train_img_dir, mapping["cat_id_to_idx"],
        transform=get_train_transform(img_size),
    )
    val_set = PillDataset(
        val_imgs, img_to_anns, train_img_dir, mapping["cat_id_to_idx"],
        transform=get_val_transform(img_size),
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    return train_loader, val_loader, mapping
