# yejin/pipeline/corrections.py
"""annotation 수정 내역(corrections) 스냅샷.

task2-masked / task3 / task4-masked 실험이 공유하는 corrections 스냅샷을 하드코딩합니다.
팀 corrections.json이 이후 바뀌어도 실험 간 데이터 조건이 흔들리지 않도록 고정하는 목적입니다.

⚠ 저장소 canonical `RF_DETR_split_ver/corrections.json`과 일부 다른 "구버전" 스냅샷입니다
  (fix_category "3444"->3351 및 "791"->31863 포함, add_boxes 1건 category 상이).
  fold-matched WBF 앙상블 파트너인 task2-masked RF-DETR 체크포인트와 학습 라벨 조건을
  완전히 일치시키기 위해 의도적으로 이 스냅샷을 유지합니다. 최신 검수 내역이 필요한
  실험은 저장소 corrections.json을 직접 사용하세요.

저장소 `dataset.apply_corrections()`는 "파일 경로"를 받는 시그니처라(저장소 함수 무수정 원칙),
`save_corrections_snapshot()`으로 dict를 json 파일로 1회 저장한 뒤 그 경로를 넘겨 사용합니다.
적용 순서(coord_fix -> remove -> modify -> add -> fix_category)는 함수 내부에서 보장됩니다.
"""
import json

# task2-masked(Kaggle) 실험과 동일한 스냅샷 (구버전)
CORRECTIONS_SNAPSHOT = {
    # 1) 좌표 오염 수정: 동일 bbox를 가진 항목을 corrected로 교체
    "coord_fix": {
        "K-003351-016262-018357_0_2_0_2_75_000_200.png": [
            {"original": [6567, 625, 311, 315], "corrected": [567, 625, 311, 315]}
        ]
    },
    # 2) 중복/오류 박스 제거: category_id + bbox가 일치하는 첫 항목 1개 제거
    "remove_boxes": {
        "K-001900-016548-019607-033009_0_2_0_2_70_000_200.png": [
            {"category_id": 16548, "bbox": [88, 255, 366, 209]}
        ]
    },
    # 3) 좌표 수정: category_id(+match_bbox) 첫 매치의 bbox를 교체하거나 directive 적용
    "modify_boxes": {
        "K-003351-020014-020238_0_2_0_2_90_000_200.png": [
            {"category_id": 3351, "match_bbox": None, "new_bbox": [390, 260, 170, 165]}
        ],
        "K-003351-019232-029667_0_2_1_2_70_000_200.png": [
            {"category_id": 19232, "match_bbox": None, "directive": "EXTEND_DOWN_95"}
        ]
    },
    # 4) 누락 박스 추가 (원본에 없던 박스라 annotation_id는 None으로 채워짐)
    "add_boxes": {
        "K-001900-016548-019607-033009_0_2_0_2_70_000_200.png": [
            {"category_id": 16548, "bbox": [90, 870, 245, 240]}
        ],
        "K-003351-013900-021325_0_2_0_2_70_000_200.png": [
            {"category_id": 3351, "bbox": [400, 830, 180, 180]}
        ],
        "K-003351-013900-036637_0_2_0_2_70_000_200.png": [
            {"category_id": 3351, "bbox": [440, 880, 175, 175]}
        ],
        "K-003351-020014-022074_0_2_0_2_90_000_200.png": [
            {"category_id": 20014, "bbox": [65, 720, 325, 315]}
        ],
        "K-003351-020238-031863_0_2_0_2_70_000_200.png": [
            {"category_id": 20238, "bbox": [580, 290, 215, 215]}
        ],
        "K-003351-021325-032310_0_2_0_2_90_000_200.png": [
            {"category_id": 32310, "bbox": [595, 830, 345, 245]}
        ],
        "K-003351-029667-031863_0_2_0_2_70_000_200.png": [
            {"category_id": 3351, "bbox": [375, 870, 165, 165]}
        ],
        "K-003351-032310-038162_0_2_0_2_70_000_200.png": [
            {"category_id": 3351, "bbox": [390, 855, 185, 185]}
        ],
        "K-003351-033880-038162_0_2_0_2_75_000_200.png": [
            {"category_id": 33880, "bbox": [70, 600, 310, 425]}
        ],
        "K-003351-035206-041768_0_2_0_2_70_000_200.png": [
            {"category_id": 3351, "bbox": [460, 875, 180, 180]}
        ],
        "K-003544-004543-012247-016548_0_2_0_2_90_000_200.png": [
            {"category_id": 4543, "bbox": [640, 195, 205, 190]}
        ]
    },
    # 5) category_id 오기재 수정: 원본 annotation_id -> 올바른 category_id
    #    (키는 문자열이어야 함 - apply_corrections 내부에서 int(k)로 변환)
    "fix_category": {
        "791": 31863,
        "3444": 3351,
        "3441": 3351,
        "1420": 35206,
        "1412": 27733
    }
}


def save_corrections_snapshot(path, corrections=None):
    """corrections 스냅샷을 json으로 저장하고 경로를 반환합니다.

    저장소 `dataset.apply_corrections(boxes, cats, ids, corrections_path)`에 넘길
    파일을 만드는 용도입니다.

    Args:
        path (str): 저장할 json 경로 (예: '/kaggle/working/corrections.json')
        corrections (dict): 저장할 내역 (기본: CORRECTIONS_SNAPSHOT)

    Returns:
        str: 저장된 파일 경로 (apply_corrections에 그대로 전달)
    """
    corrections = corrections if corrections is not None else CORRECTIONS_SNAPSHOT
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2)
    print('corrections 스냅샷 저장:', path,
          '| 항목 수:', {k: len(v) for k, v in corrections.items()})
    return path
