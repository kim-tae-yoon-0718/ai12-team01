#@title [셀 1] 마운트 + 경로 자동 탐색 — 매 세션 필수
from google.colab import drive                          # 코랩↔드라이브 연결 도구
drive.mount('/content/drive')                            # 드라이브 마운트

import os, glob                                          # 경로·탐색 도구
CANDS = [                                                # 사람마다 위치 다름 → 후보 목록
    '/content/drive/MyDrive/1팀 공유 문서/ai12-level1-project/sprint_ai_project1_data',
    '/content/drive/MyDrive/ai12-level1-project/sprint_ai_project1_data',
]
DATA_ROOT = next((c for c in CANDS if os.path.exists(c)), None)   # 존재하는 첫 경로 채택
if DATA_ROOT is None:                                    # 후보에 없으면 전체 재귀 검색
    hits = glob.glob('/content/drive/MyDrive/**/sprint_ai_project1_data', recursive=True)
    DATA_ROOT = hits[0] if hits else None
assert DATA_ROOT, "sprint_ai_project1_data 못 찾음 — 공유 폴더 바로가기 확인"
PROJ_ROOT = os.path.dirname(DATA_ROOT)                   # zip·백업 공통 상위(.../ai12-level1-project)

TRAIN_IMG = os.path.join(DATA_ROOT, 'train_images')      # 학습 이미지 232장
TRAIN_ANN = os.path.join(DATA_ROOT, 'train_annotations') # 박스당 JSON 763개
TEST_IMG  = os.path.join(DATA_ROOT, 'test_images')       # 제출용 842장
print("DATA_ROOT:", DATA_ROOT)                           # 채택 경로 확인
for p in [TRAIN_IMG, TRAIN_ANN, TEST_IMG]:               # 세 경로 점검
    print(p, '->', os.path.exists(p))                    # 전부 True여야 함