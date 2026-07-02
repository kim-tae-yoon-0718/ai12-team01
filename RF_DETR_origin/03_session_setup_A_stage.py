#@title [셀 2-B] A단계 복원 — zip 있을 때 ([2-A] 대신 이것만)
ZIP = os.path.join(PROJ_ROOT, "dataset_5fold.zip")       # PROJ_ROOT 기준(경로 자동 통일)
print("zip 존재:", os.path.exists(ZIP))                  # True 확인
!cp "$ZIP" /content/                                     # 드라이브→로컬 복사
!unzip -qo /content/dataset_5fold.zip -d /content/dataset  # 압축 해제(-o=덮어쓰기)
print("복원 fold:", sorted(d for d in os.listdir("/content/dataset") if d.startswith("fold")))