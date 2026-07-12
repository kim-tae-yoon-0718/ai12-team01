# yejin/pipeline/cloud.py
"""클라우드 실행 환경(Kaggle/Colab) 감지 + 입력 경로 탐색 유틸.

노트북들이 환경마다 반복 정의하던 경로 탐색 코드를 모듈화했습니다.
- Kaggle: `/kaggle/input` 아래에서 폴더/파일 "이름"으로 재귀 검색합니다.
  dataset 슬러그(연결된 데이터셋의 실제 폴더명)를 몰라도 동작하고, 같은 파일이
  Dataset이든 이전 커밋 Output이든 어디에 들어 있어도 찾습니다.
- Colab: 데이터 루트 탐색은 저장소 `dataset.find_data_root()`를 그대로 사용하는 것을
  전제로 하고, 여기서는 Drive 아래 파일명 재귀 검색(`find_file_under`)만 보조합니다.
"""
import glob
import os

KAGGLE_INPUT = '/kaggle/input'


def detect_platform():
    """실행 환경을 판별합니다.

    Returns:
        str: 'kaggle' | 'colab' | 'local'
    """
    if os.path.isdir('/kaggle'):
        return 'kaggle'
    if os.path.isdir('/content'):
        return 'colab'
    return 'local'


def find_input_dir(name, root=KAGGLE_INPUT):
    """root 아래에서 이름이 name인 디렉토리를 찾아 반환합니다 (여러 개면 첫 번째).

    Kaggle Input은 `/kaggle/input/<슬러그>/...` 구조라 슬러그를 몰라도 폴더 이름만으로
    찾을 수 있습니다. 탐색 실패 시 None을 반환하므로 호출부에서 assert로 점검하세요.

    Args:
        name (str): 찾을 디렉토리 이름 (예: 'train_images', 'task2_synthesized')
        root (str): 검색 루트 (기본: /kaggle/input)
    """
    hits = sorted(p for p in glob.glob(os.path.join(root, '**', name), recursive=True)
                  if os.path.isdir(p))
    if len(hits) > 1:
        print(f"'{name}' 후보 {len(hits)}개 -> 첫 번째 사용:\n  " + "\n  ".join(hits))
    return hits[0] if hits else None


def find_file_under(root, filename, required=True):
    """root 아래에서 파일명이 filename인 파일을 재귀 검색해 반환합니다 (여러 개면 첫 번째).

    fold_split_masked.json처럼 "어느 폴더에 올렸는지"가 세션/계정마다 다를 수 있는
    파일을 이름만으로 찾는 용도입니다.

    Args:
        root (str): 검색 루트 (Kaggle: /kaggle/input, Colab: Drive 프로젝트 폴더 등)
        filename (str): 찾을 파일명 (glob 패턴 가능)
        required (bool): True면 못 찾았을 때 현재 root 구성을 포함한 메시지로 AssertionError

    Returns:
        str or None: 찾은 파일 경로 (required=False이고 없으면 None)
    """
    hits = sorted(glob.glob(os.path.join(root, '**', filename), recursive=True))
    if not hits:
        if required:
            listing = sorted(os.listdir(root)) if os.path.isdir(root) else '(루트 없음)'
            raise AssertionError(
                f"{filename}을(를) {root} 아래에서 못 찾음 - 업로드/Input 연결 확인.\n"
                f"현재 {root} 구성: {listing}")
        return None
    if len(hits) > 1:
        print(f"'{filename}' 후보 {len(hits)}개 -> 첫 번째 사용:\n  " + "\n  ".join(hits))
    return hits[0]


def find_files_under(root, pattern):
    """root 아래에서 pattern과 일치하는 파일 전체를 재귀 검색합니다 (정렬, 파일명 중복 제거).

    앙상블 추론에서 체크포인트를 파일명 패턴으로 수집할 때 사용합니다.
    같은 파일명이 여러 Input에 있으면(커밋 Output 중복 연결 등) 첫 것만 남깁니다.

    Args:
        root (str): 검색 루트
        pattern (str): 파일명 glob 패턴 (예: 'medium_task2_*_fold*_best.pth')

    Returns:
        list[str]: 파일명 기준 중복 제거된 경로 리스트 (정렬 순서 유지)
    """
    hits = sorted(set(glob.glob(os.path.join(root, '**', pattern), recursive=True)))
    seen, uniq = set(), []
    for p in hits:
        fn = os.path.basename(p)
        if fn not in seen:
            seen.add(fn)
            uniq.append(p)
    return uniq
