# 코랩 셀에 통째로 붙여넣고 실행.
# 사전 준비 (Drive에 미리 업로드):
#   MyDrive/kojamonet/scripts.zip          <- colab_scripts.zip 업로드
#   MyDrive/kojamonet/train_data/          <- 로컬 train_data/ 폴더 전체 업로드
#   MyDrive/kojamonet/checkpoints/model_v13.pth  <- 로컬 checkpoints/model_v13.pth 업로드

from google.colab import drive
drive.mount('/content/drive')

import os

DRIVE_ROOT = '/content/drive/MyDrive/kojamonet'
PROJECT = '/content/kojamonet'

os.makedirs(PROJECT, exist_ok=True)

# 스크립트만 로컬(/content)로 압축 해제 — 코드 실행은 Drive 위에서 직접 하면 느림
import zipfile
with zipfile.ZipFile(f'{DRIVE_ROOT}/scripts.zip') as z:
    z.extractall(f'{PROJECT}/scripts')

# train_data, checkpoints는 용량 커서 심볼릭 링크로 Drive를 직접 참조
# -> 체크포인트 저장(10에폭마다)도 자동으로 Drive에 남아서 세션 끊겨도 안 날아감
if not os.path.islink(f'{PROJECT}/train_data'):
    os.symlink(f'{DRIVE_ROOT}/train_data', f'{PROJECT}/train_data')
if not os.path.islink(f'{PROJECT}/checkpoints'):
    os.symlink(f'{DRIVE_ROOT}/checkpoints', f'{PROJECT}/checkpoints')

os.chdir(PROJECT)
!python scripts/train.py resume
