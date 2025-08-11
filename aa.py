# 모델 상태 확인 스크립트
import torch
from mmengine.config import Config

# 설정 로드
cfg = Config.fromfile("configs/adatad/pku_mmd/e2e_pku_mmd_videomae_s_768x1_160_adapter copy.py")

# 체크포인트 로드
checkpoint = torch.load("work_dirs/e2e_pku_mmd_videomae_s_768x1_160_adapter/gpu1_id0/checkpoint/epoch_34.pth", map_location="cpu")

print(f"체크포인트 키: {list(checkpoint.keys())}")
print(f"모델 상태 키: {list(checkpoint['state_dict'].keys())[:10]}")

# 학습 가능한 파라미터 확인
from opentad.models import build_detector
model = build_detector(cfg.model)

# 파라미터 상태 확인
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"학습 가능: {name}")