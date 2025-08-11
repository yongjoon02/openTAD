import sys
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from mmengine.config import Config
from opentad.datasets import build_dataset, build_dataloader
from opentad.datasets.base import SlidingWindowDataset, PaddingDataset, filter_same_annotation
from opentad.datasets.pku import PkuSlidingDataset, PkuPaddingDataset

sys.path.append(str(Path.cwd()))
print("lib import completed")

def check_label_index_mismatch():
    """라벨 인덱스 불일치 확인 및 해결"""
    print("=" * 60)
    print(" 라벨 인덱스 불일치 확인")
    print("=" * 60)
    
    # 1. 원본 어노테이션 파일 확인
    print("\n 원본 어노테이션 파일 확인...")
    with open("data/PKU-MMD/pku_val.json", 'r') as f:
        original_annotations = json.load(f)
    
    # 2. 클래스맵 파일 확인
    print("\n 클래스맵 파일 확인...")
    with open("data/PKU-MMD/class_map.txt", 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    print(f" 클래스 개수: {len(class_names)}")
    print(f" 첫 번째 클래스: {class_names[0]} (인덱스 0)")
    print(f" 마지막 클래스: {class_names[-1]} (인덱스 {len(class_names)-1})")
    
    # 3. 원본 어노테이션의 라벨 분포 확인
    print("\n 원본 어노테이션 라벨 분포...")
    original_labels = set()
    for video_info in original_annotations:
        for anno in video_info["annotations"]:
            original_labels.add(anno["label"])
    
    print(f" 원본 어노테이션 라벨 개수: {len(original_labels)}")
    print(f" 원본 라벨 예시: {list(original_labels)[:5]}")
    
    # 4. 클래스맵과 원본 라벨 비교
    print("\n 클래스맵 vs 원본 라벨 비교...")
    class_map_set = set(class_names)
    missing_in_classmap = original_labels - class_map_set
    missing_in_original = class_map_set - original_labels
    
    if missing_in_classmap:
        print(f"  클래스맵에 없는 원본 라벨: {missing_in_classmap}")
    if missing_in_original:
        print(f"  원본에 없는 클래스맵 라벨: {missing_in_original}")
    
    if not missing_in_classmap and not missing_in_original:
        print(" 클래스맵과 원본 라벨이 일치합니다!")
    
    # 5. 라벨 인덱스 매핑 확인
    print("\n 라벨 인덱스 매핑 확인...")
    class_map_dict = {name: i for i, name in enumerate(class_names)}
    
    print("라벨명 -> 클래스맵 인덱스:")
    for i, label in enumerate(list(original_labels)[:10]):  # 처음 10개만
        if label in class_map_dict:
            print(f"  {label} -> {class_map_dict[label]}")
        else:
            print(f"  {label} ->  클래스맵에 없음")
    
    return class_map_dict, original_annotations

# 라벨 인덱스 확인
class_map_dict, original_annotations = check_label_index_mismatch()

config_path = "configs/adatad/pku_mmd/e2e_pku_mmd_videomae_s_768x1_160_adapter copy.py"
cfg = Config.fromfile(config_path)
print("config file load")
print(f"path : {config_path}")
print(f"keys : {list(cfg.keys())}")

if hasattr(cfg, "dataset"):
    dataset_cfg = cfg.dataset
    print(f"dataset_cfg keys: {list(dataset_cfg.keys())}")

    if hasattr(dataset_cfg, 'train'):
        train_cfg = dataset_cfg.train
        print(f"train_cfg settings")
        print(f"type : {train_cfg.get('type', 'Unknown')}")
        print(f"ann_file : {train_cfg.get('ann_file', 'Unknown')}")
        print(f"data path : {train_cfg.get('data_prefix', 'Unknown')}")
        print(f"pipelines : {len(train_cfg.get('pipeline', []))}")
else:
    print("dataset_cfg x")

cfg = Config.fromfile(config_path)

train_dataset = build_dataset(cfg.dataset.train)

train_loader = build_dataloader(
    train_dataset,
    rank=0,
    world_size=1,
    batch_size=2,
    num_workers=0,
    shuffle=True,
    drop_last=False
)

if 'train_dataset' in locals():
    sample = train_dataset[0]
    print(f"train dataset sample keys : {list(sample.keys())}")

    for key, value in sample.items():
        print(f"{key}")
        print(f"type: {type(value)}")

        if hasattr(value, 'shape'):
            print(f"shape: {value.shape}")
            print(f"data type: {value.dtype}")
            
            if hasattr(value, 'min') and hasattr(value, 'max'):
                print(f"value range: {value.min().item():.3f} ~ {value.max().item():.3f}")

        elif isinstance(value, list):
            print(f"list length: {len(value)}")
            if len(value) > 0:
                print(f"first element type: {type(value[0])}")
        
        elif isinstance(value, dict):
            print(f"dict keys: {list(value.keys())}")
        else:
            print(f"value: {value}")
else:
    print(f"no dataset")

# 라벨 인덱스 검증
print("\n" + "=" * 60)
print(" 데이터로더 라벨 인덱스 검증")
print("=" * 60)

if 'gt_labels' in sample:
    dataloader_labels = sample['gt_labels']
    print(f"데이터로더 라벨: {dataloader_labels}")
    print(f"라벨 범위: {dataloader_labels.min().item()} ~ {dataloader_labels.max().item()}")
    
    # 원본 어노테이션과 비교
    if 'video_name' in sample:
        video_name = sample['video_name']
        print(f"비디오명: {video_name}")
        
        # 원본 어노테이션에서 해당 비디오 찾기
        original_video_info = None
        for video_info in original_annotations:
            if video_info['video_name'] == video_name:
                original_video_info = video_info
                break
        
        if original_video_info:
            print(f"\n원본 어노테이션:")
            for anno in original_video_info['annotations']:
                label_name = anno['label']
                original_idx = class_map_dict.get(label_name, -1)
                print(f"  {label_name} -> 클래스맵 인덱스: {original_idx}")
            
            print(f"\n데이터로더 라벨:")
            for i, label_idx in enumerate(dataloader_labels):
                if label_idx < len(class_names):
                    label_name = class_names[label_idx]
                    print(f"  인덱스 {label_idx} -> {label_name}")
                else:
                    print(f"  인덱스 {label_idx} ->  범위 벗어남")

cfg = Config.fromfile(config_path)

val_dataset = build_dataset(cfg.dataset.val)

val_loader = build_dataloader(
    val_dataset,
    rank=0,
    world_size=1,
    batch_size=2,
    num_workers=0,
    shuffle=True,
    drop_last=False
)

test_dataset = build_dataset(cfg.dataset.test)

test_loader = build_dataloader(
    test_dataset,
    rank=0,
    world_size=1,
    batch_size=2,
    num_workers=0,
    shuffle=True,
    drop_last=False
)
try:
    adatad_loader = build_dataloader(
        train_dataset,
        rank=0,
        world_size=1,
        batch_size=2,
        num_workers=0,
        shuffle=True,
        drop_last=False
    )
    for batch_idx, data_dict in enumerate(adatad_loader):
        if batch_idx == 0:
            print(f" build success")
            print(f" batch keys: {list(data_dict.keys())}")
            break
except Exception as e:
    print(f"Error : {e}")
    import traceback
    traceback.print_exc()
        

first_sample = train_dataset[0]
print(f"first sample metadata:")
for key, value in first_sample.items():
    if key not in ['inputs', 'masks', 'gt_segments', 'gt_labels']:
        print(f"{key}: {value}")

if 'video_name' in first_sample:
    print(f"video name: {first_sample['video_name']}")

if 'windows_start_frame' in first_sample:
    print(f"windows start frame: {first_sample['windows_start_frame']}")



if 'duration' in first_sample:
    print(f"duration: {first_sample['duration']}")

if 'fps' in first_sample:
    print(f"fps: {first_sample['fps']}")

if 'snippet_stride' in first_sample:
    print(f"snippet stride: {first_sample['snippet_stride']}")
    
def visualize_pku_batch_sample(data_dict, sample_idx: int = 0, title: str = "sample"):
    """PKU-MMD 배치에서 영상·GT·메타데이터를 간단히 시각화한다."""
    # ────────────────────── 1. 입력 텐서 처리 ──────────────────────
    if "inputs" not in data_dict:
        print("no inputs in batch");  return

    inputs = data_dict["inputs"].cpu()          # (B, …)
    sample_inputs = inputs[sample_idx]          # (C,N,T,H,W) or (C,T,H,W)

    # (C, N, T, H, W) ⇒ (C, T, H, W) 로 변환
    if sample_inputs.dim() == 5:                # (C, N, T, H, W)
        sample_inputs = sample_inputs[:, 0]     # 첫 chunk 사용
    elif sample_inputs.dim() != 4:              # 예외 처리
        raise ValueError(f"Unexpected shape: {sample_inputs.shape}")

    C, T, H, W = sample_inputs.shape
    frame_idx = np.linspace(0, T - 1, 8, dtype=int)

    # ────────────────────── 2. 프레임 시각화 ──────────────────────
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"{title} (sample {sample_idx})", fontsize=16)

    for k, fi in enumerate(frame_idx):
        r, c = divmod(k, 4)
        frame = sample_inputs[:, fi].permute(1, 2, 0).numpy()        # HWC
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6)
        ax[r, c].imshow((frame * 255).astype("uint8"))
        ax[r, c].set_title(f"Frame {fi}")
        ax[r, c].axis("off")

    plt.tight_layout();  plt.show()

    # ────────────────────── 3. GT 구간·레이블 출력 (프레임 범위 변환 포함) ─────────────────
    if {"gt_segments", "gt_labels"} <= data_dict.keys():
        segs, lbls = data_dict["gt_segments"], data_dict["gt_labels"]
        print(f"\n[pku annotation info] sample {sample_idx}")
        if sample_idx < len(segs):
            for i, (seg, lab) in enumerate(zip(segs[sample_idx], lbls[sample_idx])):
                print(f"  #{i+1:02d}  label={lab.item():2d}  segment={seg.tolist()}")
            
            # 프레임 범위 변환 정보 출력
            if "metas" in data_dict and sample_idx < len(data_dict["metas"]):
                meta = data_dict["metas"][sample_idx]
                if "window_start_frame" in meta:
                    window_start = meta["window_start_frame"]
                    snippet_stride = meta.get("snippet_stride", 1)
                    offset_frames = meta.get("offset_frames", 0)
                    
                    print(f"\n[frame conversion info] sample {sample_idx}")
                    print(f"  window_start_frame: {window_start}")
                    print(f"  snippet_stride: {snippet_stride}")
                    print(f"  offset_frames: {offset_frames}")
                    
                    # 윈도우 기준 -> 전체 기준 복원
                    for i, seg in enumerate(segs[sample_idx]):
                        # 복원 공식: (윈도우 기준 * snippet_stride) + window_start + offset_frames
                        restored_start = seg[0] * snippet_stride + window_start + offset_frames
                        restored_end = seg[1] * snippet_stride + window_start + offset_frames
                        print(f"  #{i+1:02d}  window_segment={seg.tolist()} -> original_segment=[{restored_start:.1f}, {restored_end:.1f}]")
                    
                    # 원본 어노테이션과 비교
                    if "video_name" in meta:
                        video_name = meta["video_name"]
                        # 원본 어노테이션에서 해당 비디오 찾기
                        original_video_info = None
                        for video_info in original_annotations:
                            if video_info['video_name'] == video_name:
                                original_video_info = video_info
                                break
                        
                        if original_video_info:
                            print(f"\n[original annotation comparison] video: {video_name}")
                            for i, anno in enumerate(original_video_info['annotations']):
                                original_seg = anno['segment']
                                label_name = anno['label']
                                print(f"  #{i+1:02d}  {label_name}: {original_seg}")
                            
                            # 복원된 세그먼트와 원본 비교
                            print(f"\n[conversion accuracy check]")
                            for i, seg in enumerate(segs[sample_idx]):
                                restored_start = seg[0] * snippet_stride + window_start + offset_frames
                                restored_end = seg[1] * snippet_stride + window_start + offset_frames
                                
                                # 가장 가까운 원본 세그먼트 찾기
                                min_diff = float('inf')
                                best_match = None
                                for anno in original_video_info['annotations']:
                                    orig_seg = anno['segment']
                                    diff = abs(restored_start - orig_seg[0]) + abs(restored_end - orig_seg[1])
                                    if diff < min_diff:
                                        min_diff = diff
                                        best_match = anno
                                
                                if best_match:
                                    orig_seg = best_match['segment']
                                    accuracy = 1.0 - (min_diff / (orig_seg[1] - orig_seg[0]))
                                    print(f"  #{i+1:02d}  복원: [{restored_start:.1f}, {restored_end:.1f}]")
                                    print(f"         원본: {orig_seg}")
                                    print(f"         정확도: {accuracy:.3f}")
                                    if accuracy > 0.9:
                                        print(f"          정확한 복원!")
                                    else:
                                        print(f"           복원 오차 있음")
        else:
            print("  no annotation")

    # ────────────────────── 4. 메타데이터 출력 ─────────────────────
    if "metas" in data_dict and sample_idx < len(data_dict["metas"]):
        meta = data_dict["metas"][sample_idx]
        print(f"\n[pku metadata] sample {sample_idx}")
        for k, v in meta.items():
            print(f"  {k}: {v}")

if 'data_dict' in locals():
    visualize_pku_batch_sample(data_dict, sample_idx=0, title="sample 0")

    if data_dict['inputs'].shape[0] > 1:
        visualize_pku_batch_sample(data_dict, sample_idx=1, title="sample 1")

else:
    print("no batch data")

# Validation 데이터로도 테스트 (SlidingDataset)
print("\n" + "=" * 60)
print(" Validation 데이터 테스트 (SlidingDataset)")
print("=" * 60)

try:
    val_batch = next(iter(val_loader))
    visualize_pku_batch_sample(val_batch, sample_idx=0, title="val_sample_0")
except Exception as e:
    print(f"Validation 데이터 테스트 오류: {e}")
    import traceback
    traceback.print_exc()