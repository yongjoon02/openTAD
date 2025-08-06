
import os
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# OpenTAD 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import opentad
from opentad.datasets.pku import PkuPaddingDataset
from mmengine.config import Config

def load_class_map(class_map_file):
    with open(class_map_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def visualize_adaTAD_sample(dataset, class_names, index=0):
    """adaTAD의 실제 데이터셋에서 샘플 시각화"""
    
    print(f"=== adaTAD PKU Dataset 검증 (인덱스 {index}) ===")
    
    # 실제 adaTAD 데이터셋에서 샘플 로드
    sample = dataset[index]
    
    print(f"샘플 키들: {list(sample.keys())}")
    
    # 입력 데이터 정보
    if 'inputs' in sample:
        inputs = sample['inputs']
        print(f"입력 데이터 shape: {inputs.shape}")
        print(f"입력 데이터 타입: {inputs.dtype}")
        print(f"입력 데이터 범위: {inputs.min():.3f} ~ {inputs.max():.3f}")
    
    # 어노테이션 정보
    if 'gt_segments' in sample:
        gt_segments = sample['gt_segments']
        gt_labels = sample['gt_labels']
        print(f"어노테이션 수: {len(gt_segments)}")
        print(f"어노테이션 segments: {gt_segments}")
        print(f"어노테이션 labels: {gt_labels}")
        
        # 클래스 이름으로 변환
        for i, (segment, label) in enumerate(zip(gt_segments, gt_labels)):
            class_name = class_names[label] if label < len(class_names) else f"Unknown_{label}"
            print(f"  {i+1}. {class_name}: {segment}")
    
    # 마스크 정보
    if 'masks' in sample:
        masks = sample['masks']
        print(f"마스크 shape: {masks.shape}")
        print(f"유효한 프레임 수: {masks.sum().item()}")
    
    # 비디오 정보
    if 'video_name' in sample:
        print(f"비디오 이름: {sample['video_name']}")
    
    # 프레임 정보
    if 'frame_inds' in sample:
        frame_inds = sample['frame_inds']
        print(f"프레임 인덱스 shape: {frame_inds.shape}")
        print(f"프레임 인덱스 범위: {frame_inds.min()} ~ {frame_inds.max()}")
    
    # 윈도우 정보
    if 'window_size' in sample:
        print(f"윈도우 크기: {sample['window_size']}")
    if 'window_start_frame' in sample:
        print(f"윈도우 시작 프레임: {sample['window_start_frame']}")
    
    # 원본 어노테이션과 비교
    if 'original_anno' in sample and sample['original_anno']:
        original_anno = sample['original_anno']
        print(f"\n=== 원본 어노테이션 vs 변환된 어노테이션 ===")
        print(f"원본 segments: {original_anno['gt_segments']}")
        print(f"원본 labels: {original_anno['gt_labels']}")
        print(f"변환 segments: {gt_segments}")
        print(f"변환 labels: {gt_labels}")
    
    return sample

def visualize_frame_sequence(sample, class_names):
    """프레임 시퀀스 시각화"""
    
    if 'inputs' not in sample:
        print("입력 데이터가 없습니다.")
        return
    
    inputs = sample['inputs']
    if len(inputs.shape) != 5:  # (B, C, T, H, W)
        print(f"예상하지 못한 입력 shape: {inputs.shape}")
        return
    
    # 첫 번째 배치 선택
    video = inputs[0]  # (C, T, H, W)
    
    # 16프레임 중 12프레임 선택 (0, 1, 2, ..., 11)
    selected_frames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'adaTAD PKU Sample: {sample.get("video_name", "Unknown")}', fontsize=16)
    
    for i, frame_idx in enumerate(selected_frames):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        if frame_idx < video.shape[1]:  # T 차원
            # (C, H, W) -> (H, W, C) 변환
            frame = video[:, frame_idx, :, :].permute(1, 2, 0)
            
            # 정규화 해제 (0~1 -> 0~255)
            frame = (frame * 255).clamp(0, 255).byte().numpy()
            
            ax.imshow(frame)
            ax.set_title(f'Frame {frame_idx}')
            ax.axis('off')
            
            # 해당 프레임에 있는 어노테이션 표시
            if 'gt_segments' in sample:
                gt_segments = sample['gt_segments']
                gt_labels = sample['gt_labels']
                
                # 프레임 시간 (16프레임 기준)
                frame_time = frame_idx / 16.0
                
                for segment, label in zip(gt_segments, gt_labels):
                    if segment[0] <= frame_time <= segment[1]:
                        class_name = class_names[label] if label < len(class_names) else f"Unknown_{label}"
                        ax.text(10, 30, class_name, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
                               color='white', fontsize=8, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Frame', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # 어노테이션 정보 표시
    if 'gt_segments' in sample and len(sample['gt_segments']) > 0:
        annotation_text = "Annotations in this window:\n"
        for i, (segment, label) in enumerate(zip(sample['gt_segments'], sample['gt_labels'])):
            class_name = class_names[label] if label < len(class_names) else f"Unknown_{label}"
            annotation_text += f"{i+1}. {class_name}: {segment[0]:.2f}s-{segment[1]:.2f}s\n"
        
        fig.text(0.02, 0.02, annotation_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def main():
    """메인 함수 - adaTAD 실제 검증"""
    print("=== adaTAD PKU Dataset 실제 검증 시작 ===")
    
    # 클래스 매핑 로드
    class_map_file = "data/PKU-MMD/class_map.txt"
    class_names = load_class_map(class_map_file)
    print(f"클래스 수: {len(class_names)}")
    
    # 직접 PkuPaddingDataset 생성 (설정 파일 대신)
    print("\n=== PkuPaddingDataset 직접 생성 ===")
    
    try:
        dataset = PkuPaddingDataset(
            ann_file="data/PKU-MMD/pku_train.json",
            subset_name="training",
            data_path="F:/dataset/pku-mmd/rgb",
            class_map="data/PKU-MMD/class_map.txt",
            filter_gt=False,
            feature_stride=4,
            sample_stride=1,
            fps=-1,
            pipeline=[
                dict(type="PrepareVideoInfo", format="avi"),
                dict(type="mmaction.DecordInit", num_threads=6),
                dict(
                    type="LoadFrames",
                    num_clips=1,
                    method="random_trunc",
                    trunc_len=512,
                    trunc_thresh=0.5,
                    crop_ratio=[0.9, 1.0],
                    scale_factor=1,
                ),
                dict(type="mmaction.DecordDecode"),
                dict(type="mmaction.Resize", scale=(-1, 182)),
                dict(type="mmaction.RandomResizedCrop"),
                dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
                dict(type="mmaction.Flip", flip_ratio=0.5),
                dict(type="mmaction.ImgAug", transforms="default"),
                dict(type="mmaction.ColorJitter"),
                dict(type="mmaction.FormatShape", input_format="NCTHW"),
                dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
                dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
            ]
        )
        
        print(f"데이터셋 크기: {len(dataset)}")
        
        # 여러 샘플 검증
        for i in range(min(3, len(dataset))):
            print(f"\n{'='*50}")
            sample = visualize_adaTAD_sample(dataset, class_names, i)
            
            # 프레임 시각화 (첫 번째 샘플만)
            if i == 0:
                visualize_frame_sequence(sample, class_names)
        
    except Exception as e:
        print(f"데이터셋 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 