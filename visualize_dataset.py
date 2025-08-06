import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 백엔드 사용
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import cv2
from PIL import Image
import torch

# OpenTAD 경로 추가
sys.path.insert(0, os.path.dirname(__file__))

from opentad.datasets import build_dataset
from mmengine.config import Config

def visualize_dataset_sample():
    """데이터셋 샘플을 시각화"""
    
    # 설정 파일 로드
    config_file = "configs/adatad/pku_mmd/e2e_pku_mmd_videomae_s_768x1_160_adapter copy.py"
    cfg = Config.fromfile(config_file)
    
    print("=== 데이터셋 설정 ===")
    print(f"Train dataset type: {cfg.dataset.train.type}")
    print(f"Annotation file: {cfg.dataset.train.ann_file}")
    print(f"Video directory: {cfg.dataset.train.data_path}")
    print(f"Class map: {cfg.dataset.train.class_map}")
    
    # 클래스 맵 로드
    class_names = []
    if os.path.exists(cfg.dataset.train.class_map):
        with open(cfg.dataset.train.class_map, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"Number of classes: {len(class_names)}")
        print("First 10 classes:", class_names[:10])
    
    # 어노테이션 파일 확인
    if os.path.exists(cfg.dataset.train.ann_file):
        with open(cfg.dataset.train.ann_file, 'r') as f:
            annotations = json.load(f)
        print(f"Number of videos in train set: {len(annotations)}")
        
        # 첫 번째 비디오 정보 출력
        if len(annotations) > 0:
            first_video = annotations[0]
            print(f"\n=== 첫 번째 비디오 정보 ===")
            print(f"Video name: {first_video['video_name']}")
            print(f"Total frames: {first_video['frame']}")
            print(f"Number of annotations: {len(first_video['annotations'])}")
            
            # 어노테이션 상세 정보
            for i, anno in enumerate(first_video['annotations'][:3]):  # 처음 3개만
                print(f"  Annotation {i+1}:")
                print(f"    Label: {anno['label']}")
                print(f"    Segment: {anno['segment']} (frames)")
                print(f"    Duration: {anno['segment'][1] - anno['segment'][0]} frames")
    
    # 데이터셋 빌드
    print("\n=== 데이터셋 빌드 ===")
    try:
        dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=None))
        print(f"Dataset size: {len(dataset)}")
        
        # 첫 번째 샘플 로드
        print("\n=== 첫 번째 샘플 로드 ===")
        sample = dataset[0]
        
        print("Sample keys:", list(sample.keys()))
        
        if 'imgs' in sample:
            imgs = sample['imgs']
            print(f"Video tensor shape: {imgs.shape}")
            print(f"Video tensor dtype: {imgs.dtype}")
            print(f"Video tensor range: [{imgs.min():.3f}, {imgs.max():.3f}]")
        
        if 'gt_segments' in sample:
            gt_segments = sample['gt_segments']
            print(f"GT segments shape: {gt_segments.shape}")
            print(f"GT segments: {gt_segments}")
        
        if 'gt_labels' in sample:
            gt_labels = sample['gt_labels']
            print(f"GT labels shape: {gt_labels.shape}")
            print(f"GT labels: {gt_labels}")
            if class_names:
                print("GT label names:", [class_names[label] for label in gt_labels])
        
        if 'video_name' in sample:
            print(f"Video name: {sample['video_name']}")
        
        # 비디오 프레임 시각화
        if 'imgs' in sample:
            visualize_video_frames(sample['imgs'], sample.get('gt_segments', None), 
                                 sample.get('gt_labels', None), class_names)
        
    except Exception as e:
        print(f"데이터셋 빌드 중 오류: {e}")
        import traceback
        traceback.print_exc()

def visualize_video_frames(video_tensor, gt_segments=None, gt_labels=None, class_names=None):
    """비디오 프레임을 시각화"""
    
    # 텐서를 numpy로 변환
    if isinstance(video_tensor, torch.Tensor):
        video = video_tensor.cpu().numpy()
    else:
        video = video_tensor
    
    # NCTHW -> NHWCT로 변환 (시각화용)
    if video.ndim == 5:  # NCTHW
        video = np.transpose(video, (0, 2, 3, 1, 4))  # NHWCT
        video = video[0]  # 첫 번째 배치만
    
    print(f"Video shape for visualization: {video.shape}")
    
    # 16프레임마다 샘플링 (전체 프레임이 너무 많으므로)
    num_frames = video.shape[-1]
    sample_indices = np.linspace(0, num_frames-1, min(16, num_frames), dtype=int)
    
    # 프레임 시각화
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('PKU-MMD Video Frames Sample', fontsize=16)
    
    for i, frame_idx in enumerate(sample_indices):
        row, col = i // 4, i % 4
        frame = video[:, :, :, frame_idx]  # HWCT -> HWC
        
        # 정규화 (0-1 범위로)
        if frame.max() > 1:
            frame = frame / 255.0
        
        axes[row, col].imshow(frame)
        axes[row, col].set_title(f'Frame {frame_idx}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'video_frames_sample.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"비디오 프레임 시각화 저장: {output_path}")
    plt.close()  # 메모리 절약을 위해 닫기
    
    # GT 세그먼트 시각화
    if gt_segments is not None:
        visualize_gt_segments(gt_segments, gt_labels, class_names, num_frames)

def visualize_gt_segments(gt_segments, gt_labels, class_names, num_frames):
    """GT 세그먼트를 시각화"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 타임라인 그리기
    ax.set_xlim(0, num_frames)
    ax.set_ylim(0, len(gt_segments) + 1)
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Action Instance')
    ax.set_title('Ground Truth Action Segments')
    
    # 각 액션 세그먼트 그리기
    for i, (segment, label) in enumerate(zip(gt_segments, gt_labels)):
        start_frame, end_frame = segment
        
        # 랜덤 색상
        color = plt.cm.Set3(i % 12)
        
        # 세그먼트 박스 그리기
        rect = patches.Rectangle((start_frame, i + 0.2), end_frame - start_frame, 0.6, 
                               linewidth=2, edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        
        # 라벨 텍스트
        label_text = class_names[label] if class_names and label < len(class_names) else f'Class {label}'
        ax.text(start_frame, i + 0.5, label_text, fontsize=8, ha='left', va='center')
    
    # 그리드 추가
    ax.grid(True, alpha=0.3)
    ax.set_yticks(range(1, len(gt_segments) + 1))
    ax.set_yticklabels([f'Action {i+1}' for i in range(len(gt_segments))])
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'gt_segments.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"GT 세그먼트 시각화 저장: {output_path}")
    plt.close()  # 메모리 절약을 위해 닫기

def check_annotation_files():
    """어노테이션 파일들을 확인"""
    
    print("=== 어노테이션 파일 확인 ===")
    
    annotation_files = [
        "data/PKU-MMD/pku_train.json",
        "data/PKU-MMD/pku_val.json", 
        "data/PKU-MMD/pku_test.json",
        "data/PKU-MMD/class_map.txt"
    ]
    
    for file_path in annotation_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} - 존재함")
            
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"  데이터 개수: {len(data)}")
                
                if len(data) > 0 and isinstance(data[0], dict):
                    print(f"  첫 번째 항목 키: {list(data[0].keys())}")
                    
            elif file_path.endswith('.txt'):
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                print(f"  클래스 개수: {len(lines)}")
                print(f"  처음 5개 클래스: {[line.strip() for line in lines[:5]]}")
        else:
            print(f"✗ {file_path} - 존재하지 않음")

if __name__ == "__main__":
    print("PKU-MMD 데이터셋 시각화 시작...\n")
    
    # 어노테이션 파일 확인
    check_annotation_files()
    
    print("\n" + "="*50 + "\n")
    
    # 데이터셋 샘플 시각화
    visualize_dataset_sample()
    
    print("\n시각화 완료! 생성된 파일:")
    print("- video_frames_sample.png: 비디오 프레임 샘플")
    print("- gt_segments.png: GT 액션 세그먼트") 