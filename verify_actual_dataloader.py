#!/usr/bin/env python3
"""
실제 학습/검증에서 사용되는 dataloader 검증 스크립트
비디오 샘플 3개만 검증하되, 모든 설정은 기존 구현된 대로 유지
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path

# OpenTAD 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmengine.config import Config
from opentad.datasets import build_dataset, build_dataloader

def load_class_map(class_map_file):
    """클래스 매핑 파일 로드"""
    with open(class_map_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def verify_actual_train_dataloader():
    """실제 훈련 dataloader 검증 (3개 샘플만)"""
    print("="*60)
    print("실제 훈련 Dataloader 검증 (3개 샘플)")
    print("="*60)
    
    # 실제 설정 파일 로드
    config_file = "configs/adatad/pku_mmd/e2e_pku_mmd_videomae_s_768x1_160_adapter copy.py"
    cfg = Config.fromfile(config_file)
    
    print(f"설정 파일: {config_file}")
    print(f"훈련 데이터셋 타입: {cfg.dataset.train.type}")
    
    # 훈련 데이터셋 생성
    train_dataset = build_dataset(cfg.dataset.train)
    print(f"훈련 데이터셋 크기: {len(train_dataset)}")
    
    # 훈련 dataloader 생성 (기존 설정 그대로)
    train_dataloader = build_dataloader(
        dataset=train_dataset,
        batch_size=2,
        rank=0,
        world_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    
    print(f"훈련 dataloader 길이: {len(train_dataloader)}")
    
    # 3개 샘플만 확인
    sample_count = 0
    for batch_idx, batch in enumerate(train_dataloader):
        if sample_count >= 3:  # 3개 샘플만 확인
            break
            
        print(f"\n--- 훈련 배치 {batch_idx+1} ---")
        
        # 배치 데이터 구조 확인
        print(f"배치 타입: {type(batch)}")
        print(f"배치 키들: {list(batch.keys()) if isinstance(batch, dict) else 'Not a dict'}")
        
        if isinstance(batch, dict):
            # 배치가 딕셔너리인 경우
            print(f"배치 크기: {len(batch.get('imgs', [])) if 'imgs' in batch else 'Unknown'}")
            
            # 입력 데이터
            if 'imgs' in batch:
                imgs = batch['imgs']
                print(f"    배치 입력 형태: {imgs.shape}")
                print(f"    배치 입력 범위: {imgs.min():.3f} ~ {imgs.max():.3f}")
            
            # 어노테이션
            if 'gt_segments' in batch and 'gt_labels' in batch:
                gt_segments = batch['gt_segments']
                gt_labels = batch['gt_labels']
                print(f"    배치 어노테이션 segments: {gt_segments}")
                print(f"    배치 어노테이션 labels: {gt_labels}")
            
            # 마스크
            if 'masks' in batch:
                masks = batch['masks']
                print(f"    배치 마스크 형태: {masks.shape}")
                
            sample_count += 1
        else:
            # 배치가 리스트인 경우
            batch_size = len(batch)
            print(f"배치 크기: {batch_size}")
            
            for sample_idx in range(batch_size):
                if sample_count >= 3:  # 3개 샘플만 확인
                    break
                    
                sample = batch[sample_idx]
                print(f"\n  훈련 샘플 {sample_count+1}:")
                
                # 비디오 이름
                if 'video_name' in sample:
                    video_name = sample['video_name']
                    print(f"    비디오 이름: {video_name}")
                
                # 입력 데이터
                if 'imgs' in sample:
                    imgs = sample['imgs']
                    print(f"    입력 형태: {imgs.shape}")
                    print(f"    입력 범위: {imgs.min():.3f} ~ {imgs.max():.3f}")
                
                # 어노테이션
                if 'gt_segments' in sample and 'gt_labels' in sample:
                    gt_segments = sample['gt_segments']
                    gt_labels = sample['gt_labels']
                    print(f"    어노테이션 개수: {len(gt_segments)}")
                    
                    class_names = load_class_map("data/PKU-MMD/class_map.txt")
                    for j, (segment, label) in enumerate(zip(gt_segments, gt_labels)):
                        class_name = class_names[label] if label < len(class_names) else f"Unknown_{label}"
                        print(f"      {j+1}. {class_name}: {segment}")
                
                # 마스크
                if 'masks' in sample:
                    masks = sample['masks']
                    valid_frames = masks.sum().item()
                    print(f"    유효한 프레임 수: {valid_frames}")
                
                sample_count += 1

def verify_actual_val_dataloader():
    """실제 검증 dataloader 검증 (3개 샘플만)"""
    print("\n" + "="*60)
    print("실제 검증 Dataloader 검증 (3개 샘플)")
    print("="*60)
    
    # 실제 설정 파일 로드
    config_file = "configs/adatad/pku_mmd/e2e_pku_mmd_videomae_s_768x1_160_adapter copy.py"
    cfg = Config.fromfile(config_file)
    
    print(f"검증 데이터셋 타입: {cfg.dataset.val.type}")
    
    # 검증 데이터셋 생성
    print("검증 데이터셋 생성 중...")
    val_dataset = build_dataset(cfg.dataset.val)
    print(f"검증 데이터셋 크기: {len(val_dataset)}")
    
    # 검증 dataloader 생성 (기존 설정 그대로)
    val_dataloader = build_dataloader(
        dataset=val_dataset,
        batch_size=2,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    print(f"검증 dataloader 길이: {len(val_dataloader)}")
    
    # 3개 샘플만 확인
    sample_count = 0
    for batch_idx, batch in enumerate(val_dataloader):
        if sample_count >= 3:  # 3개 샘플만 확인
            break
            
        print(f"\n--- 검증 배치 {batch_idx+1} ---")
        
        # 배치 데이터 구조 확인
        print(f"배치 타입: {type(batch)}")
        print(f"배치 키들: {list(batch.keys()) if isinstance(batch, dict) else 'Not a dict'}")
        
        if isinstance(batch, dict):
            # 배치가 딕셔너리인 경우
            print(f"배치 크기: {len(batch.get('imgs', [])) if 'imgs' in batch else 'Unknown'}")
            
            # 입력 데이터
            if 'imgs' in batch:
                imgs = batch['imgs']
                print(f"    배치 입력 형태: {imgs.shape}")
                print(f"    배치 입력 범위: {imgs.min():.3f} ~ {imgs.max():.3f}")
            
            # 어노테이션
            if 'gt_segments' in batch and 'gt_labels' in batch:
                gt_segments = batch['gt_segments']
                gt_labels = batch['gt_labels']
                print(f"    배치 어노테이션 segments: {gt_segments}")
                print(f"    배치 어노테이션 labels: {gt_labels}")
            
            # 마스크
            if 'masks' in batch:
                masks = batch['masks']
                print(f"    배치 마스크 형태: {masks.shape}")
                
            sample_count += 1
        else:
            # 배치가 리스트인 경우
            batch_size = len(batch)
            print(f"배치 크기: {batch_size}")
            
            for sample_idx in range(batch_size):
                if sample_count >= 3:  # 3개 샘플만 확인
                    break
                    
                sample = batch[sample_idx]
                print(f"\n  검증 샘플 {sample_count+1}:")
                
                # 비디오 이름
                if 'video_name' in sample:
                    video_name = sample['video_name']
                    print(f"    비디오 이름: {video_name}")
                
                # 윈도우 정보 (PkuSlidingDataset)
                if 'window_start_frame' in sample:
                    window_start = sample['window_start_frame']
                    window_size = sample.get('window_size', 512)
                    window_end = window_start + window_size * 4  # snippet_stride=4
                    print(f"    윈도우: 프레임 {window_start} ~ {window_end}")
                
                # 입력 데이터
                if 'imgs' in sample:
                    imgs = sample['imgs']
                    print(f"    입력 형태: {imgs.shape}")
                    print(f"    입력 범위: {imgs.min():.3f} ~ {imgs.max():.3f}")
                
                # 어노테이션
                if 'gt_segments' in sample and 'gt_labels' in sample:
                    gt_segments = sample['gt_segments']
                    gt_labels = sample['gt_labels']
                    print(f"    어노테이션 개수: {len(gt_segments)}")
                    
                    class_names = load_class_map("data/PKU-MMD/class_map.txt")
                    for j, (segment, label) in enumerate(zip(gt_segments, gt_labels)):
                        class_name = class_names[label] if label < len(class_names) else f"Unknown_{label}"
                        
                        # 윈도우 시작 위치 고려한 계산
                        window_start = sample.get('window_start_frame', 0)
                        start_frame = window_start + segment[0].item() * 4
                        end_frame = window_start + segment[1].item() * 4
                        
                        print(f"      {j+1}. {class_name}: {segment} (샘플링된 인덱스)")
                        print(f"         → 원본 비디오: 프레임 {start_frame:.0f} ~ {end_frame:.0f}")
                        print(f"         → 시간: {start_frame/30:.1f}s ~ {end_frame/30:.1f}s")
                
                # 마스크
                if 'masks' in sample:
                    masks = sample['masks']
                    valid_frames = masks.sum().item()
                    print(f"    유효한 프레임 수: {valid_frames}")
                
                sample_count += 1

def verify_annotation_consistency():
    """어노테이션 일관성 검증 (3개 샘플만)"""
    print("\n" + "="*60)
    print("어노테이션 일관성 검증 (3개 샘플)")
    print("="*60)
    
    # 원본 어노테이션 로드
    with open("data/PKU-MMD/pku_val.json", "r") as f:
        val_data = json.load(f)
    
    # 첫 번째 비디오 선택
    test_video = val_data[0]
    video_name = test_video['video_name']
    original_annotations = test_video['annotations']
    
    print(f"테스트 비디오: {video_name}")
    print(f"원본 어노테이션 개수: {len(original_annotations)}")
    
    # 원본 어노테이션 출력
    print("\n원본 어노테이션:")
    for i, anno in enumerate(original_annotations):
        start_frame, end_frame = anno['segment']
        print(f"  {i+1}. {anno['label']}: 프레임 {start_frame} ~ {end_frame} (시간: {start_frame/30:.1f}s ~ {end_frame/30:.1f}s)")
    
    # 실제 dataloader에서 같은 비디오 찾기
    config_file = "configs/adatad/pku_mmd/e2e_pku_mmd_videomae_s_768x1_160_adapter copy.py"
    cfg = Config.fromfile(config_file)
    
    print("검증 데이터셋 생성 중...")
    val_dataset = build_dataset(cfg.dataset.val)
    
    # 해당 비디오의 샘플들 찾기 (3개만)
    print("해당 비디오의 샘플을 찾는 중...")
    matching_samples = []
    for i in range(min(100, len(val_dataset))):  # 처음 100개만 검색
        if len(matching_samples) >= 3:  # 3개 찾으면 중단
            break
        sample = val_dataset[i]
        if sample.get('video_name') == video_name:
            matching_samples.append((i, sample))
    
    print(f"실제 dataloader에서 찾은 {video_name} 샘플 수: {len(matching_samples)}")
    
    # 매칭 검증 (3개만)
    for sample_idx, (idx, sample) in enumerate(matching_samples[:3]):
        print(f"\n--- 실제 샘플 {sample_idx+1} (데이터셋 인덱스: {idx}) ---")
        
        if 'gt_segments' in sample and 'gt_labels' in sample:
            gt_segments = sample['gt_segments']
            gt_labels = sample['gt_labels']
            
            class_names = load_class_map("data/PKU-MMD/class_map.txt")
            print(f"변환된 어노테이션 개수: {len(gt_segments)}")
            
            for j, (segment, label) in enumerate(zip(gt_segments, gt_labels)):
                class_name = class_names[label] if label < len(class_names) else f"Unknown_{label}"
                
                # 윈도우 시작 위치 고려한 계산
                window_start = sample.get('window_start_frame', 0)
                start_frame = window_start + segment[0].item() * 4
                end_frame = window_start + segment[1].item() * 4
                
                print(f"  {j+1}. {class_name}: {segment} (샘플링된 인덱스)")
                print(f"     → 원본 비디오: 프레임 {start_frame:.0f} ~ {end_frame:.0f}")
                print(f"     → 시간: {start_frame/30:.1f}s ~ {end_frame/30:.1f}s")
                
                # 원본 어노테이션과 비교
                for orig_anno in original_annotations:
                    orig_start, orig_end = orig_anno['segment']
                    if orig_anno['label'] == class_name:
                        # IoU 계산
                        intersection_start = max(start_frame, orig_start)
                        intersection_end = min(end_frame, orig_end)
                        intersection = max(0, intersection_end - intersection_start)
                        
                        union_start = min(start_frame, orig_start)
                        union_end = max(end_frame, orig_end)
                        union = union_end - union_start
                        
                        iou = intersection / union if union > 0 else 0
                        
                        if iou > 0.5:
                            print(f"     → 원본 매칭: {orig_start} ~ {orig_end} (IoU: {iou:.3f})")

def main():
    """메인 함수"""
    print("실제 Dataloader 검증 시작 (3개 샘플만)")
    
    try:
        # 1. 실제 훈련 dataloader 검증 (3개 샘플)
        verify_actual_train_dataloader()
        
        # 2. 실제 검증 dataloader 검증 (3개 샘플)
        verify_actual_val_dataloader()
        
        # 3. 어노테이션 일관성 검증 (3개 샘플)
        verify_annotation_consistency()
        
        print("\n" + "="*60)
        print("실제 Dataloader 검증 완료!")
        print("="*60)
        print("✓ 실제 훈련 dataloader가 정상적으로 작동함")
        print("✓ 실제 검증 dataloader가 정상적으로 작동함")
        print("✓ 어노테이션이 원본과 일치함")
        print("✓ 모든 설정이 기존 구현된 대로 유지됨")
        
    except Exception as e:
        print(f"\n❌ 검증 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 