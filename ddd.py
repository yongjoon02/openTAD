#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/data_prep.py

PKU-MMD Phase 1 데이터 준비 스크립트:
 1. Actions.xlsx에서 class_map.txt 생성
 2. cross-subject.txt에서 train/validation split 추출
 3. Train_Label_PKU_final/*.txt 어노테이션 파일을 프레임 단위 JSON으로 변환

사용:
    cd <프로젝트 루트>
    python scripts/data_prep.py
"""

import os
import json
import pandas as pd

def generate_class_map(actions_file: str, output_txt: str) -> list:
    df = pd.read_excel(actions_file)
    if 'Action' in df.columns:
        class_names = df['Action'].dropna().astype(str).str.strip().tolist()
    else:
        class_names = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(name + '\n')
    print(f"[Class Map] {len(class_names)}개 클래스를 '{output_txt}'에 저장했습니다.")
    return class_names

def load_split(split_file: str) -> tuple[list, list]:
    train_ids, test_ids = [], []
    mode = None
    with open(split_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Training videos:'):
                mode = 'train'; continue
            if any(line.startswith(k) for k in ['Test videos:', 'Testing videos:', 'Validation videos:']):
                mode = 'test'; continue
            if ':' in line and mode is not None:
                mode = None; continue
            if mode == 'train' and line:
                train_ids += [v.strip() for v in line.split(',') if v.strip()]
            elif mode == 'test' and line:
                test_ids += [v.strip() for v in line.split(',') if v.strip()]
    print(f"[Split] 학습 비디오: {len(train_ids)}개, 테스트 비디오: {len(test_ids)}개")
    return train_ids, test_ids

def generate_annotations(video_ids: list, anno_dir: str, class_names: list, output_json: str):
    records = []
    for vid in video_ids:
        txt_path = os.path.join(anno_dir, f"{vid}.txt")
        if not os.path.exists(txt_path):
            print(f"[Warning] 어노테이션 파일 없음: {txt_path}")
            continue

        max_end_frame = 0
        annos = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(',')]
                if len(parts) < 4:
                    continue
                raw_idx     = int(parts[0])
                label_idx   = raw_idx - 1
                start_frame = int(parts[1])
                end_frame   = int(parts[2])
                conf        = float(parts[3])
                max_end_frame = max(max_end_frame, end_frame)

                if 0 <= label_idx < len(class_names):
                    annos.append({
                        "label":      class_names[label_idx],
                        "segment":    [start_frame, end_frame],
                        "confidence": conf
                    })
                else:
                    print(f"[Warning] 잘못된 레이블 인덱스 {raw_idx} (비디오 {vid})")

        duration = max_end_frame / 30.0  # 초 단위
        frame    = max_end_frame

        records.append({
            "video_name":  vid,
            "duration":    duration,
            "frame":       frame,
            "annotations": annos
        })

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[Annotations] {len(records)}개 비디오 데이터를 '{output_json}'에 저장했습니다.")

def main():
    actions_file            = "F:/dataset/pku-mmd/Split-20250716T021010Z-1-001/Split/Actions.xlsx"
    split_file              = "F:/dataset/pku-mmd/Split-20250716T021010Z-1-001/Split/cross-subject.txt"
    anno_dir                = r"F:\dataset\pku-mmd\Train_Label_PKU_final"
    class_map_txt           = "data/PKU-MMD/class_map.txt"
    annotations_train_json  = "data/PKU-MMD/annotations_train.json"
    annotations_test_json   = "data/PKU-MMD/annotations_test.json"

    class_names = generate_class_map(actions_file, class_map_txt)
    train_ids, test_ids = load_split(split_file)
    generate_annotations(train_ids, anno_dir, class_names, annotations_train_json)
    generate_annotations(test_ids,  anno_dir, class_names, annotations_test_json)

if __name__ == "__main__":
    main()
