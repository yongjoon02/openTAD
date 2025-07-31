#!/usr/bin/env python3
"""
모든 데이터 파일에 subset 필드 추가
"""
import json

# 파일별 subset 설정
files = [
    ("data/PKU-MMD/pku_train.json", "training"),
    ("data/PKU-MMD/pku_val.json", "validation"),
    ("data/PKU-MMD/pku_test.json", "testing"),
]

for file_path, subset_name in files:
    print(f"처리 중: {file_path}")
    
    # 데이터 로드
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # 각 비디오에 subset 필드 추가
    for video in data:
        video["subset"] = subset_name
    
    # 수정된 데이터 저장
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  - {len(data)}개 비디오에 subset='{subset_name}' 추가 완료")

print("\n모든 파일 처리 완료!") 