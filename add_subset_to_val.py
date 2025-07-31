#!/usr/bin/env python3
"""
Validation 데이터에 subset 필드 추가
"""
import json

# Validation 데이터 로드
with open("data/PKU-MMD/pku_val.json", "r") as f:
    val_data = json.load(f)

# 각 비디오에 subset 필드 추가
for video in val_data:
    video["subset"] = "validation"

# 수정된 데이터 저장
with open("data/PKU-MMD/pku_val.json", "w") as f:
    json.dump(val_data, f, indent=2, ensure_ascii=False)

print(f"Validation 데이터 {len(val_data)}개에 subset 필드 추가 완료!") 