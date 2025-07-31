import json, random, numpy as np
random.seed(42)

with open(r"F:\OpenTAD\data\PKU-MMD\annotations_train.json") as f: 
    train_data = json.load(f)

val_ratio = 0.12

# PKU-MMD는 video_name에서 subject 추출 (예: "0002-L" -> "0002")
subjects = sorted({item["video_name"].split("-")[0] for item in train_data})
val_subj = set(random.sample(subjects, int(len(subjects)*val_ratio)))

train_split, val_split = [], []
for item in train_data:
    subject = item["video_name"].split("-")[0]
    (val_split if subject in val_subj else train_split).append(item)

json.dump(train_split, open("pku_train.json","w"), indent=2)
json.dump(val_split,   open("pku_val.json","w"),   indent=2)

print(f"총 subject 수: {len(subjects)}")
print(f"Validation subject 수: {len(val_subj)}")
print(f"Train 데이터 수: {len(train_split)}")
print(f"Val 데이터 수: {len(val_split)}")
