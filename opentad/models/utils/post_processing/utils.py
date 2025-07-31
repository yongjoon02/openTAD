import os
import pickle
import torch
import torch.nn.functional as F


def boundary_choose(score):
    mask_high = score > score.max(dim=1, keepdim=True)[0] * 0.5
    mask_peak = score == F.max_pool1d(score, kernel_size=3, stride=1, padding=1)
    mask = mask_peak | mask_high
    return mask


def save_predictions(predictions, metas, folder):
    for idx in range(len(metas)):
        video_name = metas[idx]["video_name"]

        file_path = os.path.join(folder, f"{video_name}.pkl")
        prediction = [data[idx] for data in predictions]
        with open(file_path, "wb") as outfile:
            pickle.dump(prediction, outfile, pickle.HIGHEST_PROTOCOL)


def load_single_prediction(metas, folder):
    """Should not be used for sliding window. Since we saved the files with video name, and sliding window will have multiple files with the same name."""
    predictions = []
    for idx in range(len(metas)):
        video_name = metas[idx]["video_name"]
        file_path = os.path.join(folder, f"{video_name}.pkl")
        with open(file_path, "rb") as infile:
            prediction = pickle.load(infile)
        predictions.append(prediction)

    batched_predictions = []
    for i in range(len(predictions[0])):
        data = torch.stack([prediction[i] for prediction in predictions])
        batched_predictions.append(data)
    return batched_predictions


def load_predictions(metas, infer_cfg):
    if "fuse_list" in infer_cfg.keys():
        predictions = []
        predictions_list = [load_single_prediction(metas, folder) for folder in infer_cfg.fuse_list]
        for i in range(len(predictions_list[0])):
            predictions.append(torch.stack([pred[i] for pred in predictions_list]).mean(dim=0))
        return predictions
    else:
        return load_single_prediction(metas, infer_cfg.folder)


def convert_to_seconds(segments, meta):
    # 디버깅: 변환 전 상태 확인
    if segments.shape[0] > 0 and "video_name" in meta:
        print(f"DEBUG: convert_to_seconds - video: {meta['video_name']}")
        print(f"DEBUG: convert_to_seconds - segments before: {segments[:3]}")
        print(f"DEBUG: convert_to_seconds - meta fps: {meta.get('fps', 'NOT_FOUND')}")
        print(f"DEBUG: convert_to_seconds - meta duration: {meta.get('duration', 'NOT_FOUND')}")
    
    # PKU-MMD 특별 처리: 프레임 단위 유지 (시간 변환 안함)
    if "video_name" in meta and meta.get("fps", 0) == 30.0:  # PKU-MMD 특성 확인 (30fps)
        # PKU-MMD는 프레임 단위로 그대로 유지 (시간 변환 안함)
        # segments는 이미 프레임 단위이므로 그대로 사용
        if segments.shape[0] > 0:
            print(f"DEBUG: convert_to_seconds - PKU-MMD mode, segments after (frame): {segments[:3]}")
    elif meta["fps"] == -1:  # resize setting, like in anet / hacs
        segments = segments / meta["resize_length"] * meta["duration"]
    else:  # sliding window / padding setting, like in thumos / ego4d
        snippet_stride = meta["snippet_stride"]
        offset_frames = meta["offset_frames"]
        window_start_frame = meta["window_start_frame"] if "window_start_frame" in meta.keys() else 0
        segments = (segments * snippet_stride + window_start_frame + offset_frames) / meta["fps"]

    # truncate all boundaries within [0, duration] (duration도 프레임 단위로 처리)
    if segments.shape[0] > 0:
        segments[segments <= 0.0] *= 0.0
        # duration을 프레임 단위로 변환 (초 * fps)
        max_frames = meta["duration"] * meta["fps"] if meta.get("fps", 0) == 30.0 else meta["duration"]
        segments[segments >= max_frames] = segments[segments >= max_frames] * 0.0 + max_frames
        if "video_name" in meta:
            print(f"DEBUG: convert_to_seconds - final segments (frame): {segments[:3]}")
    return segments
