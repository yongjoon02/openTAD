import numpy as np
from copy import deepcopy
from .base import SlidingWindowDataset, PaddingDataset, filter_same_annotation
from .builder import DATASETS
import json

@DATASETS.register_module()
class PkuSlidingDataset(SlidingWindowDataset):
    def get_dataset(self):
        """PKU-MMD JSON 구조에 맞게 오버라이드"""
        with open(self.ann_file, "r") as f:
            anno_database = json.load(f)  
        
        # 배열을 딕셔너리로 변환
        video_dict = {}
        for video_info in anno_database:
            video_name = video_info["video_name"]
            # subset 정보 추가 (train/validation/test 구분)
            if "train" in self.subset_name.lower():
                video_info["subset"] = "training"
            elif "val" in self.subset_name.lower():
                video_info["subset"] = "validation"
            else:
                video_info["subset"] = "testing"
            video_dict[video_name] = video_info
        
        # some videos might be missed in the features or videos, we need to block them
        if self.block_list != None:
            if isinstance(self.block_list, list):
                blocked_videos = self.block_list
            else:
                with open(self.block_list, "r") as f:
                    blocked_videos = [line.rstrip("\n") for line in f]
        else:
            blocked_videos = []

        self.data_list = []
        for video_name, video_info in video_dict.items():
            if (video_name in blocked_videos) or (video_info["subset"] not in self.subset_name):
                continue

            # get the ground truth annotation
            if self.test_mode:
                video_anno = {}
            else:
                video_anno = self.get_gt(video_info)
                if video_anno == None:  # have no valid gt
                    continue

            tmp_data_list = self.split_video_to_windows(video_name, video_info, video_anno)
            self.data_list.extend(tmp_data_list)
        assert len(self.data_list) > 0, f"No data found in {self.subset_name} subset."

    def get_gt(self, video_info, thresh=0.0):
        gt_segments, gt_labels = [], []
        for anno in video_info["annotations"]:
            # JSON의 segment가 [start_frame, end_frame]인 프레임 단위 (30fps 기준)
            start_frame, end_frame = anno["segment"]
            if (not self.filter_gt) or (end_frame - start_frame > thresh):
                gt_segments.append([start_frame, end_frame])
                gt_labels.append(self.class_map.index(anno["label"]))
        if not gt_segments:
            return None
        ann = dict(
            gt_segments=np.array(gt_segments, dtype=np.float32),
            gt_labels=np.array(gt_labels, dtype=np.int32),
        )
        return filter_same_annotation(ann)

    def __getitem__(self, index):
        video_name, video_info, video_anno, window_centers = self.data_list[index]
        
        if video_anno:
            video_anno = deepcopy(video_anno)
            original_segments = video_anno["gt_segments"].copy()

            video_anno["gt_segments"] = (
                video_anno["gt_segments"]
                - window_centers[0]
                - self.offset_frames
            ) / self.snippet_stride

        data = dict(
            video_name=video_name,
            data_path=self.data_path,
            window_size=self.window_size,
            feature_start_idx=int(window_centers[0] / self.snippet_stride),
            feature_end_idx=int(window_centers[-1] / self.snippet_stride),
            sample_stride=self.sample_stride,
            fps=30.0,  # 10.0에서 30.0으로 변경
            snippet_stride=self.snippet_stride,
            window_start_frame=window_centers[0],
            duration=video_info["frame"], 
            offset_frames=self.offset_frames,
            **video_anno,
        )
        return self.pipeline(data)

@DATASETS.register_module()
class PkuPaddingDataset(PaddingDataset):
    def get_dataset(self):
        """PKU-MMD JSON 구조에 맞게 오버라이드"""
        with open(self.ann_file, "r") as f:
            anno_database = json.load(f) 
        
        # 배열을 딕셔너리로 변환
        video_dict = {}
        for video_info in anno_database:
            video_name = video_info["video_name"]
            if "subset" not in video_info:
                if "train" in self.subset_name.lower():
                    video_info["subset"] = "training"
                elif "val" in self.subset_name.lower():
                    video_info["subset"] = "validation"
                else:
                    video_info["subset"] = "testing"
            video_dict[video_name] = video_info
        

        if self.block_list != None:
            if isinstance(self.block_list, list):
                blocked_videos = self.block_list
            else:
                with open(self.block_list, "r") as f:
                    blocked_videos = [line.rstrip("\n") for line in f]
        else:
            blocked_videos = []

        self.data_list = []
        for video_name, video_info in video_dict.items():
            if (video_name in blocked_videos) or (video_info["subset"] not in self.subset_name):
                continue


            if self.test_mode:
                video_anno = {}
            else:
                video_anno = self.get_gt(video_info)
                if video_anno == None:  
                    continue

            self.data_list.append([video_name, video_info, video_anno])
        assert len(self.data_list) > 0, f"No data found in {self.subset_name} subset."

    def get_gt(self, video_info, thresh=0.0):
        gt_segments, gt_labels = [], []
        for anno in video_info["annotations"]:
            start_frame, end_frame = anno["segment"]
            if (not self.filter_gt) or (end_frame - start_frame > thresh):
                gt_segments.append([start_frame, end_frame])
                gt_labels.append(self.class_map.index(anno["label"]))
        if not gt_segments:
            return None
        ann = dict(
            gt_segments=np.array(gt_segments, dtype=np.float32),
            gt_labels=np.array(gt_labels, dtype=np.int32),
        )
        return filter_same_annotation(ann)

    def __getitem__(self, index):
        video_name, video_info, video_anno = self.data_list[index]
        
        if video_anno:
            video_anno = deepcopy(video_anno)
            # 어노테이션을 snippet_stride로 나누어 샘플링된 인덱스로 변환
            # PkuSlidingDataset과 일관성 유지 - window 시작점은 0으로 가정
            video_anno["gt_segments"] = (
                video_anno["gt_segments"] - self.offset_frames
            ) / self.snippet_stride

        # sliding_window 메서드에서 필요한 feature_start_idx와 feature_end_idx 계산
        window_size = getattr(self, 'window_size', 512)
        total_frames = video_info["frame"]
        frame_stride = self.snippet_stride // 1  # scale_factor = 1
        frame_idxs = np.arange(0, total_frames, frame_stride)
        
        # 전체 프레임 인덱스를 window_size로 나누어 feature_start_idx와 feature_end_idx 계산
        feature_start_idx = 0
        feature_end_idx = min(len(frame_idxs) - 1, window_size - 1)
        
        data = dict(
            video_name=video_name,
            data_path=self.data_path,
            window_size=window_size,
            feature_start_idx=feature_start_idx,
            feature_end_idx=feature_end_idx,
            sample_stride=self.sample_stride,
            snippet_stride=self.snippet_stride,
            fps=30.0,  # 10.0에서 30.0으로 변경
            duration=video_info["frame"],  
            offset_frames=self.offset_frames,
            **video_anno,
        )
        return self.pipeline(data)
