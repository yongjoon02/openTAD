# configs/_base_/datasets/pku-mmd/pku-mmd.py

annotation_train = "data/PKU-MMD/pku_train.json"
annotation_val   = "data/PKU-MMD/pku_val.json"
annotation_test  = "data/PKU-MMD/pku_test.json"
class_map        = "data/PKU-MMD/class_map.txt"
video_dir        = "F:/dataset/pku-mmd/rgb"
block_list       = None

window_size = 512 

dataset = dict(
    train=dict(
        type="PkuPaddingDataset",
        ann_file=annotation_train,
        subset_name="training",
        class_map=class_map,
        data_path=video_dir,
        filter_gt=False,
        feature_stride=4,  
        sample_stride=1,
        pipeline=[
            dict(type="PrepareVideoInfo", format="avi"),
            dict(type="mmaction.DecordInit", num_threads=6),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=window_size,
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
        ],
    ),
    val=dict(
        type="PkuPaddingDataset",
        ann_file=annotation_val,
        subset_name="validation",
        class_map=class_map,
        data_path=video_dir,
        filter_gt=False,
        feature_stride=4, 
        sample_stride=1,
        pipeline=[
            dict(type="PrepareVideoInfo", format="avi"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=window_size,
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
        ],
    ),
    test=dict(
        type="PkuSlidingDataset",
        ann_file=annotation_test,
        subset_name="testing",
        class_map=class_map,
        data_path=video_dir,
        filter_gt=False,
        test_mode=True,
        window_size=window_size,
        window_overlap_ratio=0.5,
        feature_stride=4,  
        sample_stride=1,
        pipeline=[
            dict(type="PrepareVideoInfo", format="avi"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=1),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)

evaluation = dict(
    type="mAP_PKU_MMD",
    subset="test",
    tiou_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
    ground_truth_filename=annotation_test,
)
