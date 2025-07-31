# _base_ = [
#     "../../_base_/datasets/activitynet-1.3/e2e_resize_768_1x224x224.py",  # 증강/파이프라인 참고용
#     "../../_base_/models/actionformer.py",
# ]

dataset_type = "PKUMMDLoader"
video_dir = "F:/dataset/pku-mmd/rgb"
anno_dir = "F:/dataset/pku-mmd/Train_Label_PKU_final"
split_file = "F:/dataset/pku-mmd/Split-20250716T021010Z-1-001/Split/cross-subject.txt"
actions_file = "F:/dataset/pku-mmd/Split-20250716T021010Z-1-001/Split/Actions.xlsx"

window_size = 512
scale_factor = 1
chunk_num = window_size * scale_factor // 16  # 768/16=48 chunks, since videomae takes 16 frames as input
img_size = 160
num_classes = 51 

dataset = dict(
    train=dict(
        type=dataset_type,
        video_dir=video_dir,
        anno_dir=anno_dir,
        split_file=split_file,
        actions_file=actions_file,
        split='train',
        fps=10,  # PKU-MMD는 10fps로 변경
        # SlidingWindowDataset 파라미터들
        feature_stride=1,  # 프레임 간격
        sample_stride=1,  # 샘플링 스트라이드
        offset_frames=0,  # 오프셋 프레임
        window_overlap_ratio=0.5,  # 윈도우 오버랩 비율 증가 (더 적은 window)
        ioa_thresh=0.5,  # IoA 임계값 낮춤 (더 많은 window 포함)
        pipeline=[
            dict(type="PrepareVideoInfo", format="avi"),
            dict(type="mmaction.DecordInit", num_threads=6),
            dict(type="LoadFrames", num_clips=1, method="random_trunc", trunc_len=window_size, trunc_thresh=0.5, crop_ratio=[0.9, 1.0], scale_factor=scale_factor),  # frame_interval 제거
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(img_size, img_size), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type=dataset_type,
        video_dir=video_dir,
        anno_dir=anno_dir,
        split_file=split_file,
        actions_file=actions_file,
        split='test',  
        window_size=window_size,  # window_size 추가
        test_mode=True,  # test_mode 추가
        fps=10,  # PKU-MMD는 10fps로 변경
        # SlidingWindowDataset 파라미터들
        feature_stride=1,  # 프레임 간격
        sample_stride=1,  # 샘플링 스트라이드
        offset_frames=0,  # 오프셋 프레임
        window_overlap_ratio=0.5,  # 윈도우 오버랩 비율 증가 (더 적은 window)
        ioa_thresh=0.5,  # IoA 임계값 낮춤 (더 많은 window 포함)
        pipeline=[
            dict(type="PrepareVideoInfo", format="avi"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),  # frame_interval 제거
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, img_size)),
            dict(type="mmaction.CenterCrop", crop_size=img_size),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),  # gt_segments, gt_labels 제거
            dict(type="Collect", inputs="imgs", keys=["masks"]),  # inputs를 "imgs"로 변경, gt 관련 키 제거
        ],
    ),
)

model = dict(
    type='ActionFormer',
    backbone=dict(
        type="VisionTransformerAdapter",
        img_size=img_size,
        patch_size=16,
        embed_dims=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        num_frames=16,
        tubelet_size=2,
        use_mean_pooling=False,
        pretrained=None,  # Adapter 모델은 별도 초기화 필요
        return_feat_map=True,
        with_cp=True,
        adapter_mlp_ratio=0.25,
        total_frames=window_size,
        adapter_index=list(range(12)),
        custom=dict(
            pretrain="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
            norm_eval=False,
            freeze_backbone=True,  # 백본을 freeze하여 어댑터만 학습
            post_processing_pipeline=[
                dict(type="Rearrange", keys=["feats"], ops="b n c t -> b (n c) t"),
            ],
        ),
    ),
    neck=dict(
        type='FPNIdentity',
        in_channels=512,
        out_channels=512,
        num_levels=6,
    ),
    projection=dict(
        type='Conv1DTransformerProj',
        in_channels=384,
        out_channels=512,
        max_seq_len=1024,
        arch=(2, 2, 5),
        attn_cfg=dict(n_head=4, n_mha_win_size=1),
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type='LN'),
        path_pdrop=0.1,
        use_abs_pe=False,
    ),
    rpn_head=dict(
        type='ActionFormerHead',
        in_channels=512,
        feat_channels=512,
        num_classes=num_classes,
        num_convs=2,
        center_sample='radius',
        center_sample_radius=1.5,
        cls_prior_prob=0.01,
        label_smoothing=0.0,
        loss=dict(
            cls_loss=dict(type='FocalLoss'),
            reg_loss=dict(type='DIOULoss'),
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        prior_generator=dict(
            type='PointGenerator',
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        ),
    ),
)

train_cfg = dict(max_epochs=60)

solver = dict(
    train=dict(batch_size=16, num_workers=4),  # 배치 사이즈 증가, num_workers 증가
    test=dict(batch_size=16, num_workers=4),   # 테스트도 동일하게
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
)

optimizer = dict(
    type="AdamW",
    lr=5e-5,  # 학습률을 1e-4에서 5e-5로 낮춤
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[dict(name="adapter", lr=2e-4, weight_decay=0.05)],  # adapter 학습률도 낮춤
        exclude=["backbone"],
    ),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=60)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    sliding_window=False,
    nms=dict(
        use_soft_nms=True,
        sigma=0.7,
        max_seg_num=2000,
        multiclass=True,
        voting_thresh=0.7,
    ),
    external_cls=None,
    save_dict=True,  # 결과 저장 활성화
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=2,
    end_epoch=60,
    # val_loss_interval 등 val 관련 키는 완전히 제거
)

work_dir = './work_dirs/e2e_pku_mmd_videomae_s_768x1_160_adapter'

evaluation = dict(
    type="mAP_PKU_MMD",  # PKU-MMD 전용 평가기 사용
    subset="test",
    tiou_thresholds=[0.1, 0.3, 0.5, 0.7],  # 원하는 정량지표로 수정
    ground_truth_filename=None,  # PKU-MMD는 개별 txt 파일 사용
    anno_dir="F:/dataset/pku-mmd/Train_Label_PKU_final",  # PKU-MMD 어노테이션 디렉토리
    split_file="F:/dataset/pku-mmd/Split-20250716T021010Z-1-001/Split/cross-subject.txt",  # PKU-MMD 분할 파일
    actions_file="F:/dataset/pku-mmd/Split-20250716T021010Z-1-001/Split/Actions.xlsx",  # PKU-MMD 액션 파일
)