
_base_ = [
    "../../_base_/datasets/pku-mmd/pku-mmd.py",  
    "../../_base_/models/actionformer.py",
]


scale_factor = 1
chunk_num = 512 * scale_factor // 16  


dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="avi"),
            dict(type="mmaction.DecordInit", num_threads=6),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=512,
                trunc_thresh=0.5,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
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
        pipeline=[
            dict(type="PrepareVideoInfo", format="avi"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="sliding_window",  
                trunc_len=512,
                scale_factor=scale_factor,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160), 
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="avi"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)

model = dict(
   
    rpn_head = dict(
    num_classes=51,
    prior_generator=dict(
        type="PointGenerator",
        strides=[4, 8, 16, 32, 64, 128],              
        regression_range=[                             
            (0, 8), (8, 16), (16, 32),
            (32, 64), (64, 128), (128, 10000)
        ],
    ),
    center_sample_radius=1.5,
    ),
    backbone=dict(
        type="VisionTransformerAdapter",
        img_size=160,
        patch_size=16,
        embed_dims=384,  
        depth=12,
        num_heads=6, 
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        num_frames=16,
        tubelet_size=2,
        use_mean_pooling=True,
        pretrained="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
        return_feat_map=True,
        with_cp=True,
        adapter_mlp_ratio=0.25,
        total_frames=512,
        adapter_index=list(range(12)),
        custom=dict(
            pretrain="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
            norm_eval=False,
            freeze_backbone=True,
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t -> b c t", reduction="mean"),
            ],
        ),
    ),
    projection=dict(
        in_channels=384,  
        max_seq_len=512,
        attn_cfg=dict(n_mha_win_size=-1),
    ),
)

solver = dict(
    train=dict(batch_size=16, num_workers=4), 
    val=dict(batch_size=8, num_workers=4),     
    test=dict(batch_size=4, num_workers=4),    
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[dict(name="adapter", lr=2e-4, weight_decay=0.05)],
        exclude=["backbone"],
    ),
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=120)

inference = dict(
    load_from_raw_predictions=False, 
    save_raw_prediction=False,
    score_thresh=0.3,  
)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.3,  
        max_seg_num=500,  
        multiclass=False, 
        voting_thresh=0.7,  
    ),
    save_dict=True,
)

workflow = dict(
    logging_interval=10,
    checkpoint_interval=2,
    val_loss_interval=5,      
    val_eval_interval=5,    
    val_start_epoch=10,       
    end_epoch=120,
)

work_dir = "work_dirs/e2e_pku_mmd_videomae_s_768x1_160_adapter"


evaluation = dict(
    type="mAP_PKU_MMD",
    subset="validation",  
    tiou_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
    ground_truth_filename="data/PKU-MMD/pku_val.json",  
)
