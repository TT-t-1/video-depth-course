CUDA_VISIBLE_DEVICES=1 python generate_flow.py \
--inference_dir /media/hdd1/kxian/data/depth/TartanAir \
--output_path /media/hdd1/kxian/data/depth/TartanAir_flow \
--resume pretrained/gmflow_things-e9887eda.pth \
--pred_bidir_flow --fwd_bwd_consistency_check
