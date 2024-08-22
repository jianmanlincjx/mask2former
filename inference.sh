CUDA_VISIBLE_DEVICES=3 python ./demo/demo.py \
--config-file /data1/JM/code/mask2former/configs/floorplan/maskformer2_R50_bs16_160k.yaml \
--input "/data1/JM/code/mask2former/datasets/FloorPlan/images/training/*.png" \
--confidence-threshold 0 --output "/data1/JM/code/mask2former/result_floorplan_trainset" \
--opts MODEL.WEIGHTS /data1/JM/code/mask2former/output/model_0004999.pth