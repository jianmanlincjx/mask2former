# CUDA_VISIBLE_DEVICES=2 python /data1/JM/code/mask2former/demo/demo.py \
# --config-file /data1/JM/code/mask2former/checkpoint_list/output_resnet50_black_w_color_1127/config.yaml \
# --input "/data1/JM/code/mask2former/datasets/annotation_dataset_586_11_black_w_color/images/validation/*.png" \
# --confidence-threshold 0 --output "/data1/JM/code/mask2former/金牌测试/result/Base_resnet50_only_black_1127" \
# --stuff-colors '{"0": [255, 255, 255], "1": [0, 255, 0], "2": [255, 0, 0], "3": [255, 191, 186], "4": [3, 152, 255], "5": [254, 153, 3], "6": [0, 255, 255], "7": [127, 127, 127], "8": [215, 210, 210], "9": [98, 126, 91], "10": [0, 0, 0]}' \
# --opts MODEL.WEIGHTS "/data1/JM/code/mask2former/checkpoint_list/output_resnet50_black_w_color_1127/model_0031499.pth"

CUDA_VISIBLE_DEVICES=2 python /data1/JM/code/mask2former/demo/demo.py \
--config-file /data1/JM/code/mask2former/checkpoint_list/output_resnet50_only_color/config.yaml \
--input "/data1/JM/code/mask2former/datasets/annotation_dataset_586_11_only_color/images/training/*.png" \
--confidence-threshold 0 \
--output "/data1/JM/code/mask2former/接口测试图像_training" \
--stuff-colors '{"0": [255, 255, 255], "1": [0, 255, 0], "2": [255, 0, 0], "3": [255, 191, 186], "4": [3, 152, 255], "5": [254, 153, 3], "6": [0, 255, 255], "7": [127, 127, 127], "8": [215, 210, 210], "9": [98, 126, 91], "10": [0, 0, 0]}' \
--opts MODEL.WEIGHTS "/data1/JM/code/mask2former/checkpoint_list/output_resnet50_only_color/model_0029499.pth"


# https://raw.githubusercontent.com/jianmanlincjx/temp/main/test_images/000384.png
# https://raw.githubusercontent.com/jianmanlincjx/temp/main/test_images_predict/000384.png
