import argparse
import glob
import multiprocessing as mp
import os
import time
import json

import cv2
import numpy as np
import tqdm
import sys
sys.path.append(os.getcwd())
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "mask2former demo"


def parse_stuff_colors(stuff_colors_str):
    """
    解析传入的 JSON 格式的颜色配置字符串为字典，并将键转换为整数。
    
    Args:
        stuff_colors_str (str): JSON 格式的颜色配置字符串
    
    Returns:
        dict: 转换后的字典形式的 stuff_colors，键为整数
    """
    try:
        stuff_colors = json.loads(stuff_colors_str)
        # 将字典的键转换为整数
        stuff_colors = {int(k): v for k, v in stuff_colors.items()}
        return stuff_colors
    except json.JSONDecodeError:
        print("Invalid stuff_colors format. Please provide a valid JSON string.")
        return {}


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/data1/JM/code/mask2former/checkpoint_list/output_resnet50_only_color/config.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        default=['/data1/JM/code/mask2former/datasets/annotation_dataset_586_11_only_color/images/validation/*.png'],
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='/data1/JM/code/mask2former/金牌测试/result/fixed_color/test_color_result',
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        "--stuff-colors",
        type=str,
        help="Stuff colors as a JSON string, e.g., '{\"0\": [255, 255, 255], \"1\": [0, 255, 0], ...}'"
    )
    
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[
            "MODEL.WEIGHTS", "/data1/JM/code/mask2former/checkpoint_list/output_resnet50_only_color/model_0029499.pth"
        ],
        nargs=argparse.REMAINDER,  # 允许接收多个参数
    )
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    if args.stuff_colors:
        stuff_colors = parse_stuff_colors(args.stuff_colors)
    else:
        print('error: No stuff_colors provided.')
        exit()

    del args.stuff_colors  # 从 args 中移除 stuff_colors

    cfg = setup_cfg(args)
    os.makedirs(args.output, exist_ok=True)
    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img, stuff_colors)

            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            out_filename = os.path.join(args.output, os.path.basename(path).split('.')[0] + '.png')
            cv2.imwrite(out_filename, visualized_output)



# # Copyright (c) Facebook, Inc. and its affiliates.
# # Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
# import argparse
# import glob
# import multiprocessing as mp
# import os

# # fmt: off
# import sys
# sys.path.insert(1, os.path.join(sys.path[0], '..'))
# # fmt: on

# import tempfile
# import time
# import warnings

# import cv2
# import numpy as np
# import tqdm
# import sys
# sys.path.append(os.getcwd())
# from detectron2.config import get_cfg
# from detectron2.data.detection_utils import read_image
# from detectron2.projects.deeplab import add_deeplab_config
# from detectron2.utils.logger import setup_logger

# from mask2former import add_maskformer2_config
# from predictor import VisualizationDemo


# # constants
# WINDOW_NAME = "mask2former demo"


# def setup_cfg(args):
#     # load config from file and command-line arguments
#     cfg = get_cfg()
#     add_deeplab_config(cfg)
#     add_maskformer2_config(cfg)
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     cfg.freeze()
#     return cfg


# def get_parser():
#     parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
#     parser.add_argument(
#         "--config-file",
#         default="/data1/JM/code/mask2former/checkpoint_list/output_resnet50_only_color/config.yaml",
#         metavar="FILE",
#         help="path to config file",
#     )
#     parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
#     parser.add_argument("--video-input", help="Path to video file.")
#     parser.add_argument(
#         "--input",
#         default=['/data1/JM/code/mask2former/datasets/annotation_dataset_586_11_only_color/images/validation/*.png'],
#         nargs="+",
#         help="A list of space separated input images; "
#         "or a single glob pattern such as 'directory/*.jpg'",
#     )
#     parser.add_argument(
#         "--output",
#         default='/data1/JM/code/mask2former/金牌测试/result/fixed_color/test_color_result',
#         help="A file or directory to save output visualizations. "
#         "If not given, will show output in an OpenCV window.",
#     )

#     parser.add_argument(
#         "--confidence-threshold",
#         type=float,
#         default=0.5,
#         help="Minimum score for instance predictions to be shown",
#     )
#     parser.add_argument(
#         "--opts",
#         help="Modify config options using the command-line 'KEY VALUE' pairs",
#         default=[
#             "MODEL.WEIGHTS", "/data1/JM/code/mask2former/checkpoint_list/output_resnet50_only_color/model_0029499.pth"
#         ],
#         nargs=argparse.REMAINDER,  # 允许接收多个参数
#     )
#     return parser

# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)
#     args = get_parser().parse_args()
#     setup_logger(name="fvcore")
#     logger = setup_logger()
#     logger.info("Arguments: " + str(args))

#     cfg = setup_cfg(args)
#     os.makedirs(args.output, exist_ok=True)
#     demo = VisualizationDemo(cfg)
#     stuff_colors= {
#                     1-1: (255, 255, 255),   
#                     2-1: (0, 255, 0),  
#                     3-1: (255, 0, 0),
#                     4-1: (255, 191, 186), 
#                     5-1: (3, 152, 255),  
#                     6-1: (254, 153, 3), 
#                     7-1: (0, 255, 255),    
#                     8-1: (127, 127, 127), 
#                     9-1: (215, 210, 210), 
#                     10-1: (98, 126, 91),  
#                     11-1: (0, 0, 0),         
#                 } 
    
#     if args.input:
#         if len(args.input) == 1:
#             args.input = glob.glob(os.path.expanduser(args.input[0]))
#             assert args.input, "The input path(s) was not found"
#         for path in tqdm.tqdm(args.input, disable=not args.output):
#             # use PIL, to be consistent with evaluation
#             img = read_image(path, format="BGR")
#             start_time = time.time()
#             predictions, visualized_output = demo.run_on_image(img, stuff_colors)

#             logger.info(
#                 "{}: {} in {:.2f}s".format(
#                     path,
#                     "detected {} instances".format(len(predictions["instances"]))
#                     if "instances" in predictions
#                     else "finished",
#                     time.time() - start_time,
#                 )
#             )

