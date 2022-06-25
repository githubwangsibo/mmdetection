# import mmcv
# import os
# import numpy as np
# from mmcv.runner import load_checkpoint
# from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, show_result_pyplot


if __name__ == "__main__": 
    config_file = './mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = './models/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'

    model = init_detector(config_file, checkpoint_file, device='cuda:1')

    # img_dir = 'data/VOCdevkit/VOC2007/JPEGImages/'
    # out_dir = 'results/'

    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)

    img = 'demo/demo.jpg'
    result = inference_detector(model, img)
    show_result_pyplot(model=model, img=img, result=result, out_file='testOut.jpg')

    print(result)
