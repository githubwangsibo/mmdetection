# import mmcv
# import os
# import numpy as np
# from mmcv.runner import load_checkpoint
# from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, show_result_pyplot


if __name__ == "__main__": 
    config_file = './configs/deepfashion/mask_rcnn_r50_fpn_15e_deepfashion.py'
    checkpoint_file = './models/mask_rcnn_r50_fpn_15e_deepfashion_20200329_192752.pth'

    model = init_detector(config_file, checkpoint_file, device='cuda:1')

    img = '../../demo_images/fashionWOMENBlouses_Shirtsid0000255001_4full.jpg'
    result = inference_detector(model, img)
    show_result_pyplot(model=model, img=img, result=result, out_file='testOut.jpg', score_thr=0.01)

    # print(result)
