import argparse
import os 
import sys
import cv2
from PIL import Image
import numpy as np 
import warnings

sys.path.append("./detectron2")
warnings.filterwarnings('ignore')

import torch
import torchvision 

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def image_resize(file_path, target_size):
    img = cv2.imread(file_path)
    h, w, _ = img.shape

    if w < h:   # vertical image
        w = int((w / h) * target_size)
        h = target_size
    else:       # horizontal image
        h = int((h / w) * target_size)
        w = target_size

    img = cv2.resize(img, dsize=(w, h))   # rescaled image

    result = np.zeros((target_size, target_size, 3), dtype=np.float32)
    result[:h, :w, :3] = img

    cv2.imwrite(file_path, result)

def merge_masks(masks):
    c, h, w = masks.size()
    result = masks[0]

    for i in range(1, c):
        result = torch.logical_or(result, masks[i])

    return result.unsqueeze(dim=0).to(torch.float32)


def crop_bboxes(pred):
    pred_boxes = pred['instances'].pred_boxes
    print(pred_boxes.shape)
    
if __name__ == '__main__': 
    # argument parsing, 이미지 경로
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user', type=str, help='enter user id')
    parser.add_argument('-f', '--filename', type=str, help='enter file name')
    parser.add_argument('-d', '--debug', action='store_true', help='debugging mode')
    args = parser.parse_args()    

    # image 경로 정의 
    base_dir = '/home/cvmlserver11/junhee/scart/images'
    userid = args.user
    filename = args.filename
    image_path = f'{base_dir}/{userid}/inputs/{filename}'

    # debug
    if args.debug:
        print('\n[image info]')
        print(f'base_dir: {base_dir}')
        print(f'user_id: {userid}')
        print(f'filename: {filename}')
        print(f'image path: {image_path}')
    
    # config 불러오기 + pretrained model weight 가져오기
    cfg = get_cfg() 
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.OUTPUT_DIR = './detectron2/output'
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "wound_seg_310.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

    # predictor 생성
    predictor = DefaultPredictor(cfg)

    # segmentation  
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    pred = predictor(image)
    pred_masks = pred['instances'].pred_masks

    if args.debug:
        print('\n[prediction info]')
        print(f'type: {type(image)}')
        print(f'predicted mask shape: {pred_masks.size()}\n')

    # 하나로 합친 mask 반환
    merged_mask = merge_masks(masks=pred_masks)

    if args.debug:
        print('\n[merged mask info]')
        print(f'type: {type(merged_mask)}')
        print(f'dtype: {merged_mask.dtype}')
        print(f'shape: {merged_mask.size()}\n')

    # mask 저장 경로 정의
    mask_dir = f'{base_dir}/{userid}/masks'
    mask_path = f'{mask_dir}/{filename}'
    
    os.makedirs(mask_dir, exist_ok=True)

    if args.debug:
        print('[output(mask) path info]')
        print(f'type: {type(base_dir)}')
        print(f'base dir: {base_dir}')
        print(f'mask dir: {mask_dir}')
        print(f'mask path: {mask_path}')
        print(f'max value: {merged_mask.max()}\n')

    # save
    cv2.imwrite(image_path, image)
    torchvision.utils.save_image(merged_mask, mask_path)
