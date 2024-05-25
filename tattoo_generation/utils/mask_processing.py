import argparse

import numpy as np
from PIL import Image, ImageChops
import cv2
from tqdm import tqdm
from typing import Tuple, List

from diffusers.utils import load_image, make_image_grid

# SAM: segmentation for body part
import torch
from torchvision.transforms.functional import to_pil_image
from transformers import SamModel, SamProcessor



def extract_bbox(back_and_mask: Image.Image) -> Tuple[List[int], Image.Image]:
    mask = np.array(back_and_mask.convert('L'))

    # find contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # extract bounding box
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append((x, y, x + w, y + h))  

    bboxes = np.array(bboxes)
    bbox = [
        np.min(bboxes[:, 0]), np.min(bboxes[:, 1]),
        np.max(bboxes[:, 2]), np.max(bboxes[:, 3])
    ]
    mask = Image.fromarray(mask[bbox[1]:bbox[3], bbox[0]:bbox[2]])

    return bbox, mask


def make_tattoo_mask(tattoo: Image.Image, tattoo_threshold: int=60) -> Image.Image:
    # 1. convert tattoo to gray scale image
    gray_tattoo = np.array(tattoo)
    gray_tattoo = cv2.cvtColor(gray_tattoo, cv2.COLOR_RGB2GRAY)

    # 2. histogram equlization
    gray_tattoo = cv2.equalizeHist(gray_tattoo)

    # 3. thresholding (threshold = 50)
    _, gray_tattoo = cv2.threshold(gray_tattoo, tattoo_threshold, 255, cv2.THRESH_BINARY_INV)

    # 4. make tattoo mask
    contours, _ = cv2.findContours(gray_tattoo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. calculate contour
    contour = max(contours, key=cv2.contourArea)
    contour_img = np.zeros_like(gray_tattoo)
    tattoo_mask = cv2.drawContours(contour_img, [contour], -1, (255), thickness=cv2.FILLED)

    return Image.fromarray(tattoo_mask)


def search_mask_coord(tattoo: Image.Image, mask: Image.Image) -> Tuple[float, float, List[int]]:
    # 1. make tattoo mask
    tattoo_mask = make_tattoo_mask(tattoo=tattoo).convert('1')
    tattoo_mask = np.array(tattoo_mask)

    # 2. mask thresholding (threshold = 1, over => 1, o.w => 0)
    mask_threshold = 0
    crop_mask = np.array(mask)
    _, crop_mask = cv2.threshold(crop_mask, mask_threshold, 1, cv2.THRESH_BINARY_INV)
    num_white_pixel = np.sum(crop_mask)

    # 3. search 
    lower_bound, upper_bound = 0.85, 0.9
    scale = 1.0 
    best = [1.0, 0.0, []]   # scale, coverage percentage, coordinate 
    mask_size = crop_mask.shape
    
    for row in tqdm(range(0, tattoo_mask.shape[0] - mask_size[0], 5), desc=f"Searching mask coord (Scale: {scale})"):
        for col in range(0, tattoo_mask.shape[1] - mask_size[1], 5):
            score = np.sum(crop_mask * tattoo_mask[row:row + mask_size[0], col:col + mask_size[1]]) / num_white_pixel
            best = (scale, score, [row, col, row + mask_size[0], col + mask_size[1]]) if best[1] < score else best

    print(f'best score: {best[1]}')        

    # Goal: lower bound <= coverage scorer <= upper bound
    if best[1] < lower_bound: 
        while 0.7 <= scale and best[1] <= lower_bound:
            scale -= 0.1
            small_mask_size = (int(mask_size[0] * scale), int(mask_size[1] * scale))
            crop_mask = Image.fromarray(crop_mask, mode='L').resize((small_mask_size[1], small_mask_size[0]))
            crop_mask = np.array(crop_mask)
            num_white_pixel = np.sum(crop_mask)
            
            for row in tqdm(range(0, tattoo_mask.shape[0] - small_mask_size[0], 5), desc=f"Searching mask coord (Scale: {scale})"):
                for col in range(0, tattoo_mask.shape[1] - small_mask_size[1], 5):
                    score = np.sum(crop_mask * tattoo_mask[row:row + small_mask_size[0], col:col + small_mask_size[1]]) / num_white_pixel
                    best = (scale, score, [row, col, row + small_mask_size[0], col + small_mask_size[1]]) if best[1] < score else best

            print(f'best score: {best[1]}')        
    else:
        while scale <= 1.2 and upper_bound <= best[1]:
            best = (best[0], lower_bound, best[2])
            scale += 0.1
            big_mask_size = (int(mask_size[0] * scale), int(mask_size[1] * scale))

            if 1024 < big_mask_size[0] or 1024 < big_mask_size[1]:
                break
            
            crop_mask = Image.fromarray(crop_mask, mode='L').resize((big_mask_size[1], big_mask_size[0]))
            crop_mask = np.array(crop_mask)
            num_white_pixel = np.sum(crop_mask)

            for row in tqdm(range(0, tattoo_mask.shape[0] - big_mask_size[0], 5), desc=f"Searching mask coord (Scale: {scale})"):
                for col in range(0, tattoo_mask.shape[1] - big_mask_size[1], 5):
                    score = np.sum(crop_mask * tattoo_mask[row:row + big_mask_size[0], col:col + big_mask_size[1]]) / num_white_pixel
                    best = (scale, score, [row, col, row + big_mask_size[0], col + big_mask_size[1]]) if best[1] < score else best

            print(f'best score: {best[1]}')        

    return best


def extract_edge(crop_mask: Image.Image) -> Image.Image:
    # edge extract (black edge and white background)
    edge = np.array(crop_mask)
    edge = cv2.Canny(edge, threshold1=100, threshold2=200)
    
    # edge dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge = cv2.dilate(edge, kernel, iterations=1)
    edge = 255 - edge
    edge = Image.fromarray(edge, mode='L')

    return edge


def overlay_edge(tattoo: Image.Image, edge: Image.Image, coord: List[int]) -> Image.Image:
    tat_mask = make_tattoo_mask(tattoo=tattoo).convert('1')
    
    moved_edge = Image.new("L", (1024, 1024), color="white")
    moved_edge.paste(edge.resize((coord[3] - coord[1], coord[2] - coord[0])), (coord[1], coord[0]))
    moved_edge = moved_edge.convert('1')
    moved_edge = ImageChops.logical_or(tat_mask, moved_edge)

    moved_edge = np.array(moved_edge)
    overlay = np.array(tattoo)
    overlay[moved_edge == False] = 0

    return Image.fromarray(overlay)


def extract_wound(input:Image.Image, mask: Image.Image) -> Image.Image:
    mask = mask.convert('L')
    result = Image.composite(input, Image.new('RGB', input.size), mask)
    
    return result


def extract_body_mask(image: Image.Image, bbox: List[int]) -> Image.Image:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    x = (bbox[0] + bbox[2]) // 2
    y = (bbox[1] + bbox[3]) // 2
    input_points = [[[x, y]]]  # 2D location of a window in the image

    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    
    inputs = processor(image, input_points=input_points, return_tensors="pt").to(device)

    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # masks.shape = (#masks, #channels, height, width)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )

    scores = outputs.iou_scores[0][0]
    idxs = [int(idx) for idx in scores.argsort(descending=True)]
    for idx in idxs:
        mask = masks[0][0][idx]
        mask = mask.to(torch.uint8) * 255
        
        mask = to_pil_image(mask)

        # check body part bounding box is larger than wound mask bounding box 
        body_bbox, _ = extract_bbox(mask)   # bbox: (left, top, right, bottom)
        if (body_bbox[0] <= bbox[0]) and (body_bbox[1] <= bbox[1]) and \
            (bbox[2] <= body_bbox[2]) and (bbox[3] <= body_bbox[3]):
            break
    
    return mask



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user', type=str)
    parser.add_argument('-f', '--file', type=str)

    args = parser.parse_args()

    user = args.user
    filename = args.file
    input_path = f'/home/cvmlserver11/junhee/scart/images/{user}/inputs/{filename}'
    mask_path = f'/home/cvmlserver11/junhee/scart/images/{user}/masks/{filename}'
    tattoo_path = f'/home/cvmlserver11/junhee/scart/images/{user}/tattoos/{filename}'

    tattoo = load_image(tattoo_path).resize((1024, 1024))   # type: PIL.Image
    mask = load_image(mask_path).resize((1024, 1024))   # type: PIL.Image

    bbox, crop_mask = extract_bbox(mask)
    print(f'bbox: {bbox}')
    print(type(crop_mask))
    print(crop_mask.size)

    make_tattoo_mask(tattoo).save('tattoo_mask.jpg')

    scale, score, coord = search_mask_coord(tattoo, crop_mask)
    print(f'Moved mask information')
    print(f'Scale: {scale}')
    print(f'Coverage score: {score}')
    print(f'Coordinate: {coord}')

    edge = extract_edge(crop_mask)  # type: PIL.Image
    print(f'edge size: {edge.size}')

    overlay = overlay_edge(tattoo, edge, coord)

    input_image = load_image(input_path).resize((1024, 1024))
    body_mask = extract_body_mask(input_image, bbox)

    body_mask.save('body_mask.jpg')

    # only_filename = filename.split('.')[0]
    # image_list = [tattoo, mask, overlay]
    # make_image_grid(image_list, rows=1, cols=len(image_list)).save(f'../results/{only_filename}_overlay_result.png')

    # input = load_image(input_path).resize((1024, 1024))
    # print(input.size)
    # mask = load_image(mask_path).resize((1024, 1024))
    # print(mask.size)
    # wound = extract_wound(input, mask)
    # wound.save(mask_path)

    # tat_mask = make_tattoo_mask(tattoo)
    # tat_mask = np.array(tat_mask)

    # print(np.unique(tat_mask, return_counts=True))

