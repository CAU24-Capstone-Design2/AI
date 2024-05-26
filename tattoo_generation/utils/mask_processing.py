import argparse
from copy import deepcopy

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


def extract_tattoo_mask(tattoo: Image.Image, tattoo_threshold: int=60) -> Image.Image:
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
    tattoo_mask = extract_tattoo_mask(tattoo=tattoo).convert('1')
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
    mask_shape = crop_mask.shape
    
    for row in tqdm(range(0, tattoo_mask.shape[0] - crop_mask.shape[0], 5), desc=f"Searching mask coord (Scale: {scale})"):
        for col in range(0, tattoo_mask.shape[1] - crop_mask.shape[1], 5):
            score = np.sum(crop_mask * tattoo_mask[row:row + crop_mask.shape[0], col:col + crop_mask.shape[1]]) / num_white_pixel
            best = (scale, score, [col, row, (col + crop_mask.shape[1]), (row + crop_mask.shape[0])]) if best[1] <= score else best
    
    print(f'best score: {best[1]}')        

    # Goal: lower bound <= coverage scorer <= upper bound
    if best[1] < lower_bound: 
        while 0.7 < scale and best[1] <= lower_bound:
            scale -= 0.1
            small_mask_shape = (int(mask_shape[0] * scale), int(mask_shape[1] * scale))
            crop_mask = Image.fromarray(crop_mask, mode='L').resize((small_mask_shape[1], small_mask_shape[0]))
            crop_mask = np.array(crop_mask)
            num_white_pixel = np.sum(crop_mask)
            
            for row in tqdm(range(0, tattoo_mask.shape[0] - crop_mask.shape[0], 5), desc=f"Searching mask coord (Scale: {scale})"):
                for col in range(0, tattoo_mask.shape[1] - crop_mask.shape[1], 5):
                    score = np.sum(crop_mask * tattoo_mask[row:row + crop_mask.shape[0], col:col + crop_mask.shape[1]]) / num_white_pixel
                    best = (scale, score, [col, row, (col + crop_mask.shape[1]), (row + crop_mask.shape[0])]) if best[1] <= score else best

            print(f'best score: {best[1]}')        
    else:
        while scale < 1.2 and upper_bound <= best[1]:
            best = (best[0], lower_bound, best[2])
            scale += 0.1
            big_mask_shape = (int(mask_shape[0] * scale), int(mask_shape[1] * scale))

            if 1024 < big_mask_shape[0] or 1024 < big_mask_shape[1]:
                break
            
            crop_mask = Image.fromarray(crop_mask, mode='L').resize((big_mask_shape[1], big_mask_shape[0]))
            crop_mask = np.array(crop_mask)
            num_white_pixel = np.sum(crop_mask)

            for row in tqdm(range(0, tattoo_mask.shape[0] - crop_mask.shape[0], 5), desc=f"Searching mask coord (Scale: {scale})"):
                for col in range(0, tattoo_mask.shape[1] - crop_mask.shape[1], 5):
                    score = np.sum(crop_mask * tattoo_mask[row:row + crop_mask.shape[0], col:col + crop_mask.shape[1]]) / num_white_pixel
                    best = (scale, score, [col, row, (col + crop_mask.shape[1]), (row + crop_mask.shape[0])]) if best[1] <= score else best

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
    tat_mask = extract_tattoo_mask(tattoo=tattoo).convert('1')
    
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


def extract_skin_mask(image: Image.Image, bbox: List[int]) -> Image.Image:
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

def synthesis_tattoo(
        tattoo: Image.Image, 
        tat_mask: Image.Image, 
        skin_mask: Image.Image, 
        wnd_tat_bbox: List[int], 
        wnd_bbox: List[int], 
        scale: int) -> Image.Image:
    width = int(tattoo.size[0] * (1.0 / scale))
    height = int(tattoo.size[1] * (1.0 / scale))
    resize_tattoo = tattoo.resize((width, height))
    wnd_tat_bbox = (np.array(wnd_tat_bbox) * (1.0 / scale)).astype(np.intc)

    skin_image_frame = (0, 0, skin_mask.size[0], skin_mask.size[1])

    dx = wnd_bbox[0] - wnd_tat_bbox[0]
    dy = wnd_bbox[1] - wnd_tat_bbox[1]
    skin_mask_frame = (max(0, dx), max(0, dy), \
                       min(skin_image_frame[2], resize_tattoo.size[0] + dx), \
                        min(skin_image_frame[3], resize_tattoo.size[1] + dy))

    dx = wnd_tat_bbox[0] - wnd_bbox[0]
    dy = wnd_tat_bbox[1] - wnd_bbox[1]
    tat_mask_frame = (max(0, dx), max(0, dy), \
                      min(resize_tattoo.size[0], skin_image_frame[2] + dx), \
                      min(resize_tattoo.size[1], skin_image_frame[3] + dy))

    intersection_mask = ImageChops.logical_and(
        skin_mask.crop(skin_mask_frame).convert('1'),
        tat_mask.crop(tat_mask_frame).convert('1')
    )

    intersection_tat_mask = Image.new('RGB', tat_mask.size)
    intersection_tat_mask.paste(intersection_mask, tat_mask_frame)

    intersection_tat = Image.new('RGB', tattoo.size)
    intersection_tat.paste(tattoo, (0, 0), intersection_tat_mask.convert("L"))
    intersection_tat = intersection_tat.crop(tat_mask_frame)

    tat_on_skin = Image.new('RGB', skin_mask.size)
    tat_on_skin.paste(intersection_tat, skin_mask_frame)

    return tat_on_skin

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

    image = load_image(input_path).resize((1024, 1024))
    wnd_mask = load_image(mask_path).resize((1024, 1024))   # type: PIL.Image
    tattoo = load_image(tattoo_path).resize((1024, 1024))   # type: PIL.Image

    wnd_bbox, crop_mask = extract_bbox(wnd_mask)
    print(f'bbox: {wnd_bbox}')

    scale, score, wnd_on_tat_bbox = search_mask_coord(tattoo, crop_mask)
    print(f'Moved mask information')
    print(f'Scale: {scale}')
    print(f'Coverage score: {score}')
    print(f'Coordinate: {wnd_on_tat_bbox}\n')

    print(f"rescaled coord: {(np.array(wnd_on_tat_bbox) * (1.0 / scale)).astype(np.intc)}")
    print(f'wnd_bbox: {wnd_bbox}\n')

    tat_mask = extract_tattoo_mask(tattoo)
    tat_mask.save("tattoo_mask.jpg")
    skin_mask = extract_skin_mask(image, wnd_bbox)
    skin_mask.save("skin_mask.jpg")
    tat_on_skin = synthesis_tattoo(tattoo=tattoo, tat_mask=tat_mask, skin_mask=skin_mask, wnd_bbox=wnd_bbox, wnd_tat_bbox=wnd_on_tat_bbox, scale=scale)
    tat_on_skin.save('tat_on_skin.jpg')


