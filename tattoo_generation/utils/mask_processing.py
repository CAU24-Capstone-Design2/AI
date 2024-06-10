import argparse
from copy import deepcopy

import numpy as np
from PIL import Image, ImageChops
import cv2
from tqdm import tqdm
from typing import Tuple, List

from diffusers.utils import load_image

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


def extract_tattoo_mask_v1(tattoo: Image.Image, tattoo_threshold: int=60) -> Image.Image:
    # 1. convert tattoo to gray scale image
    gray_tattoo = np.array(tattoo)
    gray_tattoo = cv2.cvtColor(gray_tattoo, cv2.COLOR_RGB2GRAY)

    # 2. thresholding
    _, tattoo_mask = cv2.threshold(gray_tattoo, tattoo_threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imwrite("debug/tattoo_mask.png", tattoo_mask)
    

    return Image.fromarray(tattoo_mask)


def extract_tattoo_mask(tattoo: Image.Image, tattoo_threshold: int=60) -> Image.Image:
    # 1. convert tattoo to gray scale image
    gray_tattoo = np.array(tattoo)
    gray_tattoo = cv2.cvtColor(gray_tattoo, cv2.COLOR_RGB2GRAY)

    # 2. histogram equlization
    gray_tattoo = cv2.equalizeHist(gray_tattoo)

    # 3. thresholding
    _, gray_tattoo = cv2.threshold(gray_tattoo, tattoo_threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imwrite("debug/tattoo_mask.png", gray_tattoo)
    
    # 4. make tattoo mask
    contours, _ = cv2.findContours(gray_tattoo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. calculate contour
    contour = max(contours, key=cv2.contourArea)
    contour_img = np.zeros_like(gray_tattoo)
    tattoo_mask = cv2.drawContours(contour_img, [contour], -1, (255), thickness=cv2.FILLED)

    kernel_size = 5  # 커널 크기 (너무 크게 설정하면 외곽선이 많이 줄어듦)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    tattoo_mask = cv2.erode(contour_img, kernel, iterations=1)

    return Image.fromarray(tattoo_mask)


def extract_tattoo_mask_v3(tattoo: Image) -> Image:
    # Convert PIL Image to numpy array
    gray_tattoo = np.array(tattoo.convert('L'))  # Convert to grayscale
    gray_tattoo = cv2.equalizeHist(gray_tattoo)
    # Apply Gaussian Blur to reduce noise
    # blurred_img = cv2.GaussianBlur(gray_tattoo, (5, 5), 0)

    # Find Otsu's threshold
    otsu_threshold, _ = cv2.threshold(gray_tattoo, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(otsu_threshold)

    # Apply Canny edge detection using the Otsu's threshold
    edges = cv2.Canny(gray_tattoo, otsu_threshold * 0.5, otsu_threshold)
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours from the edges
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask
    mask = np.zeros_like(gray_tattoo)

    # Draw the contours on the mask
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Convert the mask back to PIL Image
    mask_image = Image.fromarray(mask)

    return mask_image


def search_mask_coord(
        tattoo: Image.Image, 
        mask: Image.Image, 
        lower_bound: float=0.95, 
        upper_bound: float=0.98) -> Tuple[float, float, List[int]]:
    # 1. make tattoo mask
    tattoo_mask = extract_tattoo_mask(tattoo=tattoo).convert('1')
    tattoo_mask = np.array(tattoo_mask)

    # 2. mask thresholding (threshold = 1, over => 1, o.w => 0)
    mask_threshold = 1
    crop_mask = np.array(mask)
    _, crop_mask = cv2.threshold(crop_mask, mask_threshold, 1, cv2.THRESH_BINARY_INV)
    num_white_pixel = np.sum(crop_mask)

    # 3. search 
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
        while 0.5 < scale and best[1] <= lower_bound:
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
        while scale < 3.0 and upper_bound <= best[1]:
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


def make_moved_wnd_mask(crop_wnd_mask: Image.Image, wnd_on_tat_bbox: List[int]) -> Image.Image:
    moved_wnd_mask = Image.new("L", (1024, 1024))
    moved_wnd_mask.paste(
        crop_wnd_mask.resize( (wnd_on_tat_bbox[2] - wnd_on_tat_bbox[0], wnd_on_tat_bbox[3] - wnd_on_tat_bbox[1]) ), 
        (wnd_on_tat_bbox[0], wnd_on_tat_bbox[1])
        )

    return moved_wnd_mask


def overlay_edge(tattoo: Image.Image, edge: Image.Image, coord: List[int]) -> Image.Image:
    tat_mask = extract_tattoo_mask(tattoo=tattoo)
    
    moved_edge = Image.new("L", (1024, 1024), color="white")
    moved_edge.paste(edge.resize((coord[2] - coord[0], coord[3] - coord[1])), (coord[0], coord[1]))
    moved_edge = ImageChops.logical_or(tat_mask.convert('1'), moved_edge.convert('1'))

    moved_edge = np.array(moved_edge)
    overlay = np.array(tattoo)
    overlay[moved_edge == False] = 0

    return Image.fromarray(overlay)


def extract_wound(input: Image.Image, mask: Image.Image) -> Image.Image:
    mask = mask.convert('L')
    wound = deepcopy(input)
    wound.putalpha(mask)
    
    return wound


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
        input_image: Image.Image,
        tattoo: Image.Image, 
        wnd_bbox: List[int], 
        wnd_on_tat_bbox: List[int], 
        scale: int) -> Image.Image:
    width = int(tattoo.size[0] * (1.0 / scale))
    height = int(tattoo.size[1] * (1.0 / scale))
    resized_tattoo = tattoo.resize((width, height))
    wnd_on_tat_bbox = (np.array(wnd_on_tat_bbox) * (1.0 / scale)).astype(np.intc)

    skin_mask = extract_skin_mask(image=input_image, bbox=wnd_bbox)
    tat_mask = extract_tattoo_mask(tattoo=resized_tattoo)

    skin_image_frame = (0, 0, skin_mask.size[0], skin_mask.size[1])

    dx = wnd_bbox[0] - wnd_on_tat_bbox[0]
    dy = wnd_bbox[1] - wnd_on_tat_bbox[1]
    skin_mask_frame = (max(0, dx), max(0, dy), \
                       min(skin_image_frame[2], resized_tattoo.size[0] + dx), \
                        min(skin_image_frame[3], resized_tattoo.size[1] + dy))

    dx = wnd_on_tat_bbox[0] - wnd_bbox[0]
    dy = wnd_on_tat_bbox[1] - wnd_bbox[1]
    tat_mask_frame = (max(0, dx), max(0, dy), \
                      min(resized_tattoo.size[0], skin_image_frame[2] + dx), \
                      min(resized_tattoo.size[1], skin_image_frame[3] + dy))

    intersection_mask = ImageChops.logical_and(
        skin_mask.crop(skin_mask_frame).convert('1'),
        tat_mask.crop(tat_mask_frame).convert('1')
    )

    intersection_tat_mask = Image.new('RGB', tat_mask.size)
    intersection_tat_mask.paste(intersection_mask, tat_mask_frame)

    intersection_tat = Image.new('RGB', tat_mask.size)
    intersection_tat.paste(resized_tattoo, (0, 0), intersection_tat_mask.convert("L"))
    intersection_tat = intersection_tat.crop(tat_mask_frame)

    tat_on_skin = Image.new('RGB', skin_mask.size)
    tat_on_skin.paste(intersection_tat, skin_mask_frame)

    intersection_skin_mask = Image.new('RGB', skin_mask.size)
    intersection_skin_mask.paste(intersection_mask, skin_mask_frame)
    
    tat_on_skin.putalpha(intersection_skin_mask.convert("L"))

    return tat_on_skin




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user', type=str)
    parser.add_argument('-f', '--file', type=str)

    args = parser.parse_args()

    user = args.user
    filename = args.file
    base_dir = 'path/to/base/dir'
    input_path = f'{base_dir}/scart/images/{user}/inputs/{filename}'
    mask_path = f'{base_dir}/scart/images/{user}/masks/{filename}'
    tattoo_path = f'{base_dir}/scart/images/{user}/tattoos/{filename}'

    input_image = load_image(input_path).resize((1024, 1024))
    wnd_mask = load_image(mask_path).resize((1024, 1024))   # type: PIL.Image
    tattoo = load_image(tattoo_path).resize((1024, 1024))   # type: PIL.Image


    extract_tattoo_mask(tattoo=tattoo).save("tat_mask.png")

    mask = extract_tattoo_mask(tattoo)
    mask.save("tat_mask_with_canny.png")

