import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from typing import Tuple, List

from diffusers.utils import load_image


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

    mask.save('bbox.png')

    return bbox, mask


def search_mask_coord(tattoo: Image.Image, mask: Image.Image) -> Tuple[float, float, List[int]]:
    # 1. convert tattoo to gray scale image
    gray_tattoo = np.array(tattoo.convert('L'))

    # 2. thresholding (threshold = 50)
    gray_tattoo = np.where(gray_tattoo < 190, 1, 0)

    # 3. mask thresholding (threshold = 1, over => 1, o.w => 0)
    mask_threshold = 0
    crop_mask = np.array(mask)
    crop_mask = np.where(crop_mask > mask_threshold, 1, 0)
    num_white_pixel = np.sum(crop_mask)

    # 4. for loop 
    lower_bound, upper_bound = 0.9, 0.95
    scale = 1.0 
    best = [1.0, 0.0, []]
    mask_size = crop_mask.shape
    
    for row in tqdm(range(0, gray_tattoo.shape[0] - mask_size[0], 5), desc=f"Searching mask coord (Scale: {scale})"):
        for col in range(0, gray_tattoo.shape[1] - mask_size[1], 5):
            score = np.sum(crop_mask * gray_tattoo[row:row + mask_size[0], col:col + mask_size[1]]) / num_white_pixel
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
            
            for row in tqdm(range(0, gray_tattoo.shape[0] - small_mask_size[0], 5), desc=f"Searching mask coord (Scale: {scale})"):
                for col in range(0, gray_tattoo.shape[1] - small_mask_size[1], 5):
                    score = np.sum(crop_mask * gray_tattoo[row:row + small_mask_size[0], col:col + small_mask_size[1]]) / num_white_pixel
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

            for row in tqdm(range(0, gray_tattoo.shape[0] - big_mask_size[0], 5), desc=f"Searching mask coord (Scale: {scale})"):
                for col in range(0, gray_tattoo.shape[1] - big_mask_size[1], 5):
                    score = np.sum(crop_mask * gray_tattoo[row:row + big_mask_size[0], col:col + big_mask_size[1]]) / num_white_pixel
                    best = (scale, score, [row, col, row + big_mask_size[0], col + big_mask_size[1]]) if best[1] < score else best

            print(f'best score: {best[1]}')        

    Image.fromarray(crop_mask).save("mask_check.png")

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

    edge.save('edge.png')

    return edge


def overlay_edge(tattoo: Image.Image, edge: Image.Image, coord: List[int]) -> Image.Image:
    moved_edge = Image.new("L", (1024, 1024), color="white")
    moved_edge.paste(edge.resize((coord[3] - coord[1], coord[2] - coord[0])), (coord[1], coord[0]))
    moved_edge.save("moved_edge.png")
    moved_edge = np.array(moved_edge)
    
    overlay = np.array(tattoo)
    overlay[moved_edge == 0] = 0
    overlay = Image.fromarray(overlay)

    overlay.save('overlay.png')

    return overlay



if __name__ == "__main__":
    mask_path = '/home/cvmlserver11/junhee/scart/images/user1/masks/scar1.png'
    tattoo_path = '/home/cvmlserver11/junhee/scart/images/user1/tattoos/scar1.png'

    tattoo = load_image(tattoo_path).resize((1024, 1024))   # type: PIL.Image
    mask = load_image(mask_path).resize((1024, 1024))   # type: PIL.Image
    bbox, crop_mask = extract_bbox(mask)
    print(f'bbox: {bbox}')
    print(type(crop_mask))
    print(crop_mask.size)

    scale, score, coord = search_mask_coord(tattoo, crop_mask)
    print(f'Moved mask information')
    print(f'Scale: {scale}')
    print(f'Coverage score: {score}')
    print(f'Coordinate: {coord}')

    edge = extract_edge(crop_mask)  # type: PIL.Image
    print(f'edge size: {edge.size}')

    overlay = overlay_edge(tattoo, edge, coord)

