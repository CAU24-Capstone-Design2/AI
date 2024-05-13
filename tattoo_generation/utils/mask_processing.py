import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from typing import Tuple, List

from diffusers.utils import load_image


def extract_bbox(mask: Image.Image) -> Tuple[List[int], Image.Image]:
    mask = np.array(mask.convert('L'))

    # 물체의 윤곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 바운딩 박스 추출
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


def search_mask_coord(tattoo: Image.Image, crop_mask: Image.Image) -> Tuple[float, float, List[int]]:
    # 1. convert tattoo to gray scale image
    gray_tattoo = np.array(tattoo.convert('L'))

    # 2. thresholding (threshold = 50)
    gray_tattoo = np.where(gray_tattoo < 127, 1, 0)

    # print(type(gray_tattoo))
    # print(gray_tattoo.shape)
    # print(np.unique(gray_tattoo, return_counts=True))

    # 3. mask thresholding (threshold = 1, over => 1, o.w => 0)
    mask_threshold = 0
    crop_mask = np.array(crop_mask)
    crop_mask = np.where(crop_mask > mask_threshold, 1, 0)
    num_white_pixel = np.sum(crop_mask)

    # print(type(mask))
    # print(mask.shape)
    # print(np.unique(mask, return_counts=True))
    
    # 4. for loop 
    target = 0.8
    scale = 1.0 
    best = (1.0, 0.0, [])
    mask_size = crop_mask.shape
    
# for _ in range(3):
    for row in tqdm(range(0, gray_tattoo.shape[0] - mask_size[0], 5), desc="searching mask coord"):
        for col in range(0, gray_tattoo.shape[1] - mask_size[1], 5):
            score = np.sum(crop_mask * gray_tattoo[row:row+mask_size[0], col:col+mask_size[1]]) / num_white_pixel
            print(f'[{row}, {col}]: {score}')
            best = (scale, score, [row, col, row+mask_size[0], col+mask_size[1]]) if best[1] < score else best
    
    # if best[1] == 0:
    #     scale -= 0.1
    #     print(f'Reduce sacle: {scale}')
    #     mask_size = (int(mask_size[0] * scale), int(mask_size[1] * scale))
    #     print(mask_size)
    #     crop_mask = Image.fromarray(crop_mask, mode='L').resize((mask_size[1], mask_size[0]))
    #     crop_mask = np.array(crop_mask)
    #     num_white_pixel = np.sum(crop_mask)
    # else:
    #     break
        
    return best


def extract_edge(crop_mask: Image.Image) -> Image.Image:
    # load mask with gray scale

    # edge extract (black edge and white background)
    crop_mask = np.array(crop_mask)
    edge = cv2.Canny(crop_mask, threshold1=100, threshold2=200)
    
    # edge dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge = cv2.dilate(edge, kernel, iterations=1)
    edge = 255 - edge
    edge = Image.fromarray(edge)

    return edge


def overlay_edge(tattoo: Image.Image, edge: Image.Image, coord: List[int]) -> Image.Image:
    moved_edge = np.array([[255 for _ in range(1024)] for _ in range(1024)])
    moved_edge[coord[0]:coord[2], coord[1]:coord[3]] *= edge
    overlay = np.array(tattoo)
    overlay[np.array(moved_edge) == 0] = 0
    overlay = Image.fromarray(overlay)

    return overlay

if __name__ == "__main__":
    mask_path = '/home/cvmlserver11/junhee/scart/images/1/masks/1.png'
    tattoo_path = '/home/cvmlserver11/junhee/scart/images/1/tattoos/1.png'

    tattoo = load_image(tattoo_path).resize((1024, 1024))   # type: PIL.Image
    mask = load_image(mask_path).resize((1024, 1024))   # type: PIL.Image
    bbox, crop_mask = extract_bbox(mask)

    scale, score, coord = search_mask_coord(tattoo, crop_mask)
    print(f'Moved mask information')
    print(f'Scale: {scale}')
    print(f'Coverage score: {score}')
    print(f'Coordinate: {coord}')

    edge = extract_edge(crop_mask)  # type: PIL.Image
    overlay = overlay_edge(tattoo, edge, coord)
    # cv2.imwrite('asdf.png', edge)
    overlay.save('asdf.png')