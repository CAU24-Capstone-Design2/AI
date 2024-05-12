import argparse
import cv2
from PIL import Image
import numpy as np

def extract_edge(mask_path, save_path=None):
    # load mask with gray scale
    mask = cv2.imread(mask_path, 0)

    # edge extract (black edge and white background)
    edge = cv2.Canny(mask, threshold1=100, threshold2=200)
    
    # edge dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge = cv2.dilate(edge, kernel, iterations=1)
    edge = 255 - edge

    # (option) save
    if save_path:
        cv2.imwrite(save_path, edge)

    return Image.fromarray(edge).convert("RGB")


def bbox(mask_path):
    mask = cv2.imread(mask_path, 0) 

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

    return bbox, mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def overlay_edge(tattoo, edge):
    overlay = np.array(tattoo.copy())
    overlay[np.array(edge) == 0] = 0
    return Image.fromarray(overlay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mask', type=str)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-b', '--bold', type=int)
    args = parser.parse_args()
    
    mask_path = args.mask
    output_path = args.output
    iterations = args.bold

    edge = extract_edge(mask_path, iterations=iterations, save=True, save_path=output_path)

    print('[edge info]')
    print(f'type: {type(edge)}')
    print(f'shape: {edge.size}')




