import argparse
import os 

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid, numpy_to_pil

import torch

import numpy as np
import cv2
from PIL import Image

from tattoo_generation.utils.mask_processing import extract_edge, overlay_edge


parser = argparse.ArgumentParser()
parser.add_argument('-u', '--userid', type=str, help='enter user id')
parser.add_argument('-f', '--filename', type=str, help='enter file name')
parser.add_argument('-p', '--prompt', type=str, help='image generation prompt')
parser.add_argument('-l', '--lora', action='store_false', help='image generation prompt')
parser.add_argument('-d', '--debug', action='store_true', help='debugging mode')
args = parser.parse_args()    

# [default] project directory path (hardcoded)
home_dir = os.path.expanduser('~')
proj_dir = 'junhee/scart'
# [args] userid & filename
userid = args.userid if args.userid else 'junhee'
filename = args.filename if args.filename else 'burn.png'
# [args] prompt
prompt = args.prompt if args.prompt else "Astronaut floating in the space"
img_gen_prompt = prompt + "tattoo design, center align, big object" 
# [args] load lora
lora = True if args.lora else False
# [determined] base model & lora path
models_dir = f'{home_dir}/{proj_dir}/tattoo_generation/models'
model_path = f'{models_dir}/sdxl_base_1.0.safetensors'
lora_path = f'{models_dir}/lora/Traditional_Tattoos.safetensors'
# [determined] mask & tattoo & inpaint path
mask_path = f'{home_dir}/{proj_dir}/images/{userid}/masks/{filename}'
tattoo_path = f'{home_dir}/{proj_dir}/images/{userid}/tattoos/{filename}'
inpaint_path = f'{home_dir}/{proj_dir}/images/{userid}/inpaint/{filename}'
# sampling hyper parameter
num_inference_steps = 40
lora_scale = 0.65
# inpainting hyper-parameter
guidance_scale = 20.0
num_inference_steps_inpaint = 20 # steps between 15 and 30 work well for us
strength = 0.99  # make sure to use `strength` below 1.0
# integrated image path
results_path = 'results/inpaint_result.png'


device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True
).to(device)

# extract edge (black line and white background)
edge = extract_edge(mask_path).resize((1024, 1024))

print('[edge extraction info]')
print(f'type: {type(edge)}')
print(f'shape: {edge.size}')

# concat tattoo (base layer) and edge (upper layer)
tattoo = load_image(tattoo_path).resize((1024, 1024))
latent_image = overlay_edge(tattoo, edge)

print('[latent_image info]')
print(f'type: {type(latent_image)}')
print(f'shape: {latent_image.size}')


# Sampling image by img2img pipeline
image = pipe(
        prompt=img_gen_prompt, 
        image=latent_image, 
        guidance_scale=1.0
        ).images[0]


# image.save(results_path)
in_and_out = [edge, tattoo, latent_image, image]
make_image_grid(in_and_out, rows=1, cols=len(in_and_out)).save(results_path)
