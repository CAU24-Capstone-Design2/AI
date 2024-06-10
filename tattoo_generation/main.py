# diffuser
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoPipelineForInpainting, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
import torch
# default lib
import os 
import argparse
# implemented function
from utils.mask_processing import *
# wandb
import wandb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--userid', type=str, help='enter user id')
    parser.add_argument('-f', '--filename', type=str, help='enter file name')
    parser.add_argument('-p', '--prompt', type=str, help='image generation prompt')
    parser.add_argument('-l', '--lora', type=str, required=False, help='Apply tattoo style LoRA')
    parser.add_argument('-i', '--inpaint', action='store_true', help='use inpainting model instead refiner')
    parser.add_argument('-d', '--debug', action='store_true', help='debugging mode')
    args = parser.parse_args()    

    # project directory path (hardcoded)
    home_dir = os.path.expanduser('~')
    proj_dir = 'path/to/proj/dir'
    # arguments
    userid = args.userid 
    filename = args.filename
    # prompt = args.prompt + ", with white background, center align, big object" 
    # prompt = args.prompt + ", tattoo design, white background, no gradations, no shadows, clear contrast between background and foreground, intricate line work, center align, big object"
    prompt = args.prompt
    lora = args.lora.replace(' ', '').lower() if args.lora else None
    use_inpaint = args.inpaint
    # path
    input_path = f'{home_dir}/{proj_dir}/images/{userid}/inputs/{filename}'
    mask_path = f'{home_dir}/{proj_dir}/images/{userid}/masks/{filename}'
    tattoo_path = f'{home_dir}/{proj_dir}/images/{userid}/tattoos/{filename}'
    synthesis_path = f'{home_dir}/{proj_dir}/images/{userid}/synthesis/{filename}'
    # integrated image path
    results_path = 'results/result.png'
    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Initialize wandb run
    wandb.init(project='ScArt', allow_val_change=True)
    wandb.run.name = 'tattoo-generation'
    wandb.run.save() 

    config = wandb.config
    config.base_checkpoint = "stabilityai/stable-diffusion-xl-base-1.0"
    config.refiner_checkpoint = "stabilityai/stable-diffusion-xl-refiner-1.0"
    config.inpaint_checkpoint = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    config.lora_checkpoint = f'melancholic/{lora}_tattoo_lora' if lora else None
    config.prompt = f"In the style of TOK, {prompt} on the white background, main object or character on the center" if lora else prompt
    config.num_inference_steps = 30
    config.lora_scale = 0.8
    config.inpaint_num_inference_steps = 20     # steps between 15 and 30 work well for us
    config.inpaint_guidance_scale = 12.0  
    config.inpaint_strength = 0.7              # make sure to use `strength` below 1.0
    wandb.config.update(config)


    # Generate tattoo 
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    base_pipe = StableDiffusionXLPipeline.from_pretrained(
        config.base_checkpoint,
        vae=vae,
        torch_dtype=torch.float16,
        variant='fp16',
        use_safetensors=True,
    ).to(device)

    if config.lora_checkpoint:
        base_pipe.load_lora_weights(config.lora_checkpoint, adapter_name="tattoo")
        generator = torch.Generator(device="cuda").manual_seed(0)
        draft = base_pipe(prompt=config.prompt, num_inference_steps=config.num_inference_steps, generator=generator).images[0]
    else:
        draft = base_pipe(prompt=config.prompt, num_inference_steps=config.num_inference_steps).images[0]


    # Search mask position
    wnd_mask = load_image(mask_path).resize((1024, 1024))
    wnd_bbox, crop_wnd_mask = extract_bbox(wnd_mask)
    wnd_scale, coverage_score, wnd_on_tat_bbox = search_mask_coord(draft, crop_wnd_mask, lower_bound=0.95, upper_bound=0.98)

    # overlay wound edge to tattoo
    edge = extract_edge(crop_wnd_mask)
    draft_with_edge = overlay_edge(draft, edge, wnd_on_tat_bbox)

    
    if use_inpaint:
        moved_wnd_mask = make_moved_wnd_mask(crop_wnd_mask=crop_wnd_mask, wnd_on_tat_bbox=wnd_on_tat_bbox)

        # INPAINTING
        inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
            config.inpaint_checkpoint, 
            torch_dtype=torch.float16, 
            variant="fp16"
            ).to(device)
        
        tattoo = inpaint_pipe(
            prompt=args.prompt,
            image=draft,
            mask_image=moved_wnd_mask,
            guidance_scale=config.inpaint_guidance_scale,
            num_inference_steps=config.inpaint_num_inference_steps,  
            strength=config.inpaint_strength,  
        ).images[0]
    else :
        # Refine image using img2img pipeline
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            config.refiner_checkpoint, 
            torch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True
        ).to(device)
        
        tattoo = refiner_pipe(
            prompt=config.prompt, 
            image=draft_with_edge, 
        ).images[0]


    # Save results
    # 1. input image: this line is for resize the image to (1024 x 1024) -> not necessary
    input_image = load_image(input_path).resize((1024, 1024))
    input_image.save(input_path)    
    # 2. wound
    wound = extract_wound(input_image, wnd_mask)
    wound.save(mask_path)
    # 3. tattoo
    tattoo.save(tattoo_path)
    # 4. synthesis tattoo image
    tat_on_skin = synthesis_tattoo(input_image=input_image, tattoo=tattoo, wnd_bbox=wnd_bbox, wnd_on_tat_bbox=wnd_on_tat_bbox, scale=wnd_scale)
    tat_on_skin.save(synthesis_path)
    # 5. integrated results to local
    results = [extract_edge(wnd_mask).resize((1024, 1024)), draft, draft_with_edge, tattoo]
    make_image_grid(results, rows=1, cols=len(results)).save(results_path)

    # Logging the prompt and images to wandb
    table = wandb.Table(columns=[
        'Prompt', 'mask-edge', 'tattoo-design', 'tattoo-and-mask', 'refined-tattoo-design'
    ])
    table.add_data(
        config.prompt, wandb.Image(extract_edge(wnd_mask)), wandb.Image(draft), wandb.Image(draft_with_edge), wandb.Image(tattoo) 
    )
    wandb.log({"Prompt-and-Results": table})

    wandb.finish()

