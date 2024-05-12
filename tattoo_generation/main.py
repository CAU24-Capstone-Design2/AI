# diffuser
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image, make_image_grid
import torch
# default lib
import os 
import argparse
# implemented function
from utils.edge_extract import extract_edge, overlay_edge
# wandb
import wandb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--userid', type=str, help='enter user id')
    parser.add_argument('-f', '--filename', type=str, help='enter file name')
    parser.add_argument('-d', '--debug', action='store_true', help='debugging mode')
    parser.add_argument('-p', '--prompt', type=str, help='image generation prompt')
    parser.add_argument('-l', '--lora', action='store_false', help='image generation prompt')
    args = parser.parse_args()    


    # [default] project directory path (hardcoded)
    home_dir = os.path.expanduser('~')
    proj_dir = 'junhee/scart'
    # arguments
    userid = args.userid if args.userid else 'junhee'
    filename = args.filename if args.filename else 'burn.png'
    prompt = args.prompt if args.prompt else "Astronaut floating in the space"
    img_gen_prompt = prompt + ", tattoo design, center align, big object" 
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
    num_inference_steps = 30
    lora_scale = 0.8
    # integrated image path
    results_path = 'results/result.png'
    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # initialize a wandb run
    wandb.init(project='ScArt')
    wandb.run.name = 'tattoo-generation'
    wandb.run.save() 

    config = wandb.config
    config.base_checkpoint = "stabilityai/stable-diffusion-xl-base-1.0"
    config.refiner_checkpoint = "stabilityai/stable-diffusion-xl-refiner-1.0"
    config.prompt = img_gen_prompt
    config.num_inference_steps = 30
    config.lora_scale = 0.8
    wandb.config.update(config)

    base_pipe = StableDiffusionXLPipeline.from_pretrained(
        config.base_checkpoint,
        torch_dtype=torch.float16,
        variant='fp16',
        use_safetensors=True
    ).to(device)

    # Sampling tattoo 
    if lora:
        base_pipe.load_lora_weights(lora_path, adapter_name="tattoo")
        draft = base_pipe(prompt=config.prompt, num_inference_steps=config.num_inference_steps, cross_attention_kwargs={"scale": config.lora_scale}).images[0]
    else:
        draft = base_pipe(prompt=config.prompt, num_inference_steps=config.num_inference_steps).images[0]

    # save tattoo image
    draft.save(tattoo_path)

    # refiner pipeline : sdxl refiner 1.0
    refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        config.refiner_checkpoint, 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to(device)

    # extract edge (black line and white background)
    edge = extract_edge(mask_path).resize((1024, 1024))

    # concat tattoo (base layer) and edge (upper layer)
    draft = load_image(tattoo_path).resize((1024, 1024))
    draft_with_edge = overlay_edge(draft, edge)

    # Sampling image by img2img pipeline
    tattoo = refiner_pipe(
        prompt=config.prompt, 
        image=draft_with_edge, 
    ).images[0]

    tattoo.save(tattoo_path)

    # Save integrated results to local
    results = [edge, draft, draft_with_edge, tattoo]
    make_image_grid(results, rows=1, cols=len(results)).save(results_path)
    # Log the images and table to wandb
    table = wandb.Table(columns=[
        'Prompt', 'mask-edge', 'tattoo-design', 'tattoo-and-mask', 'refined-tattoo-design'
    ])
    table.add_data(
        config.prompt, wandb.Image(edge), wandb.Image(draft), wandb.Image(draft_with_edge), wandb.Image(tattoo) 
    )

    wandb.log({
        "Prompt-and-Results": table
    })

    wandb.finish()