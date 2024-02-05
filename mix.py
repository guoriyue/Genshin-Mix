
import json
import random
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from PIL.Image import Image
from collections import defaultdict
import torch
from safetensors.torch import load_file
import copy
from datetime import datetime
from collections import deque
import os


def load_lora_weights(pipeline, checkpoint_path, multiplier, device, dtype):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device=device)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    # directly update weight in diffusers model
    for layer, elems in updates.items():

        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        alpha = elems['alpha']
        if alpha:
            alpha = alpha.item() / weight_up.shape[1]
        else:
            alpha = 1.0

        # update weight
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline

# pipe = StableDiffusionPipeline.from_single_file(
#     "/Users/guomingfei/Downloads/GenshinLora/anything-v3-fp16-pruned.safetensors"
# )
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline

characters = json.load(open("lora.json", "r", encoding="utf-8"))
safetensor_dir = "~/GenshinLora"
save_dir = "~/GenshinGens"
negative_prompt = "EasyNegative, disembodied limb, worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting, bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, realistic photo, extra eyes, huge eyes, 2girl, amputation, disconnected limbs"
general_positive_prompt="Solo, beautiful, Detailed eyes, natural light, 64k resolution, beautiful, ultra detailed, colorful, rich deep color, 16k, glow effects"


# model = "CompVis/stable-diffusion-v1-4"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
# pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)

pipe = StableDiffusionPipeline.from_single_file(
    "../anything-v4.5-pruned.safetensors", vae=vae, device=device).to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# negative_prompt = "EasyNegative, disembodied limb, worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting, bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, realistic photo, extra eyes, huge eyes, 2girl, amputation, disconnected limbs"
# general_positive_prompt=", Solo, beautiful, Detailed eyes, natural light, 64k resolution, beautiful, ultra detailed, colorful, rich deep color, 16k, glow effects"
device = "cuda:0"
comb_number = 1000
while comb_number:
    print("generating ", 1000 - comb_number)
    # for each combination generate 5 images
    picked_character = random.sample(characters, 2)

    prompt = ""
    answer = []
    for character in picked_character:
        answer.append(character['name'].lower())
        prompt += character['prompt'] + ', '
        lora_model = safetensor_dir+'/'+character['path']
        print(lora_model)
    
        new_pipe = load_lora_weights(new_pipe, lora_model, 1.0, device, torch.float32)
    
    max_length = new_pipe.tokenizer.model_max_length
    input_ids = new_pipe.tokenizer(prompt + ',' + general_positive_prompt, return_tensors="pt").input_ids
    # input_ids.to(device)
    negative_ids = new_pipe.tokenizer(negative_prompt, truncation=True, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids                                                                                                     
    concat_embeds = []
    neg_embeds = []
    for i in range(0, input_ids.shape[-1], max_length):
        concat_embeds.append(new_pipe.text_encoder(input_ids[:,i: i + max_length].to(device))[0])
        neg_embeds.append(new_pipe.text_encoder(negative_ids[:, i: i + max_length].to(device))[0])
        
    prompt_embeds = torch.cat(concat_embeds, dim=1).to(device)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1).to(device)

    # 3. Forward
    image = new_pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_inference_steps=50).images[0]
    image_datetime = datetime.now().strftime('%Y%m%d-%H%M%S')
    image_file_path = save_dir + "/images/" + prompt.replace(' ', '#') + '@' + image_datetime + ".png"
    image_info_path = save_dir + "/jsons/" + image_datetime + ".json"
    
    image.save(image_file_path)
    
    image_info = {
        "image_upload": False,
        "image_url": None,
        "image_file_path": image_file_path,
        "answer": answer
    }
    json.dump(image_info, open(image_info_path, "w", encoding="utf-8"))