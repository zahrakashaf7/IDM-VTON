"""
FastAPI wrapper for IDM-VTON.

Drop this file into the root of your IDM-VTON (or IDM-VTON-hf) repo -- the same
folder that contains app.py, src/, preprocess/, apply_net.py, utils_mask.py,
configs/, and ckpt/. It reuses the exact same model-loading and inference logic
as the Gradio app, just exposed as plain HTTP endpoints instead of a UI.

Run with:
    uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --workers 1

IMPORTANT: use --workers 1. Each worker would load its own full copy of the
models onto the GPU, which will exhaust VRAM with more than one worker.

Environment variables (all optional):
    IDM_VTON_WIDTH            generation width  (default 768)
    IDM_VTON_HEIGHT           generation height (default 1024)
    IDM_VTON_VAE_SLICING      "1" to enable VAE slicing/tiling (default "1")
    IDM_VTON_ATTN_SLICING     "1" to enable attention slicing  (default "1")
"""

import io
import os
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)
from diffusers import AutoencoderKL, DDPMScheduler

import apply_net
from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from utils_mask import get_mask_location

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEN_WIDTH = int(os.environ.get("IDM_VTON_WIDTH", 768))
GEN_HEIGHT = int(os.environ.get("IDM_VTON_HEIGHT", 1024))
ENABLE_VAE_SLICING = os.environ.get("IDM_VTON_VAE_SLICING", "1") == "1"
ENABLE_ATTN_SLICING = os.environ.get("IDM_VTON_ATTN_SLICING", "1") == "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = "yisol/IDM-VTON"

tensor_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def pil_to_binary_mask(pil_image: Image.Image, threshold: int = 0) -> Image.Image:
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = (binary_mask.astype(np.uint8) * 255)
    return Image.fromarray(mask)


# ---------------------------------------------------------------------------
# Load models once, at import time
# ---------------------------------------------------------------------------

print(f"Loading IDM-VTON models onto {DEVICE} ...")

import gc

# Load each model directly onto the GPU as it's created, instead of staging all of them
# in CPU RAM first and moving them to GPU at the end. This is what was causing the Colab
# "session crashed after using all available RAM" errors -- at the old peak, every model
# (main UNet, near-duplicate UNet_Encoder, both text encoders, VAE, image encoder) sat in
# system RAM simultaneously before any GPU transfer happened. device_map loads each one's
# weights straight to the GPU, so CPU RAM never has to hold more than a sliver at a time.
_LOAD_KWARGS = {"low_cpu_mem_usage": True}
if DEVICE == "cuda":
    _LOAD_KWARGS["device_map"] = {"": 0}

unet = UNet2DConditionModel.from_pretrained(
    BASE_PATH, subfolder="unet", torch_dtype=torch.float16, **_LOAD_KWARGS
)
unet.requires_grad_(False)
gc.collect()

tokenizer_one = AutoTokenizer.from_pretrained(BASE_PATH, subfolder="tokenizer", revision=None, use_fast=False)
tokenizer_two = AutoTokenizer.from_pretrained(BASE_PATH, subfolder="tokenizer_2", revision=None, use_fast=False)
noise_scheduler = DDPMScheduler.from_pretrained(BASE_PATH, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    BASE_PATH, subfolder="text_encoder", torch_dtype=torch.float16, **_LOAD_KWARGS
)
gc.collect()
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    BASE_PATH, subfolder="text_encoder_2", torch_dtype=torch.float16, **_LOAD_KWARGS
)
gc.collect()
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    BASE_PATH, subfolder="image_encoder", torch_dtype=torch.float16, **_LOAD_KWARGS
)
gc.collect()
vae = AutoencoderKL.from_pretrained(
    BASE_PATH, subfolder="vae", torch_dtype=torch.float16, **_LOAD_KWARGS
)
gc.collect()

UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    BASE_PATH, subfolder="unet_encoder", torch_dtype=torch.float16, **_LOAD_KWARGS
)
gc.collect()

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

for m in (UNet_Encoder, image_encoder, vae, unet, text_encoder_one, text_encoder_two):
    m.requires_grad_(False)

pipe = TryonPipeline.from_pretrained(
    BASE_PATH,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder

if ENABLE_VAE_SLICING:
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
if ENABLE_ATTN_SLICING:
    pipe.enable_attention_slicing()

pipe.to(DEVICE)
pipe.unet_encoder.to(DEVICE)
openpose_model.preprocessor.body_estimation.model.to(DEVICE)

print(f"Models loaded. Generating at {GEN_WIDTH}x{GEN_HEIGHT}.")

# ---------------------------------------------------------------------------
# Core inference (same logic as app.py's start_tryon, minus the Gradio bits)
# ---------------------------------------------------------------------------


def run_tryon(
    human_img_orig: Image.Image,
    garm_img: Image.Image,
    garment_des: str,
    use_auto_mask: bool,
    use_auto_crop: bool,
    manual_mask_img: Optional[Image.Image],
    denoise_steps: int,
    seed: Optional[int],
):
    human_img_orig = human_img_orig.convert("RGB")
    garm_img = garm_img.convert("RGB").resize((GEN_WIDTH, GEN_HEIGHT))

    if use_auto_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((GEN_WIDTH, GEN_HEIGHT))
    else:
        human_img = human_img_orig.resize((GEN_WIDTH, GEN_HEIGHT))

    if use_auto_mask:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, _ = get_mask_location("hd", "upper_body", model_parse, keypoints)
        mask = mask.resize((GEN_WIDTH, GEN_HEIGHT))
    else:
        if manual_mask_img is None:
            raise HTTPException(
                status_code=400,
                detail="use_auto_mask=false requires a mask_img upload (white = area to replace).",
            )
        mask = pil_to_binary_mask(manual_mask_img.convert("RGB").resize((GEN_WIDTH, GEN_HEIGHT)))

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(
        (
            "show",
            "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
            "./ckpt/densepose/model_final_162be9.pkl",
            "dp_segm",
            "-v",
            "--opts",
            "MODEL.DEVICE",
            DEVICE,
        )
    )
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((GEN_WIDTH, GEN_HEIGHT))

    with torch.no_grad(), torch.cuda.amp.autocast():
        prompt = "model is wearing " + garment_des
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            cloth_prompt = "a photo of " + garment_des
            cloth_negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            cloth_prompt_list: List[str] = [cloth_prompt]
            cloth_negative_prompt_list: List[str] = [cloth_negative_prompt]
            with torch.inference_mode():
                (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
                    cloth_prompt_list,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=cloth_negative_prompt_list,
                )

            pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(DEVICE, torch.float16)
            garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(DEVICE, torch.float16)
            generator = torch.Generator(DEVICE).manual_seed(seed) if seed is not None else None

            images = pipe(
                prompt_embeds=prompt_embeds.to(DEVICE, torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(DEVICE, torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(DEVICE, torch.float16),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(DEVICE, torch.float16),
                num_inference_steps=denoise_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor.to(DEVICE, torch.float16),
                text_embeds_cloth=prompt_embeds_c.to(DEVICE, torch.float16),
                cloth=garm_tensor.to(DEVICE, torch.float16),
                mask_image=mask,
                image=human_img,
                height=GEN_HEIGHT,
                width=GEN_WIDTH,
                ip_adapter_image=garm_img.resize((GEN_WIDTH, GEN_HEIGHT)),
                guidance_scale=2.0,
            )[0]

    if use_auto_crop:
        out_img = images[0].resize(crop_size)
        result = human_img_orig.copy()
        result.paste(out_img, (int(left), int(top)))
        return result
    return images[0]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="IDM-VTON API", description="Virtual try-on inference over HTTP.")


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "generation_size": f"{GEN_WIDTH}x{GEN_HEIGHT}"}


@app.post("/tryon")
async def tryon(
    human_img: UploadFile = File(..., description="Photo of the person"),
    garm_img: UploadFile = File(..., description="Photo of the garment"),
    garment_des: str = Form("clothing item", description="Short garment description"),
    use_auto_mask: bool = Form(True, description="Auto-detect the region to replace"),
    use_auto_crop: bool = Form(False, description="Auto-crop/resize the person photo"),
    denoise_steps: int = Form(30, ge=1, le=100),
    seed: Optional[int] = Form(42),
    mask_img: Optional[UploadFile] = File(
        None, description="Required only if use_auto_mask=false: white = area to replace"
    ),
):
    try:
        human_pil = Image.open(io.BytesIO(await human_img.read()))
        garm_pil = Image.open(io.BytesIO(await garm_img.read()))
        mask_pil = Image.open(io.BytesIO(await mask_img.read())) if mask_img is not None else None
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded image(s): {e}")

    result_img = run_tryon(
        human_img_orig=human_pil,
        garm_img=garm_pil,
        garment_des=garment_des,
        use_auto_mask=use_auto_mask,
        use_auto_crop=use_auto_crop,
        manual_mask_img=mask_pil,
        denoise_steps=denoise_steps,
        seed=seed,
    )

    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
