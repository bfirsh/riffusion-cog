# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import torch
import numpy as np
import base64
import dataclasses
import logging
import io
from pathlib import Path
import typing as T
import PIL

from cog import BasePredictor, BaseModel, Input, Path
from io import BytesIO
from typing import List
from riffusion.audio import wav_bytes_from_spectrogram_image
from riffusion.datatypes import InferenceInput, PromptInput
from riffusion.riffusion_pipeline import RiffusionPipeline
from huggingface_hub import hf_hub_download


MODEL_ID = "riffusion/riffusion-model-v1"
MODEL_CACHE = "riffusion-cache"
UNET_CACHE = "unet-cache"

# Where built-in seed images are stored
SEED_IMAGES_DIR = Path("./seed_images")
SEED_IMAGES = ["agile", "marim", "mask_beat_lines_80", "mask_gradient_dark", "mask_gradient_top_70", "mask_graident_top_fifth_75", "mask_top_third_75", "mask_top_third_95", "motorway", "og_beat", "vibes"]


class Output(BaseModel):
    audio: Path
    spectrogram: Path


def load_model(checkpoint: str):
    """
    Load the riffusion model pipeline.
    """
    model = RiffusionPipeline.from_pretrained(
        checkpoint,
        revision="main",
        torch_dtype=torch.float16,
        # Disable the NSFW filter, causes incorrect false positives
        safety_checker=lambda images, **kwargs: (images, False),
        cache_dir=MODEL_CACHE,
        local_files_only=True
    ).to("cuda")

    @dataclasses.dataclass
    class UNet2DConditionOutput:
        sample: torch.FloatTensor

    # Using traced unet from hf hub
    unet_file = hf_hub_download(
        "riffusion/riffusion-model-v1",
        filename="unet_traced.pt", 
        subfolder="unet_traced",
        cache_dir=UNET_CACHE,
        local_files_only=True
    )
    unet_traced = torch.jit.load(unet_file)

    class TracedUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_channels = model.unet.in_channels
            self.device = model.unet.device
            self.dtype = torch.float16

        def forward(self, latent_model_input, t, encoder_hidden_states):
            sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
            return UNet2DConditionOutput(sample=sample)

    model.unet = TracedUNet()

    model = model.to("cuda")

    return model

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.MODEL = load_model(checkpoint=MODEL_ID)


    def predict(
        self,
        start_prompt: str = Input(description="The prompt for your audio", default="funky synth solo"),
        end_prompt: str = Input(description="The prompt to transition to, leave blank if no transition", default=None),
        alpha: float = Input(description="Interpolation alpha if transitioning. A value of 0 uses start fully, a value of 1 uses end fully",
            default=0.5,
            ge=0,
            le=1),
        num_inference_steps: int = Input(description="Number of steps to run the diffusion model", default=50, ge=1),
        seed_image_id: str = Input(
            description="Seed image to use",
            default="vibes",
            choices=SEED_IMAGES),

    ) -> Output:
        """
        Does all the heavy lifting of the request.
        """
        # Load the seed image by ID
        init_image_path = Path(SEED_IMAGES_DIR, f"{seed_image_id}.png")
        if not init_image_path.is_file():
            return f"Invalid seed image: {seed_image_id}", 400
        init_image = PIL.Image.open(str(init_image_path)).convert("RGB")

        # fake max ints
        start_seed = np.random.randint(0, 2147483647)
        end_seed = np.random.randint(0, 2147483647)

        start = PromptInput(prompt=start_prompt, seed=start_seed)
        if not end_prompt: # no transition
            end_prompt = start_prompt
            alpha=0
        end = PromptInput(prompt=end_prompt, seed=end_seed)
        input = InferenceInput(start=start, 
            end=end, 
            alpha=alpha, 
            num_inference_steps=num_inference_steps, 
            seed_image_id=seed_image_id)

        # Execute the model to get the spectrogram image
        image = self.MODEL.riffuse(input, init_image=init_image, mask_image=None)

        # Reconstruct audio from the image
        wav_bytes, duration_s = wav_bytes_from_spectrogram_image(image)

        out_img_path = 'out/spectrogram.jpg'
        image.save('out/spectrogram.jpg')

        out_wav_path = 'out/gen_sound.wav'
        with open(out_wav_path, "wb") as f:
            f.write(wav_bytes.getbuffer())
        return Output(audio=Path(out_wav_path), spectrogram=Path(out_img_path))

