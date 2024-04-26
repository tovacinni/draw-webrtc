import cv2
import numpy as np
from diffusers import AutoPipelineForImage2Image, StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from diffusers.utils import load_image
import torch

class ImageToImage:

    def __init__(self):
        print("Loading SD...")
        lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"
        torch_dtype=torch.float16

        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to('cuda')


        self.pipe.load_lora_weights(lcm_lora_id, adapter_name="lcm")

        self.pipe.enable_xformers_memory_efficient_attention()
        # HACK
        self.pipe.load_lora_weights(
           "_models",
           weight_name="outlinemodel.safetensors",
           adapter_name="outline")
        self.pipe.set_adapters(["lcm", "outline"], adapter_weights=[1.0, 0.8])
        # self.pipe.set_adapters(["lcm"], adapter_weights=[1.0])
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        if self.pipe.safety_checker is not None:
            self.pipe.safety_checker = lambda images, **kwargs: (images, [False])

        # HACK
        self.prompt = "PROMPT HERE"
        self.generator = torch.Generator("cuda").manual_seed(2500)

    def forward(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        # Uncomment for scribble
        # image = 1.0 - image
        image = self.pipe(
            prompt=self.prompt,
            image=image[None],
            num_inference_steps=4,
            guidance_scale=1,
            strength=0.4,
            generator=self.generator
        ).images[0]
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
