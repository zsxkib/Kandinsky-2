# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import subprocess
import time
import torch
from typing import List
from cog import BasePredictor, Input, Path

from kandinsky2.kandinsky2_1_model import Kandinsky2_1
from kandinsky2.configs import CONFIG_2_1
from omegaconf.dictconfig import DictConfig
from copy import deepcopy


MODEL_CACHE = "weights_cache"
MODEL_URL = "https://weights.replicate.delivery/default/kandinsky-2-1/2_1.tar"
VIT_L_14_URL = "https://weights.replicate.delivery/default/kandinsky-2-1/ViT-L-14.pt"

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE

def download_weights(url, dest):
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    command = ["pget", "-vf", url, dest]
    if ".tar" in url:
        command.append("-x")
    try:
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")
    
class Predictor(BasePredictor):
    def setup(self):
        model_path = os.path.join(
            MODEL_CACHE, "2_1"
        )
        if not os.path.exists(model_path):
            download_weights(MODEL_URL, model_path)

        vit_l_14_path = os.path.join(MODEL_CACHE, "ViT-L-14.pt")
        if not os.path.exists(vit_l_14_path):
            download_weights(VIT_L_14_URL, vit_l_14_path)

        config = DictConfig(deepcopy(CONFIG_2_1))
        config["model_config"]["use_flash_attention"] = False
        config["tokenizer_name"] = os.path.join(MODEL_CACHE, "2_1", "text_encoder")
        config["text_enc_params"]["model_path"] = os.path.join(
            MODEL_CACHE, "2_1", "text_encoder"
        )
        config["prior"]["clip_mean_std_path"] = os.path.join(
            MODEL_CACHE, "2_1", "ViT-L-14_stats.th"
        )
        config["image_enc_params"]["ckpt_path"] = os.path.join(
            MODEL_CACHE, "2_1", "movq_final.ckpt"
        )
        config["cache_dir"] = MODEL_CACHE
        cache_model_name = os.path.join(MODEL_CACHE, "2_1", "decoder_fp16.ckpt")
        cache_prior_name = os.path.join(MODEL_CACHE, "2_1", "prior_fp16.ckpt")
        self.model = Kandinsky2_1(
            config, cache_model_name, cache_prior_name, "cuda", task_type="text2img"
        )

    def generate_text2img_with_seed(self, seed, *args, **kwargs):
        if seed is not None:
            torch.manual_seed(seed)
        return self.model.generate_text2img(*args, **kwargs)

    def predict(
        self,
        prompt: str = Input(description="Input Prompt", default="red cat, 4k photo"),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=4
        ),
        scheduler: str = Input(
            description="Choose a scheduler",
            default="p_sampler",
            choices=["ddim_sampler", "p_sampler", "plms_sampler"],
        ),
        prior_cf_scale: int = Input(default=4),
        prior_steps: str = Input(default="5"),
        width: int = Input(
            description="Choose width. Lower the setting if out of memory.",
            default=512,
            choices=[256, 288, 432, 512, 576, 768, 1024],
        ),
        height: int = Input(
            description="Choose height. Lower the setting if out of memory.",
            default=512,
            choices=[256, 288, 432, 512, 576, 768, 1024],
        ),
        batch_size: int = Input(
            description="Choose batch size. Lower the setting if out of memory.",
            default=1,
            choices=[1, 2, 3, 4],
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        output_format: str = Input(
            description="Output image format",
            choices=["webp", "jpeg", "png"],
            default="webp",
        ),
    ) -> List[Path]:
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        output = self.generate_text2img_with_seed(
            seed,
            prompt,
            num_steps=num_inference_steps,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            h=height,
            w=width,
            sampler=scheduler,
            prior_cf_scale=prior_cf_scale,
            prior_steps=prior_steps,
        )
        output_paths = []
        for i, sample in enumerate(output):
            output_path = f"/tmp/out-{i}.{output_format}"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
 
