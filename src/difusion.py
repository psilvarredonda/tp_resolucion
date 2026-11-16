"""Super-resolución x4 con Stable Diffusion."""

from functools import lru_cache

import numpy as np
from PIL import Image

from src.imagenes import FACTOR

MODELO = "stabilityai/stable-diffusion-x4-upscaler"
REVISION = "572c99286543a273bfd17fac263db5a77be12c4c"
PERFIL_INICIAL = "Más detalle"
RUIDO_INICIAL = 20
PROMPTS = {
    PERFIL_INICIAL: "a sharp, high quality photograph",
    "Suave": "a photograph",
}
PASOS = 30
SEMILLA = 0


@lru_cache(maxsize=1)
def cargar_modelo():
    import torch
    from diffusers import StableDiffusionUpscalePipeline

    if not torch.cuda.is_available():
        raise RuntimeError("Se necesita una GPU NVIDIA con CUDA. En Colab, activá una GPU.")
    modelo = StableDiffusionUpscalePipeline.from_pretrained(
        MODELO, revision=REVISION, dtype=torch.float16, use_safetensors=True
    ).to("cuda")
    modelo.enable_attention_slicing()
    modelo.set_progress_bar_config(disable=True)
    return modelo


def agrandar(
    entrada: Image.Image, ruido: int, perfil: str = PERFIL_INICIAL
) -> Image.Image:
    import torch

    prompt = PROMPTS[perfil]
    modelo = cargar_modelo()
    # Evita que el preprocesador redimensione la foto por su cuenta.
    multiplo = modelo.vae_scale_factor
    borde_x = (-entrada.width) % multiplo
    borde_y = (-entrada.height) % multiplo
    pixeles = np.pad(
        np.asarray(entrada), ((0, borde_y), (0, borde_x), (0, 0)), mode="edge"
    )
    salida = modelo(
        prompt=prompt,
        image=Image.fromarray(pixeles),
        noise_level=ruido,
        num_inference_steps=PASOS,
        generator=torch.Generator(device="cuda").manual_seed(SEMILLA),
    )
    if salida.nsfw_content_detected and salida.nsfw_content_detected[0]:
        raise ValueError("El filtro del modelo bloqueó este resultado. Probá otra foto.")
    return salida.images[0].crop((0, 0, entrada.width * FACTOR, entrada.height * FACTOR))
