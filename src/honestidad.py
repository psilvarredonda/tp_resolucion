"""Mapa de diferencias y exportación identificada como reconstrucción."""

from pathlib import Path
from tempfile import gettempdir
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from scipy.ndimage import gaussian_filter

UMBRAL = 8
ESCALA_MAPA = 30
AVISO = "Reconstruida por IA. Los detalles finos son estimaciones."
EXPORTACIONES = Path(gettempdir()) / "album-x4"


def mapa_de_diferencias(bicubica: Image.Image, reconstruida: Image.Image):
    base = np.asarray(bicubica, dtype=np.float32)
    nueva = np.asarray(reconstruida, dtype=np.float32)
    diferencia = gaussian_filter(np.abs(nueva - base).mean(axis=2), sigma=4)
    porcentaje = float((diferencia > UMBRAL).mean() * 100)
    escala = np.clip(diferencia / ESCALA_MAPA, 0, 1)
    calor = np.stack((escala, escala**2, escala**6), axis=2)
    return Image.fromarray((calor * 255).astype(np.uint8)), porcentaje


def exportar(imagen: Image.Image) -> str:
    EXPORTACIONES.mkdir(parents=True, exist_ok=True)
    ruta = EXPORTACIONES / f"album-x4-{uuid4().hex}.png"
    letra = ImageFont.load_default(size=16)
    texto = AVISO if letra.getlength(AVISO) + 16 <= imagen.width else "Reconstruida por IA"
    ancho = max(imagen.width, round(letra.getlength(texto)) + 18)
    lienzo = Image.new("RGB", (ancho, imagen.height + 33), "#20252b")
    lienzo.paste(imagen, ((ancho - imagen.width) // 2, 0))
    dibujo = ImageDraw.Draw(lienzo)
    dibujo.text((8, imagen.height + 6), texto, fill="white", font=letra)
    metadatos = PngImagePlugin.PngInfo()
    metadatos.add_text("Aviso", AVISO)
    lienzo.save(ruta, pnginfo=metadatos)
    return str(ruta)
