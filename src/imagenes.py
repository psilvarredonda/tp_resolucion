"""Prepara la entrada y genera las referencias clásicas."""

from pathlib import Path

from PIL import Image, ImageOps

FACTOR = 4
LADO_MAXIMO = 256
TAMANO_MAXIMO = 4096


def abrir_imagen(ruta: str | Path) -> Image.Image:
    """Lee la foto con la orientación del archivo y compone los PNG sobre blanco."""
    try:
        with Image.open(ruta) as archivo:
            if archivo.format not in {"JPEG", "PNG"}:
                raise ValueError("Usá una foto en formato JPG o PNG.")
            if max(archivo.size) > TAMANO_MAXIMO:
                raise ValueError(f"Usá una imagen de hasta {TAMANO_MAXIMO} px por lado.")
            rgba = ImageOps.exif_transpose(archivo).convert("RGBA")
            fondo = Image.new("RGBA", rgba.size, "white")
            return Image.alpha_composite(fondo, rgba).convert("RGB")
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("No se pudo leer la foto. Probá con otro archivo JPG o PNG.") from error


def preparar(original: Image.Image) -> Image.Image:
    entrada = original.copy()
    entrada.thumbnail((LADO_MAXIMO, LADO_MAXIMO), Image.Resampling.LANCZOS)
    return entrada


def ampliar(imagen: Image.Image, metodo: Image.Resampling) -> Image.Image:
    return imagen.resize((imagen.width * FACTOR, imagen.height * FACTOR), metodo)
