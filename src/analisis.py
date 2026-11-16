"""Compara las clases detectadas por RT-DETR antes y después."""

from collections import Counter
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

MODELO = "PekingU/rtdetr_r50vd"
REVISION = "df939e661d8c52e80608d1ec566561aabd25a4e7"
UMBRAL = 0.50
ETIQUETAS = {
    "person": "persona", "tie": "corbata", "chair": "silla", "car": "auto",
    "bicycle": "bicicleta", "dog": "perro", "cat": "gato", "handbag": "cartera",
    "bottle": "botella", "bench": "banco", "clock": "reloj", "cake": "torta",
}


@lru_cache(maxsize=1)
def cargar_detector():
    from transformers import pipeline

    # RT-DETR usa CPU y deja la memoria de video disponible para difusión.
    return pipeline("object-detection", model=MODELO, revision=REVISION, device=-1)


def informe_detecciones(antes: list[dict], despues: list[dict]) -> str:
    inicial = Counter(deteccion["label"] for deteccion in antes)
    final = Counter(deteccion["label"] for deteccion in despues)
    if not inicial and not final:
        titulo = "RT-DETR no detectó objetos de sus categorías en ninguna de las dos imágenes."
    elif inicial != final:
        titulo = "Cambió el conteo de alguna categoría. Revisá las cajas y la foto."
    else:
        titulo = "RT-DETR obtuvo el mismo conteo por categoría en las dos imágenes."

    filas = [
        f"- **{ETIQUETAS.get(clase, clase)}** · {inicial[clase]} en bicúbica, {final[clase]} en IA"
        for clase in sorted(inicial.keys() | final.keys())
    ]
    detalle = "\n".join(filas)
    return (
        f"### Detecciones de objetos (umbral de puntuación {UMBRAL:.2f})\n\n"
        f"{titulo}\n\n{detalle}\n\n"
        "El detector puede confundir texturas con objetos o pasar por alto un objeto. "
        "Sus categorías tampoco verifican la identidad de una cara ni la lectura de un texto."
    )


def dibujar_cajas(imagen: Image.Image, detecciones: list[dict]) -> Image.Image:
    salida = imagen.copy()
    dibujo = ImageDraw.Draw(salida)
    letra = ImageFont.load_default(size=16)
    for deteccion in detecciones:
        caja = deteccion["box"]
        x, y = max(0, caja["xmin"]), max(0, caja["ymin"])
        dibujo.rectangle(
            (x, y, max(x, caja["xmax"]), max(y, caja["ymax"])),
            outline="#00e5ff", width=3,
        )
        nombre = ETIQUETAS.get(deteccion["label"], deteccion["label"])
        texto = f"{nombre} {deteccion['score']:.0%}"
        dibujo.rectangle(dibujo.textbbox((x + 3, y + 2), texto, font=letra), fill="#132129")
        dibujo.text((x + 3, y + 2), texto, fill="white", font=letra)
    return salida


def analizar(bicubica: Image.Image, reconstruida: Image.Image):
    detector = cargar_detector()
    antes, despues = detector([bicubica, reconstruida], threshold=UMBRAL)
    return (
        informe_detecciones(antes, despues),
        dibujar_cajas(bicubica, antes),
        dibujar_cajas(reconstruida, despues),
    )
