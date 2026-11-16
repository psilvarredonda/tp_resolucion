"""Interfaz de Álbum x4 y conexión de las etapas del procesamiento."""

import argparse
import logging
from pathlib import Path

import gradio as gr
from PIL import Image

from src.analisis import analizar
from src.difusion import PERFIL_INICIAL, PROMPTS, RUIDO_INICIAL, agrandar
from src.honestidad import AVISO, UMBRAL, exportar, mapa_de_diferencias
from src.imagenes import abrir_imagen, ampliar, preparar

RAIZ = Path(__file__).resolve().parent
EJEMPLOS = RAIZ / "assets" / "ejemplos"
LOG = logging.getLogger(__name__)


def procesar(archivo, ruido, perfil, progreso=gr.Progress()):
    if archivo is None:
        raise gr.Error("Subí una foto JPG o PNG, o elegí un ejemplo.")

    import torch

    try:
        progreso(0.05, desc="Preparando la foto y la ampliación bicúbica")
        original = abrir_imagen(archivo)
        entrada = preparar(original)
        bicubica = ampliar(entrada, Image.Resampling.BICUBIC)
        referencia = ampliar(entrada, Image.Resampling.NEAREST)
        progreso(0.20, desc="Reconstruyendo detalles con difusión")
        reconstruida = agrandar(entrada, int(ruido), perfil)
        progreso(0.75, desc="Calculando diferencias y buscando objetos con RT-DETR")
        mapa, porcentaje = mapa_de_diferencias(bicubica, reconstruida)
        informe, cajas_antes, cajas_despues = analizar(bicubica, reconstruida)
        progreso(0.95, desc="Preparando la descarga con el aviso de IA")
        actividad = f"**Foto recibida:** {original.width} × {original.height} px."
        if original.size != entrada.size:
            actividad += f" **Reducida antes de ampliar:** {entrada.width} × {entrada.height} px."
        actividad += f" **Resultado de difusión:** {reconstruida.width} × {reconstruida.height} px."
        detalle = (
            f"{actividad}\n\n**Perfil:** {perfil}. **Intensidad:** {int(ruido)}.\n\n"
            f"**{AVISO}** Revisá caras y letras antes de compartir.\n\n"
            f"El **{porcentaje:.1f}%** de los píxeles supera una diferencia suavizada de "
            f"{UMBRAL}/255 frente a la bicúbica. Los tonos claros marcan diferencias mayores. "
            "El porcentaje mide cambios visuales frente a esa ampliación clásica.\n\n"
            f"{informe}"
        )
        return (
            bicubica, reconstruida, (referencia, reconstruida), original, mapa, detalle,
            cajas_antes, cajas_despues, exportar(reconstruida),
        )
    except ValueError as error:
        raise gr.Error(str(error)) from error
    except torch.cuda.OutOfMemoryError as error:
        fallo_memoria = error
    except Exception as error:
        LOG.exception("Error durante el procesamiento")
        raise gr.Error(
            "Falló el procesamiento. Verificá que la sesión tenga GPU y conexión para "
            "descargar los modelos. El detalle está en la consola."
        ) from error

    # La excepción conserva el traceback y los tensores hasta que se libere.
    fallo_memoria.__traceback__ = None
    torch.cuda.empty_cache()
    raise gr.Error("Faltó memoria de GPU. Probá con una foto más chica.") from None


def crear_app():
    fotos = sorted(
        ruta for ruta in EJEMPLOS.iterdir() if ruta.suffix.lower() in {".jpg", ".png"}
    )
    with gr.Blocks(title="Álbum x4") as demo:
        gr.Markdown(
            "# Álbum x4\n"
            "Ampliá una foto de baja resolución y compará los detalles antes de compartirla.\n\n"
            f"**{AVISO}**"
        )
        with gr.Row():
            archivo = gr.File(
                label="Foto original (JPG o PNG)",
                file_types=[".jpg", ".jpeg", ".png"],
                type="filepath",
            )
            with gr.Column():
                perfil = gr.Radio(
                    choices=list(PROMPTS), value=PERFIL_INICIAL, label="Perfil de reconstrucción",
                    info="Más detalle pide bordes nítidos. Suave usa una descripción general. "
                         "En ambos perfiles la IA puede cambiar rasgos o texturas.",
                )
                ruido = gr.Slider(
                    0, 40, value=RUIDO_INICIAL, step=1, label="Intensidad del procesamiento",
                    info="La intensidad controla el ruido que recibe el modelo. "
                         "Probá Suave con 10 para comparar con el ajuste anterior.",
                )
                boton = gr.Button("Ampliar foto", variant="primary")
        gr.Examples(
            examples=[[str(foto)] for foto in fotos],
            inputs=[archivo], label="Fotos de ejemplo", cache_examples=False,
        )
        with gr.Tabs():
            with gr.Tab("Lado a lado"):
                with gr.Row():
                    bicubica = gr.Image(
                        label="Ampliación bicúbica", format="png",
                        height=450, buttons=["fullscreen"], interactive=False,
                    )
                    reconstruida = gr.Image(
                        label="Reconstruida por IA", format="png",
                        height=450, buttons=["fullscreen"], interactive=False,
                    )
            with gr.Tab("Deslizador"):
                comparador = gr.ImageSlider(
                    label="Original ampliada por vecino más cercano / Reconstruida por IA",
                    format="png", buttons=["fullscreen"], interactive=False,
                )
            with gr.Tab("Original y mapa"):
                with gr.Row():
                    original = gr.Image(label="Original recibida", format="png", interactive=False)
                    mapa = gr.Image(
                        label="Diferencias frente a bicúbica", format="png", interactive=False
                    )
        informe = gr.Markdown()
        with gr.Accordion("Ver cajas de detección", open=False):
            with gr.Row():
                cajas_antes = gr.Image(label="RT-DETR en bicúbica", interactive=False)
                cajas_despues = gr.Image(
                    label="RT-DETR en difusión", buttons=["fullscreen"], interactive=False
                )
        descarga = gr.File(label="Descargar PNG con aviso de IA", interactive=False)
        boton.click(
            procesar, [archivo, ruido, perfil],
            [bicubica, reconstruida, comparador, original, mapa, informe,
             cajas_antes, cajas_despues, descarga],
            concurrency_limit=1,
        )
    return demo.queue()


def comprobar_entorno():
    from importlib.metadata import version
    import platform

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Activá una GPU NVIDIA con CUDA antes de continuar.")
    gpu = torch.cuda.get_device_properties(0)
    print(f"{platform.system()} · Python {platform.python_version()} · CUDA {torch.version.cuda}")
    print(f"GPU {gpu.name} · {gpu.total_memory / 1024**3:.1f} GiB de VRAM")
    print("Paquetes " + ", ".join(
        f"{nombre} {version(nombre)}"
        for nombre in ("torch", "torchvision", "diffusers", "transformers", "gradio")
    ))


def main():
    parser = argparse.ArgumentParser(description="Álbum x4")
    parser.add_argument("--colab", action="store_true", help="Abre un enlace temporal de Gradio.")
    parser.add_argument("--entorno", action="store_true", help="Muestra la GPU y las versiones.")
    opciones = parser.parse_args()
    comprobar_entorno()
    if not opciones.entorno:
        crear_app().launch(share=opciones.colab, max_file_size="12mb")


if __name__ == "__main__":
    main()
