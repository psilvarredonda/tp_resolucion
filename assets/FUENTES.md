# Fuentes de las imágenes

| Archivo | Autoría, fecha y procedencia | Condición de uso |
| --- | --- | --- |
| `ejemplos/Vintage_female_photograph_2.jpg` | Autoría desconocida, 1858. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Vintage_female_photograph_2.jpg) | Dominio público |
| `ejemplos/Vintage_female_photograph_3.jpg` | Autoría desconocida, 1852. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Vintage_female_photograph_3.jpg) | Dominio público |
| `ejemplos/Vintage_female_photograph_4.jpg` | Autoría desconocida, 1855. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Vintage_female_photograph_4.jpg) | Dominio público |
| `ejemplos/Matilda_Hays_(1858).jpg` | John H. Fitzgibbon, 1858, Harvard Theatre Collection. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Matilda_Hays_(1858).jpg) | Dominio público |
| `ejemplos/Edward_Curtis_baja_res.png` y `referencia/edward-curtis.png` | Autorretrato de Edward S. Curtis, c. 1899. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:ECurtis.jpg) | Dominio público |
| `demo/detalle-prototipo.png` y `demo/rtdetr-prototipo.png` | Figuras del notebook de prueba con `Vintage_female_photograph_2.jpg`. Incluyen una reconstrucción por Stable Diffusion x4. | La fuente original está en dominio público. Ver [la ficha del modelo](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) para sus condiciones de uso. |
| `demo/colab-mas-detalle.png` y `demo/colab-deteccion-corbata.png` | Capturas de Álbum x4 en Colab con `Vintage_female_photograph_2.jpg`, perfil **Más detalle**, intensidad 20. | Retrato original de dominio público. La salida está reconstruida por Stable Diffusion x4. |

El par de Curtis permite comprobar cambios finos con una referencia. La copia de referencia se recortó y reescaló a 704 × 1024 px. La entrada de 176 × 256 px se generó reduciendo esa copia con interpolación bicúbica. En la app, cargá la entrada y mirá ojos, barba y ala del sombrero en relación con la referencia.
