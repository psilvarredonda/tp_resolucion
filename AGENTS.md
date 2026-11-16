# Álbum x4

Trabajo Integrador 2 del IFTS 24. Subtarea A, super-resolución y revisión de lo reconstruido.

- El entorno de ejecución es Python 3.12 con uv y CUDA. Para iniciar la app, usar `uv run --locked app.py`; para confirmar el entorno, `uv run --locked app.py --entorno`.
- Mantener `app.py` como entrada de Gradio. El notebook de Colab instala el entorno y llama a la misma aplicación.
- Si cambian dependencias, actualizar `uv.lock` con `uv lock`. Conservar en el README la revisión y la licencia de cada modelo.
- Comparar la misma entrada de trabajo con la bicúbica y con la difusión. Si cambia el mapa, documentar el valor umbral y la escala. Presentar las detecciones como estimaciones.
- Usar ejemplos con fuente y permiso documentados en `assets/FUENTES.md`. Registrar en el README el hardware y las pruebas efectivamente realizadas.
