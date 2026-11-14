# Atención convolucional (CBAM) vs. autoatención (DeiT-Tiny) para la roya del café

Comparación de dos mecanismos de atención para clasificar la roya del café y su severidad en imágenes de campo (RoCoLe, 6 clases): CBAM dentro de una CNN y la autoatención global de un Vision Transformer (DeiT-Tiny), entrenando desde cero y desde pesos de ImageNet. Incluye partición por planta, 69 corridas reproducibles, bootstrap pareado por planta y medidas de interpretabilidad (Grad-CAM, mapas de CBAM y *attention rollout*).

Proyecto final de Inteligencia Artificial 2026B, Universidad de Ibagué.
Autores: Rosa Peña, Yaiza Sanchez y Daniel Vanegas.

Las decisiones técnicas y su justificación están en [`docs/decisiones.md`](docs/decisiones.md).

---

## Pregunta e hipótesis

Con 1.560 imágenes de campo, ¿qué mecanismo de atención mejora más la clasificación de la roya del café y su severidad: CBAM dentro de una CNN o la autoatención global de DeiT-Tiny? ¿Cómo cambia la respuesta al entrenar desde cero o desde pesos de ImageNet?

| Hipótesis | Enunciado |
|---|---|
| H1 | CBAM mejora el F1 macro de la CNN con pocos parámetros extra |
| H2 | Desde cero DeiT-Tiny queda por debajo de las CNN; con preentrenamiento la brecha se cierra o se invierte |
| H3 | La desventaja de DeiT-Tiny crece con menos datos (25 % y 50 % del entrenamiento) |
| H4 | Los modelos con atención concentran más atención dentro de la hoja que la CNN base |

Experimentos: **E1** 6 configuraciones × 5 semillas; **E2** ablación de CBAM (solo canal, solo espacial, últimos 2 bloques) × 5 semillas; **E3** 25 % y 50 % de los datos de entrenamiento × 2 semillas.

## Resultados principales

E1, conjunto de prueba (236 imágenes de 59 plantas): media ± desviación entre 5 semillas; IC95 del F1 macro por bootstrap por planta.

| Modelo | Parámetros (M) | Exactitud | F1 macro | IC95 F1 macro | F1 sana/enferma |
|---|---|---|---|---|---|
| CNN (desde cero) | 1.17 | 0.689 ± 0.025 | 0.476 ± 0.044 | [0.392, 0.549] | 0.889 ± 0.038 |
| CNN+CBAM (desde cero) | 1.19 | 0.707 ± 0.019 | 0.500 ± 0.017 | [0.425, 0.560] | 0.918 ± 0.023 |
| DeiT-Tiny (desde cero) | 5.53 | 0.419 ± 0.023 | 0.275 ± 0.024 | [0.219, 0.332] | 0.706 ± 0.017 |
| MobileNetV2 (preentrenada) | 2.23 | 0.747 ± 0.011 | 0.502 ± 0.014 | [0.430, 0.561] | 0.929 ± 0.006 |
| MobileNetV2+CBAM (preentrenada) | 2.24 | 0.742 ± 0.015 | 0.480 ± 0.035 | [0.421, 0.534] | 0.941 ± 0.007 |
| DeiT-Tiny (preentrenada) | 5.53 | 0.743 ± 0.026 | 0.517 ± 0.051 | [0.447, 0.571] | 0.926 ± 0.023 |

Veredicto (diferencias pareadas A − B en F1 macro con IC95 por bootstrap por planta; se considera respaldada solo si el IC95 excluye el 0):

- **H1, no respaldada.** CNN+CBAM − CNN: +0.023 [−0.035, 0.083]; MobileNetV2+CBAM − MobileNetV2: −0.022 [−0.067, 0.025]. El F1 sana/enferma sí mejora con CBAM en la CNN desde cero: +0.029 [0.002, 0.058].
- **H2, respaldada.** Desde cero, DeiT-Tiny − CNN: −0.201 [−0.286, −0.113]. Con preentrenamiento, DeiT-Tiny − MobileNetV2: +0.014 [−0.041, 0.066]: la brecha se cierra, pero no se invierte.
- **H3, no respaldada.** La brecha desde cero no crece con menos datos: −0.192 con 25 %, −0.234 con 50 % y −0.201 con 100 % (E3 usa 2 semillas).
- **H4, no respaldada.** CBAM reduce la fracción de atención dentro de la hoja medida con Grad-CAM (CNN: −0.036 [−0.050, −0.022]; MobileNetV2: −0.026 [−0.032, −0.020]). Los mapas propios de CBAM (0.25–0.29) y el *rollout* de DeiT-Tiny (0.30 desde cero, 0.41 preentrenada) quedan por debajo del Grad-CAM de las CNN base (0.54 y 0.56). Línea base (área media de la hoja): 0.267.

Todas las tablas (E1, E2, E3, comparaciones, tiempos, inferencia) están en [`results/tablas/`](results/tablas) y las figuras en [`results/figuras/`](results/figuras).

## Datos

**RoCoLe** (*Robusta Coffee Leaf images*): 1.560 imágenes de hojas de café robusta de 390 plantas (4 fotos por planta), con anotación de clase y polígono de la hoja.

> J. Parraga-Alava, K. Cusme, A. Loor y E. Santander, "RoCoLe: A robusta coffee leaf images dataset for evaluation of machine learning based methods in plant diseases recognition", *Data in Brief*, vol. 25, 104414, 2019. DOI: [10.1016/j.dib.2019.104414](https://doi.org/10.1016/j.dib.2019.104414).

- Conjunto de datos: Mendeley Data, DOI [10.17632/c5yvn32dzg.2](https://doi.org/10.17632/c5yvn32dzg.2).
- Licencia: CC BY 4.0.
- Descarga automática (2,25 GB, sin cuenta) y caché de 256×256 con máscaras de la hoja:

  ```bash
  python -m src.prepare_data --data-root data/cache --download
  ```

- Descarga manual: *Download All* en https://data.mendeley.com/datasets/c5yvn32dzg/2, descomprimir y ejecutar `python -m src.prepare_data --data-root data/cache --raw-dir <carpeta descomprimida>`.

El dataset y los pesos de los modelos **no** se suben al repositorio (ver `.gitignore`).

## Requisitos e instalación

Python 3.10 o posterior. Probado con Python 3.13 y PyTorch 2.15 (CUDA 13.2) en local, y con PyTorch 2.14 (CUDA 13.0) en Colab.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; en Linux/Mac: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132   # GPU (o /whl/cpu para CPU)
pip install -r requirements.txt
pytest -q                         # pruebas con datos sintéticos (~2-3 min en CPU)
```

Para ejecutar los notebooks en local hace falta además `pip install nbconvert ipykernel`.

## Estructura del repositorio

```
configs/            base.yaml + un YAML por configuración (6 de E1 y 3 variantes de E2)
data/splits/        split.csv: partición por planta (versionada)
src/
  prepare_data.py   descarga de Mendeley y caché 256x256 (imágenes + máscaras de la hoja)
  data.py           anotaciones, ID de planta, máscaras, Dataset y aumento de datos
  split.py          partición por planta estratificada y subconjuntos de E3
  explore.py        clase por lote y por resolución (¿atajos?)
  models/           cnn.py (CNN propia), cbam.py (CBAM), factory.py (MobileNetV2, DeiT-Tiny)
  train.py          una corrida: entrenamiento, evaluación en prueba e interpretabilidad
  evaluate.py       métricas, bootstrap por planta, tiempo de inferencia
  interpret.py      Grad-CAM, mapas de CBAM, attention rollout, fracción en la hoja
  runs.py           lista de las 69 corridas, selección por experimento y semillas, cola reanudable
  merge_runs.py     une resultados de otra máquina (carpeta o .zip) sin duplicar corridas
  aggregate.py      JSON de corridas -> CSV (resumen, bootstrap, comparaciones H1-H4)
  report.py         tablas LaTeX (booktabs) y figuras IEEE (PDF + PNG 300 dpi)
  bench_inference.py  latencia y throughput de inferencia en la GPU local
notebooks/
  00_datos_y_particion.ipynb  descarga, exploración y verificación de la partición
  01_entrenamiento.ipynb      entrenamiento en Colab (opcional)
  02_resultados.ipynb         tablas y figuras
tests/              pytest (datos sintéticos, salvo test_split que usa split.csv)
results/
  runs/E1|E2|E3/    un JSON por corrida (métricas, predicciones, tiempos, entorno, commit)
  tablas/           CSV y .tex
  figuras/          PDF y PNG
figures/            figura de imágenes y máscaras (notebook 00)
docs/decisiones.md  registro de decisiones técnicas
```

Rutas: no hay rutas fijas en el código. Se pasan con `--data-root` / `--results-root` o con las variables de entorno `ROCOLE_DATA_ROOT` (carpeta con `rocole_256.npz` y `metadata.csv`) y `ROCOLE_RESULTS_ROOT` (donde se escriben `runs/` y `checkpoints/`).

## Cómo reproducir

1. **Datos y partición.** `python -m src.prepare_data --data-root data/cache --download` y, opcionalmente, el notebook `00_datos_y_particion.ipynb`, que explora los datos y verifica que el algoritmo reproduce `data/splits/split.csv`.

2. **Corridas (E1, E2, E3).** La cola es reanudable: cada corrida escribe su JSON al terminar y las que ya tienen JSON se omiten, así que tras una interrupción basta repetir el comando.

   ```bash
   python -m src.runs --status --results-root results                  # avance por experimento
   python -m src.runs --experiment E1 --execute --data-root data/cache --results-root results
   python -m src.runs --experiment E2 --execute --data-root data/cache --results-root results
   python -m src.runs --experiment E3 --execute --data-root data/cache --results-root results
   ```

   `--seeds` limita las semillas (`0-3`, `0,2,4` o `4`); sin `--experiment` se ejecutan las 69 corridas pendientes. Una corrida suelta:

   ```bash
   python -m src.train --config configs/cnn_cbam_scratch.yaml --seed 0 --fraction 1.0 --experiment E1
   ```

   Escribe `results/runs/E1/cnn_cbam_scratch_f100_s0.json`, `..._maps.npz` (mapas de ejemplo) y `results/checkpoints/E1/....pt`. Usa `--force` para repetir una corrida existente.

   **Colab (opcional).** `notebooks/01_entrenamiento.ipynb` ejecuta en Colab el experimento y las semillas elegidos (`EXPERIMENTS`, `SEEDS`) y guarda los resultados en Google Drive. Para unirlos con los locales:

   ```bash
   python -m src.merge_runs RUTA_A_LA_DESCARGA --dry-run   # ver qué haría
   python -m src.merge_runs RUTA_A_LA_DESCARGA             # copiar a results/runs/
   ```

   Las corridas idénticas se omiten; si la misma corrida aparece con resultados distintos, no se sobrescribe y se reporta como conflicto.

3. **Tablas y figuras.** Ejecutar `notebooks/02_resultados.ipynb` (en local: `jupyter nbconvert --to notebook --execute --inplace notebooks/02_resultados.ipynb`). Escribe los agregados y tablas en `results/tablas/` y las figuras en `results/figuras/`.

4. **Benchmark de inferencia.** Con todas las corridas terminadas y la GPU libre:

   ```bash
   python -m src.bench_inference
   ```

   Usa los modelos de E1 semilla 0 y escribe `results/tablas/inferencia_rtx3050.csv` y `tabla_inferencia.{csv,tex}`.

## Hardware y tiempos

| GPU | Experimentos |
|---|---|
| NVIDIA GeForce RTX 3050 6 GB | E1 semillas 0–3 y E3 |
| NVIDIA Tesla T4 (Google Colab) | E1 semilla 4 y E2 |

Las métricas de clasificación usan las 5 semillas sin importar la GPU. Los tiempos se reportan solo con la RTX 3050.

Entrenamiento (E1, semillas 0–3, media ± desviación; época estable = mediana de segundos por época sin la primera):

| Modelo | Parámetros | Parámetros CBAM | Entrenamiento (min) | Época estable (s) | Épocas |
|---|---|---|---|---|---|
| CNN (desde cero) | 1,174,758 | 0 | 4.7 ± 0.7 | 5.5 ± 0.1 | 48 ± 9 |
| CNN+CBAM (desde cero) | 1,186,540 | 11,782 | 8.9 ± 1.4 | 9.6 ± 0.2 | 54 ± 8 |
| DeiT-Tiny (desde cero) | 5,525,574 | 0 | 4.6 ± 0.2 | 4.4 ± 0.5 | 60 ± 1 |
| MobileNetV2 (preentrenada) | 2,231,558 | 0 | 2.7 ± 0.3 | 4.8 ± 0.4 | 29 ± 2 |
| MobileNetV2+CBAM (preentrenada) | 2,237,316 | 5,758 | 2.8 ± 0.1 | 5.3 ± 0.4 | 28 ± 3 |
| DeiT-Tiny (preentrenada) | 5,525,574 | 0 | 2.5 ± 0.4 | 4.6 ± 0.3 | 28 ± 3 |

Inferencia (modelos de la semilla 0, entrada 3×224×224; latencia con lote 1 = mediana de 200 pasadas; throughput con lote 32):

| Modelo | Latencia FP32 (ms) | Throughput FP32 (img/s) | Latencia FP16 (ms) | Throughput FP16 (img/s) |
|---|---|---|---|---|
| CNN (desde cero) | 2.42 | 517 | 2.79 | 874 |
| CNN+CBAM (desde cero) | 5.56 | 363 | 7.21 | 623 |
| DeiT-Tiny (desde cero) | 10.84 | 789 | 13.56 | 1625 |
| MobileNetV2 (preentrenada) | 9.89 | 715 | 12.94 | 1316 |
| MobileNetV2+CBAM (preentrenada) | 13.50 | 696 | 19.76 | 1243 |
| DeiT-Tiny (preentrenada) | 10.08 | 789 | 13.45 | 1624 |

## Reproducibilidad

- Partición fija (semilla 42) en `data/splits/split.csv`; el notebook 00 verifica que el algoritmo la reproduce. Ninguna planta aparece en dos conjuntos.
- Semillas 0–4 fijadas en Python, NumPy y PyTorch (también en los workers del DataLoader), con algoritmos deterministas de PyTorch cuando existen. Entre GPU distintas no se garantiza igualdad bit a bit.
- Hiperparámetros en `configs/` (YAML); cada corrida guarda su configuración completa.
- Cada JSON registra el commit del código, si había cambios sin confirmar, las versiones de librerías y la GPU. Las 69 corridas publicadas se ejecutaron con el mismo commit y sin cambios locales.
