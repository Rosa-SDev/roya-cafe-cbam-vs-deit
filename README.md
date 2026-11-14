# Atención convolucional (CBAM) vs. autoatención (DeiT-Tiny) para la roya del café

Código del proyecto final de Inteligencia Artificial 2026B, Universidad de Ibagué.
Integrantes: Rosa Peña, Yaiza Sanchez y Daniel Vanegas.

**Pregunta.** Con 1.560 imágenes de campo (RoCoLe), ¿qué mecanismo de atención mejora más la clasificación de la roya del café y su severidad: CBAM dentro de una CNN o la autoatención global de un Vision Transformer (DeiT-Tiny)? ¿Cómo cambia la respuesta al entrenar desde cero o desde pesos de ImageNet?

| Hipótesis | Qué se mide |
|---|---|
| H1 | CBAM mejora el F1 macro de la CNN desde cero con pocos parámetros extra |
| H2 | Desde cero DeiT-Tiny queda por debajo de las CNN; con preentrenamiento la brecha se cierra o se invierte |
| H3 | La desventaja de DeiT-Tiny crece con menos datos (25 % y 50 % del entrenamiento) |
| H4 | Los modelos con atención concentran más atención dentro de la hoja que la CNN base |

Las decisiones técnicas y su justificación (con párrafos listos para el artículo) están en [`docs/decisiones.md`](docs/decisiones.md).

---

## Estructura

```
configs/            base.yaml + un YAML por configuración (6 de E1 y 3 variantes de E2)
data/splits/        split.csv: partición por planta (versionada; la usan todos)
src/
  prepare_data.py   descarga de Mendeley y caché 256x256 (imágenes + máscaras de la hoja)
  data.py           anotaciones, ID de planta, máscaras, Dataset y aumento de datos
  split.py          partición por planta estratificada y subconjuntos de E3
  explore.py        clase por lote y por resolución (¿atajos?)
  models/           cnn.py (CNN propia), cbam.py (CBAM), factory.py (MobileNetV2, DeiT)
  train.py          una corrida: entrenamiento, evaluación en prueba e interpretabilidad
  evaluate.py       métricas, bootstrap por planta, tiempo de inferencia
  interpret.py      Grad-CAM, mapas de CBAM, attention rollout, fracción en la hoja
  runs.py           lista de las 69 corridas, reparto por integrante, cola reanudable, benchmark
  aggregate.py      JSON de corridas -> CSV (resumen, bootstrap, comparaciones H1-H4)
  report.py         tablas LaTeX (booktabs) y figuras IEEE (PDF + PNG 300 dpi)
notebooks/
  00_datos_y_particion.ipynb  descarga, exploración, verificación de la partición
  01_entrenamiento.ipynb      Colab: corre las corridas de cada integrante
  02_resultados.ipynb         tablas y figuras para el artículo
tests/              pytest (no necesitan el dataset, salvo test_split que usa split.csv)
results/            JSON por corrida y CSV agregados (sin pesos)
figures/            figuras para el artículo
docs/decisiones.md  registro de decisiones técnicas
```

El dataset y los pesos **no** se suben al repositorio (ver `.gitignore`).

## Rutas

No hay rutas fijas en el código. Se pasan como argumento o con variables de entorno:

| Variable | Argumento | Contenido |
|---|---|---|
| `ROCOLE_DATA_ROOT` | `--data-root` | carpeta con `rocole_256.npz` y `metadata.csv` |
| `ROCOLE_RESULTS_ROOT` | `--results-root` | carpeta donde se escriben `runs/` y `checkpoints/` |

## Uso rápido (local, CPU)

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows (en Linux/Mac: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest -q                                            # 52 pruebas, ~1-2 min
python -m src.prepare_data --data-root data/cache --download   # 2,25 GB desde Mendeley
python -m src.train --config configs/cnn_cbam_scratch.yaml --seed 0 --fraction 1.0 --experiment debug \
    --data-root data/cache --results-root results/debug --limit 24 --override train.epochs=1
```

Una corrida real:

```bash
python -m src.train --config configs/cnn_cbam_scratch.yaml --seed 0 --fraction 1.0 --experiment E1
```

Escribe `results/runs/E1/cnn_cbam_scratch_f100_s0.json` (métricas, matriz de confusión, predicciones de prueba, hiperparámetros, tiempos, commit), `..._maps.npz` (mapas de ejemplo) y `checkpoints/E1/....pt`. **Si el JSON ya existe, la corrida se omite** (usa `--force` para repetirla).

---

## Guía para Colab (los tres integrantes)

### Una sola vez: carpeta compartida (Rosa)

1. En Google Drive de Rosa, crear la carpeta **`IA_ProyectoFinal_RoCoLe`**.
2. Compartirla con las cuentas de Yaiza y Daniel como **Editor**.
3. Yaiza y Daniel: en *Compartido conmigo*, clic derecho sobre la carpeta → *Organizar* → **Añadir acceso directo** → *Mi unidad*. Así aparece en `/content/drive/MyDrive/IA_ProyectoFinal_RoCoLe` igual que para Rosa. El nombre del acceso directo debe ser exactamente ese.

### Una sola vez: datos (Rosa)

1. Abrir `notebooks/00_datos_y_particion.ipynb` en Colab (*Archivo → Abrir cuaderno → GitHub* → `Rosa-SDev/roya-cafe-cbam-vs-deit`).
2. *Entorno de ejecución → Ejecutar todo*. Descarga RoCoLe desde Mendeley (~2 min), genera la caché (~2 min) y la copia a `IA_ProyectoFinal_RoCoLe/data/`. También verifica la partición.

Si la descarga directa fallara: bajar el zip a mano desde https://data.mendeley.com/datasets/c5yvn32dzg/2 (*Download All*), subirlo a Drive y ejecutar `python -m src.prepare_data --data-root /content/data --raw-dir <carpeta descomprimida>`.

### Cada integrante: entrenar

1. Abrir `notebooks/01_entrenamiento.ipynb` desde GitHub en Colab.
2. *Entorno de ejecución → Cambiar tipo de entorno → GPU **T4***.
3. En la primera celda poner `MEMBER = "rosa"`, `"yaiza"` o `"daniel"`.
4. *Ejecutar todo*. El notebook monta Drive, clona el repositorio, copia la caché y ejecuta tus corridas pendientes en orden (E1 primero).
5. **Solo la primera persona, el domingo:** poner `RUN_BENCHMARK = True` una vez (~5 min) para medir los tiempos reales en la T4 y confirmar el reparto.
6. Si Colab se desconecta: volver a *Ejecutar todo*. Las corridas terminadas se omiten.
7. Avance del equipo: la última celda (`--status`).

| Integrante | Corridas | Estimado (antes del benchmark) |
|---|---|---|
| Rosa | E1 semillas 0-1 + E3 semilla 0 (24) | ~2,6 h |
| Yaiza | E1 semillas 2-3 + E3 semilla 1 (24) | ~2,6 h |
| Daniel | E1 semilla 4 + E2 completo (21) | ~2,6 h |

Consejos: Colab gratuito corta tras ~90 min sin interacción o unas 12 h de sesión; deja la pestaña abierta. Cada corrida guarda su JSON solo al terminar, así que una desconexión pierde como mucho la corrida en curso.

### Ejecución actual: GPU local + Colab

El proyecto se ejecuta en dos lugares:

- **GPU local (RTX 3050 6 GB):** corridas de `rosa` y `yaiza` (E1 semillas 0-3 y E3), escribiendo en `results/` del repositorio.
- **Colab:** corridas de `daniel` (E1 semilla 4 y E2), con `MEMBER = "daniel"` en el notebook 01.

```bash
python -m src.runs --member rosa,yaiza --execute --data-root data/cache --results-root results
```

Avance: `python -m src.runs --status --results-root results`. Si se interrumpe, se vuelve a lanzar el mismo comando y continúa donde iba.

Para unir los resultados de Colab (bajados de Drive como carpeta o como `.zip`) con los locales, sin duplicar corridas:

```bash
python -m src.merge_runs RUTA_A_LA_DESCARGA --dry-run   # ver qué haría
python -m src.merge_runs RUTA_A_LA_DESCARGA             # copiar a results/runs/
```

Las corridas que ya existen idénticas se omiten. Si la misma corrida aparece con resultados distintos, no se sobrescribe y se reporta como conflicto.

### Resultados

Con todas las corridas hechas, abrir `notebooks/02_resultados.ipynb` (en Colab o en local con Google Drive para escritorio, definiendo `ROCOLE_RESULTS_ROOT`). Genera:

- `results/aggregate/`: `runs.csv`, `summary.csv`, `bootstrap.csv`, `comparisons.csv`.
- `results/tablas/`: tablas E1, E2, parámetros y tiempos de entrenamiento (solo RTX 3050, E1 semillas 0-3) e inferencia, en CSV y `.tex` (booktabs).

Benchmark de inferencia (solo cuando hayan terminado las colas; portátil conectado al cargador):

```bash
python -m src.bench_inference
```
- `figures/`: F1 por configuración, curvas de datos, matrices de confusión, sensibilidad por clase, fracción de atención en la hoja y ejemplos de mapas (PDF y PNG a 300 dpi).
- Copia de los JSON en `results/runs/` para versionarlos.

## Reproducibilidad

- Partición fija (semilla 42) en `data/splits/split.csv`; el notebook 00 verifica que el algoritmo la reproduce.
- Semillas 0-4 fijadas en Python, NumPy y PyTorch, con modo determinista de PyTorch cuando es posible. Entre máquinas distintas de Colab no se garantiza igualdad bit a bit.
- Cada JSON registra el commit del código, versiones de librerías y la GPU.

## Datos

RoCoLe: J. Parraga-Alava, K. Cusme, A. Loor y E. Santander, "RoCoLe: A robusta coffee leaf images dataset for evaluation of machine learning based methods in plant diseases recognition", *Data in Brief* 25 (2019) 104414. Mendeley Data, DOI 10.17632/c5yvn32dzg.2, licencia CC BY 4.0.
