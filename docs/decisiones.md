# Registro de decisiones técnicas

Proyecto final de Inteligencia Artificial 2026B (Universidad de Ibagué) · Rosa Peña, Yaiza Sanchez, Daniel Vanegas.

Este documento registra qué se verificó en los datos, qué se decidió y por qué. Los párrafos marcados con **[artículo]** están redactados para copiarse a la sección de Materiales y Métodos con cambios mínimos. Todos los números de esta página provienen de ejecuciones reales sobre los datos (no son resultados de modelos; esos se generan con `notebooks/02_resultados.ipynb`).

---

## 1. Datos

### 1.1 Fuente y descarga

- RoCoLe (Parraga-Alava et al., 2019, *Data in Brief* 25:104414), Mendeley Data, DOI 10.17632/c5yvn32dzg.2, licencia CC BY 4.0.
- Se descarga sin cuenta desde el API público de Mendeley: `https://data.mendeley.com/public-api/zip/c5yvn32dzg/download/2` (zip de 2 245 588 288 bytes). El listado de archivos del API se corta en 1 000 elementos, por eso se descarga el zip completo y no foto por foto.
- El zip contiene `Photos/` (1 560 JPEG) y `Annotations/` con `RoCoLE-json.json`, `RoCoLE-csv.csv`, `RoCoLE-coco.json`, `RoCoLe-classes.xlsx` y `RoCoLe-voc.tar.gz` (700 MB, no se usa).
- `src/prepare_data.py --download` descarga, verifica el tamaño, extrae solo lo necesario y valida los conteos.

### 1.2 Archivo de anotaciones elegido

Se usa `RoCoLE-json.json`: un registro por imagen con `External ID` (nombre del archivo), `Label.classification` (una de las 6 clases) y `Label.Leaf[0].geometry` (polígono de la hoja en píxeles; entre 9 y 82 vértices; exactamente un polígono por imagen). Se verificó que coincide al 100 % con `RoCoLe-classes.xlsx` (columnas `File`, `Binary.Label`, `Multiclass.Label`). El COCO no trae la clase multiclase.

### 1.3 Conteos verificados

| Clase | Imágenes |
|---|---|
| healthy | 791 |
| red_spider_mite | 167 |
| rust_level_1 | 344 |
| rust_level_2 | 166 |
| rust_level_3 | 62 |
| rust_level_4 | 30 |
| **Total** | **1 560** |

La etiqueta binaria del propio dataset (`Binary.Label`) marca como *unhealthy* las 769 imágenes que no son *healthy*, incluido el ácaro rojo. La métrica binaria sana/enferma sigue esa misma convención.

### 1.4 Identificación de la planta **[artículo]**

> El identificador de la planta se obtuvo del nombre de cada archivo, que sigue el patrón `C<lote>P<planta><vista>.jpg`, donde la vista es una de H1, H2, E1 o E2 (por ejemplo, `C10P10E1.jpg`). Con este patrón se identificaron 390 plantas distribuidas en 12 lotes (entre 30 y 37 plantas por lote), cada una con exactamente cuatro imágenes, lo que coincide con la descripción del conjunto de datos. Un único archivo, `C11P11HE.jpg`, no sigue el patrón; corresponde a la vista H2 de la planta C11P11, la única que carecía de ella, y se trató como tal.

Notas internas:
- No se sabe con certeza qué significan H y E (posiblemente haz y envés); no hace falta para partir por planta.
- **La clase es de la imagen, no de la planta:** 379 de las 390 plantas tienen fotos de más de una clase (la combinación más común es healthy + rust_level_1). Las vistas H1 y E1 son mayoritariamente sanas (299 y 297 de 390), mientras que H2 y E2 concentran las enfermas.

### 1.5 Orientación EXIF **[artículo]**

> Todas las fotografías incluyen la etiqueta EXIF de orientación (valor 6 en 1 501 imágenes y 8 en 59), pero los polígonos de Labelbox están expresados en las coordenadas del JPEG almacenado, sin aplicar esa rotación. Se verificó superponiendo los polígonos sobre imágenes de ambas orientaciones. Por ello, las imágenes se leyeron sin aplicar la rotación EXIF, de modo que imagen y máscara quedaran alineadas.

Implementación: `src/data.py` nunca llama a `ImageOps.exif_transpose`; `tests/test_data.py::test_exif_not_applied` lo verifica. Ojo: los visores de imágenes sí aplican el EXIF y muestran las fotos verticales.

### 1.6 Resolución y redimensionado **[artículo]**

> Las imágenes tienen tres resoluciones, todas con relación de aspecto 16:9: 2048×1152 (768), 1280×720 (479) y 4128×2322 (313). Como la hoja suele estar alargada en horizontal y ocupa en promedio el 26 % del área de la imagen (área del polígono), recortar el centro para obtener un cuadrado eliminaría las puntas de la hoja. Por ello, cada imagen se redimensionó directamente a un cuadrado, deformando la relación de aspecto, y la máscara de la hoja se rasterizó a partir del polígono con la misma transformación. Para el entrenamiento se generó una única vez una versión de 256×256 píxeles de cada imagen y de su máscara; los modelos reciben entradas de 224×224.

Notas internas:
- La caché (`rocole_256.npz`, ~410 MB con máscaras) evita decodificar JPEG de hasta 4128×2322 en cada época con los 2 vCPU de Colab. Se decodifica con `Image.draft` (reducción JPEG por potencias de 2, siempre a un tamaño ≥ 512) y luego `resize` bicúbico con antialias.
- Las máscaras se rasterizan con `PIL.ImageDraw.polygon` escalando los vértices (no hacen falta pycocotools ni shapely).

### 1.7 ¿Lote o cámara como atajo? (solo se reporta)

La resolución identifica en la práctica la cámara, y la mayoría de lotes se fotografió con una sola cámara (p. ej. lotes 6–8 a 1280×720 y lotes 2–3 a 4128×2322). Asociación con la clase (prueba chi-cuadrado de independencia, `src/explore.py`):

| Factor | χ² | gl | p | V de Cramér |
|---|---|---|---|---|
| Lote (12) | 251,6 | 55 | 6,1·10⁻²⁷ | 0,180 |
| Resolución (3) | 94,6 | 10 | 6,4·10⁻¹⁶ | 0,174 |

Porcentaje de clases por resolución (filas suman 100 %):

| Resolución | healthy | ácaro | roya 1 | roya 2 | roya 3 | roya 4 |
|---|---|---|---|---|---|---|
| 1280×720 | 52,0 | 5,6 | 28,4 | 10,2 | 2,5 | 1,3 |
| 2048×1152 | 50,4 | 9,1 | 19,1 | 13,4 | 5,6 | 2,3 |
| 4128×2322 | 49,5 | **22,4** | 19,5 | 4,5 | 2,2 | 1,9 |

Lectura: la proporción de sanas es casi constante (~50 % en cada lote), pero el ácaro se concentra en la cámara de 4128×2322 y en los lotes 2–4 y 10–11, y la roya de nivel 4 en los lotes 9, 11 y 12. La asociación es débil (V ≈ 0,18), aunque significativa. Como la partición es aleatoria por planta, lotes y cámaras aparecen en entrenamiento, validación y prueba, así que un atajo de este tipo no se detectaría como caída en prueba. Las imágenes llegan al modelo a 256×256, pero la cámara también cambia el color, la nitidez y la compresión. **No se cambió la partición.** Si hiciera falta medirlo, la alternativa sería una evaluación con lotes excluidos (*leave-lot-out*), que no está en el plan.

---

## 2. Partición

### 2.1 Algoritmo **[artículo]**

> Las imágenes se dividieron por planta, de modo que las cuatro fotografías de una planta pertenecen a un único conjunto. Como la clase se asigna por imagen y la mayoría de las plantas tiene fotografías de varias clases, no es posible una estratificación exacta. Se generaron 20 000 permutaciones aleatorias de las 390 plantas (semilla 42), cada una se cortó en 70 %, 15 % y 15 % de las plantas, y se eligió la que minimizaba la mayor desviación absoluta entre la proporción de cada clase en cada conjunto y la proporción objetivo. La desviación máxima resultante fue de 1,9 puntos porcentuales.

> Para estudiar el efecto de la cantidad de datos se seleccionaron, con el mismo criterio, subconjuntos del 25 % y del 50 % de las plantas de entrenamiento. Son anidados (las plantas del 25 % están incluidas en el 50 %); validación y prueba no cambian. La desviación máxima de proporción por clase en los subconjuntos fue de 2,5 puntos porcentuales.

La partición está en `data/splits/split.csv` (versionada) y la usan todos los experimentos. `tests/test_split.py` verifica que ninguna planta aparece en dos conjuntos, que los subconjuntos están anidados dentro de entrenamiento y que cada conjunto tiene las 6 clases. El notebook 00 confirma que el algoritmo reproduce exactamente el CSV.

### 2.2 Partición final

| Conjunto | Plantas | Imágenes | healthy | ácaro | roya 1 | roya 2 | roya 3 | roya 4 |
|---|---|---|---|---|---|---|---|---|
| Entrenamiento | 273 | 1 092 | 555 | 120 | 238 | 115 | 43 | 21 |
| Validación | 58 | 232 | 113 | 22 | 58 | 24 | 10 | 5 |
| Prueba | 59 | 236 | 123 | 25 | 48 | 27 | 9 | 4 |
| Entrenamiento 50 % | 136 | 544 | 276 | 63 | 118 | 55 | 21 | 11 |
| Entrenamiento 25 % | 68 | 272 | 136 | 33 | 59 | 28 | 11 | 5 |

**Limitación importante:** prueba tiene solo 4 imágenes de roya nivel 4 y 9 de nivel 3. La sensibilidad de esas clases es muy ruidosa (cada imagen vale 25 puntos en roya 4); por eso se reportan intervalos bootstrap.

---

## 3. Preprocesamiento, aumento y desbalance (igual para todos los modelos)

- Entrada 224×224, normalización con media y desviación de ImageNet.
- Aumento solo en entrenamiento (`torchvision.transforms.v2` sobre tensores uint8): `RandomResizedCrop(224, escala 0,7–1,0)`, volteo horizontal y vertical, rotación ±20°, `ColorJitter(brillo 0,2, contraste 0,2)`. **No se altera el tono (hue)**, porque el color naranja de la roya es diagnóstico.
- Validación y prueba: solo redimensionado 256 → 224 y normalización.
- Entropía cruzada ponderada con w_c = N / (K · n_c), calculada solo con entrenamiento (equivale a `class_weight='balanced'` de scikit-learn). Con el 100 % de entrenamiento: healthy 0,328; ácaro 1,517; roya 1 0,765; roya 2 1,583; roya 3 4,233; roya 4 8,667. En E3 se recalculan con el subconjunto.

---

## 4. Modelos

### 4.1 Parámetros por configuración (cabeza de 6 clases)

| Configuración | Arquitectura | Pesos iniciales | Parámetros | De ellos, CBAM |
|---|---|---|---|---|
| cnn_scratch | CNN propia | aleatorios | 1 174 758 | 0 |
| cnn_cbam_scratch | CNN propia + CBAM (4 bloques) | aleatorios | 1 186 540 | 11 782 (+1,0 %) |
| deit_scratch | DeiT-Tiny | aleatorios | 5 525 574 | 0 |
| mbv2_pre | MobileNetV2 | ImageNet | 2 231 558 | 0 |
| mbv2_cbam_pre | MobileNetV2 + CBAM (4 bloques) | ImageNet (CBAM aleatorio) | 2 237 316 | 5 758 (+0,26 %) |
| deit_pre | DeiT-Tiny | ImageNet | 5 525 574 | 0 |
| cnn_cbam_channel_scratch (E2) | CNN + solo canal | aleatorios | 1 186 148 | 11 390 |
| cnn_cbam_spatial_scratch (E2) | CNN + solo espacial | aleatorios | 1 175 150 | 392 |
| cnn_cbam_last2_scratch (E2) | CNN + CBAM bloques 3–4 | aleatorios | 1 185 602 | 10 844 |

### 4.2 CNN propia

Cuatro bloques [Conv3×3 → BatchNorm → ReLU] ×2 con 32, 64, 128 y 256 canales, cada uno seguido de MaxPool 2×2; Global Average Pooling, Dropout 0,3 y capa lineal. Convoluciones sin sesgo (lo absorbe BatchNorm), inicialización de Kaiming. 1,17 M de parámetros, dentro del rango pedido; no se propusieron cambios.

### 4.3 CBAM **[artículo]**

> Se implementó CBAM siguiendo a Woo et al. (2018). La atención de canal aplica un perceptrón multicapa compartido (C → C/16 → C con ReLU) a los vectores obtenidos por *average-pooling* y *max-pooling* espaciales, suma ambas salidas y aplica una sigmoide. La atención espacial concatena la media y el máximo a lo largo de los canales, aplica una convolución 7×7 y una sigmoide. Ambas se aplican en secuencia (primero canal y luego espacial) multiplicando el mapa de características. En la CNN propia, CBAM se insertó al final de cada bloque convolucional, antes del *max-pooling*.

> En MobileNetV2, CBAM se insertó dentro de la rama residual, antes de la suma, del último bloque de las etapas de 32, 64, 96 y 160 canales (`features[6]`, `[10]`, `[13]` y `[16]`, con mapas de 28², 14², 14² y 7²), del mismo modo que Woo et al. lo integran en ResNet. Así, la conexión identidad conserva las características preentrenadas mientras los parámetros de CBAM, inicializados aleatoriamente, se ajustan durante el entrenamiento.

Por qué la rama residual: si CBAM se pusiera en serie con inicialización aleatoria, las sigmoides (≈0,5 al inicio) multiplicarían las activaciones por ≈0,25 en cada punto de inserción desde la primera iteración y dañarían el preentrenamiento. Se omitió la etapa de 24 canales porque con r = 16 su MLP tendría una sola neurona. Con 32 canales (bloque 1 de la CNN propia) la capa oculta tiene 2 neuronas; se mantuvo r = 16 por fidelidad al artículo.

### 4.4 DeiT-Tiny

`timm.create_model('deit_tiny_patch16_224', pretrained=..., num_classes=6, drop_path_rate=0.1)`. El *drop path* 0,1 es el de la receta de DeiT para el modelo Tiny y se usa igual desde cero y preentrenado.

---

## 5. Entrenamiento

| | CNN / CNN+CBAM cero | DeiT cero | MBV2 / MBV2+CBAM pre | DeiT pre |
|---|---|---|---|---|
| Épocas (100 %) | 60 | 60 | 30 | 30 |
| lr inicial | 1e-3 | **3e-4** | 3e-4 | 5e-5 |
| Weight decay | **0,01** | 0,05 | **0,01** | 0,05 |
| Recorte de gradiente | — | **1,0** | — | 1,0 |

Común: AdamW, batch 32, precisión mixta (AMP, fp16) en GPU, warmup lineal de 5 épocas y luego coseno hasta el 1 % del lr inicial (por paso), selección del mejor checkpoint por F1 macro en validación, early stopping con paciencia de 15 épocas **contadas después del warmup**.

Ajustes aprobados respecto a la propuesta original:
- **DeiT desde cero, lr 3e-4 (antes 5e-4) y recorte de gradiente 1,0:** con batch 32 un lr de 5e-4 suele ser inestable en ViT; la regla de escalado lineal de DeiT (5e-4 · 32/512) daría incluso menos.
- **Weight decay de las CNN 0,01 (antes 1e-4):** en AdamW el decaimiento es desacoplado y se multiplica por el lr en cada paso, así que 1e-4 no tiene efecto práctico.
- **Sin weight decay** en sesgos, parámetros de BatchNorm/LayerNorm, `cls_token` y `pos_embed`.

### 5.1 E3: se igualan los pasos de optimización **[artículo]**

> En los experimentos con el 25 % y el 50 % de los datos de entrenamiento se mantuvo constante el número de pasos de optimización en lugar del número de épocas: las épocas, el warmup, la paciencia y la frecuencia de validación se multiplicaron por el inverso de la fracción (por ejemplo, 240 épocas con validación cada 4 para un modelo entrenado desde cero con el 25 %). Así, las diferencias reflejan la cantidad de datos y no una menor duración del entrenamiento, y el número de evaluaciones en validación, es decir, de oportunidades de seleccionar el checkpoint, es el mismo en todas las fracciones.

### 5.2 Reproducibilidad

Semillas fijadas en Python, NumPy y PyTorch (también en cada worker del DataLoader), `cudnn.benchmark = False`, `cudnn.deterministic = True`, `torch.use_deterministic_algorithms(True, warn_only=True)` y `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Algunas operaciones de CUDA no tienen versión determinista (solo emiten aviso), y entre máquinas de Colab distintas no se garantiza igualdad bit a bit. Cada JSON guarda el commit del código, versiones de librerías y GPU.

---

## 6. Evaluación

- Sobre prueba, con el mejor checkpoint: exactitud, F1 macro, precisión, sensibilidad y F1 por clase, matriz de confusión.
- Binaria sana/enferma colapsando las predicciones de 6 clases (cualquier clase distinta de healthy = enferma). Se reporta el F1 de la clase enferma.
- Parámetros, tiempo de entrenamiento y latencia de inferencia (mediana de 100 pasadas con lote 1 tras 20 de calentamiento, en la GPU de entrenamiento).
- Agregación: media ± desviación estándar entre semillas.

### 6.1 Bootstrap por planta **[artículo]**

> Los intervalos de confianza del 95 % se obtuvieron por bootstrap de percentiles con 2 000 réplicas, remuestreando plantas del conjunto de prueba con reemplazo (cada planta aporta todas sus imágenes), ya que las cuatro fotografías de una planta no son independientes. En cada réplica se calculó la métrica para cada semilla y se promedió entre semillas. Para comparar configuraciones se usaron las mismas réplicas en ambas, lo que da intervalos de diferencias pareadas.

---

## 7. Interpretabilidad (H4)

- **Grad-CAM** sobre la última capa convolucional (CNN propia: salida del bloque 4, 28×28; MobileNetV2: `features[18]`, 7×7), para la clase predicha.
- **CBAM:** mapa de atención espacial del CBAM más profundo (`cbam_last`) y promedio de todos los CBAM (`cbam_mean`).
- **DeiT:** *attention rollout* (Abnar y Zuidema, 2020). Se promedian las cabezas, se suma la identidad, se renormalizan las filas, se multiplican las 12 capas y se toma la fila del token CLS hacia los 196 parches (14×14). Para capturar las matrices se desactiva la atención "fused" de timm.
- **Métrica:** cada mapa se normaliza a [0, 1] (min-max), se lleva a 224×224 (bilineal) y fracción en la hoja = Σ(mapa·máscara) / Σ(mapa), promediada sobre las 236 imágenes de prueba. **Referencia de azar:** un mapa uniforme daría el área relativa de la hoja (también se reporta).
- Se calcula al final de cada corrida, con el mejor checkpoint. Para las figuras se guardan los mapas de 12 imágenes de prueba fijas (las 2 primeras de cada clase en orden alfabético) a 56×56.
- **Advertencia para el artículo:** Grad-CAM, CBAM y rollout son mapas de naturaleza distinta; la comparación entre familias es indicativa. La comparación más limpia de H4 es Grad-CAM de CNN+CBAM contra Grad-CAM de la CNN base (misma técnica).
- El mapa espacial de CBAM es una compuerta multiplicativa sobre las características, no un mapa de saliencia. En MobileNetV2 además actúa solo sobre la rama residual, así que un valor alto en el fondo puede significar "dejar pasar la corrección residual" y no "el modelo mira aquí". En una prueba de depuración en CPU (2 épocas, sin valor como resultado) el Grad-CAM se concentró sobre la hoja, mientras que el mapa espacial de CBAM resaltaba bastante fondo. Hay que interpretarlo con este cuidado en el artículo.

---

## 8. Plan de cómputo

### 8.1 Corridas

| Experimento | Corridas |
|---|---|
| E1: 6 configuraciones × 5 semillas | 30 |
| E2: 3 variantes de CBAM × 5 semillas | 15 |
| E3: 6 configuraciones × {25 %, 50 %} × 2 semillas | 24 |
| **Total** | **69** |

### 8.2 Estimación de tiempos (antes de medir; se corrige con `python -m src.runs --benchmark`)

Supuestos: T4, AMP, caché en RAM. CNN y CNN+CBAM desde cero ≈ 8–9 min por corrida; DeiT desde cero ≈ 10 min; preentrenados ≈ 4 min. Como E3 iguala los pasos, cada corrida de E3 cuesta lo mismo que su equivalente de E1.

| Integrante | Corridas | Estimado |
|---|---|---|
| Rosa | E1 semillas 0–1 (12) + E3 semilla 0 (12) | ~2,6 h |
| Yaiza | E1 semillas 2–3 (12) + E3 semilla 1 (12) | ~2,6 h |
| Daniel | E1 semilla 4 (6) + E2 completo (15) | ~2,6 h |
| **Total** | 69 | **~7,8 h de GPU** |

Cabe en el límite de ~6 h por cuenta. Si el benchmark en Colab da tiempos mucho mayores, primero se consulta al equipo antes de recortar (orden acordado: semillas de E3, épocas de los preentrenados, variantes de E2).

---

## 9. Limitaciones conocidas

1. Prueba con 4 imágenes de roya 4 y 9 de roya 3: métricas por clase muy ruidosas.
2. Posible atajo por lote o cámara (sección 1.7), no evaluado con lotes excluidos.
3. Deformación 16:9 → 1:1: cambia la geometría de la hoja y de las lesiones (igual para todos los modelos).
4. Los mapas de interpretabilidad de familias distintas no son estrictamente comparables.
5. E3 usa solo 2 semillas por el plazo.
