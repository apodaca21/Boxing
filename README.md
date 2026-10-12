# EvoBox AI

Seguimiento corporal, recopilación de muestras, entrenamiento NEAT y un
entrenador en vivo. La webcam y MediaPipe Pose Landmarker alimentan el
esqueleto y las características biomecánicas. El modo en vivo cuenta golpes
con un detector provisional. NEAT solo se usa en ese modo cuando existe un
genoma entrenado.

## Requisitos

- macOS (probado para instalación en Apple Silicon `arm64`).
- Python 3.14 de 64 bits. El entorno de desarrollo detectado fue
  CPython 3.14.7 en `/opt/homebrew/bin/python3`.
- Permiso de cámara para la aplicación desde la que se ejecuta Python.

Las versiones están fijadas en `requirements.txt`. `opencv-contrib-python`
proporciona el módulo `cv2`; no instales `opencv-python` en el mismo entorno.

## Instalación

Desde la raíz del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

El modelo oficial ligero debe existir como
`models/pose_landmarker_lite.task`. El proyecto ya lo incluye. Si falta,
descárgalo una sola vez:

```bash
mkdir -p models
curl --fail --location \
  --output models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

La aplicación valida que el archivo exista y no esté vacío antes de abrir la
cámara.

## Ejecución

Seguimiento original:

```bash
source .venv/bin/activate
python main.py
```

Recopilación de muestras etiquetadas:

```bash
python main.py --mode collect
```

Ambos modos aceptan `--camera-index`, `--width`, `--height`, `--no-mirror` y
`--model`. La recopilación también acepta `--data-dir`; su valor predeterminado
es `data/`.

Controles:

- `Q`, `q` o `Escape`: cerrar.
- La vista aparece como espejo de forma predeterminada, pero las coordenadas y
  los nombres izquierda/derecha conservan la anatomía reportada por MediaPipe.

Opciones útiles:

```bash
python main.py --camera-index 1
python main.py --width 640 --height 480 --no-mirror
python main.py --help
```

## Permiso de cámara en macOS

Si la cámara no abre:

1. Cierra aplicaciones que puedan estar usando la webcam.
2. Abre **Ajustes del Sistema > Privacidad y seguridad > Cámara**.
3. Permite la cámara para Terminal, iTerm, Cursor o la aplicación desde la que
   ejecutas el comando.
4. Cierra y vuelve a abrir esa aplicación.

En macOS se intenta primero AVFoundation y después el backend automático de
OpenCV. El programa solicita 640x480 de forma predeterminada y muestra un error
accionable si ningún backend entrega fotogramas.

## Qué muestra

- Estado real: `CUERPO DETECTADO` o `BUSCANDO PERSONA`.
- FPS calculado con un reloj monotónico y una ventana móvil.
- Ángulos de los codos izquierdo y derecho.
- Brazos izquierdo y derecho con colores distintos.
- Torso y cadera como referencias, omitiendo conexiones con baja visibilidad o
  presencia.

Las distancias hombro–muñeca de ambos lados también se calculan y normalizan
con la distancia entre hombros. Se conservan para el futuro vector de
características, pero no saturan el HUD.

## Recopilación

La postura de esta fase es `orthodox`: pie y brazo izquierdos adelantados. Las
teclas `0` a `4` seleccionan la clase y `Espacio` inicia la captura. Durante la
cuenta regresiva y la grabación no se puede cambiar la clase.

1. Ponte frente a la cámara con todo el tren superior visible.
2. Selecciona la clase:
   - `0`: Guardia / Neutral, incluyendo guardia y movimientos suaves.
   - `1`: Jab izquierdo.
   - `2`: Cross derecho.
   - `3`: Gancho izquierdo.
   - `4`: Gancho derecho.
3. Presiona `Espacio`.
4. Espera la cuenta de 3 segundos.
5. Ejecuta el movimiento completo durante unos 2 segundos, incluyendo la
   preparación, el golpe y el regreso a guardia.
6. Si la captura es válida, el contador aumenta. Si se rechaza, la ventana
   muestra el motivo y puedes repetirla con `Espacio`.

La etiqueta siempre es la elegida manualmente. No se infiere el tipo de golpe.
Una muestra rechazada no se escribe.

## Dataset

Cada ejecución completa es una unidad indivisible. Sus fotogramas no deben
separarse entre entrenamiento, validación y prueba; el campo `split_unit` vale
`sample` para dejarlo explícito.

```text
data/
├── index.csv
├── sessions/<session_id>.json
└── samples/<session_id>/<sample_id>.json
```

`index.csv` contiene `sample_id`, `session_id`, `label_id`, `label_name`,
`created_at`, `duration_ms`, `frame_count`, `approximate_fps`, `stance`,
`relative_path` y `schema_version`.

Cada JSON de muestra contiene:

- `schema_version` y `split_unit`.
- `sample_id`, `session_id`, `label_id` y `label_name`.
- `created_at`, `duration_ms`, `frame_count` y `approximate_fps`.
- `stance` con el valor `orthodox` y su descripción.
- `camera`: índice, resolución pedida, resolución real, espejo y
  `coordinate_space` con valor `anatomical_normalized`.
- `privacy`: imágenes, audio y video desactivados.
- `frames`: por fotograma, `timestamp_ms`, `elapsed_ms`, `pose_detected`,
  articulaciones y características.

Las articulaciones guardadas son hombros, codos, muñecas y cadera. Cada una
tiene `x`, `y`, `z`, `visibility`, `presence` y `reliable`, o `null` si falta.
Las coordenadas pertenecen al espacio anatómico de MediaPipe aunque la ventana
se muestre como espejo.

Las características de cada fotograma son ángulos de ambos codos, distancias
normalizadas de hombro a muñeca, posición de cada muñeca respecto al punto medio
de los hombros y separación entre muñecas. Los valores ausentes se guardan como
`null`. Los timestamps permiten calcular después velocidades y cambios angulares
sin asumir un FPS constante. El FPS aproximado es
`(fotogramas - 1) * 1000 / duration_ms`.

Una muestra válida necesita persona en al menos el 80 % de los fotogramas,
brazos y cadera fiables en al menos el 70 %, 12 fotogramas como mínimo, duración
real entre 1.5 y 2.8 segundos, timestamps estrictamente crecientes y únicamente
números finitos o `null`.

## Entrenamiento NEAT

El modo de entrenamiento no abre la webcam. Lee `data/`, comprueba integridad y
solo continúa cuando hay muestras reales válidas de las cinco clases y al menos
tres sesiones. Si falta información, escribe `results/data_audit.json` y se
detiene sin inventar métricas.

```bash
python main.py --mode train --experiment quick
python main.py --mode train --experiment initial --seed 42
python main.py --mode train --population 30 --generations 5
```

`quick` evoluciona 30 redes durante 5 generaciones. `initial` evoluciona 100
durante 50. La población y las generaciones de la línea de comandos reemplazan
esos valores. La semilla predeterminada es 42 y mantiene fijo el conjunto de
prueba.

## Reporte académico

`python main.py --mode dataset-report` no abre la webcam. Audita las muestras
reales de `data/` y escribe `results/academic/`. Si hay clases y sesiones
suficientes para la división existente, ejecuta el mismo entrenamiento NEAT.
El conjunto de prueba no elige el genoma ni cambia hiperparámetros.

Sin un entrenamiento real, el reporte deja pendientes `neat_evolution.csv`,
`evaluation_metrics.json`, `confusion_matrix.png`, `fitness_evolution.png` y
`model_topology.png`. No se inventan precisiones ni se completan clases con
datos sintéticos. El detector heurístico y NEAT no se comparan como si contar
eventos en video y clasificar una secuencia recortada fueran la misma tarea.

La función objetivo, calculada solo sobre entrenamiento, es:

```text
F(g) = 0.65 * R_punch(g) + 0.30 * S_neutral(g) + 0.05 / (1 + C)
```

`R_punch` es el recall promedio de las cuatro clases de golpe, `S_neutral` es
el acierto sobre neutral y `C` es el número de conexiones habilitadas. Los
coeficientes son valores iniciales y todavía deben validarse experimentalmente.
La clase predicha es la salida de mayor activación; esas salidas no se tratan
como probabilidades calibradas.

Cada muestra se remuestrea a 8 instantes y produce un vector de 92 valores:
ángulos, distancias, posiciones relativas, velocidades de muñeca y tasa de
extensión. Los datos ausentes no se reemplazan por cero. Tras normalizar con
estadísticas exclusivas del entrenamiento, una ausencia se codifica como `-4`,
fuera del rango recortado `[-3, 3]`.

La división predeterminada entrega sesiones completas a entrenamiento,
validación o prueba. El esquema actual no contiene identificador de persona; la
opción `--split person` solo funciona si todas las muestras lo incluyen y hay
al menos tres personas. El ganador se elige por fitness de entrenamiento. Las
gráficas, matrices y el modelo quedan en `results/`, directorio excluido de Git.

Una red MLP fija de scikit-learn se evalúa sobre la misma partición como línea
base. La comparación es descriptiva y no afirma que NEAT sea superior.

## Entrenador en vivo

```bash
python main.py --mode live --classifier heuristic
python main.py --mode live --classifier neat
python main.py --mode live --classifier external
```

El modo abre la webcam, dibuja el seguimiento y muestra contadores. La sesión
empieza contando. `P` o la barra espaciadora pausan y reanudan, `R` reinicia y
`Q` cierra. En el perfil normal un golpe se cuenta cuando el brazo completa
salida y regreso; dejar el brazo extendido no acumula repeticiones. La
extensión se mide en tres dimensiones, porque un golpe hacia la cámara casi no
alarga el brazo en la imagen plana.

Con `--classifier heuristic` el panel indica `MODELO: HEURISTICO`. Reconoce
guardia, jab izquierdo, cross derecho, gancho izquierdo y gancho derecho en
postura orthodox. Si el movimiento no se puede clasificar, muestra
`MOVIMIENTO NO IDENTIFICADO`. No muestra porcentajes.

Sparring relaja el regreso a la guardia y confirma cada brazo cuando el golpe
frena. La sensibilidad inicial de ese modo es `HIGH`. `1`, `2` y `3` cambian
entre `NORMAL`, `HIGH` y `MAXIMUM` durante la sesión. `0` restaura el perfil
con el que arrancó. `D` muestra el diagnóstico y `C` ejecuta una calibración
de unos 14 segundos. Esa rutina solo propone umbrales; no crea muestras para
NEAT.

```bash
python main.py --mode live --classifier heuristic --training-style sparring
```

Estos perfiles son una hipótesis inicial. Las pruebas automáticas usan
secuencias sintéticas y no demuestran la precisión frente a la webcam.

## Evaluación guiada

```bash
python main.py --mode benchmark --classifier heuristic --sensitivity normal
python main.py --mode benchmark --classifier heuristic --sensitivity high
python main.py --mode benchmark --classifier external --sensitivity maximum
```

La pantalla pide prepararse, ejecutar el movimiento, volver a guardia y pasar
a la siguiente repetición. Son 10 repeticiones de jab izquierdo, cross derecho,
gancho izquierdo, gancho derecho y guardia. `--repetitions`, `--prepare-ms`,
`--active-ms` y `--return-ms` cambian esos intervalos.

`Y` confirma que ejecutaste el movimiento pedido. `0` a `4` corrigen la
etiqueta antes de confirmar. `X` excluye la repetición. Una predicción durante
el intervalo no cuenta como acierto hasta ese paso. Cada predicción se empareja
como máximo con un golpe verificado.

Los informes quedan en `results/benchmarks/<sensibilidad>/<fecha>/` como JSON,
CSV y, si hay al menos una repetición verificada de cada clase, una matriz.
La comparación entre perfiles no elige un ganador por haber detectado más
golpes. Las ventanas confirmadas se guardan en `data/` con el esquema de NEAT,
sin imágenes ni video. `--no-save-samples` omite ese guardado. Si el intervalo
activo no dura entre 1.5 y 2.8 segundos, la muestra no supera la validación
existente.

`--classifier neat` carga el genoma de `--neat-model` o, si no se indica, el
`winner.pkl` más reciente de `results/`. Reutiliza el orden de características,
la normalización y la ventana temporal guardadas en el entrenamiento. Si no hay
modelo, el programa se detiene y no sustituye el resultado por el detector
provisional.

El clasificador externo usa el LSTM de
[ACM40960/Boxing](https://github.com/ACM40960/Boxing), licencia MIT, copyright
2025 de Jayaganeshan Thanga Kumar y Kabilesh Sekar:

```bash
python main.py --mode live --classifier external
```

Ese modo abre la webcam una sola vez. MediaPipe dibuja el esqueleto y YOLO11
pose alimenta una ventana deslizante de 25 fotogramas con 16 valores
normalizados. El LSTM tiene dos capas de 128 unidades y cuatro salidas: jab,
cross, hook y uppercut. La ventana no se vacía después de cada golpe. El
perfil normal espera 20 fotogramas antes de otro evento; alta espera 10 y
máxima espera 5. El archivo y el orden de las clases no cambian. Los pesos se cargan con `weights_only=True`. YOLO se carga con la
opción restringida de Ultralytics. No se instaló CUDA ni TensorRT. Ultralytics
y el archivo `yolo11m-pose.pt` están bajo AGPL-3.0. El panel dice
`MODELO: YOLO + LSTM`. Jab y cross conservan el nombre del modelo; un hook
solo indica el lado cuando un brazo se desplaza con claridad; el uppercut
tiene su propio contador.

[just-box](https://github.com/nlevarun/just-box) y
[CVboxing](https://github.com/JamesHandler28/CVboxing) aportaron ideas de
detección temporal, pero no publican una licencia identificable y su código no
se incorporó.

## Target Training

```bash
python main.py --mode targets
python main.py --mode targets --training-order blocked
```

`--training-order blocked` practica un solo golpe por bloque: 10 jab, 10 cross,
10 ganchos izquierdos, 10 ganchos derechos y 10 guardias. La guardia es el
último bloque. `--target-goal` cambia esa cantidad. `--only-class 0` practica
solo guardia y `--only-class 4` solo el gancho derecho. Después de cada intento
la pantalla se detiene y muestra la técnica que sigue; espacio la inicia. Al
terminar se ve cuántas capturas son válidas, rechazadas o pendientes. `P`
repite las clases incompletas. `0`–`4` y después `G` guardan una captura ya
revisada. El modo mezclado anterior sigue disponible.

La webcam muestra el esqueleto y un objetivo por golpe: jab a la izquierda,
cross a la derecha y ganchos más laterales. La guardia pide mantenerse sin
golpear. El círculo sigue al torso, en anchos de hombro, y no se cuenta un
acierto solo porque la muñeca lo cruce: hace falta el evento del detector
heurístico, la muñeca del brazo correspondiente y que no sea un duplicado.

`ESPACIO` inicia o pausa, `R` dos veces reinicia la ronda, `H` abre sesiones ya
guardadas, `V` abre la revisión y `Q` sale. Al terminar la ronda se abre la
revisión. Cada ciclo pide las cinco clases, incluida la guardia, hasta reunir
10 verificadas de cada una. Una ventana con seguimiento válido queda pendiente
aunque el detector no acierte. `0`–`4` eligen la etiqueta, `G` la guarda y `X`
la descarta. `N` deja la captura sin decidir. `ESPACIO` reproduce el esqueleto
guardado y la coma o el punto avanzan un fotograma. No se dibujan articulaciones
que no estén en el archivo, ni la imagen de la cámara.

```bash
python main.py --mode targets --review-session list
python main.py --mode targets --review-session SESION
```

Esa revisión no abre la cámara ni borra capturas. No hay video de los golpes ya
hechos: se muestran duración, ángulos, alcance y la clase pedida frente a la
del detector. Por eso no se aprueban solas.

En las dos sesiones reales, 18 de 26 capturas quedaron dudosas porque el
detector nombró otra clase, 2 porque la muñeca quedó fuera del objetivo, 1
porque no hubo evento y 1 porque hubo un golpe durante la guardia. Las 4
provisionales coincidieron en clase y distancia. La duración era válida, así
que no se relajaron los umbrales.

Esas capturas viven en `data/review/` hasta aprobarse. Solo entonces entran al
dataset de NEAT, con la etiqueta que tú confirmaste, y sin video. La puntuación
del juego no es la precisión del modelo evolutivo. Si se pierde el cuerpo, la
ronda se detiene y no suma fallos.

La meta inicial es 10 muestras verificadas por clase. Hacen falta al menos tres
sesiones, cada una con su propio identificador, antes de
`python main.py --mode dataset-report` y `python main.py --mode train`. Usar el
modelo NEAT dentro del juego queda para la entrega final.

## Arquitectura

- `main.py`: argumentos, composición y bucle de ejecución.
- `app/config.py`: parámetros configurables y validación.
- `app/camera.py`: `VideoCapture`, fallback y timestamps.
- `app/pose_tracker.py`: MediaPipe Tasks y observaciones estructuradas.
- `app/geometry.py`: ángulos, distancias, posiciones relativas y vector biomecánico.
- `app/overlay.py`: esqueleto y HUD de seguimiento o recopilación.
- `app/data_collector.py`: cuenta regresiva, grabación y etiquetas manuales.
- `app/sample_validator.py`: reglas de calidad de una ejecución.
- `app/dataset_manager.py`: JSON atómico, sesiones e índice CSV.
- `app/evolution/`: auditoría, vectores temporales, división, NEAT, métricas, gráficas y reporte académico.
- `app/trainer/`: detector provisional, adaptador NEAT, YOLO + LSTM y sesión.
- `app/benchmark/`: rutina guiada, emparejamiento y reportes de evaluación.
- `app/targets/`: objetivos, impactos, captura provisional y revisión.
- `config/neat_feedforward.cfg`: plantilla de evolución feedforward.
- `models/`: modelo oficial Pose Landmarker Lite y pesos externos en `models/external/`.
- `tests/`: pruebas unitarias y sintéticas, sin webcam ni ventanas.

`PoseObservation` entrega coordenadas normalizadas, coordenadas de mundo,
visibilidad, presencia y timestamp. `BiomechanicalFeatures` consume ese
contrato sin depender de MediaPipe. Esta separación permite añadir ventanas
temporales y un clasificador NEAT en otra fase sin reescribir la captura.

## Pruebas

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
python -m compileall -q main.py app tests
```

## Privacidad

Los fotogramas se procesan localmente. La aplicación no guarda imágenes, video
ni audio, no crea grabaciones y no sube información a ningún servidor. El
directorio `data/` y los resultados de `results/` permanecen fuera de Git. La red
solo es necesaria durante la instalación y para descargar el modelo si aún no
existe.

## Alcance actual

El modo en vivo cuenta golpes sueltos. El detector provisional, NEAT y
YOLO + LSTM se eligen con `--classifier`. No hay rutinas, aplicación móvil,
servidor ni base de datos. El modo de seguimiento muestra la pose, la
recopilación guarda muestras que superan las reglas de calidad y el
entrenamiento NEAT solo continúa cuando el dataset real permite una división
por sesiones. El reconocimiento frente a la webcam con el LSTM depende de
que haya una persona en cuadro; una imagen vacía no produce golpes.
