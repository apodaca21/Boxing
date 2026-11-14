# Vision Lab — Fase 1

La Fase 1 añade una señal auxiliar para la webcam RGB externa (la Steren se
asigna normalmente al rol `depth` del `CameraRig`). La cámara frontal continúa
siendo la fuente del motor de pose y del clasificador. La segmentación no
modifica `ClassificationResult`, `PunchLedger`, los contadores ni las reglas de
los modos existentes.

## Demo

Conecta la MacBook y la Steren, comprueba los índices y arranca `live`:

```bash
./.venv/bin/python main.py --list-cameras
./.venv/bin/python main.py --mode live --camera-setup depth \
  --camera-index <macbook> --camera-depth-index <steren> \
  --classifier heuristic
```

En la ventana del entrenador:

- `B`: recalibra el fondo; deja la Steren inmóvil y sin persona durante 36
  fotogramas.
- `V`: cambia `RGB`, `FONDO`, `SILUETA`, `MOVIMIENTO`, `VALIDO` y
  `SUPERPOSICION`.
- `+` / `-`: sube o baja el umbral de diferencia.
- `I`: alterna ROI completa y ROI central.
- `F`, `P`, `R`, `Q` y los controles existentes conservan su significado.

Los mapas no acumulan cualquier cambio de píxel: para exportarlos se cruza el
movimiento con la silueta detectada, se exige sincronización válida y los mapas
por muñeca sólo reciben frames con movimiento intencional/evento. El fondo se
congela tras calibrar y se compensa una variación global de exposición; si la
luz cambia bruscamente, la actividad válida queda en cero y se solicita
recalibrar. Así una pared que cambia de iluminación no domina el resultado.

Si sólo hay una cámara, el motor muestra la sesión normal y no inventa una
segmentación externa. Si la auxiliar deja de entregar frames, `CameraRig` la
retira del rig y el frontal sigue capturando.

## Implementación evaluada

- `background_model.py`: mediana por píxel y actualización exponencial sólo en
  píxeles confiables; permite `freeze_background`.
- `morphology.py`: erosión, dilatación, apertura, cierre y BFS de componentes
  conexos, todo con NumPy/Python.
- `segmentation.py`: diferencia absoluta media o máxima, umbral, ROI y
  selección explicable de la componente más plausible.
- `motion_detector.py`: diferencia entre frames actuales, separada de la
  máscara de primer plano.
- `ui_integration.py`: procesa a 320 px de ancho por defecto y sólo mantiene
  arrays en memoria; no guarda video crudo.

La telemetría observable incluye `external_motion_score`,
`segmentation_confidence`, `sync_valid`, `frame_id`, `camera_id`, dimensiones de
origen y la desviación temporal frente al frame frontal. La sincronización sólo
valida cercanía temporal (100 ms por defecto); no implica profundidad ni fusión
geométrica.

## Fases 2 y 3

`trajectories.py`, `biomechanics.py` y `heatmaps.py` contienen buffers y APIs
puras para puntos de muñeca, derivadas con `dt` real, ángulos de codo y
acumuladores separados. `SessionRecorder` conecta esas APIs en `live`,
`sparring`, `targets` y `game`, sin tocar las decisiones del clasificador.

Al cerrar una sesión con `E` (o `Q`) se guarda:

```text
results/sessions/<session-id>/
  metadata.json
  punch_events.csv
  wrist_trajectories.csv
  biomechanics.json
  heatmap_external_activity.npy   # sólo si hubo Steren
  heatmap_left.npy
  heatmap_right.npy
  exports/
    session_summary.json
    trajectory_arms.png
    reference_macbook.png
    reference_steren.png       # si hubo Steren
    heatmap_activity.png          # sólo si hubo Steren
    heatmap_left.png
    heatmap_right.png
```

La pantalla final usa `1` Resumen, `2` Trayectorias/Biomecánica, `3` Mapas,
`4` Exportar y `Q` cerrar. Los mapas de manos usan coordenadas anatómicas de
la MacBook y el mapa general usa exclusivamente el canvas de la Steren; no se
superponen cámaras sin calibración geométrica.

La Steren RGB no genera profundidad 3D métrica. El reporte marca
`three_dimensional_metric_data=false` y expresa las métricas como `px/s`,
`px/s²`, grados y anchos de hombro.

Los PNG de heatmap se dibujan sobre una única captura representativa de la
sesión (`reference_macbook.png` y, si aplica, `reference_steren.png`), con una
realce suave de exposición para que una webcam oscura no produzca un panel
negro. La escala de cada panel hace visibles también las trayectorias débiles;
no se guarda video crudo. Como esas capturas pueden contener un rostro, esta
vista contextual se genera porque fue solicitada explícitamente y queda
indicada en `metadata.json`.
