# Auditoría de datasets externos de boxeo

Fecha de auditoría: 2026-10-19. El conjunto original de EvoBox permanece en
`data/` y tiene 136 muestras en 8 sesiones orthodox. No se mezcló con datos
externos y no se modificó `results/20261012T001458_initial`.

También se inspeccionó [moakun/BoxingPunchDetection](https://github.com/moakun/BoxingPunchDetection)
como referencia de ingeniería. Su README describe el flujo webcam → MediaPipe
→ características normalizadas/kinemática → buffer temporal → detector por brazo
con máquina de estados → clasificador, recomienda 30--45 grados para golpes
rectos y ofrece un `calibrate.py` que deriva umbrales del ruido de guardia,
velocidad y ángulos de codo. El árbol público inspeccionado no muestra un
archivo `LICENSE`; por eso EvoBox no copió código, pesos ni datos de ese
repositorio. Las ideas se conservaron como procedencia y se implementaron de
forma independiente en `preflight.py` y `calibration_session.py`. El repositorio
tampoco se trata como un dataset.

## Resultado de la selección

| Fuente | Volumen y etiquetas declaradas | Temporal / pose | Lateralidad útil | Licencia y acceso | Decisión |
|---|---|---|---|---|---|
| [Punch_DL](https://github.com/balezz/Punch_DL) | README: 240 golpes en v0.1; no punch, jab izquierdo, cross derecho, hook izquierdo/derecho y uppercuts; también variantes strong | Videos enlazados en Google Drive; código usa MoveNet | La mejor coincidencia nominal para 1–4; uppercuts no entran; `no punch` no se convierte en guardia | No se encontró licencia independiente de los videos enlazados | Candidato futuro, bloqueado hasta confirmar derechos |
| [BoxingVI](https://github.com/Bikudebug/BoxingVI) / [paper](https://arxiv.org/abs/2511.16524) | 6,915 clips de 20 videos de YouTube; Cross, Jab, Lead Hook, Lead Uppercut, Rear Hook, Rear Uppercut | Límites temporales; AlphaPose 2D en formato COCO, hasta 25 frames a 30 FPS | Lead/rear solo podría mapearse con postura orthodox declarada; Jab/Cross genéricos no dan lado; uppercuts excluidos | No se encontró `LICENSE` en el repositorio ni permiso independiente para los videos de YouTube; descarga enlazada en Google Drive | No seleccionado para entrenar EvoBox |
| [FightFlow](https://github.com/krishmula/fightflow) | 3,320 clips, 20 videos, straight/hook/uppercut/none, 830 por clase | Anotaciones por frame; secuencias de pose MediaPipe de 11 frames | `straight` no distingue jab/cross; `hook` no da lado; uppercut y none no se convierten | No se encontró licencia de dataset/vídeos; el repositorio sí publica `data/annotations.csv` | No seleccionado; útil solo como referencia metodológica |
| [Zenodo boxing kinetic](https://zenodo.org/records/22299415) | 1,013 filas de resumen, 27 atletas; jab, cross, lead hook y rear hook; protocolo orthodox/southpaw | IMU y plataforma de fuerza, no webcam, video ni landmarks | Sí aporta lead/rear y postura, pero no puede producir el vector MediaPipe-92 | Dataset marcado como Open en Zenodo; derechos de los sensores sí están descritos en el registro | Alternativa legalmente más clara, pero incompatible con entrenamiento visual; solo referencia biomecánica |

El código de un repositorio no se interpreta como licencia de los videos. Por esa
razón, la auditoría no descargó videos, no ejecutó scripts externos y no instaló
dependencias de terceros. La carpeta `data/external/` queda en estado de registro,
con cero archivos de datos descargados.

## Política de conversión

La clase canónica es:

```text
0 guardia / neutral
1 jab izquierdo
2 cross derecho
3 gancho izquierdo
4 gancho derecho
```

Solo se aceptan `left/right` explícitos, o `lead/rear` con orthodox declarada.
`HOOK` genérico, `straight`, `uppercut`, `block` y `none` se excluyen. La función
`app.external_data.conversion.convert_external_label` conserva la etiqueta original,
el resultado y el motivo de exclusión. No convierte una etiqueta por la posición
visual del cuerpo.

## Compatibilidad de características

El modelo académico de EvoBox conserva el vector temporal de 92 entradas: 8
instantes de 8 canales estáticos y 7 intervalos de 4 canales dinámicos. El orden
exacto está en `results/20261012T001458_initial/model/feature_spec.json` y en
`app/evolution/features.py`.

AlphaPose/COCO, MoveNet y MediaPipe no son intercambiables por índice. Una futura
ingesta debe convertir primero a los ocho puntos MediaPipe usados por EvoBox,
conservar visibilidad/presencia y demostrar equivalencia temporal antes de crear
una muestra. Si solo hay pose 2D AlphaPose, no se presenta como una muestra
compatible con MediaPipe-92.

## Requisitos antes de descargar o entrenar

1. Confirmar por escrito la licencia o permiso de los videos subyacentes.
2. Registrar URLs, hash, tamaño, número real de clips, sujetos y sesiones.
3. Convertir etiquetas con la función separada y conservar las excluidas.
4. Separar sesiones/personas completas entre entrenamiento, validación y prueba.
5. Ejecutar primero una auditoría; si falta alguna clase o hay contaminación de
   sesiones, no entrenar una nueva versión NEAT.
