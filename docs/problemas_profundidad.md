# Problemas conocidos: profundidad y detección de golpes

Documento de estado real del proyecto para la entrega y para quien continúe
el trabajo. No describe una profundidad métrica comercial; describe lo que
hace hoy EvoBox con una laptop + webcam USB (p. ej. Steren).

## Resumen corto

| Qué parece | Qué es en realidad |
|---|---|
| “Cámara de profundidad” | Webcam RGB auxiliar (Steren). **No** mide metros. |
| MediaPipe `z` | Estimación relativa, no calibrada en el espacio real. |
| Depth ayuda a jab vs gancho | A veces sí; a menudo confunde o inventa golpes. |
| Muñecas al 100 % | En sesión típica bajan (~60–70 %) al pegar fuerte. |

La cámara frontal de la Mac sigue siendo la autoridad para izquierda/derecha
y para el conteo principal. La Steren es apoyo, no un sensor TrueDepth /
RealSense / Kinect.

## 1. La Steren no es profundidad métrica

- Entrega RGB 2D. No hay nube de puntos ni distancia en centímetros.
- El modo `--camera-setup depth` usa esa vista para:
  - estimar **alcance** (hombro–muñeca en imagen),
  - estimar **desplazamiento lateral**,
  - opcionalmente correr un segundo clasificador heurístico.
- Sin calibración extrínseca (posición, orientación, sincronización entre
  cámaras) no se puede afirmar “el puño avanzó 40 cm hacia la bolsa”.

Consecuencia: el HUD puede decir `DEPTH LISTA` y aun así **no** estar midiendo
profundidad real. Solo indica que hay persona, brazos y un encuadre estable.

## 2. Por qué confunde jab y gancho

Problemas observados en sesiones reales:

1. **Proyección frontal.** Un jab hacia la cámara se ve un poco de lado en 2D
   y el detector lo puede marcar como gancho.
2. **La Steren pisaba la laptop.** Cuando ambas veían el mismo brazo con
   etiquetas distintas, a veces ganaba depth y convertía un recto largo en
   gancho. Eso ya se endureció: la frontal conserva jab/gancho; depth solo
   corrige con una señal de movimiento clara.
3. **Depth sola inventaba golpes.** Con pose lateral ruidosa salían muchos
   “ganchos” con alcance casi nulo. Ahora depth sola solo suma si está lista
   o el cue de movimiento confirma el brazo.
4. **Umbrales HIGH demasiado permisivos** para lateral. Se subió el umbral de
   gancho y se exige que el lateral **domine** al alcance.

Aun con esos cambios, un candidato de prueba que se acerca, gira el torso o
sale del cuadro puede seguir mezclando clases.

## 3. Por qué “al principio sí y después no”

En una sesión reciente el resumen mostró:

- tracking general ~100 %,
- **muñecas ~67 %**,
- 3D métrico: no disponible (Steren RGB).

Interpretación:

- Al empezar, la persona está centrada y las muñecas se ven.
- Al pegar, el brazo sale del marco, se acerca demasiado a la laptop o
  MediaPipe baja la visibilidad de la muñeca.
- Si se pierde la muñeca, el ciclo del brazo se corta o se etiqueta mal.
  Izquierda/derecha se separan peor aunque el torso siga detectado.

No hace falta cuerpo completo. Hace falta **pecho + ambos brazos** (hombros,
codos, muñecas) visibles **también en la extensión del golpe**.

## 4. Limitaciones de hardware / software

- OpenCV + AVFoundation: los índices no coinciden siempre con el nombre de
  la cámara; a veces hay que pulsar `F` para intercambiar frontal y depth.
- Continuity Camera / iPhone puede aparecer como índice negro e inutilizable.
- MediaPipe Pose Landmarker Lite prioriza velocidad; en golpes rápidos las
  muñecas son el landmark más inestable.
- No hay sincronización temporal fina entre Mac y Steren (timestamps
  independientes por `VideoCapture`).
- El pipeline de Vision Lab (segmentación / actividad Steren) aporta mapas 2D
  de apoyo; **no** convierte la Steren en sensor 3D.

## 5. Qué sí hace el sistema hoy

- Bloqueo de inicio hasta `FRONT LISTA` (+ `DEPTH LISTA` si hay Steren) y
  `ENTER`.
- Indicadores en vivo de brazo izquierdo / derecho y avisos si sales del
  cuadro o te acercas demasiado.
- Fusión soft: depth puede apoyar, no debe pisar a ciegas jab↔gancho.
- Sesiones guardadas en `results/sessions/` con trayectorias, biomecánica y
  mapas (sin video crudo).

## 6. Cómo probar con menos dolor

```bash
# Dual cam (laptop frontal + Steren depth)
python main.py --mode live --classifier heuristic --camera-setup depth \
  --camera-index 1 --camera-depth-index 0 --depth-gate soft \
  --training-style sparring

# Si solo quieres estabilizar brazos I/D en una webcam
python main.py --mode live --classifier heuristic --arm-only \
  --camera-setup single --depth-gate off
```

Recomendaciones de encuadre:

1. Mac de frente a ~1.5–2 m.
2. Que se vean hombros, codos y muñecas al estirar.
3. Steren de lado, misma zona de brazos, sin contraluz fuerte.
4. Si la miniatura DEPTH muestra tu cara, pulsa `F`.
5. No empieces a contar hasta el aviso de cámaras listas + `ENTER`.

## 7. Trabajo pendiente (honesto)

- Sensor de profundidad real **o** calibración estereoscópica si se quiere
  distancia métrica.
- Benchmark guiado nuevo con las dos cámaras y verificación humana por golpe
  (jab/cross/gancho × lado).
- Más muestras de gancho derecho en sesiones distintas si se sigue con NEAT.
- Mejorar sincronización temporal Mac ↔ Steren.
- Evaluar si `--arm-only` debe ser el modo demo por defecto cuando no hay
  Steren bien colocada.

## 8. Relación con otros documentos

- Corrección del heurístico: [`heuristic_correction.md`](heuristic_correction.md)
- Vision Lab auxiliar: [`vision_lab.md`](vision_lab.md)
- README principal: sección “Dos o tres webcams” y “Entrenador en vivo”
