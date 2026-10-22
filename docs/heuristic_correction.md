# Corrección del detector heurístico

Fecha: 2026-10-19.

## Evidencia revisada

Se revisó `results/benchmarks/normal/20261019T190350/` sin convertir la
predicción automática en verdad. El benchmark tiene 50 intentos verificados:
40 golpes, 10 intervalos de guardia, 3 aciertos, 5 clases incorrectas, 32
golpes sin detección, 2 falsos positivos en guardia y 0 duplicados. La
reconciliación separa el fallo del detector del contador: los eventos que
llegaron al contador fueron 10 y no hubo eventos ignorados.

Los registros conservan una evidencia temporal resumida por fotograma, pero no
el video ni todos los landmarks originales con los que MediaPipe produjo cada
valor. Por eso no existe una reproducción offline exacta comparable antes vs.
después. El benchmark histórico no se modifica.

## Problemas encontrados en el código

1. La clasificación anterior convertía cualquier trayectoria lateral amplia en
   gancho. Un recto que se proyecta lateralmente por el ángulo de la cámara
   podía terminar como gancho del mismo brazo.
2. La clasificación no usaba el cambio temporal del ángulo del codo: no
   distinguía apertura progresiva de un recto frente a un codo que permanece
   flexionado en un gancho.
3. Un intento ambiguo o un ciclo excedido podía renovar el cooldown aunque no
   se hubiese emitido ningún evento.
4. La ruta heurística prefería landmarks mundiales por estar presentes, aunque
   no verificaba visibilidad/presencia. Así podía usar una profundidad poco
   fiable en vez de la geometría 2D válida.
5. Un candidato se podía descartar demasiado pronto al caer un fotograma por
   debajo del desplazamiento inicial. Ahora permanece en `iniciando` hasta el
   timeout de arranque o hasta comprometerse.

## Cambio de reconocimiento

El detector mantiene tres responsabilidades separadas dentro de la misma
implementación:

- detector de movimiento: desplazamiento, velocidad calculada con el delta de
  los timestamps y compromiso temporal;
- clasificador biomecánico: extensión radial, apertura relativa del codo y
  trayectoria lateral;
- emisor de eventos: confirma una sola vez al retorno y deja el conteo a la
  capa común de eventos.

Un recto requiere extensión y evidencia de apertura del codo respecto al inicio
del ciclo. Esa evidencia tiene prioridad sobre una proyección lateral. Un
gancho requiere trayectoria lateral amplia y codo aún flexionado, con poca
apertura relativa. Si ninguna geometría es suficiente, el resultado queda
ambiguo y no se inventa una clase.

Los estados públicos conservan compatibilidad con la aplicación:

```text
neutral -> iniciando -> extendiendo -> regresando -> completado
```

El estado es independiente por brazo. Se toleran pérdidas breves de landmarks;
la pérdida prolongada cancela el ciclo y devuelve `no_detection`. El cooldown
solo se aplica al emitir un evento confirmado. Los intentos ambiguos, sin
compromiso o con tracking perdido no lo prolongan.

## Geometría y seguimiento

El ángulo se calcula como S-E-W con denominadores validados. Las coordenadas
mundiales de MediaPipe solo se usan si hombros, codo y muñeca cumplen
visibilidad/presencia mínima; de lo contrario se usa la fuente normalizada en
2D y se fuerza la profundidad a cero. No se rellenan articulaciones ni se
fabrica profundidad.

MediaPipe documenta que Pose Landmarker entrega landmarks de imagen y
landmarks mundiales 3D, incluyendo visibilidad y presencia:
[documentación oficial de Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python).

El repositorio externo consultado se conserva solo como referencia de
arquitectura temporal y no se copió:
[BoxingPunchDetection](https://github.com/moakun/BoxingPunchDetection).

## Validación pendiente

Las pruebas sintéticas y unitarias verifican la máquina de estados, no la
precisión de webcam. No se declara una mejora real hasta ejecutar una nueva
sesión webcam con 10 jabs izquierdos, 10 cross derechos, 10 ganchos izquierdos,
10 ganchos derechos y 10 guardias, verificando cada intento de forma humana.
