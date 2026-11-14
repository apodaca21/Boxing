# Auditoría de cámaras, golpes y mapas

Fecha de auditoría: 2026-11-11.

## Evidencia reproducida

- La suite existente pasó antes de la corrección: 197 pruebas OK.
- `main.py --list-cameras --camera-max-scan-index 2` no pudo obtener un frame
  porque macOS devolvió `camera access has been denied`. La validación física
  de índices, FPS real y latencia queda pendiente hasta habilitar el permiso de
  Cámara para la aplicación que ejecuta Python.
- `app/vision_lab/session_recorder.py` creaba dos `HeatmapAccumulator`
  independientes; la sospecha de memoria compartida no se reproduce con
  `np.shares_memory`.
- El espejado se aplicaba al render/export: se volteaban simultáneamente la
  captura y el mapa, mientras las coordenadas anatómicas se conservaban sin
  espejo. No se encontró intercambio de LEFT/RIGHT en el clasificador.

## Causas confirmadas

1. Las manos se depositaban en el mapa en cada frame válido, incluyendo la
   guardia. El mapa representaba presencia y no actividad intencional.
2. Si la segmentación Steren no estaba calibrada, `activity_mask` devolvía la
   máscara de movimiento cruda. Eso permitía que ruido de pared llegara al
   acumulador externo.
3. El fondo se actualizaba continuamente desde `ExternalVisionLab`, por lo que
   una segmentación imperfecta podía absorber gradualmente al usuario.
4. La presentación de mapas con distinto tamaño usaba índices nearest y la
   exportación por mano no usaba una escala compartida.
5. El informe dibujaba tres paneles en una grilla 2×2 y dejaba una cuarta celda
   negra; además, `cv2.putText` no representa UTF-8 de forma fiable.

## Correcciones aplicadas

- Fondo Steren congelado durante la sesión y calibración con mediana más MAD
  robusta por píxel.
- Umbral de segmentación adaptado al ruido local; cambios globales de luz se
  marcan como baja confiabilidad y solicitan recalibración.
- `activity_mask` es nula hasta tener fondo y silueta confiables, y el mapa
  externo sólo acumula frames sincronizados.
- Mapas de muñeca independientes, con puntos válidos únicamente dentro del
  frame y durante movimiento/evento intencional; se conserva la trayectoria
  completa en CSV para auditoría.
- Interpolación bilineal NumPy para presentar mapas y escala máxima compartida
  entre izquierda y derecha.
- Leyenda baja/alta en los PNG, unidad de mapa y placeholder honesto cuando no
  hay datos.
- Informe de análisis con tres paneles horizontales, sin cuarta celda negra, y
  texto UTF-8 mediante Pillow.
- Filas de trayectoria con `frame_id`, `timestamp_ns`, `camera_id`, resolución,
  presencia, visibilidad, espejo de presentación y `tracking_valid`.
- Etiquetas anatómicas `L/R` sobre las muñecas; el espejo sólo afecta el dibujo.
- El modo `calibrate` muestra también la miniatura Steren y permite capturar el
  fondo con `B` antes de comenzar la rutina.

## Pendiente de validación física

1. Activar Cámara en macOS y repetir el inventario de dispositivos.
2. Ejecutar el protocolo unilateral: guardia inmóvil, cinco movimientos sólo
   con izquierda y cinco sólo con derecha.
3. Probar luz estable, cambio brusco de luz, brazo fuera de cuadro y desconexión
   de Steren.
4. Comparar jabs, crosses y ganchos etiquetados; no inferir mejora de precisión
   desde una sola sesión.

Los cambios no alteran la lógica principal de conteo ni los modelos NEAT,
MediaPipe o YOLO. La Steren sigue siendo RGB auxiliar y no se presenta como
sensor de profundidad métrica.
