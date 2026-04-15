---
title: Detección vs Clasificación
sidebar_position: 1
---

## Clasificación a detección con YOLO

Objetivo de esta sesión: entender por qué **clasificar** una imagen no es lo mismo que **detectar** objetos, y cómo YOLO representa esto con **bounding boxes** en imágenes y vídeo.

## Guion por minutos

### Clasificación vs detección

- **Clasificación**: responde "qué hay en la imagen" (una etiqueta global).
- **Detección**: responde "qué hay" y "dónde está" (clase + posición por objeto).

Ejemplo:

- Imagen con dos coches y una persona.
- Clasificación: `"street"` o `"car"`.
- Detección: `car(x1,y1,x2,y2)`, `car(...)`, `person(...)`.

Si quieres actuar sobre el mundo (contar, seguir, evitar colisiones, vigilar zonas), necesitas ubicación, no solo etiqueta.

### Qué es una bounding box

Una **bounding box** es un rectángulo que delimita un objeto.

YOLO suele devolver, por cada objeto:

- `class_id`: clase predicha (persona, coche, etc.).
- `confidence`: probabilidad/confianza.
- `bbox`: coordenadas (`x1, y1, x2, y2`) o formato centro-ancho-alto (`x, y, w, h`).

Parámetros que incluye YOLO:

- **IoU (Intersection over Union)**: cuánto se solapan dos cajas.
- **NMS (Non-Max Suppression)**: elimina cajas duplicadas para un mismo objeto.
- Umbrales típicos de inferencia:
  - `conf`: filtro por confianza.
  - `iou`: control del NMS.

### Demostración con imagen

Este ejemplo carga un modelo YOLO preentrenado y detecta objetos en una imagen estática.

```python
from ultralytics import YOLO

# Modelo ligero para demo rápida en clase
model = YOLO("yolov8n.pt")

# Inferencia sobre una imagen
results = model("imagen_aula.jpg", conf=0.35, iou=0.5)

# Mostrar y guardar la imagen anotada con bounding boxes
results[0].show()
results[0].save(filename="salida_imagen_boxes.jpg")

# Inspección de detecciones por consola
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"clase={cls_id} conf={conf:.2f} bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
```

Una imagen puede tener **múltiples detecciones**.
Cada detección tiene su propia caja y confianza.
Cambiar `conf` altera cantidad/calidad de cajas.

### Mismo concepto con vídeo

Aquí no cambia la idea, solo se repite frame a frame.

```python
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("video_calle.mp4")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    # Inferencia sobre el frame actual
    result = model(frame, conf=0.35, iou=0.5, verbose=False)[0]

    # Dibuja bounding boxes + etiquetas directamente en el frame
    frame_annotated = result.plot()

    cv2.imshow("YOLO deteccion en video", frame_annotated)

    # Salir con tecla q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

Qué diferencia hay:

- Detección en vídeo = detección en imagen + tiempo.
- La latencia depende del modelo y del hardware.
- Modelos "nano" son buenos.

## Requisitos mínimos para ejecutar los ejemplos

```bash
pip install ultralytics opencv-python
```
Si no hay GPU disponible, los ejemplos también funcionan con CPU, pero más lentos.
