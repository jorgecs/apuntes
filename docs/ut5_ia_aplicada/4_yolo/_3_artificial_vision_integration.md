---
title: Integración de visión artificial
sidebar_position: 3
---

# Integración de OCR y YOLO en una aplicación FastAPI

Hasta ahora hemos probado OCR en scripts sueltos. El siguiente paso realista es exponerlo como servicio para que lo consuma un frontend, otra API o una app móvil.

Una forma habitual de hacerlo en Python es con **FastAPI**.

## ¿Por qué FastAPI para OCR?

1. Validación de entrada con `Pydantic` (tipos, campos obligatorios, límites).
2. Endpoints HTTP claros para imagen única, lotes o vídeo.
3. Documentación automática en `/docs` para probar sin cliente adicional.
4. Fácil despliegue con `uvicorn` y buen rendimiento para inferencia ligera.

---

## Arquitectura básica recomendada

Pipeline típico en producción:

1. Cliente sube imagen (ticket, matrícula, cartel, documento).
2. API valida tipo de archivo y tamaño.
3. YOLO localiza zonas relevantes (bounding boxes).
4. Preprocesado básico del recorte (grises, umbral, limpieza).
5. Motor OCR (Tesseract / EasyOCR) extrae texto de cada ROI.
6. API responde JSON con bbox, texto, confianza y metadatos.
7. Logs y manejo de errores para trazabilidad.

## 1. Instalación de dependencias

```bash
pip install fastapi "fastapi[standard]" uvicorn python-multipart opencv-python pytesseract ultralytics
sudo apt-get install tesseract-ocr tesseract-ocr-spa
```

`python-multipart` es necesario para subir archivos desde formularios HTTP.

## 2. API mínima de OCR por imagen (`main.py`)

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import pytesseract

app = FastAPI(title="OCR API", version="1.0.0")


class OCRResponse(BaseModel):
	texto: str
	motor: str


def preprocess_image(image_bytes: bytes) -> np.ndarray:
	"""Decodifica y aplica un preprocesado basico para estabilizar OCR."""
	np_arr = np.frombuffer(image_bytes, np.uint8)
	img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
	if img is None:
		raise ValueError("No se pudo decodificar la imagen")

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return th


@app.get("/")
def root():
	return {"status": "ok", "service": "ocr-api"}


@app.post("/api/ocr", response_model=OCRResponse)
async def ocr_image(file: UploadFile = File(...)):
	if file.content_type not in {"image/png", "image/jpeg", "image/jpg", "image/webp"}:
		raise HTTPException(status_code=400, detail="Formato no soportado")

	try:
		image_bytes = await file.read()
		img_pre = preprocess_image(image_bytes)

		config = "--oem 3 --psm 6"
		text = pytesseract.image_to_string(img_pre, lang="spa+eng", config=config)

		return OCRResponse(texto=text.strip(), motor="tesseract")
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Error OCR: {e}")
```

## 3. Ejecutar y probar

```bash
fastapi dev main.py
```

Prueba en `http://127.0.0.1:8000/docs`:

1. Abre `POST /api/ocr`.
2. Sube una imagen (`.jpg` o `.png`).
3. Ejecuta y revisa el JSON de respuesta.

## 4. Integrar OCR + detección (YOLO + OCR)

Si ya tienes YOLO, el patrón típico es:

1. YOLO detecta la región de interés (por ejemplo, cartel o etiqueta).
2. Recortas ese `bbox`.
3. Ejecutas OCR solo en ese recorte.

Esto reduce ruido y mejora precisión respecto a hacer OCR sobre la imagen completa.

Esquema de endpoint:

```python
@app.post("/api/detect-and-read")
async def detect_and_read(file: UploadFile = File(...)):
	# 1) leer imagen
	# 2) ejecutar yolo -> obtener bounding boxes
	# 3) recortar ROI mas relevante
	# 4) aplicar OCR en ROI
	# 5) devolver bbox + texto
	return {
		"bbox": [120, 80, 420, 170],
		"texto": "ABC-1234",
		"motor": "yolo+tesseract",
	}
```

## 5. Subir un vídeo y aplicar YOLO + OCR frame a frame

En vídeo, la idea es la misma, pero repetida por frame:

1. leer frame,
2. detectar ROI con YOLO,
3. recortar cada caja,
4. aplicar OCR en ese recorte,
5. devolver resultados agregados.

Ejemplo de endpoint en FastAPI:

```python
from pathlib import Path
from tempfile import NamedTemporaryFile
from ultralytics import YOLO

yolo_model = YOLO("yolov8n.pt")


@app.post("/api/video-detect-and-read")
async def video_detect_and_read(file: UploadFile = File(...), sample_every: int = 5):
	if file.content_type not in {"video/mp4", "video/quicktime", "video/x-msvideo"}:
		raise HTTPException(status_code=400, detail="Formato de video no soportado")

	# Guardamos temporalmente el video subido
	suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
	with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
		tmp.write(await file.read())
		tmp_path = tmp.name

	cap = cv2.VideoCapture(tmp_path)
	if not cap.isOpened():
		raise HTTPException(status_code=400, detail="No se pudo abrir el video")

	frame_idx = 0
	resultados = []

	try:
		while True:
			ok, frame = cap.read()
			if not ok:
				break

			frame_idx += 1
			if frame_idx % sample_every != 0:
				continue

			det = yolo_model(frame, conf=0.35, iou=0.5, verbose=False)[0]

			for box in det.boxes:
				x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
				roi = frame[y1:y2, x1:x2]
				if roi.size == 0:
					continue

				gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
				_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
				text = pytesseract.image_to_string(th, lang="spa+eng", config="--oem 3 --psm 6").strip()

				if text:
					resultados.append(
						{
							"frame": frame_idx,
							"bbox": [x1, y1, x2, y2],
							"class_id": int(box.cls[0]),
							"det_confidence": float(box.conf[0]),
							"texto": text,
							"motor": "yolo+tesseract",
						}
					)

		return {
			"total_lecturas": len(resultados),
			"sample_every": sample_every,
			"resultados": resultados,
		}
	finally:
		cap.release()
		Path(tmp_path).unlink(missing_ok=True)
```

1. `sample_every=5` acelera la inferencia (no procesa todos los frames).
2. Conviene filtrar por clase YOLO antes de OCR para reducir ruido.
3. En producción, lo normal es ejecutar este endpoint en background (cola de tareas).

## 6. Ejemplo frontend mínimo (subida de imagen)

```html
<!doctype html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>OCR Demo</title>
</head>
<body>
  <h1>OCR API</h1>
  <input id="img" type="file" accept="image/*" />
  <button id="enviar">Procesar</button>
  <pre id="out"></pre>

  <script>
	const out = document.getElementById("out");

	document.getElementById("enviar").addEventListener("click", async () => {
	  const fileInput = document.getElementById("img");
	  if (!fileInput.files.length) return;

	  const fd = new FormData();
	  fd.append("file", fileInput.files[0]);

	  const res = await fetch("/api/ocr", { method: "POST", body: fd });
	  const data = await res.json();
	  out.textContent = JSON.stringify(data, null, 2);
	});
  </script>
</body>
</html>
```

### Variante: subida de vídeo para `/api/video-detect-and-read`

```html
<!doctype html>
<html lang="es">
<head>
	<meta charset="UTF-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1.0" />
	<title>Video YOLO + OCR</title>
</head>
<body>
	<h1>Video -> YOLO + OCR</h1>
	<input id="video" type="file" accept="video/*" />
	<button id="procesar">Procesar vídeo</button>
	<pre id="out"></pre>

	<script>
		const out = document.getElementById("out");
		const btn = document.getElementById("procesar");

		btn.addEventListener("click", async () => {
			const input = document.getElementById("video");
			if (!input.files.length) return;

			const fd = new FormData();
			fd.append("file", input.files[0]);

			out.textContent = "Procesando... puede tardar unos segundos.";

			const res = await fetch("/api/video-detect-and-read?sample_every=5", {
				method: "POST",
				body: fd,
			});

			const data = await res.json();
			out.textContent = JSON.stringify(data, null, 2);
		});
	</script>
</body>
</html>
```

Si frontend y backend están en dominios distintos, añade CORS:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)
```

## 7. Buenas prácticas de producción

1. Limitar tamaño de archivo (evita abuso y OOM).
2. Registrar tiempos de preprocesado e inferencia (monitorización).
3. Normalizar salida (`texto`, `confidence`, `bbox`, `engine`).
4. Separar lógica OCR en módulo propio (`services/ocr_service.py`).
5. Preparar fallback de motor OCR (`tesseract` -> `easyocr`) cuando falle.
6. Para vídeo largo, evitar espera síncrona: usar colas (Celery/RQ) y endpoint de estado.
7. Guardar resultados por timestamp/frame para auditoría.

## Cierre

Integrar OCR en FastAPI convierte una demo local en un componente reutilizable de backend.
Combinado con YOLO, tienes un pipeline completo de visión aplicada:

1. detectar,
2. recortar,
3. leer,
4. usar el texto para reglas de negocio.
