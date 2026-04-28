---
title: Integración de visión artificial
sidebar_position: 3
---

import testocr from "./img/testocr.png";
import video1 from "./img/video.mp4";
import video2 from "./img/video2.mp4";

# Integración de OCR y YOLO en una aplicación FastAPI

Hasta ahora hemos probado OCR en scripts sueltos. El siguiente paso realista es exponerlo como servicio para que lo consuma un frontend.

Una forma habitual de hacerlo en Python es con **FastAPI**.

## Arquitectura básica recomendada

Pipeline típico en producción:

1. Cliente sube imagen (ticket, matrícula, cartel, documento).
2. API valida tipo de archivo y tamaño.
3. YOLO localiza zonas relevantes (bounding boxes).
4. Preprocesado básico del recorte (grises, umbral, limpieza).
5. Motor OCR (Tesseract / PaddleOCR) extrae texto de cada ROI.
6. API responde JSON con bbox, texto, confianza, metadatos...

## 1. Instalación de dependencias

```bash
pip install fastapi "fastapi[standard]" opencv-python pytesseract "paddleocr<3" "paddlepaddle<3" ultralytics huggingface_hub
```

Para instalar Tesseract:

- Windows: https://github.com/UB-Mannheim/tesseract/wiki
Instala la ruta:
```
C:\Program Files\Tesseract-OCR
```

Si no se añade al PATH, en el código de Python usa esto:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\RutaDeLaDescarga\Tesseract-OCR\tesseract.exe"
```

- Linux:
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-spa
```

OpenCV puede usar FFmpeg para leer vídeos. Normalmente no es necesario instalarlo porque se incluye en la librería de Python, pero si diese error con vídeos `.mp4`, sería necesario instalarlo:

Linux:
```bash
sudo apt-get install ffmpeg
```
Windows: https://ffmpeg.org/download.html


## 2. API de OCR por imagen (`main.py`)

Vamos a utilizar FastAPI para crear una API que nos permita subir una imagen y obtener el texto extraído por OCR.

Para ello, añadiremos las dependencias necesarias y crearemos la APP de FastAPI. Como vamos a utilizar PaddleOCR, importaremos la clase `PaddleOCR` y `cv2` para poder leer la imagen.

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
from paddleocr import PaddleOCR

app = FastAPI(title="OCR API", version="1.0.0")
```

A continuación podemos definir el motor OCR que se va a utilizar, en este caso PaddleOCR. Usaremos los parámetros `use_angle_cls=True` para detectar el ángulo de la imagen y `lang="es"` para detectar el idioma español.

```python
ocr = PaddleOCR(use_angle_cls=True, lang="es")
```

PaddleOCR devuelve el texto de una forma estructurada, así que crearemos un método para extraer todo el texto en una sola cadena.

```python
def extraer_texto(resultado_ocr):
	textos = []
	
	if resultado_ocr and len(resultado_ocr) > 0:
		for linea in resultado_ocr[0]:
			textos.append(linea[1][0])

	return " ".join(textos)
```

A continuación crearemos la API con su respectivo endpoint para procesar imágenes. Igual que con la API que ya creamos, vamos a utilizar Pydantic para definir el contrato de la API. En este caso, la respuesta será el texto extraído por OCR y la entrada será la imagen que se envíe al endpoint.

```python
class OCRResponse(BaseModel):
	texto: str

class OCRRequest(BaseModel):
	image: UploadFile = File(...)
```

Ahora ya podemos crear el endpoint utilizando estos modelos y el OCR.

```python
@app.post("/api/ocr", response_model=OCRResponse)
```

Hay algo muy importante, la lectura de los archivos puede ser un proceso lento, por lo que necesitamos hacerlo de forma asíncrona. Añadiremos la palabra `async` antes de `def` y `await` antes de la lectura de los archivos. Además de controlar el formato de entrada. Para leer el fichero usaremos `file.read()` que nos devuelve el contenido del fichero en bytes. Si no se soporta el formato, lanzaremos una excepción.

```python
async def ocr_image(file: UploadFile = File(...)):
	if file.content_type not in {"image/png", "image/jpeg", "image/jpg", "image/webp"}:
		raise HTTPException(status_code=400, detail="Formato no soportado")

	try:
		image_bytes = await file.read()


	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Error OCR: {e}")
```

Una vez tenemos el contenido del fichero en bytes, necesitamos convertirlo para que OpenCV pueda leerlo. Para ello, utilizaremos `cv2.imdecode()`.

```python
np_arr = np.frombuffer(image_bytes, np.uint8)
img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
```

Ahora ya podemos utilizar PaddleOCR, extraer el texto y devolverlo en el formato que definimos con Pydantic.

```python
result = ocr.ocr(img)

texto = extraer_texto(result)

return OCRResponse(texto=texto)
```

## 3. Ejecutar y probar

```bash
fastapi dev
```

Prueba en `http://127.0.0.1:8000/docs`:

1. Abre `POST /api/ocr`.
2. Sube una imagen (`.jpg` o `.png`).
3. Ejecuta y revisa el JSON de respuesta.

<a href={testocr} download>
  📥 Descargar imagen de prueba
</a>

## 4. Integrar OCR + detección (YOLO + OCR)

Usando YOLO, el patrón típico es:

1. YOLO detecta la región de interés (por ejemplo, cartel o etiqueta).
2. Recortas ese `bbox`.
3. Ejecutas OCR solo en ese recorte.

Esto reduce ruido y mejora precisión respecto a hacer OCR sobre la imagen completa.

En vídeo, la idea es la misma, pero repetida por frame:

1. Leer frame.
2. Detectar ROI con YOLO.
3. Recortar cada caja.
4. Aplicar OCR en ese recorte.
5. Devolver resultados agregados.

Como ahora vamos a trabajar con YOLO, necesitamos importar las librerías necesarias y el modelo de matrículas

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import tempfile
import os
from collections import Counter

model_path = hf_hub_download(
    repo_id="morsetechlab/yolov11-license-plate-detection",
    filename="license-plate-finetune-v1s.pt"
)
model = YOLO(model_path)
```

Queremos devolver la matrículas de cada uno de los coches, por tanto, devolveremos una lista en lugar de un string. Para ello hay que definirlo en nuestro contrato de la API

```python
class YOLOResponse(BaseModel):
	plate_per_car: list[str] = []
```

Ahora ya podemos crear el nuevo endpoint que utilice YOLO, este endpoint:
1. Recibe un vídeo
2. Detecta matrículas con YOLO
3. Aplica OCR a cada una
4. Devuelve la matrícula más probable por coche

```python
@app.post("/api/video-detect-and-read" response_model=YOLOResponse)
```

Igual que antes, hay que leer un fichero, así que será asíncrono

```python
async def video_detect_and_read(file: UploadFile = File(...)):
	if file.content_type not in {"video/mp4", "video/quicktime", "video/x-msvideo"}:
		raise HTTPException(status_code=400, detail="Formato de video no soportado")
```

Como hicimos con el endpoint de OCR, leeremos el contenido del fichero, pero al estar usando OpenCV, hay que leerlo desde disco. Para poder hacer esto, crearemos un fichero temporal y lo leeremos.

```python
video_bytes = await file.read()
fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
	
with os.fdopen(fd, 'wb') as tmp_file:
	tmp_file.write(video_bytes)

cap = cv2.VideoCapture(tmp_path)
```

Una vez tenemos el vídeo en OpenCV, podemos comenzar a procesarlo. Crearemos la variable `plate_per_car`, que es un diccionario de listas para almacenar las matrículas de cada uno de los coches que se han visualizado. Guardamos múltiples resultados por coche porque el OCR puede fallar en algunos frames. Después nos quedamos con el valor más frecuente (la moda), que suele ser el correcto.

Se usará `model.track()` para observar cada matrícula individualmente. Y por cada una, obtendremos el recorte de la bounding box de la matrícula y añadiremos el track_id a nuestra lista de coches, para tenerlo controlado.

```python
plate_per_car = {}

while True:
	ok, frame = cap.read()
	if not ok:
			break

	#Detectar matrículas
	results = model.track(frame, persist=True)[0]

	boxes = results.boxes

	for box in boxes:
		if box.id is None:
			continue

		track_id = int(box.id[0])

		x1, y1, x2, y2 = box.xyxy[0].tolist()
			
		roi = frame[int(y1):int(y2), int(x1):int(x2)]

		if track_id not in plate_per_car:
			plate_per_car[track_id] = []
```

Una vez ya tenemos el recorte, simplemente tenemos que aplicar el OCR sobre este y almacenar el resultado en `plate_per_car` para relacionar el coche con su matrícula. Añadiremos qué matrícula tiene en cada frame para después poder saber cuál es la que más se repite de cada coche.

```python
texto = ocr.ocr(roi)

if not texto or not texto[0]:
	continue

texto = extraer_texto(texto)

if text:
	plate_per_car[track_id].append(text)
```

Una vez procesados todos los frames, podemos cerrar el vídeo y borrar el fichero temporal

```python
cap.release()
os.remove(tmp_path)
```

Finalmente, ya tendremos en `plate_per_car` una lista de todas las matrículas detectadas por cada coche, ahora queremos saber cuál es la correcta. Para ello aplicaremos la función de la moda sobre cada coche y finalmente devolveremos la lista.

```python
plate_per_car = {
	k: Counter(v).most_common(1)[0][0]
	for k, v in plate_per_car.items()
	if v
}
return YOLOResponse(plate_per_car=list(plate_per_car.values()))
```

En producción, lo normal es ejecutar este endpoint en background (cola de tareas). Por eso se usa async

<a href={video2} download>
  📥 Descargar vídeo de prueba
</a>

<a href={video1} download>
  📥 Descargar otro vídeo de prueba
</a>


## 5. Ejemplo frontend para subida de imagen

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

### Ejemplo frontend para subida de vídeo

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

Para no tener que montar CORS, vamos a servir el frontend directamente desde FastAPI, añadimos lo siguiente:
```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")
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

## 6. Buenas prácticas de producción

1. Limitar tamaño de archivo.
2. Registrar tiempos de preprocesado e inferencia (monitorización).
3. Usar confidence para saber qué resultado aceptar.
4. Para vídeos largos, evitar espera síncrona: usar colas (Celery/RQ) y endpoint de estado.

## Actividad 1: Preprocesado
Normalmente es buena idea preprocesar las imágenes antes de procesarlas con OCR.
Crea un método para preprocesar los recortes de matrículas antes de que se procesen con el motor OCR.