---
title: OCR
sidebar_position: 2
---

## OCR

Tras YOLO, el flujo natural es:

1. detectar región de interés (matrícula, etiqueta, cartel, documento),
2. recortar esa región,
3. aplicar OCR para extraer texto.

### Qué es OCR y cuándo usarlo

OCR (**Optical Character Recognition**) convierte texto en imágenes a texto editable.

Casos típicos:

- lectura de matrículas,
- extracción de datos en facturas,
- digitalización de formularios,
- lectura de cartelería en vídeo.

### OCR clásico vs OCR moderno

Dos enfoques que merece la pena explicar:

- **OCR clásico (Tesseract)**:
	- rápido de poner en marcha,
	- funciona muy bien con texto impreso y limpio,
	- sensible a ruido, inclinación, baja resolución.
- **OCR moderno con redes neuronales (EasyOCR, PaddleOCR, etc.)**:
	- más robusto en escenarios complejos,
	- suele rendir mejor en texto natural de escenas y casos difíciles,
	- mayor coste computacional.

### OCR con Tesseract (texto impreso)

```python
import cv2
import pytesseract

# Opcional: ruta si tesseract no esta en PATH
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

img = cv2.imread("ticket_supermercado.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Preprocesado basico para mejorar contraste
gray = cv2.GaussianBlur(gray, (3, 3), 0)
_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# psm 6: bloque de texto uniforme
config = "--oem 3 --psm 6"
text = pytesseract.image_to_string(th, lang="spa+eng", config=config)

print("Texto extraido:\n")
print(text)
```

El preprocesado suele ser tan importante como el OCR.
Ajustar `psm` cambia mucho el resultado (linea, bloque, palabra).
Con imágenes limpias, Tesseract es muy competitivo.

### OCR alternativo con EasyOCR (escena real y casos complejos)

Si el texto viene de fotos reales (carteles, etiquetas, capturas móviles), **EasyOCR** suele ser una buena alternativa para ejercicios prácticos.

```python
import cv2
import easyocr

img = cv2.imread("cartel_tienda.jpg")

# Idiomas del modelo (espanol + ingles)
reader = easyocr.Reader(["es", "en"], gpu=False)

# Devuelve: bounding box, texto reconocido y confianza
results = reader.readtext(img)

for bbox, text, conf in results:
	print(f"texto='{text}' conf={conf:.2f} bbox={bbox}")
```

Oara texto impreso limpio, usa Tesseract,
Para escenas reales o documentos difíciles, usa modelos tipo EasyOCR/PaddleOCR.

### Pipeline típico en visión artificial

1. YOLO detecta objetos o zonas relevantes,
2. recorte de ROI,
3. OCR extrae texto,
4. una regla de negocio usa ese texto (registro, validación, alerta, analítica).

## Errores típicos de OCR

- baja resolución,
- movimiento/blur en vídeo,
- mala iluminación,
- perspectiva inclinada,
- tipografías raras o escritura poco legible.

Por eso en proyectos reales casi siempre hay preprocesado: escalado, binarización, corrección de orientación y contraste.

## Instalación mínima

### Opción Tesseract

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-spa
pip install pytesseract opencv-python
```

### Opción EasyOCR

```bash
pip install easyocr torch
```
