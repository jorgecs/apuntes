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

### La importancia del Preprocesado

En OCR (especialmente con Tesseract), la calidad de la imagen de entrada es crítica. El preprocesado busca convertir la imagen original en algo que el motor de OCR pueda entender fácilmente (generalmente texto negro sobre fondo blanco puro).

Técnicas comunes:
1.  **Escala de grises**: Elimina información de color innecesaria.
2.  **Desenfoque (Blurring)**: Reduce el ruido digital y las imperfecciones del papel.
3.  **Binarización (Thresholding)**: Convierte la imagen a blanco y negro puro. El método de **Otsu** es muy popular porque calcula automáticamente el umbral óptimo.

```python
# Ejemplo de preprocesado avanzado
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Eliminar ruido conservando bordes
gray = cv2.bilateralFilter(gray, 9, 75, 75)
# Binarización adaptativa para iluminación no uniforme
th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
```

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

### OCR moderno con redes neuronales (PaddleOCR)

Si el texto viene de escenas reales (carteles, matrículas, imágenes con ruido o perspectiva), **PaddleOCR** es una alternativa muy robusta frente a Tesseract.

A diferencia del OCR clásico, PaddleOCR no solo reconoce texto, sino que primero **detecta las regiones donde hay texto** y luego lo interpreta, lo que lo hace más fiable en entornos complejos.

```python
import cv2
from paddleocr import PaddleOCR

img = cv2.imread("cartel_tienda.jpg")

# Inicialización del modelo
ocr = PaddleOCR(use_angle_cls=True, lang="es")

# Ejecuta OCR sobre la imagen completa
results = ocr.ocr(img, cls=True)

for line in results[0]:
    bbox, (text, conf) = line
    print(f"texto='{text}' conf={conf:.2f} bbox={bbox}")
```

### PaddleOCR vs Tesseract
Tesseract:
 - Funciona mejor en texto limpio y documentos estructurados.
 - Necesita preprocesado para rendir bien.
 - Muy útil para entender los fundamentos del OCR.

PaddleOCR:
 - Basado en deep learning.
 - Detecta texto en escenas reales de forma automática.
 - Mucho más robusto ante ruido, inclinación y fondos complejos.
 - Menor dependencia del preprocesado.

### Pipeline típico en visión artificial

1. YOLO detecta objetos o zonas relevantes.
2. Recorte de ROI.
3. OCR extrae texto.
4. Una regla de negocio usa ese texto (registro, validación, alerta, analítica).

## Errores típicos de OCR

- Baja resolución.
- Movimiento/blur en vídeo.
- Mala iluminación.
- Perspectiva inclinada.
- Tipografías raras o escritura poco legible.

Por eso en proyectos reales casi siempre hay preprocesado: escalado, binarización, corrección de orientación y contraste.

## Instalación mínima

### Opción Tesseract

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-spa
pip install pytesseract opencv-python
```

### Opción EasyOCR

```bash
pip install "paddleocr<3" "paddlepaddle<3"
pip install "paddlepaddle-gpu<3"
```

## Ejercicios prácticos

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgecs/apuntes/blob/main/docs/ut5_ia_aplicada/4_yolo/notebooks/OCR.ipynb)

**IMPORTANTE**: Guarda una copia en Drive antes de empezar (Archivo → Guardar una copia)