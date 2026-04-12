---
title: HuggingFace, Model Hub y Pipelines
sidebar_position: 1
---

import task from "./img/task.png";
import lang from "./img/lang.png";
import filter from "./img/filter.png";

# Hugging Face

## ¿Qué es Hugging Face?

Hugging Face es una empresa y plataforma que se ha convertido en el **centro de referencia** para la comunidad de Machine Learning. Aunque empezó enfocándose en NLP (procesamiento de lenguaje natural), hoy abarca prácticamente todas las áreas del aprendizaje automático.

**¿Qué ofrece Hugging Face?**

1. **Model Hub**: Un repositorio gigantesco de modelos preentrenados (más de 500.000 modelos)
2. **Datasets Hub**: Miles de datasets listos para usar
3. **Spaces**: Demos interactivas de modelos
4. **Bibliotecas**: `transformers`, `datasets`, `tokenizers`, `diffusers`, etc.
5. **Comunidad**: Foros, papers, cursos gratuitos

En esencia, Hugging Face facilita el uso de Machine Learning, permitiendo usar modelos avanzados sin tener que entrenarlos desde cero.

## Model Hub

El [Model Hub](https://huggingface.co/models) es un repositorio donde la comunidad sube y comparte modelos preentrenados.

![image](img/hf.png)

### ¿Para qué sirve?

- **Descargar modelos** listos para usar en tareas específicas
- **Comparar modelos** por métricas, popularidad, tamaño
- **Filtrar** por tarea (clasificación, traducción, generación de texto, etc.), framework (PyTorch, TensorFlow), idioma, licencia...
- **Ver la documentación** de cada modelo: cómo usarlo, ejemplos, limitaciones

### Estructura de un modelo en el Hub

Cada modelo tiene su propia Model Card con:

| Sección | Descripción |
|---------|-------------|
| **Model Card** | Descripción, uso previsto, limitaciones, sesgos |
| **Files** | Pesos del modelo, configuración, tokenizer |
| **Usage** | Código de ejemplo para usarlo |
| **Metrics** | Resultados en benchmarks |
| **Community** | Discusiones y feedback |

![image](img/model_card.png)

### Ejemplo de búsqueda

Si necesitas un modelo para **análisis de sentimiento en español**:

1. Ir a [huggingface.co/models](https://huggingface.co/models)
2. Filtrar por tarea: `Text Classification`
3. Filtrar por idioma: `Spanish`
4. Ordenar por descargas o likes
5. Revisar las Model Cards de los candidatos

Los modelos más descargados suelen ser los más probados por la comunidad.

<div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px", margin: "30px 0" }}>
  <div style={{ textAlign: "center" }}>
    <strong>Paso 2: Filtrar por tarea (Task)</strong>
    <img src={task} alt="Filtrar por tarea" style={{ width: "100%" }} />
  </div>
  <div style={{ textAlign: "center" }}>
    <strong>Paso 3: Filtrar por idioma (Language)</strong>
    <img src={lang} alt="Filtrar por idioma" style={{ width: "100%" }} />
  </div>
</div>

<div style={{ textAlign: "center", margin: "30px 0" }}>
  <strong>Paso 4: Ver resultados filtrados</strong><br />
  <img src={filter} alt="Resultados filtrados" style={{ width: "100%" }} />
</div>

## Pipelines

Los **pipelines** son la forma más sencilla de usar modelos de Hugging Face. Abstraen toda la complejidad (tokenización, modelo, post-procesado) en una sola línea de código. Es una API de alto nivel para usar modelos fácilmente.

### ¿Por qué usar pipelines?

- **Simplicidad**: No necesitas conocer los detalles internos del modelo
- **Productividad**: En pocas líneas tienes un modelo funcionando
- **Flexibilidad**: Puedes cambiar de modelo fácilmente

### Instalación

```bash
pip install transformers
```

### Ejemplo básico: Análisis de sentimiento

```python
from transformers import pipeline

# Crear el pipeline (descarga el modelo automáticamente la primera vez)
clasificador = pipeline("sentiment-analysis")

# Usar el pipeline
resultado = clasificador("Me encanta este helado de chocolate")
print(resultado)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

### Tareas disponibles

Los pipelines cubren muchas tareas comunes:

| Tarea | Pipeline | Ejemplo de uso |
|-------|----------|----------------|
| Análisis de sentimiento | `sentiment-analysis` | Detectar si un texto es positivo/negativo |
| Clasificación de texto | `text-classification` | Clasificar texto en categorías |
| Generación de texto | `text-generation` | Completar o generar texto |
| Respuesta a preguntas | `question-answering` | Responder preguntas sobre un texto |
| Resumen | `summarization` | Resumir textos largos |
| Traducción | `translation` | Traducir entre idiomas |
| Reconocimiento de entidades | `ner` | Extraer personas, lugares, organizaciones... |
| Clasificación de imágenes | `image-classification` | Clasificar imágenes |
| Detección de objetos | `object-detection` | Detectar objetos en imágenes |
| Transcripción de audio | `automatic-speech-recognition` | Convertir audio a texto |

### Especificar un modelo concreto

Por defecto, cada pipeline usa un modelo predeterminado. Pero puedes especificar cualquier modelo del Hub:

```python
from transformers import pipeline

# Usar un modelo específico para español
clasificador = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

resultado = clasificador("Esto lo cambia todo")
print(resultado)
```

### Ejemplo: Generación de texto
**No** es lo mismo que [Fill-Mask](https://colab.research.google.com/github/jorgecs/apuntes/blob/main/docs/ut5_ia_aplicada/1_transformers/notebooks/Transformer.ipynb#scrollTo=rcag6ntkta), la generación de texto continúa donde termina la frase, mientras que Fill-Mask puede rellenar palabras en el medio (tarea de comprensión vs tarea de generación).

```python
from transformers import pipeline

generador = pipeline("text-generation", model="gpt2")

texto = generador(
    "The cake is",
    max_length=50,
    num_return_sequences=1
)
print(texto[0]['generated_text'])
```

### Ejemplo: Respuesta a preguntas

```python
from transformers import pipeline

qa = pipeline("question-answering")

contexto = """
Hugging Face fue fundada en 2016 en Nueva York.
Inicialmente era una app de chatbot, pero pivotó hacia
herramientas de NLP y se convirtió en la plataforma
de referencia para modelos de Machine Learning.
"""

pregunta = "¿Cuándo se fundó Hugging Face?"

respuesta = qa(question=pregunta, context=contexto)
print(respuesta)
# {'answer': '2016', 'score': 0.98, 'start': 30, 'end': 34}
```

### Ejemplo: Resumidor

```python
from transformers import pipeline

summarizer = pipeline("summarization")

texto = """
Hugging Face es una empresa especializada en inteligencia artificial fundada en 2016.
Comenzó como una aplicación de chatbot, pero con el tiempo evolucionó hacia el desarrollo
de herramientas avanzadas para procesamiento del lenguaje natural (NLP). Hoy en día,
ofrece una amplia variedad de modelos preentrenados que pueden utilizarse para tareas
como traducción, clasificación de texto, generación de texto y análisis de sentimientos.
Su plataforma se ha convertido en un estándar dentro de la comunidad de machine learning.
"""

resumen = summarizer(texto, max_length=50, min_length=20, do_sample=False)

print(resumen)
# [{'summary_text': 'Hugging Face es una empresa de inteligencia artificial fundada en 2016. Actualmente ofrece modelos para múltiples tareas de NLP.'}]
```

### Ejemplo: Reconocimiento de entidades
```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)

texto = """
Pedro trabaja en Hugging Face y vive en Madrid.
La empresa fue fundada en Nueva York en 2016.
"""

entidades = ner(texto)

print(entidades)
# [
#   {'entity_group': 'PER', 'word': 'Pedro', 'score': 0.99},
#   {'entity_group': 'ORG', 'word': 'Hugging Face', 'score': 0.98},
#   {'entity_group': 'LOC', 'word': 'Madrid', 'score': 0.97},
#   {'entity_group': 'LOC', 'word': 'Nueva York', 'score': 0.96},
#   {'entity_group': 'DATE', 'word': '2016', 'score': 0.95}
# ]
```

### Ejemplo: Traducción

```python
from transformers import pipeline

translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M"
)

texto = "La inteligencia artificial está transformando el mundo rápidamente."

traduccion = translator(
    texto,
    src_lang="spa_Latn",
    tgt_lang="eng_Latn"
)

print(traduccion)
# [{'translation_text': 'Artificial intelligence is rapidly transforming the world.'}]
```

### Ejemplo: Clasificación en categorías

```python
from transformers import pipeline

zero_shot = pipeline("zero-shot-classification")

textos = [
    "El equipo ganó el campeonato después de una final emocionante",
    "La nueva actualización del sistema operativo incluye mejoras de seguridad",
    "El presidente anunció nuevas medidas económicas"
]

categorias = ["deportes", "tecnología", "política", "entretenimiento"]

for texto in textos:
    resultado = zero_shot(texto, categorias)
    print(f"Texto: {texto}")
    print(f"Categoría predicha: {resultado['labels'][0]}")
    print(f"Confianza: {resultado['scores'][0]:.2f}")
    print()

```

### Pipeline con imágenes

```python
from transformers import pipeline

# Clasificación de imágenes
clasificador_img = pipeline("image-classification")

resultado = clasificador_img("imagen_gato.jpg")
print(resultado)
# [{'label': 'Egyptian cat', 'score': 0.89}, ...]
```

### Pipeline con audio

```python
from transformers import pipeline

# Transcripción de audio (Speech-to-Text)
transcriptor = pipeline("automatic-speech-recognition", model="openai/whisper-base")

resultado = transcriptor("audio.mp3")
print(resultado["text"])
```

## ¿Qué hace un pipeline?

Un pipeline hace automáticamente estos pasos:

1. **Preprocesado**: Tokeniza el texto (o procesa la imagen/audio)
2. **Modelo**: Pasa los datos por el modelo (inferencia). Cada tarea usa un modelo diferente:
  - Clasificación: AutoModelForSequenceClassification
  - NER: AutoModelForTokenClassification
  - QA: AutoModelForQuestionAnswering
  - Summarization / Traducción: AutoModelForSeq2SeqLM
3. **Postprocesado**: Convierte la salida en algo legible (etiquetas, scores, texto...)

```
Entrada -> Tokenizer -> Modelo -> Postprocesado -> Salida
```

### Qué ocurre internamente

Cuando escribes `pipeline("sentiment-analysis")`, por debajo ocurren tres fases:

1. **Tokenización**: el `tokenizer` convierte el texto en IDs numéricos.
2. **Inferencia** (depende de la tarea): El modelo procesa la entrada y genera una salida:
  - En tareas como clasificación, NER o QA → produce `logits` (son puntuaciones sin procesar por clase)
  - En tareas como traducción o resumen → genera el texto de la tarea directamente, usando `generate()`, en formato de tokens
  3. **Postprocesado** (depende de la tarea):  
Se convierte la salida del modelo en una representación interpretable (etiquetas, spans o texto).  
 - Clasificación: 
   - Se aplica `softmax` a los logits para obtener probabilidades. 
   - `argmax` selecciona la clase con mayor probabilidad. 
   - `id2label` traduce el índice a una etiqueta legible. 
   - Finalmente se obtiene la etiqueta predicha y su score de confianza.
 - NER (token classification):
    - Se asigna una etiqueta a cada token. 
    - Luego se agrupan tokens consecutivos para formar entidades completas (por ejemplo, "Nueva York").
 - Question Answering: 
    - El modelo produce `start_logits` y `end_logits`. 
    - Se selecciona el span de texto con mayor probabilidad dentro del contexto.
 - Summarization / Traducción / Generación: 
    - Se decodifican los tokens de `generate()` usando `tokenizer.decode()` para conocer a qué palabras corresponden los tokens generados.

Un ejemplo mínimo equivalente al pipeline sería:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

#Carga del tokenizer y modelo
modelo_nombre = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(modelo_nombre)
modelo = AutoModelForSequenceClassification.from_pretrained(modelo_nombre)

#Tokenización
entrada = tokenizer("I love this movie", return_tensors="pt")

#Inferencia
with torch.no_grad():
  logits = modelo(**entrada).logits

#Postprocesado

#Convertir logits a probabilidades usando softmax
probabilidades = torch.nn.functional.softmax(logits, dim=-1)
#Obtener la clase predicha
clase_predicha = torch.argmax(probabilidades, dim=-1).item()
#Mapear el indice a la etiqueta
etiquetas = modelo.config.id2label
#Obtener la etiqueta correspondiente a la clase predicha
etiqueta = etiquetas[clase_predicha]
#Obtener el score (probabilidad) de la clase predicha
score = probabilidades[0][clase_predicha].item()

print(etiqueta, score)
```

La ventaja de `pipeline` es que automatiza estos pasos.

## Resumen

### Flujo pipeline
| Etapa | Qué hace | Depende de la tarea | Ejemplo de salida |
|------|----------|---------------------|--------------------|
| **1. Tokenización** | Convierte el texto en IDs numéricos entendibles por el modelo | No | `input_ids`, `attention_mask` |
| **2. Inferencia** | El modelo procesa los inputs | Sí | logits o texto generado |
| **3. Postprocesado** | Convierte la salida en algo interpretable | Sí | etiquetas, spans o texto final |

### Inferencia
| Tipo de tarea | Modelo típico | Tipo de salida |
|--------------|--------------|----------------|
| Clasificación | `AutoModelForSequenceClassification` | logits (por clase) |
| NER | `AutoModelForTokenClassification` | logits por token |
| Question Answering | `AutoModelForQuestionAnswering` | start_logits y end_logits |
| Summarization / Traducción | `AutoModelForSeq2SeqLM` | texto generado con `generate()` |

### Postprocesado
| Tarea | Qué ocurre en el postprocesado |
|------|--------------------------------|
| Clasificación | softmax → probabilidades → argmax → `id2label` → etiqueta final + score |
| NER | etiquetas por token → agrupación de tokens en entidades completas |
| Question Answering | selección del span con mayor probabilidad (start + end logits) |
| Summarization / Traducción / Generación | `generate()` → `decode()` a texto final |

## Ejercicios prácticos

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgecs/apuntes/blob/main/docs/ut5_ia_aplicada/2_huggingface/notebooks/Pipeline.ipynb)

**IMPORTANTE**: Guarda una copia en Drive antes de empezar (Archivo → Guardar una copia)

## Bibliografía

- [Documentación oficial de Hugging Face](https://huggingface.co/docs)
- [Curso gratuito de Hugging Face](https://huggingface.co/learn/nlp-course)
- [Model Hub](https://huggingface.co/models)
