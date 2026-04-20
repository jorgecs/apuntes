---
title: Fundamentos de LLMs
sidebar_position: 1
---

# Teoría: Fundamentos de LLMs y Configuración de la API

## ¿Qué es un LLM?

### El cambio de paradigma

**Pipelines Tradicionales:**
```
Entrada → [Modelo Específico: Clasificador] → Categoría (Una tarea)
Entrada → [Modelo Específico: Sentiment] → Positivo/Negativo (Una tarea)
```
Cada modelo = una tarea. Le entrenan o adaptan para hacer *exactamente* una cosa, y la hace muy bien.

**Large Language Models (LLMs):**
```
Entrada → [Un Único Modelo Gigante] → Lo que necesites (Infinitas tareas)
         ↓
    Traducción, resumen, código, preguntas, creatividad...
```

Por dentro, los LLMs son sorprendentemente simples. Están entrenados para hacer **una sola cosa**: predecir la siguiente palabra (o *token*) en una secuencia.

**Ejemplo en tiempo real:**
```
"El cielo es de color..."

Internamente calcula probabilidades:
┌─────────────────────────┐
│ azul ████████████ 85%   │
│ gris ██ 10%             │
│ rojo █ 4%               │
│ elefante ▌ 0.001%       │
└─────────────────────────┘

→ Elige "azul"
```

Luego repite con: *"El cielo es de color azul..."* y predice la siguiente, y así sucesivamente.

**Por qué esto es tan poderoso:** Al hacer esto token a token durante meses de entrenamiento en miles de millones de palabras de internet, aprende patrones tan profundos que puede:
- Escribir código correcto
- Explicar conceptos complejos
- Razonar sobre problemas
- Traducir entre idiomas
- Todo con el *mismo* modelo

### Predecir tokens acaba en capacidades complejas

Que el objetivo de entrenamiento sea simple **no** significa que el modelo simplemente suelte palabras sin sentido.

**1) Aprenden representaciones muy complejas**
- Para predecir bien el siguiente token, el modelo acaba aprendiendo gramática, semántica, relaciones lógicas e incluso cierto razonamiento emergente.

**2) No solo repiten palabras**
- Internamente construyen representaciones del contexto usando mecanismos como la **atención**.
- Esto les permite manejar dependencias largas y relaciones complejas entre partes del texto.

**3) Se adaptan a múltiples tareas**
- Gracias a técnicas como **fine-tuning** o **RLHF** (aprendizaje por refuerzo con feedback humano), un mismo modelo puede especializarse mejor en:
  - Responder preguntas
  - Programar
  - Traducir
  - Razonar (hasta cierto punto)

#### Importante: no funciona por Fill-Mask

El uso de `[MASK]` es propio de modelos tipo **BERT**, que usan **Masked Language Modeling (MLM)**:
- Se ocultan palabras en una frase.
- El modelo intenta predecirlas.
- Ejemplo: `"El gato está en el [MASK]"` -> `"tejado"`.

Esto es distinto del objetivo típico de los LLM generativos.

En LLMs tipo **GPT**, la predicción es **secuencial (autoregresiva)**:
- Ven el texto anterior.
- Predicen el siguiente token.
- Ejemplo: `"El gato está en el..."` -> predice `"tejado"`.
- Luego: `"El gato está en el tejado..."` -> predice lo siguiente, y así sucesivamente.

Entonces, ¿por qué parecen tan "inteligentes"? No es por `[MASK]`, sino por la combinación de:
- **Arquitectura Transformer + attention**: relacionan palabras entre sí y capturan contexto complejo.
- **Escala**: muchísimos datos y miles de millones de parámetros.
- **Objetivo + datos -> comportamiento emergente**: para predecir bien, se ven forzados a aprender estructuras profundas de lenguaje, lógica y estilo.

### Diferencias clave con pipelines clásicos

| Aspecto | Pipeline Clásico | LLM |
|---------|-----------------|-----|
| **Tareas** | Una específica | Múltiples (sin reentrenamiento) |
| **Adaptación** | Requiere reentrenar | Solo necesita instrucciones en lenguaje natural |
| **Flexibilidad** | Baja | Muy alta (Zero/Few-Shot Learning) |
| **Computación** | A veces local | Casi siempre en la nube (API) |
| **Output** | Estructura fija | Texto flexible |

---

## Tres pilares para controlar LLMs en producción

Ahora bien, ¿cómo trabajamos con este tipo de modelos? Aquí están los tres conceptos más importantes:

### 1. Prompting

**El prompt es el 80% del resultado.** La calidad de lo que le pides determina directamente la calidad de lo que recibes.

**Mal prompt:**
```
"Dime algo sobre la IA"
```
→ Respuesta confusa, genérica.

**Buen prompt (estructura RTCF):**
```
[ROL] Actúa como un profesor de informática con 10 años de experiencia.
[TAREA] Explica qué es una red neuronal profunda en máximo 5 líneas.
[CONTEXTO] Tu audiencia son alumnos de secundaria (14-16 años).
[FORMATO] Usa una analogía con algo cotidiano. Devuelve en Markdown.
```
→ Respuesta exacta, adaptada.

Cuanto más específico y claro eres con el modelo, mejor resultado obtienes

---

### 2. Temperatura

La distribución de probabilidades que vimos antes se puede "manipular" con un parámetro llamado **temperature**.

```python
generation_config = {
  "temperature": ?, 
  "max_output_tokens": 800,
}
```

**Temperature BAJA (0.0 - 0.3):**
- El modelo elige **casi siempre** la palabra más probable
- **Resultado:** Respuestas deterministas, precisas, repetibles
- **Uso ideal:**
  - Generar código
  - Cálculos matemáticos
  - Extraer datos (JSON)
  - Análisis técnico
- **Ejemplo:** Temperature = 0.0 → Siempre "azul"

**Temperature ALTA (0.7 - 1.0+):**
- El modelo escoge palabras **menos probables**, se atreve a "arriesgar"
- **Resultado:** Respuestas creativas, variadas, sorpresas
- **Uso ideal:**
  - Escribir historias
  - Brainstorming de ideas
  - Marketing
- **Ejemplo:** Temperature = 1.0 → Podría elegir "elefante"

---

### 3. Alucinaciones

**Definición:** El LLM genera información falsa, inventada, pero la presenta como si fuera cierta.

#### ¿Por qué ocurren?

El LLM **NO es una base de datos**. No busca en Google ni tiene acceso a internet por defecto. Es un **motor probabilístico**.

Cuando le preguntas algo que no ha "visto" durante su entrenamiento:
```
Pregunta: "¿Cuál es la biografía de XYZ (persona inventada)?"

El modelo piensa:
"Mmm, XYZ... no la conozco exactamente, pero sé que las biografías
tienen esta estructura: nombre + fecha de nacimiento + trabajos + logros.
Voy a predecir tokens que 'suenen correctamente estadísticamente'..."

→ Genera una biografía completamente falsa pero que SUENA real
```

El modelo SÍ sabe que está inventando (en teoría), pero está obligado a generar algo, así que lo hace de forma convincente.

#### 4 estrategias para mitigar alucinaciones

| Estrategia | Cómo funciona | Cuándo usar |
|-----------|--------------|-----------|
| **RAG (Retrieval-Augmented Generation)** | Le pasas el texto fuente JUNTO con la pregunta: *"Basándote SOLO en este texto, responde..."* | Preguntas sobre documentos específicos |
| **Vías de escape explícitas** | Instrucción: *"Si no sabes, responde: 'No tengo información'"* | Siempre (es una buena práctica) |
| **Temperature baja** | Reduce su "creatividad" y lo mantiene en caminos seguros | Información crítica o técnica |
| **Validación externa** | Verifica respuestas contra APIs, DBs o fuentes confiables | En producción, siempre |

---

## Gemini (SDK oficial)

En práctica trabajaremos con Gemini usando el SDK `google-genai`. Esta es la base mínima que necesitas:

### 1. Crear un cliente

```python
from google import genai
from google.colab import userdata

GOOGLE_API_KEY = userdata.get("GEMINI_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

MODEL_ID = "models/gemini-3.1-flash-lite-preview"
```

### 2. Enviar un mensaje y mostrar respuesta

```python
prompt = "Explica qué es un LLM en 3 frases"

response = client.models.generate_content(
  model=MODEL_ID,
  contents=prompt,
)

print(response.text)
```

### 3. Abrir un chat y enviar mensajes

Con `generate_content()` cada petición es aislada. Si quieres memoria conversacional, usa `chat`:

```python
chat = client.chats.create(model=MODEL_ID)

resp_1 = chat.send_message("Hola, me llamo Ana y me gusta Python.")
print(resp_1.text)

resp_2 = chat.send_message("¿Cómo me llamo y qué lenguaje me gusta?")
print(resp_2.text)
```

### 4. Añadir configuración: temperatura y contexto

```python
from google.genai import types

config_generacion = types.GenerateContentConfig(
  temperature=0.2,
  max_output_tokens=300,
)

contexto = "Tu audiencia son estudiantes de secundaria. Usa lenguaje claro y ejemplos cotidianos."
pregunta = "¿Qué diferencia hay entre IA tradicional y LLM?"

response = client.models.generate_content(
  model=MODEL_ID,
  contents=f"Contexto:\n{contexto}\n\nPregunta:\n{pregunta}",
  config=config_generacion,
)

print(response.text)
```

## Ejercicios prácticos

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgecs/apuntes/blob/main/docs/ut5_ia_aplicada/3_llm/notebooks/Llm_basics.ipynb)

**IMPORTANTE**: Guarda una copia en Drive antes de empezar (Archivo → Guardar una copia)