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

### Diferencias clave con pipelines clásicos

| Aspecto | Pipeline Clásico | LLM |
|---------|-----------------|-----|
| **Tareas** | Una específica | Múltiples (sin reentrenamiento) |
| **Adaptación** | Requiere reentrenar | Solo necesita instrucciones en lenguaje natural |
| **Flexibilidad** | Baja | Muy alta (Zero/Few-Shot Learning) |
| **Computación** | A veces local | Casi siempre en la nube (API) |
| **Output** | Estructura fija | Texto flexible |

---

---

## Tres pilares para controlar LLMs en producción

Ahora bien, ¿cómo trabajamos con un modelo tan poderoso y "impredecible"? Aquí están los tres conceptos que dominaréis hoy:

### 1. PROMPTING

**El prompt es el 80% del resultado.** La calidad de lo que le pides determina directamente la calidad de lo que recibes.

**Mal prompt:**
```
"Dime algo sobre la IA"
```
→ Respuesta confusa, genérica y useless.

**Buen prompt (estructura RTCF):**
```
[ROL] Actúa como un profesor de informática con 10 años de experiencia.
[TAREA] Explica qué es una red neuronal profunda en máximo 5 líneas.
[CONTEXTO] Tu audiencia son alumnos de secundaria (14-16 años).
[FORMATO] Usa una analogía con algo cotidiano. Devuelve en Markdown.
```
→ Respuesta exacta, adaptada, memorable.

Cuanto más específico y claro eres con el modelo, mejor resultado obtienes. La especificidad es tu aliada.

---

### 2. TEMPERATURE

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
- **Resultado:** Respuestas creativas, variadas, sorpresas (a veces peligrosas)
- **Uso ideal:**
  - Escribir historias
  - Brainstorming de ideas
  - Marketing copy
- **Ejemplo:** Temperature = 1.0 → Podría elegir "elefante"

**Regla práctica para tu carrera:**
```
Código, datos → Temperature baja
Creatividad → Temperature alta
```

---

### 3. ALUCINACIONES

**Definición:** El LLM genera información falsa, inventada, pero la presenta como si fuera cierta.

#### ¿Por qué ocurren?

El LLM **NO es una base de datos**. No busca en Google ni tiene acceso a internet por defecto. Es un **motor probabilístico puro**.

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

## Ejercicios prácticos

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgecs/apuntes/blob/main/docs/ut5_ia_aplicada/23_llm/notebooks/Llm_basics.ipynb)

**IMPORTANTE**: Guarda una copia en Drive antes de empezar (Archivo → Guardar una copia)