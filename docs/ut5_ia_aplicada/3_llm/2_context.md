---
title: Contexto y RAG
sidebar_position: 2
---

# Teoría: Introducción al Contexto y "RAG Simplificado"

Es bastante común necesitar que un LLM responda basándose en documentos o bases de datos. Hoy veremos cómo lograr que Gemini pueda utilizar información que nunca ha visto durante su entrenamiento.

## El problema de la memoria estática

Como vimos, los LLMs son motores probabilísticos. Sin embargo, tienen dos grandes limitaciones respecto al conocimiento:
1. **Corte de conocimiento (Knowledge Cutoff):** Su entrenamiento terminó en una fecha concreta. No saben de forma nativa qué pasó la semana pasada.
2. **Privacidad:** No tienen acceso a vuestros archivos locales, código propietario o bases de datos de clientes.

Si le preguntas a Gemini por un documento interno de tu empresa, o te dirá que no lo sabe, o peor aún, **alucinará** una respuesta intentando adivinarla.

## ¿Cómo le enseñamos datos nuevos? Finetuning vs Contexto (RAG)

Para solucionar esto, históricamente se planteaban dos caminos:

### 1. Finetuning (Reentrenamiento)
Consiste en tomar un modelo base y seguir entrenándolo con miles de ejemplos nuevos para modificar permanentemente sus "pesos" internos (las probabilidades). Técnicas como **LoRA** (Low-Rank Adaptation) hicieron esto computacionalmente viable, pero sigue teniendo problemas graves para nuestra necesidad:
- **Es caro y complejo:** Requiere preparar datasets estructurados masivos (pares de instrucción-respuesta).
- **No es dinámico:** Si un dato de tu empresa cambia (ej. un precio), tienes que volver a reentrenar.
- **Sigue alucinando:** El modelo *memoriza* conceptos difusos, pero no tiene una "fuente de verdad" estricta a la que mirar al responder.
*(Por esto, para responder preguntas factuales fiables, no es la mejor herramienta).*

### 2. Generación Aumentada por Recuperación (RAG)
En lugar de reentrenar el modelo, **le damos el contexto para que lo lea en el momento de preguntar**. 
Extraemos el texto relevante de nuestros archivos o bases de datos y lo pegamos directamente en el *prompt* junto a la pregunta del usuario.
- **Ventajas:** Es barato, inmediato (si modificas el archivo, la próxima respuesta cambia al instante) y reduce drásticamente las alucinaciones al obligarle a ceñirse al texto.
- **Desventajas:** Hay que tener cuidado con qué información se le da al LLM si no se está usando en local, es posible que utilicen tus datos para volver a entrenar el modelo.

---

## "RAG Simplificado" con Gemini (Context Stuffing)

En un sistema RAG corporativo complejo (con miles de PDFs u hojas de cálculo), se utilizan Bases de Datos Vectoriales para buscar matemáticamente solo los 3 o 4 párrafos más relevantes a la pregunta, para no saturar al LLM.

Sin embargo, gracias a modelos modernos como **Gemini 3.0**, tenemos ventanas de contexto enormes (hasta 1 o 2 millones de *tokens*, que equivalen a libros enteros). Aplicaremos un abordaje supereficiente llamado **"RAG Simplificado"** (o *Context Stuffing*):
En lugar de trocear archivos y usar bases de datos vectoriales, **leeremos el archivo de texto entero y lo inyectaremos de golpe en el prompt.**

### Cómo es un Prompt con Contexto
Para evitar que el modelo se confunda entre cuáles son tus instrucciones, cuál es la referencia y qué quiere el usuario, debemos ser extremadamente metódicos estructurando el *string*, como vimos, usando la estructura **RTCF**.

En el código se crea un prompt con esta estructura:

```text
Actúa como un asistente de soporte de nuestra aplicación.
    
REGLAS:
1. Responde a la pregunta del usuario utilizando ÚNICAMENTE la INFORMACIÓN proporcionada.
2. Si la respuesta no está en la información, di "Lo siento, esa información no consta en la base de datos." No inventes nada bajo ningún concepto.

--- INICIO DE LA INFORMACIÓN ---
[Variable de Python con el texto inyectado leyendo directamente tu archivo .md, .txt o tu JSON]
--- FIN DE LA INFORMACIÓN ---

Pregunta del usuario: [Variable de input del usuario]

Formato: ...
```

## Elegir el formato de salida

Cuando usamos un LLM dentro de una app real, no siempre queremos una respuesta "bonita" para leer, sino una salida que otro sistema pueda consumir de forma automática y fiable. Ahí es donde entra JSON.

JSON (JavaScript Object Notation) es un formato de datos estructurado, ligero y estándar. En la práctica, permite representar información como pares clave-valor (objetos) o como listas (arrays), de forma que Python, JavaScript, APIs y bases de datos puedan procesarla directamente sin tener que "interpretar" texto libre.

Controlar el formato de salida del LLM tiene ventajas muy importantes:
- **Automatización robusta:** Si la respuesta llega siempre con el mismo esquema, tu código puede parsearla y seguir el flujo fácilmente.
- **Menos errores en producción:** Evitas depender de frases naturales ambiguas o cambiantes.
- **Integración sencilla con frontend y backend:** Un JSON se puede enviar tal cual por una API REST y mostrarlo en interfaz sin transformaciones manuales.
- **Validación y testing:** Puedes comprobar fácilmente si faltan campos, si los tipos son correctos o si el modelo se ha salido del formato esperado.

En un prototipo puede bastar con texto natural, pero en una aplicación profesional suele ser mejor pedir al LLM una salida estructurada y estable. Esto convierte al modelo en una pieza dentro de un pipeline, no solo en un chatbot.

## Gemini

Todo esto se puede hacer con el SDK `google-genai` de Gemini:

### 1. Inyectar contexto

Inyectando el contexto inicial:

```python
config_generacion = types.GenerateContentConfig(
	system_instruction=instrucciones
)

```

### 2. Forzar la salida con formato JSOM

Para forzar la salida, es buena idea usar un JSON Schema como el que indica a continuación. Eso junto con el MIME type fuerzan el formato.

```python
from google.genai import types

schema = types.Schema(
	type="OBJECT",
	properties={
		"dias_max_teletrabajo": types.Schema(type="INTEGER"),
		"dias_vacaciones_2026": types.Schema(type="INTEGER"),
	},
	required=["dias_max_teletrabajo", "dias_vacaciones_2026"],
)

config_generacion = types.GenerateContentConfig(
	temperature=0.0,
	response_mime_type="application/json",
	response_schema=schema,
)

response = client.models.generate_content(
	model=MODEL_ID,
	contents=prompt_extraccion,
	config=config_generacion,
)

print(response.text)
```

## Ejercicios prácticos

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgecs/apuntes/blob/main/docs/ut5_ia_aplicada/3_llm/notebooks/Context.ipynb)

**IMPORTANTE**: Guarda una copia en Drive antes de empezar (Archivo → Guardar una copia)