---
title: Contexto y RAG
sidebar_position: 2
---

# Teoría: Introducción al Contexto y "RAG Simplificado"

En el proyecto final necesitaréis que el LLM responda basándose en vuestros propios documentos o bases de datos. Hoy veremos cómo lograr que Gemini actúe como un experto en información que nunca ha visto durante su entrenamiento.

## El problema de la memoria estática

Como vimos, los LLMs son motores probabilísticos. Sin embargo, tienen dos grandes limitaciones respecto al conocimiento:
1. **Corte de conocimiento (Knowledge Cutoff):** Su entrenamiento terminó en una fecha concreta. No saben de forma nativa qué pasó la semana pasada.
2. **Privacidad:** No tienen acceso a vuestros archivos locales, código propietario o bases de datos de clientes.

Si le preguntas a Gemini por un documento interno de tu empresa, o te dirá que no lo sabe, o peor aún, **alucinará** una respuesta intentando adivinarla.

## ¿Cómo le enseñamos datos nuevos? Finetuning vs Contexto (RAG)

Para solucionar esto, históricamente se planteaban dos caminos. Es crucial entender por qué nos decantaremos por el segundo para vuestro proyecto:

### 1. Finetuning (Reentrenamiento)
Consiste en tomar un modelo base y seguir entrenándolo con miles de ejemplos nuevos para modificar permanentemente sus "pesos" internos (las probabilidades). Técnicas como **LoRA** (Low-Rank Adaptation) hicieron esto computacionalmente viable, pero sigue teniendo problemas graves para nuestra necesidad:
- **Es caro y complejo:** Requiere preparar datasets estructurados masivos (pares de instrucción-respuesta).
- **No es dinámico:** Si un dato de tu empresa cambia (ej. un precio), tienes que volver a reentrenar.
- **Sigue alucinando:** El modelo *memoriza* conceptos difusos, pero no tiene una "fuente de verdad" estricta a la que mirar al responder.
*(Por esto no dedicaremos tiempo a programar LoRA en este módulo, para responder preguntas factuales fiables, no es la mejor herramienta).*

### 2. Generación Aumentada por Recuperación (RAG)
En lugar de reentrenar el "cerebro" del modelo, **le damos el libro abierto para que lo lea en el momento de preguntar**. 
Extraemos el texto relevante de nuestros archivos o bases de datos y lo pegamos directamente en el *prompt* junto a la pregunta del usuario.
- **Ventajas:** Es barato, inmediato (si modificas el archivo, la próxima respuesta cambia al instante) y reduce drásticamente las alucinaciones al obligarle a ceñirse al texto.

---

## "RAG Simplificado" con Gemini (Context Stuffing)

En un sistema RAG corporativo complejo (con miles de PDFs u hojas de cálculo), se utilizan Bases de Datos Vectoriales para buscar matemáticamente solo los 3 o 4 párrafos más relevantes a la pregunta, para no saturar al LLM.

Sin embargo, gracias a modelos modernos como **Gemini 1.5**, tenemos ventanas de contexto de locura (hasta 1 o 2 millones de *tokens*, que equivalen a libros enteros). Aplicaremos un abordaje supereficiente llamado **"RAG Simplificado"** (o *Context Stuffing*):
En lugar de trocear archivos y usar bases de datos vectoriales, **leeremos el archivo de texto entero mediante Python y lo inyectaremos de golpe en el prompt.**

### La anatomía de un Prompt con Contexto
Para evitar que el modelo se confunda entre cuáles son tus instrucciones, cuál es la referencia y qué quiere el usuario, debemos ser extremadamente metódicos estructurando el *string*. 

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
```

En la próxima práctica, escribiremos un script en Python que lea un archivo local de forma dinámica, ensamble este "Super-prompt" y se lo envíe a Gemini para hacerle preguntas precisas sobre su contenido.

## Ejercicios prácticos

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgecs/apuntes/blob/main/docs/ut5_ia_aplicada/23_llm/notebooks/Context.ipynb)

**IMPORTANTE**: Guarda una copia en Drive antes de empezar (Archivo → Guardar una copia)