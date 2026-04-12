---
title: Integración de un LLM en un proyecto
sidebar_position: 3
---

# Teoría y Práctica: Integración de LLMs en Producción con FastAPI

Hasta ahora hemos ejecutado nuestros scripts de Gemini localmente en Jupyter Notebooks. Sin embargo, para que un modelo de Inteligencia Artificial sea verdaderamente útil en el mundo real (como en el backend de una web o en tu proyecto final), necesitamos exponerlo a través de una API.

Aquí es donde entra **FastAPI**.

## ¿Qué es FastAPI?

FastAPI es un framework web moderno y ultra-rápido para construir APIs con Python. Se ha convertido en el estándar de la industria para aplicaciones de Inteligencia Artificial por varias razones:

1. **Rapidez:** Es uno de los frameworks más rápidos de Python (comparable a NodeJS o Go).
2. **Validación Estricta:** Usa la librería `Pydantic` para validar de forma automática que los datos que te envían desde el Frontend (el JSON) tienen el formato estructural correcto antes de que lleguen a la IA.
3. **Documentación Automática:** Autogenera una web interactiva (Swagger) en `/docs` para probar tus endpoints fácilmente sin necesidad de programar tú mismo un cliente ni usar Postman.

---

## Ejemplo Práctico: Un Endpoint de RAG con Gemini

Vamos a crear un servidor backend que reciba un **contexto** (ej. un texto largo extraído de una base de datos) y una **pregunta** mediante una petición HTTP `POST`. El servidor unirá ambas piezas en un "Super-Prompt", llamará a Gemini y devolverá la respuesta al cliente.

### 1. Instalación de dependencias

Asegúrate de preparar tu entorno de terminal instalando lo necesario:

```bash
pip install fastapi uvicorn google-generativeai pydantic
```

### 2. Código del Servidor (`main.py`)

Crea un archivo local llamado `main.py` y añade la siguiente estructura básica. Observa detenidamente cómo usamos las clases *BaseModel* de Pydantic para tipar de forma estricta qué es lo que esperamos recibir:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os

# 1. Configuración de Gemini
# En producción, usa variables de entorno (NUNCA expongas tu API Key subiéndola a GitHub)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "TU_API_KEY_ACÁ")
genai.configure(api_key=GOOGLE_API_KEY)

# Instanciamos el modelo base
modelo_base = genai.GenerativeModel('gemini-1.5-flash')

# 2. Inicialización de la app API
app = FastAPI(title="Mi API con RAG y Gemini", version="1.0.0")

# 3. Definir la estructura de datos esperada desde el cliente Frontend (Pydantic)
class RAGRequest(BaseModel):
    contexto: str
    pregunta: str

class RAGResponse(BaseModel):
    respuesta: str

# 4. Crear el Endpoint de tipo POST
@app.post("/api/chat", response_model=RAGResponse)
async def procesar_pregunta(request: RAGRequest):
    try:
        # Ensamblamos el Super-Prompt que vimos en la teoría inyectando el body del JSON
        prompt_gigante = f"""
        Eres un asistente experto. Utiliza ÚNICAMENTE la siguiente información para responder a la pregunta del usuario. 
        Si la respuesta no se encuentra en el texto, debes responder "Lo siento, no dispongo de esa información." y no debes inventar nada.
        
        --- INICIO DEL CONTEXTO ---
        {request.contexto}
        --- FIN DEL CONTEXTO ---
        
        Pregunta del usuario: {request.pregunta}
        """
        
        # Llamar a Gemini. 
        # NOTA PRO: En web (FastAPI) usamos 'await generate_content_async' 
        # para no bloquear ni congelar a otros usuarios esperando la respuesta.
        response = await modelo_base.generate_content_async(prompt_gigante)
        
        # Devolver el objeto de respuesta al cliente web/móvil
        return RAGResponse(respuesta=response.text)
        
    except Exception as e:
        # Manejo de errores básicos (ej. API Key inválida, internet caído)
        raise HTTPException(status_code=500, detail=str(e))
```

### 3. Ejecutar y Probar tu API

Para arrancar el servidor web, abre tu terminal en la misma carpeta donde salvaste el `main.py` y ejecuta el servidor local de desarrollo `uvicorn`:

```bash
uvicorn main:app --reload
```
*(El subfijo `--reload` le indica a uvicorn que debe reiniciar el servidor automáticamente cada vez que pulses "Guardar" en tu código, facilitando el desarrollo).*

**Para probar el endpoint interactivo:**
1. Abre tu navegador web favorito y ve a `http://127.0.0.1:8000/docs`
2. Verás la interfaz gráfica verde y blanca autogenerada por Swagger. Busca y despliega tu ruta `POST /api/chat`.
3. Haz clic en el botón blanco **"Try it out"** (situado arriba a la derecha).
4. En el campo "Request body", escribe un JSON de prueba como este para someter al sistema a un test:

```json
{
  "contexto": "La empresa cierra todos los fines de semana de agosto. Las vacaciones de verano se piden al departamento de RRHH con al menos un mes de antelación estricta.",
  "pregunta": "¿A quién debo pedirle información sobre el stock de folios?"
}
```

5. Haz clic en el botón azul **"Execute"**. Verás cómo el backend empaqueta el contenido que has escrito por pantalla, se lo manda a la IA de Google y en unos segundos la caja inferior de "Server response" te contestará con el HTTP Status 200 y el cuerpo:

```json
{
  "respuesta": "Lo siento, no dispongo de esa información."
}
```

**¡Felicidades!** Acabas de lograr integrar el cerebro generativo de un LLM comercial por detrás de tu propia API REST segura y validada. Este es el primer bloque arquitectónico y el más crítico de cómo se construyen el 90% de las startups de Inteligencia Artificial que verás ahí fuera.
