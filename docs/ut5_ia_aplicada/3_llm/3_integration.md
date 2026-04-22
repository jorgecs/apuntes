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
pip install fastapi "fastapi[standard]" google-genai pydantic python-dotenv
```

### 2.1 Código simple del Servidor (`main.py`)

Crea un archivo local llamado `main.py` y añade la siguiente estructura básica. 

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
```

### 2.2 Ejecutar tu API

```bash
fastapi dev
```

Si entramos en `localhost:8000` veremos la respuesta de la API

### 3.1 Configurar archivo `.env` y probar un endpoint simple

Antes de meter RAG y lógica compleja, conviene validar que la API funciona con un endpoint mínimo y una llamada real a Gemini, leyendo la API Key desde un fichero local.

Crea un archivo llamado `.env` en la raíz del proyecto:

```bash
GEMINI_API_KEY=TU_API_KEY
```

Modifica `main.py` con un endpoint de prueba:

```python
from fastapi import FastAPI, HTTPException
from google import genai
from dotenv import load_dotenv
import os

MODEL_ID = "models/gemini-2.5-flash"

# Carga variables desde .env
load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Falta GEMINI_API_KEY en el archivo .env")

client = genai.Client(api_key=api_key)
app = FastAPI(title="API LLM - Base", version="1.0.0")

@app.post("/api/prompt")
def prompt_simple(request):
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=request,
        )
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

Ejecuta y prueba:

```bash
fastapi dev
```

En `localhost:8000/docs`, prueba `POST /api/prompt` con:

```json
"Dame una definición corta de FastAPI"
```

Si esto funciona, ya tienes verificado SDK, lectura de API key desde `.env` y endpoint.

### 3.2 Añadir RAG + chat con documento (contexto persistente)

Ahora damos el salto, en vez de enviar prompts sueltos, creamos un chat con `system_instruction` y le inyectamos un documento local como contexto estable.

Primero creamos un fichero `politicas_empresa.md` con el siguiente contenido:

```bash
# Políticas Internas EmpresaFalsa123.

## Trabajo Remoto
- Los empleados pueden trabajar desde casa un máximo de 3 días a la semana.
- Es obligatorio estar online en Slack entre las 10:00 y las 14:00.

## Vacaciones 2026
- Todos tienen 25 días laborables de vacaciones pagadas.
- Durante el mes de Agosto no se pueden tomar más de 2 semanas consecutivas.

## Comidas y Oficina
- El menú de la cafetería cambia cada semana. El menú del viernes es siempre "Día de Pizza".
- La oficina cierra a las 20:00 y requiere tarjeta magnética para el acceso de fin de semana.
```

El código para leer el fichero y utilizarlo quedaría de la siguiente forma:

```python
import os

with open("politicas_empresa.md", "r", encoding="utf-8") as f:
    documento = f.read()
```

Una vez hemos cargado el fichero, es necesario inicializar Gemini usando la API Key
```python
from google import genai
from google.genai import types

MODEL_ID = "models/gemini-2.5-flash"

api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
```

Una posible opción ahora es utilizar un chat, podemos darle de forma simple este contexto.

```python
config = types.GenerateContentConfig(
    temperature=0,
    system_instruction=instrucciones_sistema,
)

chat_model = client.chats.create(
    model=MODEL_ID,
    config=config,
)
```

Ahora ya podemos cambiar la lógica del endpoint para enviar un mensaje al chat, pero antes vamos a intentar gestionar los datos de entrada y salida, ya que Python no es tipado, pero para una API robusta, es muy importante que se conozca cuáles son las entradas y salidas de cada endpoint, lovamos a hacer con Pydantic.

```python
from pydantic import BaseModel
```

Crearemos un modelo para la entrada de la API (un string con la pregunta) y la salida (un string con la respuesta).

```python
class ChatRequest(BaseModel):
    pregunta: str


class ChatResponse(BaseModel):
    respuesta: str
```

Ahora ya podemos definir el endpoint indicando el tipo de entrada
```python
@app.post("/api/chat", response_model=ChatResponse)
```

La lógica de este endpoint será un método que envía la pregunta al chat de Gemini y devuelve la respuesta (tipo ChatResponse)
```python
def chat_con_contexto(request: ChatRequest):
    try:
        respuesta = chat_model.send_message(request.pregunta)
        return ChatResponse(respuesta=respuesta.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

Prueba rápida en Swagger (`POST /api/chat`):

```json
{
  "pregunta": "¿Cuántos días de teletrabajo máximo permite la empresa?"
}
```

Importante:

1. Aquí el "RAG" está simplificado: no hay vector DB, pero sí inyección de contexto documental real.
2. El chat mantiene historial en memoria del servidor, por eso puede responder repreguntas.
3. En producción multiusuario, se separan chats por sesión/usuario (no un único `chat_model` global).

### 3.3 Gestión stateless de la aplicación
Si usamos la versión anterior y probamos en dos pestañas diferentes, te responderá teniendo en cuenta lo que se le ha dicho. Esto NO es lo ideal, una forma de solucionarlo sería crear un chat en el endpoint, pero saturaríamos la memoria creando chats. Vamos a hacerlo de una forma más profesional, haciendo cambios para que sea `stateless` (el frontend envía el historial y el backend lo recibe)

Para ello, la entrada al endpoint no tendrá únicamente la pregunta, deberá incluir el historial (lista de strings). Para ello tenemos que cambiar la definición de la entrada.

```python
class ChatRequest(BaseModel):
    pregunta: str
    historial: list[str] = []
```

¿Vale la pena seguir utilizando chats? Si lo pensamos, realmente no, tendríamos que CREAR y GUARDAR cada chat de cada usuario, una mejor aproximación es que cada mensaje a Gemini tenga su propio contexto, por tanto eliminamos la creación del chat. 
Además, usaremos prompts extendidos con el contexto y el historial. 

Al modificar ChatRequest, ahora el endpoint espera la pregunta y el historial.

```python
def chat_con_contexto(request: ChatRequest):
```

De aquí podemos obtener la pregunta y el historial
```python
pregunta = request.pregunta
historial = request.historial
```

Ahora podemos crear el prompt extendido con el contexto y el historial recibidos del frontend.

```python
prompt = f"""
Eres un asistente que responde preguntas usando SOLO información del documento.

REGLAS IMPORTANTES:
- No copies el documento completo.
- No repitas texto largo del documento.
- Extrae SOLO la información necesaria.
- Si la respuesta no está en el documento, di: "No aparece en el documento".
- Responde de forma breve y directa.

DOCUMENTO:
\"\"\"{documento}\"\"\"

HISTORIAL:
{historial}

PREGUNTA:
{pregunta}

RESPUESTA:
"""
```

Y finalmente enviarlo a Gemini
```python
resp = client.models.generate_content(
  model=MODEL_ID,
  contents=prompt
)
```        

Y devolver el resultado

```python
return ChatResponse(respuesta=respuesta.text)
```

Ejemplo de request desde frontend:

```json
"historial": [
  "USER: ¿Cuántos días de vacaciones hay?",
  "ASSISTANT: Hay 25 días laborables."
]
```

O cambiar `ChatRequest` para permitir versiones más limpias como:
```json
{
  "pregunta": "¿Y en agosto?",
  "historial": [
    {"role": "user", "parts": "¿Cuántos días de vacaciones hay?"},
    {"role": "model", "parts": "Hay 25 días laborables."}
  ]
}
```

### Ejemplo frontend (HTML)
Para no tener que montar CORS, vamos a servir el frontend directamente desde FastAPI, añadimos lo siguiente:
```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")
```

Y ya podemos cargar este HTML dentro de un directorio `/static`
```html
<!doctype html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Chat Stateless (Historial en Memoria)</title>
  <style>
    body { font-family: sans-serif; max-width: 760px; margin: 24px auto; padding: 0 12px; }
    #log { border: 1px solid #ccc; border-radius: 8px; padding: 12px; min-height: 220px; white-space: pre-wrap; }
    .row { display: flex; gap: 8px; margin-top: 12px; }
    input { flex: 1; padding: 10px; }
    button { padding: 10px 14px; cursor: pointer; }
  </style>
</head>
<body>
  <h1>Chat con historial en memoria</h1>
  <p>Este ejemplo guarda localmente el historial y lo manda al backend en cada petición.</p>

  <div id="log"></div>

  <div class="row">
    <input id="pregunta" type="text" placeholder="Escribe tu pregunta..." />
    <button id="enviar">Enviar</button>
    <button id="limpiar" type="button">Limpiar historial</button>
  </div>

  <script>
    // Historial en memoria (se pierde al recargar)
    let historial = [];

    const log = document.getElementById("log");
    const input = document.getElementById("pregunta");
    const btnEnviar = document.getElementById("enviar");
    const btnLimpiar = document.getElementById("limpiar");

    function pintarLog() {
      log.textContent = historial.join("\n");
    }

    async function preguntar() {
      const pregunta = input.value.trim();
      if (!pregunta) return;

      // 1) Añadimos turno de usuario al historial local
      historial.push(`USER: ${pregunta}`);
      pintarLog();
      input.value = "";
      input.focus();

      try {
        // 2) Enviamos pregunta + historial completo al backend stateless
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            pregunta: pregunta,
            historial: historial
          })
        });

        if (!res.ok) {
          const errText = await res.text();
          throw new Error(`HTTP ${res.status} - ${errText}`);
        }

        const data = await res.json();

        // 3) Añadimos respuesta del asistente al historial local
        historial.push(`ASSISTANT: ${data.respuesta}`);
        pintarLog();
      } catch (error) {
        historial.push(`ASSISTANT: Error al consultar API (${error.message})`);
        pintarLog();
      }
    }

    btnEnviar.addEventListener("click", preguntar);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") preguntar();
    });

    btnLimpiar.addEventListener("click", () => {
      historial = [];
      pintarLog();
      input.focus();
    });

    pintarLog();
  </script>
</body>
</html>
```

En caso de querer usar otro frontend, hay que configurar CORS:
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

### Actividad 1: Fallback

Hemos utilizado un modelo específico de Gemini, pero, como vimos en las prácticas anteriores, hacer fallback a otros modelos es bastante importante. Puedes intentar modificar el código para aplicar fallback, incluso si falla, pasar a modelos de Hugging Face (usando pipeline, sin necesidad de cambiar el prompt de entrada).

### Actividad 2: Fuentes

Ya sabemos que en aplicaciones reales, no basta con que el modelo responda bien, necesitamos que responda en un formato que podamos procesar automáticamente desde el frontend. Modifica el endpoint para que Gemini devuelva siempre una respuesta en formato JSON con esta 
estructura:
```json
{
  "respuesta": string,
  "fuente": string
}
```
Donde:
- "respuesta": respuesta corta a la pregunta
- "fuente": sección del documento de donde se ha extraído la información (por ejemplo: "Vacaciones 2026")
