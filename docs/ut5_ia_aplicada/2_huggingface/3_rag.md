---
title: RAG (Retrieval-Augmented Generation)
sidebar_position: 3
---

# RAG: Retrieval-Augmented Generation

## El problema de los modelos generativos

Los modelos de generación de texto tienen limitaciones importantes:

| Problema | Ejemplo |
|----------|---------|
| **Conocimiento desactualizado** | "¿Quién ganó el Mundial 2026?" → No lo sabe |
| **No conocen datos privados** | "¿Qué dice el manual de mi empresa?" → No lo sabe |
| **Alucinaciones** | Inventa respuestas cuando no sabe |
| **Sin fuentes** | No puede citar de dónde sacó la información |

**RAG soluciona estos problemas** dándole al modelo acceso a información externa.

## ¿Qué es RAG?

**RAG (Retrieval-Augmented Generation)** es una **arquitectura** que combina:

1. **Retrieval**: Buscar información relevante en documentos
2. **Augmented**: Añadir esa información al prompt
3. **Generation**: El modelo genera una respuesta basada en esa información

```
Pregunta del usuario
        ↓
   [RETRIEVAL] ← Busca en documentos
        ↓
   Contexto relevante
        ↓
   [AUGMENTED] ← Combina pregunta + contexto
        ↓
   Prompt enriquecido → Modelo → Respuesta
```

:::info RAG es una arquitectura, no una propiedad del modelo
RAG funciona con **cualquier modelo de generación de texto**: modelos locales de Hugging Face (GPT-2, LLaMA, Mistral) o APIs en la nube. En esta sesión veremos RAG con **modelos locales**. En el bloque de LLM veremos cómo usar RAG con APIs cloud.
:::

## RAG vs Fine-tuning: ¿Por qué RAG domina en empresas?

| Aspecto | RAG | Fine-tuning |
|---------|-----|-------------|
| **Datos actualizados** | ✅ Añades documentos y listo | ❌ Hay que reentrenar |
| **Coste** | ✅ Bajo (solo inferencia) | ❌ Alto (GPU, tiempo, datos) |
| **Tiempo de implementación** | ✅ Horas/días | ❌ Semanas/meses |
| **Citar fuentes** | ✅ Puedes mostrar de dónde viene | ❌ No es posible |
| **Datos privados/sensibles** | ✅ Se quedan en tu infra | ⚠️ Se "mezclan" en el modelo |
| **Alucinaciones** | ✅ Reducidas (tiene contexto real) | ⚠️ Puede seguir alucinando |
| **Escala** | ✅ Millones de documentos | ❌ Limitado por entrenamiento |

### ¿Cuándo usar cada uno?

| Situación | Recomendación |
|-----------|---------------|
| Chatbot con documentación de empresa | **RAG** |
| Asistente con datos que cambian | **RAG** |
| Necesitas citar fuentes | **RAG** |
| Cambiar el estilo/tono del modelo | **Fine-tuning** |
| Dominio muy específico (médico, legal) | **Fine-tuning** (o ambos) |
| Poco presupuesto | **RAG** |

**En la práctica**: La mayoría de empresas usan RAG porque es más rápido, barato y flexible. Fine-tuning se reserva para casos muy específicos.

## Componentes de RAG

### 1. Documentos fuente

Tu base de conocimiento:
- PDFs, Word, TXT, Markdown
- Páginas web
- Bases de datos
- APIs

### 2. Chunking (dividir documentos)

Los documentos se dividen en "chunks" porque:
- Los modelos tienen límite de tokens
- Chunks pequeños = búsqueda más precisa

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Caracteres por chunk
    chunk_overlap=50     # Solapamiento entre chunks
)

chunks = splitter.split_text(documento_largo)
# ["chunk 1...", "chunk 2...", ...]
```

### 3. Embeddings (vectorizar texto)

Convertimos texto en vectores numéricos. **Textos similares = vectores cercanos**.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Texto → Vector
embedding = model.encode("¿Qué es Python?")
print(embedding.shape)  # (384,)
```

**Modelos de embeddings recomendados:**

| Modelo | Idiomas | Uso |
|--------|---------|-----|
| `all-MiniLM-L6-v2` | Inglés | Rápido, prototipos |
| `paraphrase-multilingual-MiniLM-L12-v2` | Multi (incluye español) | Producción multilingüe |
| `BAAI/bge-m3` | Multi | Alta calidad |

### 4. Vector Store (base de datos vectorial)

Almacena embeddings y permite buscar por similitud.

```python
import chromadb

# Crear base de datos
client = chromadb.Client()
collection = client.create_collection("mis_docs")

# Guardar chunks
collection.add(
    documents=["chunk 1", "chunk 2", "chunk 3"],
    ids=["id1", "id2", "id3"]
)

# Buscar similares a una pregunta
results = collection.query(
    query_texts=["mi pregunta"],
    n_results=3  # Top 3 más relevantes
)
```

**Opciones de vector stores:**

| Vector Store | Tipo | Cuándo usarlo |
|--------------|------|---------------|
| **ChromaDB** | Local | Prototipos, proyectos pequeños |
| **FAISS** | Local | Alto rendimiento, grandes datasets |
| **Pinecone** | Cloud | Producción, escalabilidad |

### 5. El prompt aumentado

Combinamos la pregunta con el contexto recuperado:

```python
contexto = "Python fue creado por Guido van Rossum en 1991..."

prompt = f"""Usa ÚNICAMENTE el siguiente contexto para responder.
Si no encuentras la respuesta en el contexto, di "No tengo esa información".

CONTEXTO:
{contexto}

PREGUNTA: ¿Quién creó Python?

RESPUESTA:"""
```

### 6. Generación (modelo local)

El modelo recibe el prompt enriquecido y genera la respuesta. Usamos un pipeline de Hugging Face:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
respuesta = generator(prompt, max_new_tokens=100)
```

Para mejores resultados, puedes usar modelos más potentes:

```python
# Modelo más capaz (requiere más RAM/VRAM)
generator = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-medium"  # O cualquier modelo de HF
)
```

## Ejemplo completo: RAG con modelos locales

```python
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

# ============ FASE 1: INDEXACIÓN (una vez) ============

# Documentos de ejemplo (tu base de conocimiento)
documentos = [
    "Python fue creado por Guido van Rossum en 1991.",
    "Python es un lenguaje interpretado de alto nivel.",
    "JavaScript fue creado por Brendan Eich en 1995.",
    "JavaScript se usa principalmente en navegadores web.",
    "RAG significa Retrieval-Augmented Generation.",
]

# Crear vector store y añadir documentos
client = chromadb.Client()
collection = client.create_collection("mi_conocimiento")

collection.add(
    documents=documentos,
    ids=[f"doc_{i}" for i in range(len(documentos))]
)

print(f"✅ Indexados {len(documentos)} documentos")

# ============ FASE 2: MODELO DE GENERACIÓN ============

# Cargar modelo local de Hugging Face
generator = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=50
)

# ============ FASE 3: FUNCIÓN RAG ============

def rag_responder(pregunta, n_chunks=2):
    # 1. Buscar chunks relevantes
    results = collection.query(
        query_texts=[pregunta],
        n_results=n_chunks
    )
    chunks = results['documents'][0]

    # 2. Construir prompt con contexto
    contexto = "\n".join(chunks)

    prompt = f"""Basándote SOLO en el siguiente contexto, responde la pregunta.

Contexto:
{contexto}

Pregunta: {pregunta}

Respuesta:"""

    # 3. Generar respuesta con modelo local
    respuesta = generator(prompt)[0]['generated_text']

    # 4. Mostrar trazabilidad (de dónde vino la info)
    print(f"📚 Chunks usados:")
    for chunk in chunks:
        print(f"   - {chunk}")

    return respuesta

# ============ USAR ============

pregunta = "¿Quién creó Python?"
respuesta = rag_responder(pregunta)
print(f"\n🤖 Respuesta:\n{respuesta}")
```

## RAG con LangChain y Hugging Face

LangChain simplifica la implementación y se integra perfectamente con modelos de Hugging Face:

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# 1. Cargar documento
loader = TextLoader("mi_documento.txt")
docs = loader.load()

# 2. Dividir en chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Crear embeddings y vector store (todo local)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. Crear retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Crear modelo local con Hugging Face
pipe = pipeline("text-generation", model="gpt2", max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=pipe)

# 6. Crear cadena RAG
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# 7. Preguntar
respuesta = qa.run("¿De qué trata el documento?")
print(respuesta)
```

## Modelos locales recomendados para RAG

| Modelo | Tamaño | RAM/VRAM | Calidad | Velocidad |
|--------|--------|----------|---------|-----------|
| `gpt2` | 124M | ~500MB | Baja | Muy rápida |
| `gpt2-medium` | 355M | ~1.5GB | Media | Rápida |
| `microsoft/DialoGPT-medium` | 355M | ~1.5GB | Media (chat) | Rápida |
| `google/flan-t5-base` | 250M | ~1GB | Buena | Rápida |
| `google/flan-t5-large` | 780M | ~3GB | Muy buena | Media |
| `mistralai/Mistral-7B-v0.1` | 7B | ~14GB | Excelente | Lenta |

:::tip Para esta práctica
Usa `gpt2` o `google/flan-t5-base` para prototipos rápidos. Si tienes GPU con más de 8GB VRAM, prueba modelos más grandes.
:::

## Resumen visual

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG (Modelos Locales)                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   INDEXACIÓN (una vez):                                      │
│   Documentos → Chunks → Embeddings → Vector Store            │
│                         (sentence-transformers)  (ChromaDB)  │
│                                                              │
│   CONSULTA (cada pregunta):                                  │
│   Pregunta → Buscar similares → Top K chunks                 │
│                    ↓                                         │
│   Prompt + Chunks → pipeline("text-generation") → Respuesta  │
│                     (Hugging Face)                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Librerías necesarias

```bash
pip install langchain langchain-community chromadb sentence-transformers transformers
```

| Librería | Para qué |
|----------|----------|
| `transformers` | Modelos de generación (Hugging Face) |
| `sentence-transformers` | Embeddings |
| `chromadb` | Vector store local |
| `langchain` | Framework RAG |

## Ejercicios prácticos

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgecs/apuntes/blob/main/docs/ut5_ia_aplicada/2_huggingface/notebooks/RAG.ipynb)

**IMPORTANTE**: Guarda una copia en Drive antes de empezar (Archivo → Guardar una copia)
