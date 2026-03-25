---
title: Gestión de modelos preentrenados
sidebar_position: 2
---

# Gestión de modelos preentrenados

Ya sabéis usar pipelines y entendéis cómo funcionan los transformers por dentro. En esta sesión veremos cómo **gestionar modelos** para trabajar offline, optimizar el uso de memoria y ejecutar modelos grandes.

## Guardar y cargar modelos localmente

Útil para trabajar **offline** o desplegar en **producción**.

### Guardar un modelo

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Cargar modelo desde Hugging Face
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Guardar localmente
tokenizer.save_pretrained("./mi_modelo_local")
model.save_pretrained("./mi_modelo_local")
```

### Cargar un modelo local

```python
# Cargar desde la carpeta local (sin internet)
tokenizer = AutoTokenizer.from_pretrained("./mi_modelo_local")
model = AutoModelForCausalLM.from_pretrained("./mi_modelo_local")

# O usar en un pipeline
generator = pipeline("text-generation", model="./mi_modelo_local")
```

### Estructura de archivos

```
mi_modelo_local/
├── config.json              # Arquitectura del modelo
├── model.safetensors        # Pesos (puede ser .bin en modelos antiguos)
├── tokenizer.json           # Tokenizer
├── tokenizer_config.json
├── vocab.json / vocab.txt
└── special_tokens_map.json
```

### ¿Por qué guardar modelos localmente?

| Ventaja | Descripción |
|---------|-------------|
| **Sin internet** | Funciona offline, no depende de HuggingFace Hub |
| **Velocidad** | No hay descarga cada vez que ejecutas |
| **Reproducibilidad** | Siempre usas la misma versión del modelo |
| **Despliegue** | Empaquetas el modelo con tu aplicación |

## Uso eficiente con GPU

### Verificar y usar GPU

```python
import torch

# ¿Hay GPU disponible?
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cargar modelo directamente en GPU
model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")

# O especificar dispositivo en el pipeline
generator = pipeline("text-generation", model="gpt2", device=0)  # GPU 0
```

### Procesar en batches

Procesar múltiples textos a la vez es más eficiente:

```python
generator = pipeline("text-generation", model="gpt2", device=0)

# En lugar de un bucle...
prompts = [
    "The future of AI is",
    "Python is great because",
    "Machine learning helps"
]

# Procesar todos a la vez (más rápido)
results = generator(prompts, max_new_tokens=20, batch_size=3)

for prompt, result in zip(prompts, results):
    print(f"{prompt} → {result[0]['generated_text']}")
```

## Modelos grandes: Cuantización

Los modelos grandes (7B, 13B, 70B parámetros) no caben en GPUs normales. La **cuantización** reduce la precisión de los pesos para que ocupen menos memoria.

### ¿Qué es la cuantización?

Normalmente los pesos de un modelo son números de 32 bits (float32). La cuantización los convierte a menos bits:

| Precisión | Bits por peso | Memoria ~7B modelo | Calidad |
|-----------|---------------|-------------------|---------|
| float32 | 32 | ~28 GB | 100% |
| float16 | 16 | ~14 GB | ~99% |
| int8 | 8 | ~7 GB | ~97% |
| int4 | 4 | ~3.5 GB | ~93% |

### Cargar modelo en 8 bits

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    load_in_8bit=True,          # Cuantización a 8 bits
    device_map="auto"
)
```

### Cargar modelo en 4 bits (BitsAndBytes)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### Modelos ya cuantizados (GGUF)

Muchos modelos en Hugging Face ya vienen cuantizados, listos para usar:

```python
# Buscar modelos GGUF (cuantizados para CPU/GPU limitada)
# Ejemplo: TheBloke tiene muchos modelos cuantizados
# https://huggingface.co/TheBloke
```

**Nomenclatura común:**
- `Q4_K_M` - Cuantización 4 bits, calidad media
- `Q5_K_M` - Cuantización 5 bits, mejor calidad
- `Q8_0` - Cuantización 8 bits, casi sin pérdida

## Resumen

| Qué quieres | Cómo hacerlo |
|-------------|--------------|
| Trabajar offline | `save_pretrained()` / `from_pretrained("./local")` |
| Usar GPU | `device_map="auto"` o `device=0` |
| Más velocidad | `batch_size > 1` |
| Modelo grande en GPU pequeña | `load_in_8bit=True` o `load_in_4bit=True` |
| Modelo muy grande en CPU | Buscar versiones GGUF cuantizadas |

## Próximos pasos

Con lo que habéis aprendido, podéis:

1. **Descargar modelos** y trabajar sin conexión
2. **Optimizar memoria** con cuantización
3. **Usar GPU** eficientemente

En la siguiente sesión veremos **RAG** para dar conocimiento externo a los modelos.
