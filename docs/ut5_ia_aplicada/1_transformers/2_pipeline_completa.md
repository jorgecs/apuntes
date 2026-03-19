---
title: La pipeline completa de transformers
sidebar_position: 2
---

# La pipeline completa de un transformer

Aquí vamos a entender **por qué** cada paso existe y **qué hace realmente**.

## ¿Por qué usar transformers?

Los transformers fueron diseñados para resolver un problema concreto: **las máquinas no entienden texto, solo números**. Y no solo eso, sino que necesitan entender el **contexto** entre todas las palabras simultáneamente.

La pipeline es la solución a esto:

```
Texto legible por humanos 
    ↓ (Tokenización)
Palabras/subpalabras 
    ↓ (Embeddings)
Vectores numéricos = espacio multidimensional 
    ↓ (Atención)
Relaciones entre tokens = contexto 
    ↓ (Capa de salida)
Predicción/clasificación/generación
```

Veamos cada paso.

---

## 1. Tokenización: "Economía lingüística"

### ¿Qué pasa?

Conviertes el texto en **tokens** (trozos pequeños).

```
"Programación" → ["Programación"]  (el tokenizer puede dejarlo entero)
"Programación" → ["Program", "##ación"]  (o dividirlo si es raro)
"token" → ["token"]  (una palabra común se mantiene)
"token" → ["to", "##ken"]  (algunos tokenizers lo dividen en unidades comunes)
```

### ¿Por qué se fragmenta?

**Razón 1: Vocabulario limitado**  
Un tokenizer tiene un vocabulario fijo (p. ej., 30,000 palabras/subpalabras). No puede guardar todas las palabras existentes. Entonces divide las raras en partes comunes.

Ejemplo:
- "Electroencefalografista" → ["Electro", "##encefalo", "##grafista"]

**Razón 2: Reutilización de patrones**  
Si el modelo ve "programa", "programación", "reprogramar", todas comparten la raíz "program". Dividir así le ayuda a entender que todas están relacionadas.

**Razón 3: Eficiencia**  
Guardar 500,000 palabras distintas costaría mucha memoria. Con 30,000 tokens puedes construir casi cualquier palabra. Es como usar un "juego de piezas" que se reutilizan.

### ¿Diferentes tokenizers, diferentes cortes?

Sí. Algunos dividen más, otros menos:

```
Frase: "Necesito dinero"

Tokenizer A (BERT): ["Necesito", "dinero"]
Tokenizer B (GPT2): ["Neces", "##ito", "dinero"]
Tokenizer C (Sentencepiece): ["Necesito", "▁dinero"]  (nota: ▁ = espacio)
```

Diferentes idiomas también se dividen diferente. El inglés tiende a usar menos tokens, lenguas aglutinantes (turco, finlandés) usan más.

### Siguiente paso: Conocer qué token es

Una lista de tokens es solo eso: una lista. Pero para que una máquina trabaje con texto, necesita mapearlos a **números**.

```
Tokens: ["El", "modelo", "entiende", "...]
IDs:    [1034, 4521, 7892, ...]
```

Cada token tiene asignado un número único en el diccionario.

---

## 2. Embeddings: "De palabras a números"

### ¿Qué pasa?

Conviertes cada token (número) en un **vector multidimensional** (una lista de números).

Piensa en ello así: cada token se mapea a un "perfil numérico" que el modelo **ya aprendió** durante su preentrenamiento.

```
Token: "banco"  (ID = 5234)
    ↓ (embedding)
Vector: [0.2, -0.5, 0.8, 0.1, ..., -0.3]  (p. ej., 768 números)
```

Ese vector no se inventa en el momento. Sale de una tabla de embeddings aprendida previamente (y, si reentrenas o haces fine-tuning, puede ajustarse).

### ¿Por qué números en lugar de palabras?

**Las máquinas entienden geometría, no semántica.**

Si tienes la palabra "banco" como string, ¿cómo le dices a una red neuronal que se relaciona con "dinero"? No puedes. Las redes neuronales solo entienden operaciones matemáticas: multiplicación, suma, distancia.

Pero si "banco" es un vector en un espacio de 768 dimensiones:

```
"banco" = [0.2, -0.5, 0.8, 0.1, ...]
"dinero" = [0.1, -0.6, 0.7, 0.15, ...]
```

Puedes calcular la **distancia** entre ellos. Si están cerca, significa que el modelo los considera relacionados.

En resumen: **token -> ID -> vector (embedding)**. Ese mapeo convierte texto en una forma que la red puede procesar.

### ¿Cómo se forman los embeddings?

**Durante el entrenamiento**, el modelo aprende a colocar palabras relacionadas cerca una de otra en este espacio multidimensional.

Ejemplo simple (imagina solo 2 dimensiones):

```
        ↑ dimensión 2
        |
    "reina"  ●
        |
        |     ● "rey"
        |
    "mujer"  |  ● "hombre"
        |
    ────┼──────────→ dimensión 1
```

Las palabras relacionadas están cerca. El modelo aprendió esto de millones de textos.

### Importante: ¿esto ya representa el significado de toda la frase?

Parcialmente. El embedding inicial captura significado del token, pero todavia sin todo el contexto fino de la frase.

Ejemplo:
- "banco" (dinero)
- "banco" (asiento)

El token inicial puede parecerse, pero tras la **atencion** se vuelve contextualizado y ya distingue mejor el significado segun la frase.

### ¿Para qué sirven los embeddings?

1. **Capturan significado**: Dos palabras con significado parecido tienen vectores parecidos.
2. **Permiten operaciones**: Puedes calcular similitud, distancia, relaciones.
3. **Son compactos**: En lugar de un diccionario enorme, solo tienes la matriz de embeddings (768 numeros por palabra).
4. **Abren la puerta al contexto**: La atencion toma esos vectores y los combina para construir el significado final de la frase.

---

## 3. Atención: "El modelo descubre qué mirar"

### El problema de antes (sin atención)

Imagina una máquina que procesa una frase palabra por palabra de forma **secuencial**:

```
"El gato está en el tejado"
 ↓ 
(lee "El")
 ↓ 
(lee "gato")
 ↓ 
(lee "está")
...
```

El problema: cuando procesa "gato", ¿recuerda qué fue "El"? ¿Recuerda qué viene después? En redes recurrentes (RNN), la memoria se va borrando.

**La atención lo resuelve permitiendo que el modelo "mire" todas las palabras a la vez.**

### ¿Cómo funciona la atención?

La idea: **cada token calcula una relación ponderada con todos los demás**.

```
Frase: "El gato está en el tejado"

Paso 1: El modelo pregunta: "¿Qué otras palabras debo considerar al procesar 'gato'?"

Paso 2: Asigna "pesos" (importancia):
   - "gato" ← "El" (peso = 0.9)  → muy importante, está relacionado
   - "gato" ← "está" (peso = 0.7)  → algo importante, es el verbo de la oración
   - "gato" ← "en" (peso = 0.3)  → menos importante
   - "gato" ← "tejado" (peso = 0.8)  → importante, dice dónde está

Paso 3: "Mezcla" todos esos vectores ponderados
```

### Visualización: multiplicación ponderada sin fórmulas

Imagina que tienes información sobre cada palabra:

```
"El":     [información sobre artículos]
"gato":   [información sobre animales]
"está":   [información sobre acciones]
"tejado": [información sobre lugares]
```

La atención dice: "Para entender 'gato', tomo el 90% de su propia información y mezclo un poco de 'El' (70%), 'está' (70%), 'tejado' (80%) y poco de 'en' (30%)."

Es como preguntar: "¿Este gato es independiente (100% su propia esencia) o depende mucho del contexto?"

La respuesta: **depende del contexto**. "Gato" necesita saber que hay un artículo ("El"), una acción ("está"), un lugar ("tejado").

### Por eso es revolucionario

Antes (con convoluciones o RNN):
- Veías solo una "ventana" alrededor de cada palabra
- Información lejana se perdía

Con atención:
- Cada palabra ve **toda la frase**
- El modelo aprende automáticamente qué es importante
- Es más eficiente (todo en paralelo, no secuencial)

---

## 4. Output (Salida): Cómo el modelo genera su predicción

### ¿Qué pasa después de toda la atención?

Después de pasar por embeddings y atención, cada token tiene un **vector contextualizado** (un número por cada dimensión que contiene información sobre su contexto).

Pero aquí viene lo importante: **ese vector contextualizado NO es la respuesta final**. El modelo aún no ha respondido tu pregunta.

### La capa de salida: "De contexto a predicción"

La **capa de salida** es lo que hace que el modelo dé una respuesta específica. Transforma los vectores contextualizados en **números que representan predicciones** para tu tarea.

```
Vector contextualizado de "película":
[0.5, -0.2, 0.8, ..., 0.3]  (768 números conteniendo contexto)
  ↓ (Capa de salida)
Números de salida: [2.3, -5.1]
  ↓ (Interpretación)
Predicción: "Positivo" (98%)
```

**¿Qué hace la capa de salida?**

La capa de salida es una **transformación final** (típicamente una multiplicación de matrices muy simple). Dice:

- Si la tarea es **clasificación de sentimiento**: genera un número para cada clase posible (positivo, neutral, negativo)
- Si la tarea es **traducción**: genera un número para cada palabra del vocabulario destino
- Si la tarea es **preguntas y respuestas**: genera puntuaciones de inicio y fin para cada token

### De números brutos a probabilidades interpretables

Los números que sale la capa de salida no tienen significado directo. ¿Qué significa [2.3, -5.1]? ¿Es mucho? ¿Poco?

Por eso usamos funciones especiales que **convierten esos números brutos en probabilidades claras**:

**Softmax:** Para tareas con **múltiples opciones excluyentes**

```
Salida bruta:  [2.3, -5.1]
     ↓ (Softmax)
Probabilidades: [0.98, 0.02]
                (98% positivo, 2% negativo)
```

Softmax amplifica las diferencias y las normaliza para que sumen 100%.

**Sigmoid:** Para tareas con **sí/no** o **etiquetas independientes**

```
Salida bruta: 2.3
    ↓ (Sigmoid)
Probabilidad: 0.91  (91% de chance)
```

Sigmoid convierte a un número entre 0 y 1.

### Ejemplo: Análisis de sentimientos

```
Frase: "El producto es excelente pero el envío fue muy lento"

Paso 1: El token "excelente" se procesa
  - Embeddings: se convierte en vector
  - Atención: se mezcla con el contexto de "producto", "envío", etc.
  - Vector contextualizado: [0.7, -0.3, 0.9, ...]

Paso 2: Capa de salida transforma en predicción
  - Números brutos: [positivo = 3.5, negativo = 1.2]
  
Paso 3: Softmax interpreta
  - Probabilidades: [positivo = 73%, negativo = 27%]
  - Resultado: Sentimiento "mixto pero más positivo"
```

Sin la capa de salida, el modelo solo tendría contexto. Sin Softmax, no sabría interpretar la predicción.

---

## ¿Por qué no usar solo una de estas?

### Sin tokenización
- El modelo recibiría texto sin procesar. No sabría dónde empieza y termina cada palabra.
- Se desperdicia memoria si se almacenan todas las palabras existentes.

### Sin embeddings
- Los tokens son solo números arbitrarios (1, 2, 3). El modelo no sabe que "gato" está relacionado con "felino".
- La red neuronal no podría capturar la semántica.

### Sin atención
- Las convoluciones/RNN solo ven el contexto local. Una palabra al principio no influye en el final.
- Sacrificas precisión y velocidad.

### Sin Output (Salida)
- El modelo solo tendrá representaciones contextuales, pero nunca responderá tu pregunta.
- No puede hacer predicciones específicas para la tarea (clasificación, traducción, etc.).

---

## Resumen: Por qué funciona todo junto

1. **Tokenización**: Conviertes texto en piezas que el modelo puede entender.
2. **Embeddings**: Conviertes piezas en números que contienen el significado.
3. **Atención**: Relacionas números para obtener el contexto (¿qué importa de cada palabra?).
4. **Output (Salida)**: Transformas contexto en predicciones, e interpretas esas predicciones con Softmax/Sigmoid.

Cada paso resuelve un problema específico, todos los componentes son necesarios.

---

## Aplicación práctica

**Lo más importante**: Los transformers son una **secuencia de transformaciones** muy bien diseñadas que permiten a máquinas:

1. Entender texto (tokens → embeddings)
2. Entender contexto (atención)
3. Tomar decisiones claras (Softmax/Sigmoid)

La práctica es solo aplicar estas transformaciones. Los modelos preentrenados ya tienen todo aprendido en los embeddings y los pesos de atención.


## Bibliografía adicional

* [Attention is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
* [Jay Alammar's Visual Guide to Transformers](http://jalammar.github.io/illustrated-transformer/)
* [Hugging Face Course - NLP](https://huggingface.co/course/chapter1/1)
