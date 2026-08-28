# Henry M3 L4: Observabilidad y Evaluación de Agentes con Gemini 3.7 Flash

Un agente pequeño con un aparato de medida grande alrededor. El agente busca en
Wikipedia y responde; todo lo demás existe para saber **si lo hace bien**, y para
notarlo cuando deja de hacerlo.

El proyecto cubre el ciclo completo que plantea la lección: trazar, instrumentar,
probar contra casos fijos, mejorar a partir de lo medido y volver a medir.

## 🚀 Arquitectura del Proyecto

1.  **Capa de Infraestructura (`conexion.py`)**: la única puerta al modelo. Crea el
    cliente de Gemini, el *handler* de Langfuse y activa el autologging de MLflow.
    Todo lo demás lo importa de aquí.
2.  **Capa de Orquestación**:
    * **Grafo mínimo (`main.py`, `grafo.py`)**: un solo nodo — pregunta, respuesta,
      traza. Sirve para comprobar que el cableado funciona.
    * **Agente Reactivo (`agente.py`)**: `create_react_agent` con la herramienta de
      Wikipedia y una instrucción de sistema que gobierna cuándo debe usarla.
3.  **Capa de Evaluación (`evaluar.py`, `dataset_dorado.json`)**: corre el agente
    contra casos con respuesta esperada y sube los resultados a Langfuse.

## 🛠️ Stack Tecnológico

* **LLM**: Google Gemini 3.7 Flash (configurable con `GEMINI_MODEL`).
* **Frameworks**: LangChain 1.x & LangGraph 1.x.
* **Observabilidad**:
    * **Langfuse**: trazas y experimentos.
    * **LangSmith**: debugging nativo del ecosistema LangChain.
    * **MLflow**: histórico local en SQLite.

## 📋 Requisitos Previos

- Python 3.10 o superior.
- Entorno virtual (recomendado).

## ⚙️ Configuración e Instalación

```bash
git clone git@github.com:Ivan-Cepeda/AI-Observability.git
cd AI-Observability

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` fija las versiones exactas con las que el proyecto está
verificado. Un `pip install -U` a pelo instala versiones distintas cada vez y las
APIs de LangChain y Langfuse se mueven rápido.

Después, copia `.env_example` a `.env` y completa tus credenciales:

```bash
cp .env_example .env
```

## ▶️ Ejecución

```bash
python main.py     # grafo mínimo de un nodo
python grafo.py    # idéntico a main.py, con otro prompt
python agente.py   # agente ReAct con la herramienta de Wikipedia
python evaluar.py  # evaluación contra el dataset dorado
```

## 🎯 Evaluación con Casos Dorados

Trazar sin evaluar es mirar el tablero sin decidir nada. `evaluar.py` cierra el
ciclo: corre el agente contra un conjunto fijo de casos con respuesta esperada,
los puntúa y sube los scores a Langfuse, de modo que dos corridas se comparen
entre sí.

```bash
python evaluar.py                      # nombre de corrida automático
python evaluar.py base --sin-prompt    # variante sin instrucción de sistema
python evaluar.py v2-con-prompt        # variante actual
```

`--sin-prompt` reproduce el agente original, sin instrucciones de uso de
herramientas. No es una opción de conveniencia: es la línea base contra la que se
compara todo lo demás. Sin un «antes», un «después» no significa nada.

### El dataset

`dataset_dorado.json` tiene seis casos. Cuatro miden competencia. **Dos son
trampas deliberadas**:

- `alucinacion-fecha` pregunta por un dato que no existe públicamente: mide
  **honestidad**, no conocimiento. Una respuesta con cifra es un fallo aunque
  suene impecable.
- `aritmetica-sin-herramienta` pregunta cuánto es 17 × 23: mide el criterio para
  **no** buscar.

Un dataset sin trampas solo confirma lo que ya creías.

### Los evaluadores

| Evaluador | Tipo | Qué mide |
|---|---|---|
| `cobertura_de_hechos` | determinista | Fracción de hechos esperados presentes, normalizando tildes. |
| `uso_de_herramienta` | determinista | Si buscó cuando debía *y se abstuvo cuando no*. |
| `juez_llm` | Gemini, temp. 0 | Si la respuesta cumple el criterio escrito del caso. |
| `tasa_de_aprobados` | de corrida | Fracción de casos con nota del juez ≥ 0,7. |

Cuatro y no uno porque miden cosas distintas y cuestan cosas distintas. Los
deterministas son gratis y no opinan; el juez cuesta una llamada al modelo y ve lo
que ningún *keyword matching* alcanza; el de corrida existe para que haya **un solo
número** que comparar entre versiones.

## 🔍 Cómo interpretar los resultados

La primera corrida devolvió un pleno — y aun así escondía un fallo grave:

```
OK   rag-definicion             juez=1.0  hechos=0.67  herramienta=0
OK   transformer-autores        juez=1.0  hechos=1.00  herramienta=0
OK   alucinacion-fecha          juez=1.0  hechos=1.00  herramienta=1
OK   aritmetica-sin-herramienta juez=1.0  hechos=1.00  herramienta=1
OK   langgraph-vs-langchain     juez=1.0  hechos=1.00  herramienta=0
OK   observabilidad-llm         juez=1.0  hechos=1.00  herramienta=0

==> tasa_de_aprobados: 1.00  (6/6 casos con juez_llm >= 0.7)
```

**En cuatro casos que exigían consultar Wikipedia, el agente hizo cero llamadas.**
Contestó de memoria. El prompt original decía literalmente *«Busca en Wikipedia qué
es…»*; con esa orden explícita obedecía, pero con preguntas naturales Gemini decide
que ya lo sabe y se salta la herramienta. La demo funcionaba por el prompt, no por
el agente.

Un fallo así es **invisible en una traza suelta**: abre cualquiera de las seis en
Langfuse y verás una respuesta correcta. Solo existe en el agregado.

### El arreglo, medido

`INSTRUCCION_SISTEMA` en `agente.py` obliga al agente a consultar antes de afirmar
hechos verificables, y a abstenerse en aritmética y reformulaciones:

| Caso | Base | Con prompt | Esperado |
|---|:--:|:--:|---|
| `rag-definicion` | 0 | **1** | buscar ✓ |
| `transformer-autores` | 0 | **1** | buscar ✓ |
| `langgraph-vs-langchain` | 0 | **1** | buscar ✓ |
| `observabilidad-llm` | 0 | **1** | buscar ✓ |
| `aritmetica-sin-herramienta` | 1 | 1 | no buscar ✓ |
| `alucinacion-fecha` | 1 | **0** | no buscar ✗ |

`uso_de_herramienta` pasa de **2/6 a 5/6**. Y el dato que más dice:
`tasa_de_aprobados` se quedó en **1,00 en las dos corridas**. La métrica de titular
no se movió ni un punto mientras el comportamiento real cambiaba por completo. Esa
insensibilidad es, en sí misma, lo que hay que aprender a detectar.

### La regresión que no lo es

`alucinacion-fecha` empeora: ahora busca en Wikipedia un dato que no existe, y el
dataset lo marca como fallo porque dice `debe_usar_herramienta: false`.

Pero mira lo que hizo el agente: buscó, no encontró nada y admitió que no lo sabía.
El juez le dio 1,0. **El que está equivocado es el caso, no el agente** — verificar
antes de negar es mejor comportamiento que el que se especificó. Se deja así a
propósito.

Un dataset dorado no es una verdad revelada: es una hipótesis sobre lo que quieres,
escrita antes de ver los resultados. Cuando una corrida y el dataset discrepan, la
pregunta correcta no es «¿cómo arreglo el agente?» sino «**¿quién de los dos tiene
razón?**». Ajustar el caso porque el agente demostró mejor criterio es progreso;
ajustarlo para que salgan los números que querías es engañarse con pasos extra.

### Qué mirar en la próxima corrida

- **Lee la columna determinista antes que la del juez.** El juez aprueba respuestas
  plausibles; las deterministas comprueban conducta. Un pleno del juez con ceros al
  lado es una señal de alarma, no de éxito.
- **Desconfía de una métrica que no se mueve.** Si cambias el agente y el número de
  titular sigue idéntico, o el cambio no hizo nada o la métrica no mide lo que crees.
- **Guarda siempre la línea base.**
- **Añade un caso cada vez que encuentres un fallo.** Es la única manera de que ese
  fallo concreto no vuelva sin avisar.

## 📈 Visualización de Trazas

### Langfuse & LangSmith
Las trazas se envían automáticamente a las plataformas configuradas en el `.env`.
Los resultados de `evaluar.py` quedan en Langfuse bajo **Experiments**, agrupados
por nombre de corrida.

### MLflow (Local)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow_henry.db
```

Abre [http://127.0.0.1:5000](http://127.0.0.1:5000). Ojo: en MLflow 3.x el
autologging de LangChain escribe **traces**, no **runs** — que `search_runs()`
devuelva cero es lo esperado, no un fallo. Mira `search_traces()`.

## 🐛 Averías conocidas

Ninguna es teórica: todas rompieron la ejecución y todas están corregidas aquí.

| Síntoma | Causa real |
|---|---|
| `JSONDecodeError: Expecting value: line 1 column 1` | Wikimedia devuelve **403** al User-Agent por defecto de la librería `wikipedia` ([T400119](https://phabricator.wikimedia.org/T400119)), y **429** si el tuyo no lleva datos de contacto. Ambos llegan como texto plano y la librería los parsea como JSON. Arreglo: `set_user_agent()` con contacto, `set_rate_limiting(True)` y evaluación secuencial. |
| `UnicodeEncodeError: '✅'` | El emoji del `print` final revienta en la consola cp1252 de Windows, así que *toda* ejecución terminaba con traceback aunque el grafo hubiera ido bien. Arreglo: reconfigurar la salida a UTF-8 en `conexion.py`. |
| `search_runs()` devuelve 0 | No es un fallo. Ver la nota de MLflow 3.x más arriba. |
| Host de Langfuse | El código leía `LANGFUSE_HOST` y el `.env` definía `LANGFUSE_BASE_URL`. Hoy no rompe porque el SDK v4 deduce la región del prefijo de la clave pública, pero eso es suerte, no diseño. Ahora acepta las dos. |
| `'dict' object has no attribute 'name'` | La docstring del SDK de Langfuse dice que los evaluadores devuelven diccionarios; la implementación accede por atributo. Hay que devolver `Evaluation(...)`. |

## 📁 Estructura de Archivos

- `conexion.py` — clientes de Gemini y Langfuse, `autolog` de MLflow, salida UTF-8.
- `main.py` / `grafo.py` — grafo mínimo de un nodo. Hoy son el mismo archivo con
  distinto prompt.
- `agente.py` — agente ReAct, la herramienta de Wikipedia y `INSTRUCCION_SISTEMA`.
  Expone `construir_agente()` para que el evaluador monte el mismo agente que corre
  en producción.
- `dataset_dorado.json` — los seis casos con respuesta esperada.
- `evaluar.py` — la corrida de evaluación y sus cuatro evaluadores.
- `requirements.txt` — versiones verificadas.
- `.env` — llaves de API. **No se sube a GitHub.**

## 👤 Autor
Proyecto desarrollado por **Iván Miguel Cepeda**.
