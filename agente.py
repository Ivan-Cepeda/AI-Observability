from conexion import obtener_infraestructura

import wikipedia
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage

# Wikimedia rechaza con 403 el User-Agent por defecto de la libreria `wikipedia`
# (phabricator.wikimedia.org/T400119). Hay que declarar uno propio y descriptivo,
# CON datos de contacto: sin ellos caes en la cuota anonima y llegan 429 en cuanto
# haces unas pocas busquedas seguidas. Ambos errores llegan como HTML/texto plano
# y la libreria los intenta parsear como JSON, de ahi el confuso JSONDecodeError.
wikipedia.set_user_agent(
    "HenryM3L4-Agente/1.0 (proyecto educativo Henry; icepeda@vikua.com)"
)
wikipedia.set_rate_limiting(True)

# Dejamos solo Wikipedia para aislar el error de YouTube
tools = [
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="es", top_k_results=2)),
]

# Sin esta instruccion el agente contesta de memoria y no consulta nunca la
# herramienta: lo midio evaluar.py (uso_de_herramienta = 0/4). No basta con darle
# una herramienta; hay que decirle cuando esta obligado a usarla.
INSTRUCCION_SISTEMA = """Eres un asistente que solo afirma hechos que puede respaldar.

Reglas de uso de herramientas:
1. Si la pregunta es sobre hechos verificables (definiciones tecnicas, fechas,
   personas, eventos, terminologia), consulta Wikipedia ANTES de responder,
   aunque creas saber la respuesta. Tu memoria puede estar desactualizada.
2. No consultes Wikipedia para calculos aritmeticos, reformulaciones ni
   preguntas sobre la propia conversacion: resuelvelas directamente.
3. Si tras consultar sigues sin el dato, di que no lo sabes. Nunca inventes
   cifras, fechas ni nombres para rellenar un hueco.

Responde en espanol, de forma concisa."""


def construir_agente(llm=None, instruccion: str = INSTRUCCION_SISTEMA):
    """Devuelve el agente ReAct ya compilado.

    Lo separamos de ejecutar_agente() para que evaluar.py pueda montar el mismo
    agente que corre en produccion, sin duplicar su definicion. El parametro
    `instruccion` permite medir variantes del prompt: pasar None reproduce el
    agente original, sin instrucciones de uso de herramientas.
    """
    if llm is None:
        llm, _, _ = obtener_infraestructura()
    if instruccion is None:
        return create_react_agent(llm, tools=tools)
    return create_react_agent(llm, tools=tools, prompt=instruccion)


def ejecutar_agente():
    llm, handler, langfuse_client = obtener_infraestructura()
    
    print("--- Iniciando Agente LangGraph con Gemini 3.7 Flash ---")
    
    agente = construir_agente(llm)

    try:
        respuesta = agente.invoke(
            {"messages": [HumanMessage(content="Busca en Wikipedia qué es el patrón RAG (Retrieval-Augmented Generation) y resúmelo en una oración.")]},
            config={"callbacks": [handler]}
        )
        
        print("\nResultado Final:")
        print("-" * 40)
        print(respuesta["messages"][-1].text)
        print("-" * 40)
        
    except Exception as e:
        print(f"Error durante la ejecución del agente: {e}")
        
    finally:
        langfuse_client.flush()
        print("\n✅ Ejecución finalizada. Revisa la traza en Langfuse.")

if __name__ == "__main__":
    ejecutar_agente()