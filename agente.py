from conexion import obtener_infraestructura
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage

# Dejamos solo Wikipedia para aislar el error de YouTube
tools = [
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
]

def ejecutar_agente():
    llm, handler, langfuse_client = obtener_infraestructura()
    
    print("--- Iniciando Agente LangGraph con Gemini 2.5 Pro ---")
    
    agente = create_react_agent(llm, tools=tools)

    try:
        respuesta = agente.invoke(
            {"messages": [HumanMessage(content="Busca en Wikipedia qué es el patrón RAG (Retrieval-Augmented Generation) y resúmelo en una oración.")]},
            config={"callbacks": [handler]}
        )
        
        print("\nResultado Final:")
        print("-" * 40)
        print(respuesta["messages"][-1].content)
        print("-" * 40)
        
    except Exception as e:
        print(f"Error durante la ejecución del agente: {e}")
        
    finally:
        langfuse_client.flush()
        print("\n✅ Ejecución finalizada. Revisa la traza en Langfuse.")

if __name__ == "__main__":
    ejecutar_agente()