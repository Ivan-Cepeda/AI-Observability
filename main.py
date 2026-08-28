from conexion import obtener_infraestructura
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# 1. Definimos el estado de nuestro grafo
class AgentState(TypedDict):
    messages: List[str]

def ejecutar_grafo():
    llm, handler, langfuse_client = obtener_infraestructura()
    print("--- Iniciando Workflow LangGraph con Gemini ---")

    # 2. Definimos el nodo principal
    def nodo_ia(state: AgentState):
        pregunta = state["messages"][-1]
        
        # Inyectamos la observabilidad directamente en la invocación del nodo
        respuesta = llm.invoke(pregunta, config={"callbacks": [handler]})
        
        return {"messages": [respuesta.text]}

    # 3. Construimos y compilamos el grafo
    workflow = StateGraph(AgentState)
    workflow.add_node("asistente", nodo_ia)
    workflow.set_entry_point("asistente")
    workflow.add_edge("asistente", END)

    app = workflow.compile()

    try:
        # 4. Ejecución del flujo
        inputs = {"messages": ["Explica en una oración cómo la podemos ser mejores programadores en IA."]}
        
        for output in app.stream(inputs, config={"callbacks": [handler]}):
            # Imprimimos la salida de cada nodo que se vaya ejecutando
            for key, value in output.items():
                print(f"\nSalida del nodo '{key}':")
                print("-" * 40)
                print(value["messages"][-1])
                print("-" * 40)
                
    except Exception as e:
        print(f"Error en la ejecución del grafo: {e}")
        
    finally:
        # 5. Garantizamos el envío de la traza
        langfuse_client.flush()
        print("\n✅ Grafo finalizado. Traza enviada a Langfuse.")

if __name__ == "__main__":
    ejecutar_grafo()