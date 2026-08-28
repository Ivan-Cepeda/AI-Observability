"""Evaluacion del agente contra el dataset dorado.

Cierra el ciclo que pide la leccion: el agente ya deja trazas (Langfuse, MLflow),
pero nadie las leia. Aqui corremos el agente contra un conjunto fijo de casos con
respuesta esperada, lo puntuamos y subimos los scores a Langfuse para poder
comparar una corrida contra otra.

    python evaluar.py                 # corrida con nombre automatico
    python evaluar.py mi-experimento  # corrida con nombre propio
"""

import io
import json
import os
import re
import sys
import unicodedata
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage
from langfuse import Evaluation

from agente import INSTRUCCION_SISTEMA, construir_agente
from conexion import GEMINI_MODEL, obtener_infraestructura

RUTA_DATASET = "dataset_dorado.json"

# Prompt del juez. Temperatura 0 y una sola cifra de salida: queremos un
# instrumento de medida repetible, no una segunda opinion creativa.
PROMPT_JUEZ = """Eres un evaluador estricto. Puntua si la RESPUESTA cumple el CRITERIO.

PREGUNTA: {pregunta}
CRITERIO: {criterio}
RESPUESTA: {respuesta}

Responde UNICAMENTE con un numero entre 0 y 1 con un decimal.
1.0 = cumple el criterio por completo. 0.5 = lo cumple a medias.
0.0 = no lo cumple, o inventa datos que el criterio no respalda."""


def normalizar(texto: str) -> str:
    """Minusculas y sin tildes, para que 'Recuperacion' case con 'recuperación'."""
    sin_tildes = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in sin_tildes if not unicodedata.combining(c))


def texto_de(mensaje) -> str:
    """Gemini 3.x devuelve .content como lista de bloques; .text lo aplana."""
    return mensaje.text if hasattr(mensaje, "text") else str(mensaje.content)


def cargar_casos(ruta: str = RUTA_DATASET) -> list:
    datos = json.load(io.open(ruta, encoding="utf-8"))
    return [
        {
            "id": c["id"],
            "input": {"pregunta": c["pregunta"]},
            "expected_output": {
                "hechos_esperados": c["hechos_esperados"],
                "debe_usar_herramienta": c["debe_usar_herramienta"],
                "criterio": c["criterio"],
            },
            "metadata": {"nota": c["nota"], "modelo": GEMINI_MODEL},
        }
        for c in datos["casos"]
    ]


# ---------------------------------------------------------------- la tarea

def construir_tarea(agente, handler):
    def tarea(*, item, **_):
        pregunta = item["input"]["pregunta"] if isinstance(item, dict) else item.input["pregunta"]
        try:
            resultado = agente.invoke(
                {"messages": [HumanMessage(content=pregunta)]},
                config={"callbacks": [handler]},
            )
        except Exception as error:
            # Un fallo de red o una cuota agotada no deben tumbar la corrida entera:
            # el caso se puntua como fallo y queda registrado el motivo.
            return {"respuesta": f"[ERROR] {type(error).__name__}: {error}",
                    "llamadas_herramienta": 0}
        mensajes = resultado["messages"]
        # Contamos las llamadas a herramienta que el agente decidio hacer.
        llamadas = sum(
            len(m.tool_calls) for m in mensajes
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
        )
        return {"respuesta": texto_de(mensajes[-1]), "llamadas_herramienta": llamadas}

    return tarea


# ---------------------------------------------------------- los evaluadores

def cobertura_de_hechos(*, input, output, expected_output, metadata=None, **_):
    """Determinista: que fraccion de los hechos esperados aparece en la respuesta."""
    esperados = expected_output["hechos_esperados"]
    respuesta = normalizar(output["respuesta"])
    presentes = [h for h in esperados if normalizar(h) in respuesta]
    faltan = [h for h in esperados if h not in presentes]
    return Evaluation(
        name="cobertura_de_hechos",
        value=len(presentes) / len(esperados),
        comment="todos presentes" if not faltan else f"faltan: {', '.join(faltan)}",
    )


def uso_de_herramienta(*, input, output, expected_output, metadata=None, **_):
    """Determinista: ¿busco cuando debia, y se abstuvo cuando no hacia falta?"""
    debia = expected_output["debe_usar_herramienta"]
    uso = output["llamadas_herramienta"] > 0
    acerto = uso == debia
    esperado = "buscar" if debia else "no buscar"
    hizo = f"{output['llamadas_herramienta']} llamada(s)"
    return Evaluation(
        name="uso_de_herramienta",
        value=1.0 if acerto else 0.0,
        comment=f"se esperaba {esperado}; hizo {hizo}",
    )


def construir_juez(llm_juez):
    def juez_llm(*, input, output, expected_output, metadata=None, **_):
        """LLM como juez: lo que ningun keyword matching puede medir."""
        veredicto = llm_juez.invoke(PROMPT_JUEZ.format(
            pregunta=input["pregunta"],
            criterio=expected_output["criterio"],
            respuesta=output["respuesta"],
        ))
        crudo = texto_de(veredicto).strip()
        encontrado = re.search(r"[01](?:[.,]\d+)?", crudo)
        if not encontrado:
            # Sin nota parseable preferimos declararlo que fingir un 0.
            return Evaluation(name="juez_llm", value=0.0,
                              comment=f"veredicto no parseable: {crudo[:80]}")
        nota = min(1.0, float(encontrado.group().replace(",", ".")))
        return Evaluation(name="juez_llm", value=nota, comment=crudo[:120])

    return juez_llm


def tasa_de_aprobados(*, item_results, **_):
    """Evaluador de corrida: un solo numero para comparar entre versiones."""
    notas = [
        e.value for r in item_results for e in r.evaluations
        if e.name == "juez_llm" and isinstance(e.value, (int, float))
    ]
    aprobados = [n for n in notas if n >= 0.7]
    return Evaluation(
        name="tasa_de_aprobados",
        value=len(aprobados) / len(notas) if notas else 0.0,
        comment=f"{len(aprobados)}/{len(notas)} casos con juez_llm >= 0.7",
    )


# ------------------------------------------------------------------- main

def main():
    # --sin-prompt reproduce el agente original (sin instruccion de sistema).
    # Es la variante contra la que comparamos, no una opcion de conveniencia.
    argumentos = [a for a in sys.argv[1:] if not a.startswith("--")]
    con_prompt = "--sin-prompt" not in sys.argv
    variante = "con-prompt" if con_prompt else "sin-prompt"

    nombre_corrida = argumentos[0] if argumentos else (
        f"{GEMINI_MODEL}-{variante}-{datetime.now():%Y%m%d-%H%M%S}")

    llm, handler, langfuse_client = obtener_infraestructura()
    casos = cargar_casos()

    print(f"--- Evaluando {len(casos)} casos dorados ---")
    print(f"    modelo:   {GEMINI_MODEL}")
    print(f"    variante: {variante}")
    print(f"    corrida:  {nombre_corrida}\n")

    resultado = langfuse_client.run_experiment(
        name="agente-wikipedia-dorado",
        run_name=nombre_corrida,
        description="Agente ReAct de Wikipedia contra el dataset dorado de la L4.",
        metadata={"variante": variante, "modelo": GEMINI_MODEL},
        data=casos,
        task=construir_tarea(
            construir_agente(llm, instruccion=INSTRUCCION_SISTEMA if con_prompt else None),
            handler),
        evaluators=[cobertura_de_hechos, uso_de_herramienta, construir_juez(llm)],
        run_evaluators=[tasa_de_aprobados],
        # Secuencial a proposito: la API de Wikimedia devuelve 429 en cuanto varios
        # hilos buscan a la vez, y la cuota gratuita de Gemini tampoco lo agradece.
        max_concurrency=1,
    )

    for r in resultado.item_results:
        notas = {e.name: e.value for e in r.evaluations}
        juez = notas.get("juez_llm", 0)
        marca = "OK  " if juez >= 0.7 else "FALLA"
        ident = r.item["id"] if isinstance(r.item, dict) else r.item.id
        print(f"{marca} {ident:26} juez={juez:.1f}  "
              f"hechos={notas.get('cobertura_de_hechos', 0):.2f}  "
              f"herramienta={notas.get('uso_de_herramienta', 0):.0f}")
        for e in r.evaluations:
            if isinstance(e.value, (int, float)) and e.value < 1.0 and e.comment:
                print(f"        {e.name}: {e.comment}")

    print()
    for e in resultado.run_evaluations:
        print(f"==> {e.name}: {e.value:.2f}  ({e.comment})")

    langfuse_client.flush()
    print(f"\n✅ Corrida '{nombre_corrida}' enviada a Langfuse (seccion Experiments).")
    print("   Compara corridas ahí, o localmente con: mlflow ui "
          "--backend-store-uri sqlite:///mlflow_henry.db")


if __name__ == "__main__":
    main()
