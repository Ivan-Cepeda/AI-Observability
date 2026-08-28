import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler 
import mlflow # 1. Importamos MLflow

load_dotenv()

# La consola de Windows usa cp1252 y revienta con los emojis de los prints.
# Reconfiguramos la salida a UTF-8 una sola vez, aqui, para los tres scripts.
for _flujo in (sys.stdout, sys.stderr):
    try:
        _flujo.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Modelo Gemini desde variables de entorno (Zero-Hardcoding)
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")

# ==========================================
# CONFIGURACIÓN DE MLFLOW (MLOps Estándar)
# ==========================================
# 2. Configuramos una base de datos local para guardar los experimentos
mlflow.set_tracking_uri("sqlite:///mlflow_henry.db")

# 3. Nombramos el experimento
mlflow.set_experiment("Agentes_Gemini_Henry")

# 4. LA MAGIA: Activamos el autologging para LangChain
# Esto intercepta todo automáticamente, igual que LangSmith
mlflow.langchain.autolog()
# ==========================================

def obtener_infraestructura():
    # El LLM (Gemini 3.7 Flash por defecto; se cambia con GEMINI_MODEL en el .env)
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )
    
    # Cliente de Langfuse (Rastreo Explícito)
    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        # El SDK llama a esta variable LANGFUSE_HOST; el .env de la leccion,
        # LANGFUSE_BASE_URL. Aceptamos las dos para no depender del nombre.
        host=os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")
    )
    handler = CallbackHandler()
    
    # LangSmith funciona de fondo por las variables del .env
    # MLflow funciona de fondo gracias al autolog()
    
    return llm, handler, langfuse_client