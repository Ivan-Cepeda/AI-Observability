import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler 
import mlflow # 1. Importamos MLflow

load_dotenv()

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
    # El LLM (Gemini 2.5 Pro)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )
    
    # Cliente de Langfuse (Rastreo Explícito)
    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST")
    )
    handler = CallbackHandler()
    
    # LangSmith funciona de fondo por las variables del .env
    # MLflow funciona de fondo gracias al autolog()
    
    return llm, handler, langfuse_client