# Henry M3 L4: Observabilidad y Orquestación con Gemini 2.5 Pro

Este repositorio contiene un framework profesional para el desarrollo de aplicaciones de IA Generativa, integrando un stack de observabilidad triple (Langfuse, LangSmith y MLflow) y orquestación avanzada mediante LangChain y LangGraph.

Diseñado originalmente para el programa de **AI Engineering** en Henry, este proyecto sirve como guía técnica para implementar flujos de trabajo robustos y monitoreables.

## 🚀 Arquitectura del Proyecto

El proyecto implementa una separación de responsabilidades clara:

1.  **Capa de Infraestructura (`conexion.py`)**: Centraliza la conexión con el modelo Gemini 2.5 Pro y configura los motores de observabilidad de forma simultánea.
2.  **Capa de Orquestación**:
    * **Consulta Simple (`main.py`)**: Implementación básica de chat con trazabilidad.
    * **Agente Reactivo (`agente.py`)**: Uso de `create_react_agent` con herramientas de búsqueda (Wikipedia, YouTube).
    * **Grafos de Estado (`grafo.py`)**: Orquestación granular mediante LangGraph para flujos no lineales.

## 🛠️ Stack Tecnológico

* **LLM**: Google Gemini 2.5 Pro.
* **Frameworks**: LangChain & LangGraph.
* **Observabilidad**:
    * **Langfuse**: Trazabilidad agnóstica de código abierto.
    * **LangSmith**: Debugging nativo del ecosistema LangChain.
    * **MLflow**: Registro de experimentos y MLOps (Local SQLite).

## 📋 Requisitos Previos

- Python 3.10 o superior.
- Entorno virtual (recomendado).

## ⚙️ Configuración e Instalación

1.  **Clonar el repositorio e instalar dependencias:**

    ```bash
    git clone <tu-repo-url>
    cd <nombre-carpeta>
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate

    pip install -U langchain langchain-google-genai langgraph langfuse mlflow langchainhub wikipedia youtube-search python-dotenv
    ```

2.  **Configurar Variables de Entorno:**
    Renombra el archivo `.env_example` a `.env` y completa tus credenciales.

    ```bash
    cp .env_example .env
    ```

## 📈 Visualización de Trazas

### Langfuse & LangSmith
Las trazas se envían automáticamente a las plataformas en la nube configuradas en el `.env`. Accede a sus respectivos dashboards web para monitorear latencia, costos y prompts.

### MLflow (Local)
Para visualizar los experimentos registrados localmente en MLflow:

1.  Asegúrate de haber ejecutado al menos una vez el código.
2.  Lanza el servidor de la UI:
    ```bash
    mlflow ui --backend-store-uri sqlite:///mlflow_henry.db
    ```
3.  Abre [http://127.0.0.1:5000](http://127.0.0.1:5000) en tu navegador.

## 📁 Estructura de Archivos

- `conexion.py`: Inicialización de clientes y configuración de `autolog` de MLflow.
- `main.py`: Punto de entrada para validación de conectividad.
- `agente.py`: Implementación de agente con capacidad de uso de herramientas.
- `grafo.py`: Ejemplo de flujo controlado por estados.
- `.env`: Archivo sensible con llaves de API (no subir a GitHub).

## 👤 Autor
Proyecto desarrollado por **Iván Miguel Cepeda**.