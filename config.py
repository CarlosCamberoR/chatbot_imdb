"""
Configuración de desarrollo para el Chatbot IMDB
"""

import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de desarrollo
DEV_CONFIG = {
    # Configuración de datos limitada para desarrollo
    "MAX_MOVIES_DEV": 1000,  # Menos películas para desarrollo rápido
    "CACHE_ENABLED": True,
    "DEBUG_MODE": True,
    
    # Modelos más ligeros para desarrollo
    "DEV_MODEL_NAME": "distilgpt2",
    "DEV_EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    
    # Configuración de respuesta para desarrollo
    "DEV_MAX_LENGTH": 200,
    "DEV_TEMPERATURE": 0.8,
    "DEV_TOP_K": 3,
    
    # Configuración de streamlit
    "STREAMLIT_PORT": 8501,
    "STREAMLIT_HOST": "localhost"
}

# Configuración de producción
PROD_CONFIG = {
    "MAX_MOVIES_PROD": 10000,
    "CACHE_ENABLED": True,
    "DEBUG_MODE": False,
    
    "PROD_MODEL_NAME": "microsoft/DialoGPT-medium",
    "PROD_EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    
    "PROD_MAX_LENGTH": 500,
    "PROD_TEMPERATURE": 0.7,
    "PROD_TOP_K": 5,
    
    "STREAMLIT_PORT": 8501,
    "STREAMLIT_HOST": "0.0.0.0"
}

def get_config(mode="dev"):
    """Obtener configuración según el modo"""
    if mode == "prod":
        return PROD_CONFIG
    return DEV_CONFIG

def setup_dev_environment():
    """Configurar entorno de desarrollo"""
    config = get_config("dev")
    
    # Establecer variables de entorno para desarrollo
    os.environ["MODEL_NAME"] = config["DEV_MODEL_NAME"]
    os.environ["EMBEDDING_MODEL"] = config["DEV_EMBEDDING_MODEL"]
    os.environ["MAX_RESPONSE_LENGTH"] = str(config["DEV_MAX_LENGTH"])
    os.environ["TEMPERATURE"] = str(config["DEV_TEMPERATURE"])
    os.environ["TOP_K"] = str(config["DEV_TOP_K"])
    
    print("🔧 Entorno de desarrollo configurado")
    print(f"📊 Modelo: {config['DEV_MODEL_NAME']}")
    print(f"🔍 Embeddings: {config['DEV_EMBEDDING_MODEL']}")
    print(f"🎬 Máximo películas: {config['MAX_MOVIES_DEV']}")

if __name__ == "__main__":
    setup_dev_environment()
