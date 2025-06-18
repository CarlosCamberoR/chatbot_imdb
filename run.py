#!/usr/bin/env python3
"""
Script de inicio rápido para el Chatbot IMDB
"""

import os
import sys
import subprocess
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_conda():
    """Verificar si conda está disponible"""
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_environment():
    """Verificar si el entorno existe"""
    try:
        result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
        return 'chatbot_imdb' in result.stdout
    except:
        return False

def create_environment():
    """Crear entorno conda"""
    logger.info("Creando entorno conda...")
    try:
        subprocess.run(['conda', 'env', 'create', '-f', 'environment.yml'], check=True)
        logger.info("✅ Entorno creado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error creando entorno: {e}")
        return False

def install_dependencies():
    """Instalar dependencias adicionales si es necesario"""
    logger.info("Verificando dependencias...")
    try:
        # Activar entorno e instalar dependencias
        subprocess.run([
            'conda', 'run', '-n', 'chatbot_imdb', 
            'pip', 'install', '-r', 'requirements.txt'
        ], check=True)
        logger.info("✅ Dependencias verificadas")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error instalando dependencias: {e}")
        return False

def run_app():
    """Ejecutar la aplicación Streamlit"""
    logger.info("🚀 Iniciando aplicación Streamlit...")
    try:
        subprocess.run([
            'conda', 'run', '-n', 'chatbot_imdb',
            'streamlit', 'run', 'app.py'
        ])
    except KeyboardInterrupt:
        logger.info("Aplicación detenida por el usuario")
    except Exception as e:
        logger.error(f"Error ejecutando aplicación: {e}")

def main():
    """Función principal"""
    print("🎬 Chatbot IMDB - Sistema RAG")
    print("=" * 50)
    
    # Verificar conda
    if not check_conda():
        logger.error("❌ Conda no está instalado. Por favor instala Anaconda o Miniconda.")
        sys.exit(1)
    
    logger.info("✅ Conda encontrado")
    
    # Verificar/crear entorno
    if not check_environment():
        logger.info("Entorno no encontrado, creando...")
        if not create_environment():
            sys.exit(1)
    else:
        logger.info("✅ Entorno encontrado")
    
    # Instalar dependencias
    if not install_dependencies():
        logger.warning("⚠️ Algunas dependencias podrían fallar, continuando...")
    
    # Crear directorios necesarios
    os.makedirs('data', exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    logger.info("✅ Directorios creados")
    
    # Verificar archivo .env
    if not os.path.exists('.env'):
        logger.warning("⚠️ Archivo .env no encontrado. Crea uno con tu HUGGINGFACE_TOKEN")
    
    print("\n" + "=" * 50)
    print("🎉 Sistema listo!")
    print("📝 Asegúrate de configurar tu HUGGINGFACE_TOKEN en .env")
    print("🌐 La aplicación se abrirá en http://localhost:8501")
    print("=" * 50 + "\n")
    
    # Ejecutar aplicación
    run_app()

if __name__ == "__main__":
    main()
