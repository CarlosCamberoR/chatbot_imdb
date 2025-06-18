#!/bin/bash

# Script de instalación y configuración del Chatbot IMDB

echo "🎬 Configurando Chatbot IMDB..."

# Verificar si conda está instalado
if ! command -v conda &> /dev/null; then
    echo "❌ Conda no está instalado. Por favor instala Anaconda o Miniconda primero."
    exit 1
fi

echo "✅ Conda encontrado"

# Crear entorno conda
echo "📦 Creando entorno conda..."
conda env create -f environment.yml

# Activar entorno
echo "🔧 Activando entorno..."
eval "$(conda shell.bash hook)"
conda activate chatbot_imdb

# Verificar instalación
echo "🧪 Verificando instalación..."
python -c "import streamlit, transformers, torch, sentence_transformers, faiss; print('✅ Todas las dependencias instaladas correctamente')"

# Crear directorios necesarios
echo "📁 Creando directorios..."
mkdir -p data
mkdir -p cache

echo "🎉 ¡Instalación completada!"
echo ""
echo "Para usar el chatbot:"
echo "1. Activa el entorno: conda activate chatbot_imdb"
echo "2. Configura tu token en .env: HUGGINGFACE_TOKEN=tu_token"
echo "3. Ejecuta la app: streamlit run app.py"
echo ""
echo "🚀 ¡Disfruta tu chatbot de IMDB!"
