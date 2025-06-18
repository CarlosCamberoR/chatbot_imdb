import streamlit as st
import os
import time
from rag_system import RAGChatbot
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la página
st.set_page_config(
    page_title="Chatbot IMDB",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: right;
    }
    .bot-message {
        background-color: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .source-box {
        background-color: #e9ecef;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        font-size: 0.9em;
    }
    .stats-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_chatbot():
    """Inicializa el chatbot optimizado para RTX 4070 (solo una vez)"""
    chatbot = RAGChatbot()
    # Usar configuración optimizada desde .env
    max_movies = int(os.getenv("MAX_MOVIES", "25000"))
    success = chatbot.initialize(max_movies=max_movies, use_cache=True)
    return chatbot, success

def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">
            🎬 Chatbot IMDB - Asistente de Películas
        </h1>
        <p style="color: white; text-align: center; margin: 0;">
            Pregúntame cualquier cosa sobre películas, series y la base de datos de IMDB
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración del Sistema")
        
        # Información del hardware
        st.subheader("�️ Hardware Detectado")
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                st.success(f"🚀 GPU: {gpu_name}")
                st.info(f"💾 VRAM: {gpu_memory:.1f} GB")
            else:
                st.warning("⚠️ GPU no detectada, usando CPU")
        except:
            st.warning("⚠️ No se pudo detectar hardware")
        
        st.divider()
        
        # Configuración del modelo
        st.subheader("🤖 Configuración del Modelo")
        temperature = st.slider("Temperatura", 0.1, 1.0, 0.7, 0.1)
        max_length = st.slider("Longitud máxima de respuesta", 200, 1200, 800, 50)
        top_k = st.slider("Número de documentos a recuperar", 3, 15, 8)
        
        # Actualizar variables de entorno
        os.environ["TEMPERATURE"] = str(temperature)
        os.environ["MAX_RESPONSE_LENGTH"] = str(max_length)
        os.environ["TOP_K"] = str(top_k)
        
        st.divider()
        
        # Información del sistema
        st.subheader("📊 Estado del Sistema")
        if st.button("🔍 Mostrar Información GPU"):
            try:
                import torch
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(0) / 1e9
                    memory_cached = torch.cuda.memory_reserved(0) / 1e9
                    st.metric("VRAM Usada", f"{memory_allocated:.1f} GB")
                    st.metric("VRAM Reservada", f"{memory_cached:.1f} GB")
                else:
                    st.info("GPU no disponible")
            except:
                st.error("Error obteniendo información de GPU")
        
        # Botón para reinicializar
        if st.button("🔄 Reinicializar Sistema"):
            st.cache_resource.clear()
            st.rerun()
    
    # Inicializar chatbot
    with st.spinner("Inicializando sistema RAG con OpenHermes-2.5-Mistral-7B... Esto puede tomar unos minutos la primera vez."):
        chatbot, success = initialize_chatbot()
    
    if not success:
        st.error("❌ Error inicializando el sistema. Por favor, verifica tu configuración.")
        return
    
    # Mostrar estadísticas del sistema
    stats = chatbot.get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "✅ Optimizado" if stats["sistema_inicializado"] else "❌ Error"
        st.metric("Estado del Sistema", status)
    
    with col2:
        docs_count = f"{stats['documentos_en_base']:,}"
        st.metric("Documentos en Base", docs_count)
    
    with col3:
        modelo = "OpenHermes-2.5-Mistral-7B"
        if stats["modelo_activo"]:
            modelo_completo = stats["modelo_activo"]
            if "OpenHermes" in modelo_completo:
                modelo = "OpenHermes-2.5-Mistral-7B"
            else:
                modelo = modelo_completo.split("/")[-1]
        st.metric("Modelo", modelo)
    
    with col4:
        kb_status = "✅ OpenHermes Optimizada" if stats["base_conocimiento_lista"] else "❌ Error"
        st.metric("Base de Conocimiento", kb_status)
    
    st.divider()
    
    # Interfaz principal de chat
    st.header("💬 Chat con el Asistente")
    
    # Inicializar historial de chat en session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "¡Hola! Soy tu asistente de IMDB potenciado por OpenHermes-2.5-Mistral-7B. Puedo ayudarte con información detallada sobre películas, series, actores, directores y cualquier tema relacionado con el cine. ¿En qué puedo ayudarte hoy?"
        })
    
    # Mostrar historial de chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Mostrar fuentes si las hay
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📚 Fuentes consultadas"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Fuente {i}:**")
                            st.markdown(f"```\n{source}\n```")
    
    # Input del usuario
    if prompt := st.chat_input("Escribe tu pregunta sobre películas o IMDB..."):
        # Añadir mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generar respuesta del asistente
        with st.chat_message("assistant"):
            with st.spinner("Buscando información y generando respuesta..."):
                # Consultar al chatbot
                result = chatbot.query(prompt)
                
                # Mostrar respuesta
                st.markdown(result["response"])
                
                # Mostrar fuentes si las hay
                if result["sources"]:
                    with st.expander("📚 Fuentes consultadas"):
                        for i, source in enumerate(result["sources"], 1):
                            st.markdown(f"**Fuente {i}:**")
                            st.markdown(f"```\n{source}\n```")
                
                # Añadir respuesta al historial
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "sources": result["sources"]
                })
    
    # Sección de herramientas adicionales
    st.divider()
    st.header("🔧 Herramientas Adicionales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Búsqueda de Películas")
        search_query = st.text_input("Buscar película por título:")
        
        if search_query:
            with st.spinner("Buscando películas..."):
                movies = chatbot.search_movies(search_query, limit=5)
            
            if movies:
                for movie in movies:
                    with st.expander(f"🎬 {movie.get('primaryTitle', 'Título desconocido')} ({movie.get('startYear', 'N/A')})"):
                        st.write(f"**Título original:** {movie.get('originalTitle', 'N/A')}")
                        st.write(f"**Año:** {movie.get('startYear', 'N/A')}")
                        st.write(f"**Géneros:** {movie.get('genres', 'N/A')}")
                        st.write(f"**Duración:** {movie.get('runtimeMinutes', 'N/A')} minutos")
                        st.write(f"**ID IMDB:** {movie.get('tconst', 'N/A')}")
            else:
                st.warning("No se encontraron películas con ese título.")
    
    with col2:
        st.subheader("📊 Información del Sistema")
        with st.container():
            st.markdown("""
            <div class="stats-box">
                <h4>🤖 Sistema OpenHermes-2.5-Mistral-7B:</h4>
                <ul>
                    <li><strong>Modelo:</strong> OpenHermes-2.5-Mistral-7B (conversacional avanzado)</li>
                    <li><strong>Embeddings:</strong> MPNet-v2 (mejor comprensión semántica)</li>
                    <li><strong>Base de datos:</strong> 25,000+ películas de IMDB</li>
                    <li><strong>GPU:</strong> Cuantización 4-bit optimizada para RTX 4070</li>
                    <li><strong>Respuestas:</strong> Hasta 1200 tokens conversacionales</li>
                    <li><strong>Retrieval:</strong> 8 documentos por consulta</li>
                </ul>
                <h4>💡 Consejos de uso:</h4>
                <ul>
                    <li>Pregunta sobre películas específicas, actores o directores</li>
                    <li>Puedes preguntar sobre géneros, años o ratings</li>
                    <li>El sistema funciona mejor con preguntas específicas</li>
                    <li>Usa la búsqueda de películas para encontrar títulos exactos</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        🎬 Chatbot IMDB - Potenciado por OpenHermes-2.5-Mistral-7B<br>
        🚀 OpenHermes + MPNet-v2 | 25K+ películas | Cuantización 4-bit<br>
        ⚡ Powered by Hugging Face, FAISS y Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
