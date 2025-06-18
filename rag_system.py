import os
import logging
from typing import List, Dict, Any
from imdb_loader import IMDBDataLoader
from retriever import MixedRetriever
from chatbot_model import ChatbotModel
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChatbot:
    """Sistema RAG completo para consultas sobre IMDB"""
    
    def __init__(self):
        load_dotenv()
        
        # Configuración desde variables de entorno
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.model_name = os.getenv("MODEL_NAME", "microsoft/DialoGPT-medium")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.max_response_length = int(os.getenv("MAX_RESPONSE_LENGTH", "500"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.top_k = int(os.getenv("TOP_K", "5"))
        
        # Inicializar componentes
        self.imdb_loader = None
        self.retriever = None
        self.chatbot_model = None
        self.knowledge_base_ready = False
        
        logger.info("Inicializando sistema RAG para IMDB...")
    
    def initialize(self, max_movies: int = None, use_cache: bool = True):
        """Inicializa todos los componentes del sistema con configuración optimizada"""
        try:
            # Usar configuración del .env para max_movies
            if max_movies is None:
                max_movies = int(os.getenv("MAX_MOVIES", "25000"))
            
            # 1. Inicializar cargador de IMDB
            logger.info("Inicializando cargador de IMDB...")
            self.imdb_loader = IMDBDataLoader()
            
            # 2. Inicializar retriever
            logger.info("Inicializando retriever...")
            self.retriever = MixedRetriever(embedding_model_name=self.embedding_model)
            
            # 3. Cargar o crear base de conocimiento
            if use_cache and self.retriever.load_cache():
                logger.info("Base de conocimiento cargada desde cache")
                self.knowledge_base_ready = True
            else:
                logger.info(f"Creando base de conocimiento con {max_movies} películas...")
                knowledge_texts = self.imdb_loader.create_knowledge_base(max_movies)
                
                if knowledge_texts:
                    self.retriever.add_documents(knowledge_texts)
                    self.knowledge_base_ready = True
                    logger.info(f"Base de conocimiento creada con {len(knowledge_texts)} documentos")
                else:
                    logger.error("No se pudo crear la base de conocimiento")
            
            # 4. Inicializar modelo de chatbot
            logger.info("Inicializando modelo de chatbot optimizado...")
            self.chatbot_model = ChatbotModel(
                model_name=self.model_name,
                hf_token=self.hf_token
            )
            
            logger.info("Sistema RAG inicializado exitosamente con optimizaciones para RTX 4070")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando sistema RAG: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Verifica si el sistema está listo para uso"""
        return (self.knowledge_base_ready and 
                self.retriever is not None and 
                self.chatbot_model is not None)
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """Procesa una consulta del usuario y devuelve la respuesta"""
        if not self.is_ready():
            return {
                "response": "El sistema no está completamente inicializado. Por favor, espera un momento.",
                "sources": [],
                "error": "Sistema no inicializado"
            }
        
        try:
            logger.info(f"Procesando consulta: {user_query}")
            
            # 1. Verificar si la consulta está relacionada con IMDB
            if not self.chatbot_model.is_imdb_related(user_query):
                return {
                    "response": ("Lo siento, solo puedo responder preguntas relacionadas con películas, "
                               "series y la base de datos de IMDB. ¿Tienes alguna pregunta sobre cine?"),
                    "sources": [],
                    "error": None
                }
            
            # 2. Recuperar documentos relevantes
            relevant_docs = self.retriever.retrieve(
                user_query, 
                k=self.top_k, 
                method="hybrid"
            )
            
            if not relevant_docs:
                return {
                    "response": "No encontré información relevante en la base de datos de IMDB para tu consulta.",
                    "sources": [],
                    "error": None
                }
            
            # 3. Generar respuesta usando el modelo
            response = self.chatbot_model.generate_response(
                query=user_query,
                context=relevant_docs,
                max_length=self.max_response_length,
                temperature=self.temperature
            )
            
            return {
                "response": response,
                "sources": relevant_docs[:3],  # Mostrar solo las 3 fuentes más relevantes
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error procesando consulta: {e}")
            return {
                "response": "Lo siento, ocurrió un error procesando tu consulta. Por favor, inténtalo de nuevo.",
                "sources": [],
                "error": str(e)
            }
    
    def search_movies(self, title: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Busca películas por título"""
        if not self.imdb_loader:
            return []
        
        try:
            return self.imdb_loader.search_movies(title, limit)
        except Exception as e:
            logger.error(f"Error buscando películas: {e}")
            return []
    
    def get_movie_info(self, tconst: str) -> Dict[str, Any]:
        """Obtiene información completa de una película"""
        if not self.imdb_loader:
            return {}
        
        try:
            return self.imdb_loader.get_movie_info(tconst)
        except Exception as e:
            logger.error(f"Error obteniendo información de película: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema"""
        stats = {
            "sistema_inicializado": self.is_ready(),
            "base_conocimiento_lista": self.knowledge_base_ready,
            "documentos_en_base": len(self.retriever.documents) if self.retriever else 0,
            "modelo_activo": self.model_name if self.chatbot_model else None
        }
        return stats
