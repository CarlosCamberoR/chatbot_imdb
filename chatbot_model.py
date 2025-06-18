import os
import logging
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class ChatbotModel:
    """Modelo de chatbot usando Hugging Face Transformers"""
    
    def __init__(self, 
                 model_name: str = None,
                 hf_token: str = None,
                 device: str = None):
        
        self.model_name = model_name or os.getenv("MODEL_NAME", "microsoft/DialoGPT-large")
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        
        # Configurar dispositivo - Priorizar GPU para RTX 4070
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                # Configurar memoria GPU
                torch.cuda.set_per_process_memory_fraction(0.9)  # Usar 90% de VRAM
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Inicializando modelo {self.model_name} en {self.device}")
        if self.device == "cuda":
            logger.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Inicializar tokenizer y modelo
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo y tokenizer optimizado para RTX 4070"""
        try:
            # Configurar argumentos de autenticación
            auth_kwargs = {}
            if self.hf_token:
                auth_kwargs['token'] = self.hf_token
            
            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **auth_kwargs
            )
            
            # Configurar pad_token si no existe
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configuración optimizada para RTX 4070
            model_kwargs = {
                **auth_kwargs,
                'torch_dtype': torch.float16,  # Usar FP16 para ahorrar VRAM
                'low_cpu_mem_usage': True,     # Optimizar uso de RAM
            }
            
            if self.device == "cuda":
                model_kwargs.update({
                    'device_map': "auto",           # Auto-distribución en GPU
                    'offload_folder': "cache/offload"  # Offload a disco si es necesario
                })
            
            # Cargar modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Crear pipeline optimizado
            pipeline_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            # No especificar device cuando se usa device_map="auto"
            if self.device == "cpu":
                pipeline_kwargs["device"] = -1
            # Para CUDA con device_map, no especificar device explícitamente
            
            self.pipeline = pipeline(
                "text-generation",
                **pipeline_kwargs
            )
            
            logger.info("Modelo cargado exitosamente con optimizaciones para RTX 4070")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            # Fallback a un modelo más simple
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Carga un modelo de respaldo optimizado"""
        try:
            logger.info("Cargando modelo de respaldo optimizado: microsoft/DialoGPT-medium")
            
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            model_kwargs = {
                'torch_dtype': torch.float16 if self.device == "cuda" else torch.float32,
                'low_cpu_mem_usage': True
            }
            
            if self.device == "cuda":
                model_kwargs['device_map'] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/DialoGPT-medium",
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Crear pipeline de respaldo
            pipeline_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32
            }
            
            # Solo especificar device para CPU
            if self.device == "cpu":
                pipeline_kwargs["device"] = -1
            
            self.pipeline = pipeline(
                "text-generation",
                **pipeline_kwargs
            )
            
            logger.info("Modelo de respaldo cargado exitosamente")
            
        except Exception as e:
            logger.error(f"Error cargando modelo de respaldo: {e}")
            raise
    
    def is_imdb_related(self, query: str) -> bool:
        """Verifica si la consulta está relacionada con IMDB/películas"""
        imdb_keywords = [
            'película', 'filme', 'movie', 'film', 'cinema', 'actor', 'actriz',
            'director', 'rating', 'imdb', 'serie', 'tv', 'show', 'drama',
            'comedia', 'acción', 'terror', 'romance', 'thriller', 'aventura',
            'ciencia ficción', 'fantasía', 'animación', 'documental',
            'reparto', 'cast', 'protagonista', 'género', 'año', 'estreno',
            'oscar', 'premio', 'nominación', 'crítica', 'reseña'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in imdb_keywords)
    
    def generate_response(self, 
                         query: str, 
                         context: List[str], 
                         max_length: int = 500,
                         temperature: float = 0.7) -> str:
        """Genera una respuesta usando el contexto recuperado"""
        
        # Verificar si la consulta está relacionada con IMDB
        if not self.is_imdb_related(query):
            return ("Lo siento, solo puedo responder preguntas relacionadas con películas, "
                   "series y la base de datos de IMDB. ¿Tienes alguna pregunta sobre cine?")
        
        if not self.pipeline:
            return "Error: El modelo no está disponible."
        
        try:
            # Crear el prompt con contexto
            context_text = "\n".join(context[:3])  # Usar solo los 3 primeros contextos
            
            prompt = f"""Contexto sobre películas de IMDB:
{context_text}

Pregunta del usuario: {query}

Respuesta basada en el contexto anterior:"""
            
            # Generar respuesta
            outputs = self.pipeline(
                prompt,
                max_length=len(prompt.split()) + max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extraer solo la parte generada
            generated_text = outputs[0]['generated_text']
            response = generated_text[len(prompt):].strip()
            
            # Limpiar la respuesta
            response = self._clean_response(response)
            
            return response if response else "No pude generar una respuesta apropiada."
            
        except Exception as e:
            logger.error(f"Error generando respuesta: {e}")
            return self._generate_simple_response(query, context)
    
    def _clean_response(self, response: str) -> str:
        """Limpia y mejora la respuesta generada"""
        # Eliminar texto repetitivo o no relevante
        lines = response.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Pregunta:') and not line.startswith('Contexto:'):
                clean_lines.append(line)
        
        return '\n'.join(clean_lines[:5])  # Máximo 5 líneas
    
    def _generate_simple_response(self, query: str, context: List[str]) -> str:
        """Genera una respuesta simple cuando falla el modelo principal"""
        if not context:
            return "No encontré información relevante en la base de datos de IMDB."
        
        # Respuesta basada en el contexto más relevante
        best_context = context[0] if context else ""
        
        if "título:" in best_context.lower():
            return f"Basándome en la información de IMDB: {best_context}"
        else:
            return f"Según la base de datos de IMDB: {best_context}"
