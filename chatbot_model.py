import os
import logging
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
import torch
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class ChatbotModel:
    """Modelo de chatbot optimizado para RAG usando modelos generativos potentes"""
    
    def __init__(self, 
                 model_name: str = None,
                 hf_token: str = None,
                 device: str = None):
        
        self.model_name = model_name or os.getenv("MODEL_NAME", "teknium/OpenHermes-2.5-Mistral-7B")
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        
        # Detectar tipo de modelo
        self.is_seq2seq = any(model_type in self.model_name.lower() 
                             for model_type in ['t5', 'bart', 'pegasus', 'flan'])
        
        # Detectar si es un modelo instruccional moderno (como Mistral, Hermes)
        self.is_instruct_model = any(keyword in self.model_name.lower() 
                                   for keyword in ['instruct', 'chat', 'mistral', 'llama', 'hermes', 'nous'])
        
        # Configurar dispositivo - Priorizar GPU para RTX 4070
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                # Configurar memoria GPU
                torch.cuda.set_per_process_memory_fraction(0.85)  # Usar 85% de VRAM para modelos más grandes
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Inicializando modelo {self.model_name} ({'Seq2Seq' if self.is_seq2seq else 'Causal'}) en {self.device}")
        if self.device == "cuda":
            logger.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Inicializar tokenizer y modelo
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo y tokenizer optimizado para generación tipo RAG con soporte para cuantización"""
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
            
            # Configuración optimizada para RTX 4070 con soporte para cuantización
            model_kwargs = {
                **auth_kwargs,
                'low_cpu_mem_usage': True,     # Optimizar uso de RAM
            }
            
            # Configurar cuantización 4-bit si está habilitada
            load_in_4bit = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"
            quantization = os.getenv("QUANTIZATION", "none")
            low_cpu_mem_usage = os.getenv("LOW_CPU_MEM_USAGE", "true").lower() == "true"
            device_map = os.getenv("DEVICE_MAP", "auto")
            
            if load_in_4bit and quantization == "4bit":
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs['quantization_config'] = quantization_config
                model_kwargs['low_cpu_mem_usage'] = low_cpu_mem_usage
                logger.info("Usando cuantización 4-bit para optimización de memoria")
            else:
                model_kwargs['torch_dtype'] = torch.float16  # Usar FP16 sin cuantización
            
            if self.device == "cuda":
                model_kwargs.update({
                    'device_map': device_map,           # Auto-distribución en GPU
                    'low_cpu_mem_usage': low_cpu_mem_usage,
                    'offload_folder': "cache/offload"  # Offload a disco si es necesario
                })
            
            # Cargar modelo según tipo
            if self.is_seq2seq:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                task_type = "text2text-generation"
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                task_type = "text-generation"
            
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
            
            self.pipeline = pipeline(
                task_type,
                **pipeline_kwargs
            )
            
            logger.info(f"Modelo {task_type} cargado exitosamente con optimizaciones para RTX 4070")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            # Fallback a un modelo más simple
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Carga un modelo de respaldo optimizado para RAG"""
        try:
            logger.info("Cargando modelo de respaldo optimizado: HuggingFaceH4/zephyr-7b-beta")
            
            self.model_name = "HuggingFaceH4/zephyr-7b-beta"  # Actualizar nombre
            self.is_seq2seq = False  # Zephyr es causal
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            model_kwargs = {
                'torch_dtype': torch.float16 if self.device == "cuda" else torch.float32,
                'low_cpu_mem_usage': True
            }
            
            if self.device == "cuda":
                model_kwargs['device_map'] = "auto"
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
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
                "text2text-generation",
                **pipeline_kwargs
            )
            
            logger.info("Modelo de respaldo Zephyr cargado exitosamente")
            
        except Exception as e:
            logger.error(f"Error cargando modelo de respaldo: {e}")
            # Último fallback a GPT-2
            self._load_minimal_fallback()
    
    def _load_minimal_fallback(self):
        """Último fallback a GPT-2"""
        try:
            logger.info("Cargando fallback mínimo: distilgpt2")
            
            self.model_name = "distilgpt2"
            self.is_seq2seq = False
            
            self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            model_kwargs = {
                'torch_dtype': torch.float16 if self.device == "cuda" else torch.float32,
                'low_cpu_mem_usage': True
            }
            
            if self.device == "cuda":
                model_kwargs['device_map'] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                "distilgpt2",
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            pipeline_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32
            }
            
            if self.device == "cpu":
                pipeline_kwargs["device"] = -1
            
            self.pipeline = pipeline(
                "text-generation",
                **pipeline_kwargs
            )
            
            logger.info("Fallback mínimo cargado exitosamente")
            
        except Exception as e:
            logger.error(f"Error en fallback mínimo: {e}")
            raise
    
    def is_imdb_related(self, query: str) -> bool:
        """Siempre devuelve True - el LLM decidirá la relevancia basándose en el contexto"""
        return True
    
    def _is_cinema_related_query(self, query: str) -> bool:
        """Detecta si una consulta está relacionada con cine/entretenimiento"""
        cinema_keywords = [
            # Español
            'película', 'serie', 'actor', 'actriz', 'director', 'directora', 'cine', 'film', 
            'drama', 'comedia', 'terror', 'acción', 'romance', 'thriller', 'documental',
            'oscar', 'premio', 'rating', 'imdb', 'netflix', 'estreno', 'taquilla',
            'protagonista', 'reparto', 'personaje', 'trama', 'argumento', 'género',
            # Inglés
            'movie', 'film', 'series', 'show', 'actor', 'actress', 'director', 'cinema',
            'drama', 'comedy', 'horror', 'action', 'romance', 'thriller', 'documentary',
            'oscar', 'award', 'rating', 'cast', 'character', 'plot', 'genre'
        ]
        
        non_cinema_keywords = [
            # Deportes
            'futbol', 'fútbol', 'football', 'soccer', 'messi', 'cristiano', 'ronaldo',
            'barcelona', 'madrid', 'liga', 'champions', 'mundial', 'gol', 'partido',
            # Otros temas
            'política', 'politics', 'economía', 'economy', 'ciencia', 'science',
            'medicina', 'medicine', 'tecnología', 'technology', 'deportes', 'sports'
        ]
        
        query_lower = query.lower()
        
        # Si contiene palabras claramente no relacionadas con cine, devolver False
        for keyword in non_cinema_keywords:
            if keyword in query_lower:
                return False
        
        # Si contiene palabras de cine, devolver True
        for keyword in cinema_keywords:
            if keyword in query_lower:
                return True
        
        # Por defecto, asumir que sí es de cine (para mantener funcionalidad)
        return True

    def _create_robust_prompt(self, query: str, context: List[str]) -> str:
        """Crea un prompt robusto y completo para modelos generativos"""
        
        # Verificar si la consulta es sobre cine
        if not self._is_cinema_related_query(query):
            # Respuesta clara para consultas no cinematográficas
            if self.is_instruct_model:
                return f"""<s>[INST] {query} [/INST]

Soy un asistente especializado exclusivamente en información cinematográfica de IMDB. Tu pregunta parece estar relacionada con un tema diferente al cine, series, actores o directores.

Mi especialidad es proporcionar información detallada sobre:
- 🎬 Películas y series (títulos, años, géneros, sinopsis, reparto)
- 👥 Actores y actrices (filmografía, biografías, premios)
- 🎭 Directores (obras, estilo, reconocimientos)
- ⭐ Calificaciones y reseñas de IMDB
- 🎯 Recomendaciones personalizadas

¿Hay alguna película, serie, actor o director específico sobre el que te gustaría saber más?</s>"""
            elif self.is_seq2seq:
                return f"""Pregunta: {query}

Respuesta: Soy un asistente especializado exclusivamente en información cinematográfica de IMDB. Tu pregunta parece estar relacionada con un tema diferente al cine, series, actores o directores. 

Puedo ayudarte con:
- Información sobre películas y series
- Datos de actores y directores  
- Calificaciones y reseñas de IMDB
- Recomendaciones cinematográficas
- Géneros y años de estreno

¿Hay alguna película o serie específica sobre la que te gustaría saber más?"""
            else:
                return f"""Pregunta: {query}

Soy un experto en cine y no puedo responder sobre otros temas. ¿Hay alguna película que te interese?"""
        
        # Combinar y optimizar contexto 
        context_text = ""
        if context:
            # Tomar los 3 contextos más relevantes para más información
            selected_contexts = context[:3]
            context_parts = []
            for i, ctx in enumerate(selected_contexts, 1):
                # Aumentar límite para más detalles en modelos más capaces
                limited_ctx = ctx[:400] + "..." if len(ctx) > 400 else ctx
                context_parts.append(f"Fuente {i}: {limited_ctx}")
            context_text = "\n\n".join(context_parts)
        
        # Prompts específicos para diferentes tipos de modelo
        if self.is_instruct_model:
            # Prompt optimizado para OpenHermes-2.5-Mistral-7B (muy conversacional y entusiasta)
            if context_text:
                prompt = f"""<s>[INST] Eres un experto en cine con acceso a información de IMDB. IMPORTANTE: Solo usa la información que te proporciono, no inventes datos.

Información real de IMDB:
{context_text}

Pregunta: {query}

INSTRUCCIONES:
- USA SOLO los datos reales que aparecen en la información de IMDB
- Si no tienes el nombre del director o actor, NO lo inventes, simplemente no lo menciones
- NO uses placeholders como [Director] o [Actor Principal]
- Sé específico con años, duraciones y calificaciones reales
- Si falta información, di "no tengo esa información específica"
- Mantén un tono entusiasta pero basado en hechos reales

[/INST]"""
            else:
                prompt = f"""<s>[INST] Eres un cinéfilo apasionado con acceso a IMDB. 

Pregunta: {query}

No tengo información específica sobre esa consulta en mi base de datos, pero como cinéfilo entusiasta puedo ayudarte con:

- Análisis detallados de películas y series
- Biografías completas de actores y directores  
- Recomendaciones personalizadas por género, década o tema
- Curiosidades y trivia cinematográfica
- Comparaciones entre películas similares
- Historia del cine y evolución de géneros

¿Qué te parece si me das más detalles sobre lo que buscas? ¿Un género específico, una época, un director favorito? ¡Vamos a encontrar algo perfecto para ti! [/INST]"""
        
        return prompt
    
    def generate_response(self, 
                         query: str, 
                         context: List[str], 
                         max_length: int = 800,
                         temperature: float = 0.7) -> str:
        """Genera una respuesta usando el contexto recuperado con prompt robusto"""
        
        # Filtro temprano: si no es sobre cine, cortar inmediatamente
        if not self._is_cinema_related_query(query):
            return (
                "🎬 Lo siento, soy un asistente especializado exclusivamente en temas cinematográficos. "
                "Puedo ayudarte con películas, series, actores, directores, géneros y todo lo que esté en IMDB. "
                "Si tienes una pregunta sobre cine, estaré encantado de ayudarte. 🎞️"
            )
        
        # Si no tenemos modelo disponible, usar respuesta simple
        if not self.pipeline:
            return self._generate_simple_response(query, context)
        
        # Para consultas sin contexto relevante
        if not context or len(context) == 0:
            return "No encontré información específica sobre eso en mi base de datos de películas y series de IMDB. ¿Hay alguna película, serie, actor o director específico sobre el que te gustaría saber más?"
        
        # Intentar generar con el modelo
        try:
            # Crear prompt robusto
            prompt = self._create_robust_prompt(query, context)
            
            # Configurar parámetros según tipo de modelo
            if self.is_instruct_model:
                # Parámetros optimizados para OpenHermes-2.5-Mistral-7B (muy charlatán y detallado)
                generation_kwargs = {
                    "max_new_tokens": 600,  # Aumentado para aprovechar naturaleza charlatana
                    "min_new_tokens": 100,  # Mínimo alto para respuestas extensas
                    "temperature": 0.7,     # Más creativo para conversación natural
                    "do_sample": True,
                    "top_p": 0.95,          # Alta diversidad para naturalidad
                    "top_k": 60,            # Vocabulario amplio
                    "repetition_penalty": 1.1,  # Bajo para mantener fluidez natural
                    "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "return_full_text": False,
                    "num_return_sequences": 1
                }
            elif self.is_seq2seq:
                # Parámetros optimizados para respuestas más completas y detalladas
                generation_kwargs = {
                    "max_new_tokens": 500,  # Aumentado significativamente para respuestas extensas
                    "min_new_tokens": 80,   # Mínimo más alto para evitar respuestas cortas
                    "temperature": 0.4,     # Ligeramente más creativo pero controlado
                    "do_sample": True,
                    "top_p": 0.9,           # Más diversidad en vocabulario
                    "top_k": 50,
                    "repetition_penalty": 1.2,  # Evitar repeticiones
                    "length_penalty": 1.5,  # Fuertemente favorecer respuestas más largas
                    "early_stopping": True,
                    "no_repeat_ngram_size": 3,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "num_return_sequences": 1
                }
            else:
                # Parámetros para modelos causales (GPT-2, etc.)
                generation_kwargs = {
                    "max_new_tokens": 250,  # Aumentado para respuestas más extensas
                    "min_new_tokens": 40,   # Mínimo más alto
                    "temperature": 0.6,     # Más creativo
                    "do_sample": True,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repetition_penalty": 1.3,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "return_full_text": False,
                    "num_return_sequences": 1
                }
            
            # Generar respuesta
            outputs = self.pipeline(prompt, **generation_kwargs)
            
            if outputs and len(outputs) > 0:
                if self.is_seq2seq:
                    response = outputs[0]['generated_text']
                else:
                    full_text = outputs[0]['generated_text']
                    
                    # Extracción inteligente basada en el tipo de modelo
                    if self.is_instruct_model:
                        # Para modelos instruccionales (Mistral), extraer después de [/INST]
                        if "[/INST]" in full_text:
                            response = full_text.split("[/INST]")[-1].strip()
                        else:
                            response = full_text.strip()
                    else:
                        # Para otros modelos causales con nuevo formato
                        if "Respuesta:" in full_text:
                            response = full_text.split("Respuesta:")[-1].strip()
                        elif "Respuesta completa:" in full_text:
                            response = full_text.split("Respuesta completa:")[-1].strip()
                        elif "Mi respuesta:" in full_text:
                            response = full_text.split("Mi respuesta:")[-1].strip()
                        else:
                            # Extraer solo la parte generada (después del prompt)
                            response = full_text[len(prompt):].strip() if len(full_text) > len(prompt) else full_text.strip()
                
                # Limpiar y verificar respuesta
                response = self._clean_response(response)
                
                # Si la respuesta está bien, usarla
                if len(response.strip()) > 15 and self._is_coherent_response(response):
                    return response
        
        except Exception as e:
            logger.error(f"Error generando respuesta con modelo: {e}")
        
        # Fallback a respuesta basada en contexto
        return self._generate_contextual_response(query, context)
    
    def _clean_response(self, response: str) -> str:
        """Limpia y mejora la respuesta generada de manera más robusta (versión mejorada)"""
        if not response:
            return ""
        
        # Artefactos específicos del prompt a eliminar (más selectivo)
        artifacts_to_remove = [
            "Pregunta del usuario:", "Contexto relacionado (extraído de IMDB):",
            "Instrucción: Responde como un experto cinéfilo,",
            "Respuesta:", "Response:", "Answer:",
            "con detalles técnicos, anécdotas si las hay, y un tono cercano pero informativo.",
            "Sé claro, pero no demasiado escueto.",
            "Como experto cinéfilo,", "Mantén un tono cercano y profesional.",
            "Información 1:", "Información 2:", "Información 3:",
            "Fuente 1:", "Fuente 2:", "Fuente 3:"
        ]
        
        # Limpiar artefactos específicos
        cleaned = response
        for artifact in artifacts_to_remove:
            cleaned = cleaned.replace(artifact, "").strip()
        
        # Limpiar líneas de guiones que son del prompt
        import re
        cleaned = re.sub(r'^-\s*(Detalles específicos|Información adicional|Contexto cinematográfico).*$', '', cleaned, flags=re.MULTILINE)
        
        # Limpiar solo numeración clara del prompt (más conservador)
        cleaned = re.sub(r'^(\d+)\.\s*(?=Detalles|Información|Contexto)', '', cleaned, flags=re.MULTILINE)
        
        # Dividir en oraciones de manera más inteligente
        sentences = []
        current_sentence = ""
        
        for char in cleaned:
            current_sentence += char
            if char in '.!?':
                sentence = current_sentence.strip()
                if sentence and len(sentence) > 8:  # Oraciones mínimamente útiles
                    sentences.append(sentence)
                current_sentence = ""
        
        # Agregar la última oración si no termina en puntuación
        if current_sentence.strip() and len(current_sentence.strip()) > 8:
            sentences.append(current_sentence.strip())
        
        # Filtrar solo artefactos obvios del prompt
        clean_sentences = []
        bad_patterns = [
            'analiza cuidadosamente', 'proporciona una respuesta', 'incluye detalles relevantes',
            'mantén un tono profesional', 'provider', 'coherente manera', 'basándote en esta información'
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Solo eliminar artefactos muy obvios
            is_clear_artifact = any(pattern in sentence_lower for pattern in bad_patterns)
            
            if not is_clear_artifact and len(sentence) > 8:
                clean_sentences.append(sentence)
        
        # Reconstruir respuesta manteniendo MÁS contenido
        if clean_sentences:
            # Mantener hasta 15 oraciones para respuestas más completas
            final_response = ' '.join(clean_sentences[:15])
            
            # Limpiar espacios extra pero mantener estructura
            final_response = re.sub(r'\s+', ' ', final_response).strip()
            
            # Mejorar formato final
            if final_response and not final_response.endswith(('.', '!', '?')):
                final_response += '.'
            
            return final_response
        
        return ""
    
    def _is_coherent_response(self, response: str) -> bool:
        """Verifica si una respuesta es coherente y útil (mejorado)"""
        if not response or len(response.strip()) < 20:
            return False
        
        # Verificar que no contenga artefactos comunes
        bad_patterns = [
            "Question:", "Answer:", "Context:", "Information:",
            "CONTEXTO DE IMDB:", "CONSULTA DEL USUARIO:", "RESPUESTA DETALLADA:",
            "INSTRUCCIONES PARA TU RESPUESTA:", "INFORMACIÓN DISPONIBLE:",
        ]
        
        response_upper = response.upper()
        for pattern in bad_patterns:
            if pattern.upper() in response_upper:
                return False
        
        # Verificar patrones problemáticos
        problem_checks = [
            response.count('.') > 20,  # Demasiados puntos
            response.count(',') > 25,  # Demasiadas comas
            len(response.split()) < 8   # Muy pocas palabras
        ]
        
        return not any(problem_checks)
    
    def _generate_contextual_response(self, query: str, context: List[str]) -> str:
        """Genera una respuesta contextual mejorada cuando falla el modelo principal"""
        if not context or len(context) == 0:
            return "No encontré información específica sobre eso en mi base de datos de películas y series de IMDB. ¿Hay alguna película, serie, actor o director específico sobre el que te gustaría saber más?"
        
        # Usar el contexto más relevante
        best_context = context[0] if context else ""
        
        # Intentar extraer información estructurada del contexto
        response_parts = []
        
        # Procesar el contexto para extraer información clave
        if "Título:" in best_context:
            # Es información de película/serie
            lines = best_context.split('\n')
            title_info = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    title_info[key.strip()] = value.strip()
            
            # Construir respuesta estructurada
            if "Título" in title_info:
                response_parts.append(f"Te puedo contar sobre '{title_info['Título']}'.")
                
                if "Año" in title_info:
                    response_parts.append(f"Se estrenó en {title_info['Año']}.")
                
                if "Rating IMDB" in title_info:
                    response_parts.append(f"Tiene una calificación de {title_info['Rating IMDB']} en IMDB.")
                
                if "Género" in title_info:
                    response_parts.append(f"Pertenece al género {title_info['Género']}.")
                
                if "Director" in title_info:
                    response_parts.append(f"Fue dirigida por {title_info['Director']}.")
                
                if "Actores principales" in title_info:
                    response_parts.append(f"Entre los actores principales están: {title_info['Actores principales']}.")
                
                if "Duración" in title_info:
                    response_parts.append(f"Tiene una duración de {title_info['Duración']} minutos.")
        
        # Si no pudimos extraer información estructurada, usar el contexto directamente
        if not response_parts:
            # Limitar el contexto y estructurarlo mejor
            context_limited = best_context[:300] + "..." if len(best_context) > 300 else best_context
            response_parts.append(f"Según la información de IMDB que tengo: {context_limited}")
        
        return " ".join(response_parts)
    
    def _generate_simple_response(self, query: str, context: List[str]) -> str:
        """Genera una respuesta simple cuando falla todo lo demás"""
        return self._generate_contextual_response(query, context)
