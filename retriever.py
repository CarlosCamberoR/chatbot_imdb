import os
import pickle
import numpy as np
import faiss
import torch
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import logging

logger = logging.getLogger(__name__)

class MixedRetriever:
    """Retriever híbrido que combina búsqueda semántica y BM25"""
    
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 cache_dir: str = "cache"):
        self.embedding_model_name = embedding_model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Inicializar modelo de embeddings optimizado
        logger.info(f"Cargando modelo de embeddings: {embedding_model_name}")
        
        # Configurar dispositivo para embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            device=device,
            trust_remote_code=True
        )
        
        # Optimizar para RTX 4070
        if device == "cuda":
            logger.info(f"Usando GPU para embeddings: {torch.cuda.get_device_name(0)}")
            # Configurar batch size óptimo para RTX 4070
            self.batch_size = int(os.getenv("BATCH_SIZE", "32"))
        else:
            self.batch_size = 16
        
        # Inicializar componentes
        self.documents = []
        self.embeddings = None
        self.faiss_index = None
        self.bm25 = None
        
    def add_documents(self, documents: List[str], save_cache: bool = True):
        """Añade documentos al retriever con procesamiento optimizado"""
        logger.info(f"Añadiendo {len(documents)} documentos al retriever")
        
        self.documents = documents
        
        # Crear embeddings para búsqueda semántica con batch processing optimizado
        logger.info("Creando embeddings con procesamiento por lotes optimizado...")
        self.embeddings = self.embedding_model.encode(
            documents, 
            show_progress_bar=True,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalizar directamente
        )
        
        # Crear índice FAISS optimizado
        logger.info("Creando índice FAISS optimizado...")
        dimension = self.embeddings.shape[1]
        
        # Usar índice más eficiente para datasets grandes
        if len(documents) > 10000:
            # Para datasets grandes, usar HNSW para mejor rendimiento
            self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)
            self.faiss_index.hnsw.efConstruction = 200
            self.faiss_index.hnsw.efSearch = 100
        else:
            # Para datasets pequeños, usar índice plano
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product para cosine similarity
        
        # Los embeddings ya están normalizados
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        # Crear índice BM25
        logger.info("Creando índice BM25...")
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        if save_cache:
            self.save_cache()
        
        logger.info("Retriever configurado exitosamente con optimizaciones para RTX 4070")
    
    def save_cache(self):
        """Guarda el estado del retriever en cache"""
        cache_file = os.path.join(self.cache_dir, "retriever_cache.pkl")
        
        cache_data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'embedding_model_name': self.embedding_model_name
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Guardar índices
        if self.faiss_index:
            faiss.write_index(self.faiss_index, os.path.join(self.cache_dir, "faiss_index.idx"))
        
        if self.bm25:
            with open(os.path.join(self.cache_dir, "bm25_index.pkl"), 'wb') as f:
                pickle.dump(self.bm25, f)
        
        logger.info("Cache guardado exitosamente")
    
    def load_cache(self) -> bool:
        """Carga el estado del retriever desde cache"""
        cache_file = os.path.join(self.cache_dir, "retriever_cache.pkl")
        
        if not os.path.exists(cache_file):
            return False
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verificar compatibilidad del modelo
            if cache_data['embedding_model_name'] != self.embedding_model_name:
                logger.warning("Modelo de embeddings ha cambiado, recreando cache")
                return False
            
            self.documents = cache_data['documents']
            self.embeddings = cache_data['embeddings']
            
            # Cargar índices
            faiss_path = os.path.join(self.cache_dir, "faiss_index.idx")
            if os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path)
            
            bm25_path = os.path.join(self.cache_dir, "bm25_index.pkl")
            if os.path.exists(bm25_path):
                with open(bm25_path, 'rb') as f:
                    self.bm25 = pickle.load(f)
            
            logger.info("Cache cargado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando cache: {e}")
            return False
    
    def semantic_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Búsqueda semántica usando FAISS"""
        if self.faiss_index is None:
            return []
        
        # Crear embedding de la query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Buscar en FAISS
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def bm25_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Búsqueda BM25"""
        if self.bm25 is None:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Obtener top k resultados
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                results.append((self.documents[idx], float(scores[idx])))
        
        return results
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Tuple[str, float]]:
        """Búsqueda híbrida combinando semántica y BM25"""
        # Obtener resultados de ambos métodos
        semantic_results = self.semantic_search(query, k * 2)  # Obtener más para mezclar
        bm25_results = self.bm25_search(query, k * 2)
        
        # Combinar resultados usando un diccionario
        combined_scores = {}
        
        # Normalizar y combinar scores semánticos
        if semantic_results:
            max_semantic = max([score for _, score in semantic_results])
            for doc, score in semantic_results:
                normalized_score = score / max_semantic if max_semantic > 0 else 0
                combined_scores[doc] = alpha * normalized_score
        
        # Normalizar y combinar scores BM25
        if bm25_results:
            max_bm25 = max([score for _, score in bm25_results]) 
            for doc, score in bm25_results:
                normalized_score = score / max_bm25 if max_bm25 > 0 else 0
                if doc in combined_scores:
                    combined_scores[doc] += (1 - alpha) * normalized_score
                else:
                    combined_scores[doc] = (1 - alpha) * normalized_score
        
        # Ordenar por score combinado y devolver top k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]
    
    def retrieve(self, query: str, k: int = 5, method: str = "hybrid") -> List[str]:
        """Recupera documentos relevantes"""
        if method == "semantic":
            results = self.semantic_search(query, k)
        elif method == "bm25":
            results = self.bm25_search(query, k)
        else:  # hybrid
            results = self.hybrid_search(query, k)
        
        return [doc for doc, _ in results]
