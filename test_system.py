"""
Pruebas unitarias para el sistema RAG de IMDB
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TestIMDBLoader(unittest.TestCase):
    """Pruebas para el cargador de IMDB"""
    
    def setUp(self):
        from imdb_loader import IMDBDataLoader
        self.loader = IMDBDataLoader(data_dir="test_data")
    
    def test_initialization(self):
        """Probar inicialización del loader"""
        self.assertEqual(self.loader.data_dir, "test_data")
        self.assertIn("title.basics", self.loader.imdb_files)
    
    def test_imdb_files_mapping(self):
        """Probar mapeo de archivos IMDB"""
        expected_files = [
            "title.basics", "title.ratings", "title.crew",
            "name.basics", "title.principals"
        ]
        for file in expected_files:
            self.assertIn(file, self.loader.imdb_files)

class TestRetriever(unittest.TestCase):
    """Pruebas para el retriever híbrido"""
    
    def setUp(self):
        # Mock para evitar cargar modelos reales en tests
        with patch('sentence_transformers.SentenceTransformer'):
            from retriever import MixedRetriever
            self.retriever = MixedRetriever()
    
    def test_initialization(self):
        """Probar inicialización del retriever"""
        self.assertIsNotNone(self.retriever.embedding_model_name)
        self.assertEqual(self.retriever.documents, [])

class TestChatbotModel(unittest.TestCase):
    """Pruebas para el modelo de chatbot"""
    
    def setUp(self):
        # Mock para evitar cargar modelos reales
        with patch('transformers.AutoTokenizer.from_pretrained'), \
             patch('transformers.AutoModelForCausalLM.from_pretrained'), \
             patch('transformers.pipeline'):
            from chatbot_model import ChatbotModel
            self.model = ChatbotModel()
    
    def test_is_imdb_related(self):
        """Probar detección de consultas relacionadas con IMDB"""
        # Casos positivos
        self.assertTrue(self.model.is_imdb_related("¿Cuál es la mejor película de 2020?"))
        self.assertTrue(self.model.is_imdb_related("Información sobre el actor Tom Hanks"))
        self.assertTrue(self.model.is_imdb_related("Películas de ciencia ficción"))
        
        # Casos negativos
        self.assertFalse(self.model.is_imdb_related("¿Cómo está el clima hoy?"))
        self.assertFalse(self.model.is_imdb_related("Receta de pasta"))
        self.assertFalse(self.model.is_imdb_related("Problemas matemáticos"))

class TestRAGSystem(unittest.TestCase):
    """Pruebas para el sistema RAG completo"""
    
    def setUp(self):
        with patch('rag_system.IMDBDataLoader'), \
             patch('rag_system.MixedRetriever'), \
             patch('rag_system.ChatbotModel'):
            from rag_system import RAGChatbot
            self.rag = RAGChatbot()
    
    def test_initialization(self):
        """Probar inicialización del sistema RAG"""
        self.assertIsNotNone(self.rag.hf_token)
        self.assertIsNotNone(self.rag.model_name)
        self.assertFalse(self.rag.knowledge_base_ready)

if __name__ == '__main__':
    # Configurar variables de entorno para tests
    os.environ['HUGGINGFACE_TOKEN'] = 'test_token'
    os.environ['MODEL_NAME'] = 'distilgpt2'
    
    unittest.main()
