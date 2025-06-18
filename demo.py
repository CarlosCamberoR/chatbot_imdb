#!/usr/bin/env python3
"""
Script de demostración del Chatbot IMDB
"""

def demo_queries():
    """Consultas de demostración"""
    return [
        "¿Cuáles son las mejores películas de Christopher Nolan?",
        "Información sobre la película Inception",
        "¿Qué películas de acción tienen el mejor rating?",
        "Actores principales de The Dark Knight",
        "Películas de ciencia ficción de los años 80",
        "¿Cuál es la película con mejor rating de 2020?",
        "Directores más famosos de Hollywood",
        "Películas de terror más populares",
        "¿Qué son los Premios Oscar?",
        "Información sobre actores de Marvel"
    ]

def run_demo():
    """Ejecutar demostración del sistema optimizado para RTX 4070"""
    print("🎬 Demo del Chatbot IMDB - RTX 4070 Optimizado")
    print("=" * 60)
    
    try:
        from rag_system import RAGChatbot
        
        print("🚀 Inicializando sistema RAG optimizado...")
        chatbot = RAGChatbot()
        
        # Inicializar con configuración optimizada
        success = chatbot.initialize(max_movies=5000, use_cache=True)  # 5K para demo rápida
        
        if not success:
            print("❌ Error inicializando el sistema")
            return
        
        print("✅ Sistema inicializado con optimizaciones RTX 4070!")
        print("\n📊 Estadísticas del sistema:")
        stats = chatbot.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Mostrar información de GPU si está disponible
        try:
            import torch
            if torch.cuda.is_available():
                print(f"\n🖥️ GPU: {torch.cuda.get_device_name(0)}")
                print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                print("\n⚠️ GPU no detectada, usando CPU")
        except:
            print("\n⚠️ No se pudo verificar GPU")
        
        print("\n🎯 Ejecutando consultas de demostración...")
        print("-" * 60)
        
        queries = demo_queries()
        
        for i, query in enumerate(queries[:3], 1):  # Solo las primeras 3 para la demo
            print(f"\n💬 Consulta {i}: {query}")
            print("🤔 Procesando con modelo optimizado...")
            
            result = chatbot.query(query)
            
            print(f"🤖 Respuesta: {result['response']}")
            
            if result['sources']:
                print("\n📚 Fuentes (top 2):")
                for j, source in enumerate(result['sources'][:2], 1):
                    print(f"  {j}. {source[:120]}...")
            
            print("-" * 60)
        
        print("\n🎉 Demo completada!")
        print("🚀 Para usar la interfaz completa optimizada:")
        print("   streamlit run app.py")
        print("💡 El sistema ahora usa DialoGPT-Large y MPNet-v2 para mejores resultados")
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("Asegúrate de haber instalado todas las dependencias:")
        print("conda env create -f environment.yml")
        print("conda activate chatbot_imdb")
    except Exception as e:
        print(f"❌ Error ejecutando demo: {e}")

if __name__ == "__main__":
    run_demo()
