import os
import gzip
import shutil
import requests
import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class IMDBDataLoader:
    """Cargador de datos de IMDB desde los datasets oficiales"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.base_url = "https://datasets.imdbws.com"
        os.makedirs(data_dir, exist_ok=True)
        
        # Mapeo de archivos IMDB
        self.imdb_files = {
            "title.basics": "title.basics.tsv.gz",
            "title.ratings": "title.ratings.tsv.gz",
            "title.crew": "title.crew.tsv.gz",
            "name.basics": "name.basics.tsv.gz",
            "title.principals": "title.principals.tsv.gz",
            "title.episode": "title.episode.tsv.gz",
            "title.akas": "title.akas.tsv.gz"
        }
    
    def download_file(self, filename: str) -> str:
        """Descarga un archivo de IMDB si no existe"""
        local_path = os.path.join(self.data_dir, filename)
        if os.path.exists(local_path.replace('.gz', '')):
            logger.info(f"Archivo {filename} ya existe, saltando descarga")
            return local_path.replace('.gz', '')
            
        url = f"{self.base_url}/{filename}"
        logger.info(f"Descargando {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Descomprimir
        with gzip.open(local_path, 'rb') as f_in:
            with open(local_path.replace('.gz', ''), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Eliminar archivo comprimido
        os.remove(local_path)
        return local_path.replace('.gz', '')
    
    def load_dataset(self, dataset_name: str, nrows: int = None) -> pd.DataFrame:
        """Carga un dataset específico de IMDB"""
        if dataset_name not in self.imdb_files:
            raise ValueError(f"Dataset {dataset_name} no disponible")

        filename = self.imdb_files[dataset_name]
        local_path = self.download_file(filename)

        logger.info(f"Cargando dataset {dataset_name}")
        # Cargar todo el dataset sin límite de filas para tener más datos
        df = pd.read_csv(local_path, sep='\t', low_memory=False, nrows=nrows)

        # Reemplazar '\\N' con NaN
        df = df.replace('\\N', pd.NA)

        return df
    
    def get_movie_info(self, tconst: str) -> Dict[str, Any]:
        """Obtiene información completa de una película"""
        try:
            # Cargar información básica
            basics = self.load_dataset("title.basics", nrows=100000)
            movie = basics[basics['tconst'] == tconst].iloc[0].to_dict()
            
            # Cargar ratings
            try:
                ratings = self.load_dataset("title.ratings", nrows=100000)
                rating_info = ratings[ratings['tconst'] == tconst]
                if not rating_info.empty:
                    movie.update(rating_info.iloc[0].to_dict())
            except:
                pass
            
            # Cargar crew
            try:
                crew = self.load_dataset("title.crew", nrows=100000)
                crew_info = crew[crew['tconst'] == tconst]
                if not crew_info.empty:
                    movie.update(crew_info.iloc[0].to_dict())
            except:
                pass
            
            return movie
        except Exception as e:
            logger.error(f"Error obteniendo información de {tconst}: {e}")
            return {}
    
    def search_movies(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Busca películas por título"""
        try:
            basics = self.load_dataset("title.basics", nrows=100000)
            
            # Filtrar solo películas y series
            movies = basics[basics['titleType'].isin(['movie', 'tvSeries', 'tvMovie'])]
            
            # Buscar por título
            mask = movies['primaryTitle'].str.contains(query, case=False, na=False) | \
                   movies['originalTitle'].str.contains(query, case=False, na=False)
            
            results = movies[mask].head(limit)
            return results.to_dict('records')
        except Exception as e:
            logger.error(f"Error buscando películas: {e}")
            return []
    
    def create_knowledge_base(self, max_movies: int = 50000) -> List[str]:
        """Crea una base de conocimiento completa con información detallada de películas y series"""
        try:
            logger.info("Creando base de conocimiento completa de IMDB")
            
            # Cargar datasets principales de manera inteligente
            logger.info("Cargando información básica de títulos (chunk optimizado)...")
            basics = self.load_dataset("title.basics", nrows=500000)  # Cargar muchos títulos
            
            logger.info("Cargando ratings...")
            ratings = self.load_dataset("title.ratings", nrows=200000)  # Ratings disponibles
            
            # Pre-filtrar para obtener solo contenido de calidad antes de cargar más datos
            logger.info("Pre-filtrando contenido de calidad...")
            relevant_types = ['movie', 'tvSeries', 'tvMovie', 'documentary', 'tvMiniSeries']
            filtered_basics = basics[
                (basics['titleType'].isin(relevant_types)) & 
                basics['primaryTitle'].notna() & 
                basics['startYear'].notna() &
                (basics['startYear'].astype(str).str.match(r'^\d{4}$', na=False))  # Años válidos
            ]
            
            # Convertir startYear a entero para filtrar
            filtered_basics = filtered_basics.copy()
            filtered_basics['startYear'] = pd.to_numeric(filtered_basics['startYear'], errors='coerce')
            filtered_basics = filtered_basics[
                (filtered_basics['startYear'] >= 1980) & 
                (filtered_basics['startYear'] <= 2025)
            ]
            
            # Combinar con ratings para filtrar por calidad
            combined_initial = filtered_basics.merge(ratings, on='tconst', how='inner')
            
            # Filtrar por calidad mínima
            quality_filter = (
                (pd.to_numeric(combined_initial['numVotes'], errors='coerce') >= 50) & 
                (pd.to_numeric(combined_initial['averageRating'], errors='coerce') >= 5.0)
            )
            high_quality_titles = combined_initial[quality_filter]
            
            # Ordenar por popularidad y tomar los mejores
            high_quality_titles['numVotes'] = pd.to_numeric(high_quality_titles['numVotes'], errors='coerce')
            high_quality_titles['averageRating'] = pd.to_numeric(high_quality_titles['averageRating'], errors='coerce')
            high_quality_titles = high_quality_titles.sort_values(
                ['numVotes', 'averageRating'], 
                ascending=False
            ).head(max_movies)
            
            logger.info(f"Seleccionados {len(high_quality_titles)} títulos de alta calidad")
            
            # Obtener IDs relevantes para cargar solo los datos necesarios
            relevant_tconsts = set(high_quality_titles['tconst'].values)
            
            logger.info("Cargando información de crew (directores, escritores)...")
            crew = self.load_dataset("title.crew", nrows=100000)
            crew = crew[crew['tconst'].isin(relevant_tconsts)]
            
            logger.info("Cargando información de actores principales...")
            principals = self.load_dataset("title.principals", nrows=200000)
            principals = principals[
                (principals['tconst'].isin(relevant_tconsts)) & 
                (principals['category'].isin(['actor', 'actress', 'director']))
            ]
            
            logger.info("Cargando información de personas (nombres)...")
            # Obtener IDs de personas relevantes
            relevant_nconsts = set()
            if not crew.empty and 'directors' in crew.columns:
                for directors_str in crew['directors'].dropna():
                    relevant_nconsts.update(directors_str.split(','))
            if not crew.empty and 'writers' in crew.columns:
                for writers_str in crew['writers'].dropna():
                    relevant_nconsts.update(writers_str.split(','))
            if not principals.empty:
                relevant_nconsts.update(principals['nconst'].values)
            
            names = self.load_dataset("name.basics", nrows=300000)
            names = names[names['nconst'].isin(relevant_nconsts)]
            
            # Usar los datos ya filtrados
            combined = high_quality_titles
            
            logger.info(f"Procesando {len(combined)} títulos de calidad...")
            
            # Combinar con información de crew
            combined = combined.merge(crew, on='tconst', how='left')
            
            knowledge_texts = []
            
            for idx, movie in combined.iterrows():
                try:
                    # Información básica
                    title = movie['primaryTitle']
                    year = movie['startYear']
                    title_type = movie['titleType']
                    genres = movie['genres'] if pd.notna(movie['genres']) else 'No especificado'
                    runtime = movie['runtimeMinutes'] if pd.notna(movie['runtimeMinutes']) else 'No especificado'
                    rating = movie['averageRating'] if pd.notna(movie['averageRating']) else 'Sin rating'
                    votes = movie['numVotes'] if pd.notna(movie['numVotes']) else 'Sin votos'
                    
                    # Información de crew
                    directors = []
                    writers = []
                    
                    if pd.notna(movie.get('directors')):
                        director_ids = str(movie['directors']).split(',')
                        for dir_id in director_ids[:3]:  # Máximo 3 directores
                            dir_info = names[names['nconst'] == dir_id.strip()]
                            if not dir_info.empty:
                                directors.append(dir_info.iloc[0]['primaryName'])
                    
                    if pd.notna(movie.get('writers')):
                        writer_ids = str(movie['writers']).split(',')
                        for writer_id in writer_ids[:3]:  # Máximo 3 escritores
                            writer_info = names[names['nconst'] == writer_id.strip()]
                            if not writer_info.empty:
                                writers.append(writer_info.iloc[0]['primaryName'])
                    
                    # Información de actores principales
                    main_actors = []
                    movie_principals = principals[
                        (principals['tconst'] == movie['tconst']) & 
                        (principals['category'].isin(['actor', 'actress']))
                    ].head(5)  # Top 5 actores
                    
                    for _, actor_row in movie_principals.iterrows():
                        actor_info = names[names['nconst'] == actor_row['nconst']]
                        if not actor_info.empty:
                            actor_name = actor_info.iloc[0]['primaryName']
                            character = actor_row['characters'] if pd.notna(actor_row['characters']) else ''
                            if character:
                                character = str(character).replace('["', '').replace('"]', '').replace('"', '')
                                main_actors.append(f"{actor_name} ({character})")
                            else:
                                main_actors.append(actor_name)
                    
                    # Determinar tipo de contenido en español
                    type_spanish = {
                        'movie': 'Película',
                        'tvSeries': 'Serie de TV',
                        'tvMovie': 'Película de TV',
                        'documentary': 'Documental',
                        'tvMiniSeries': 'Miniserie'
                    }.get(title_type, title_type)
                    
                    # Crear descripción completa
                    text_parts = [
                        f"Título: {title}",
                        f"Tipo: {type_spanish}",
                        f"Año: {year}",
                        f"Géneros: {genres}",
                        f"Duración: {runtime} minutos" if runtime != 'No especificado' else "Duración: No especificada",
                        f"Calificación IMDb: {rating}/10 ({votes} votos)",
                    ]
                    
                    if directors:
                        text_parts.append(f"Director(es): {', '.join(directors)}")
                    
                    if writers:
                        text_parts.append(f"Guionista(s): {', '.join(writers)}")
                    
                    if main_actors:
                        text_parts.append(f"Reparto principal: {', '.join(main_actors)}")
                    
                    # Información adicional basada en datos
                    try:
                        rating_float = float(rating)
                        if rating_float >= 8.0:
                            text_parts.append("Esta es una producción altamente valorada por la audiencia.")
                    except:
                        pass
                    
                    try:
                        votes_int = int(votes)
                        if votes_int >= 100000:
                            text_parts.append("Esta es una producción muy popular con gran número de votaciones.")
                    except:
                        pass
                    
                    # Contexto de géneros
                    if 'Action' in genres:
                        text_parts.append("Película/Serie de acción con secuencias emocionantes.")
                    if 'Drama' in genres:
                        text_parts.append("Drama que explora temas profundos y emocionales.")
                    if 'Comedy' in genres:
                        text_parts.append("Comedia diseñada para entretener y hacer reír.")
                    if 'Horror' in genres:
                        text_parts.append("Película/Serie de terror con elementos de suspenso.")
                    if 'Sci-Fi' in genres:
                        text_parts.append("Ciencia ficción que explora conceptos futuristas.")
                    
                    final_text = '\n'.join(text_parts)
                    knowledge_texts.append(final_text)
                    
                except Exception as e:
                    logger.warning(f"Error procesando título {movie.get('primaryTitle', 'desconocido')}: {e}")
                    continue
            
            logger.info(f"Base de conocimiento completa creada con {len(knowledge_texts)} títulos")
            return knowledge_texts
            
        except Exception as e:
            logger.error(f"Error creando base de conocimiento: {e}")
            return []
