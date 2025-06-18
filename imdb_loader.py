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
    
    def create_knowledge_base(self, max_movies: int = 10000) -> List[str]:
        """Crea una base de conocimiento con información de películas"""
        try:
            logger.info("Creando base de conocimiento de IMDB")
            
            # Cargar datasets principales
            basics = self.load_dataset("title.basics", nrows=max_movies)
            ratings = self.load_dataset("title.ratings", nrows=max_movies)
            
            # Combinar información
            combined = basics.merge(ratings, on='tconst', how='left')
            
            # Filtrar solo películas con información completa
            movies = combined[
                (combined['titleType'] == 'movie') & 
                combined['primaryTitle'].notna() & 
                combined['startYear'].notna()
            ].head(max_movies)
            
            knowledge_texts = []
            
            for _, movie in movies.iterrows():
                text = f"""
                Título: {movie['primaryTitle']}
                Año: {movie['startYear']}
                Géneros: {movie['genres'] if pd.notna(movie['genres']) else 'No especificado'}
                Duración: {movie['runtimeMinutes'] if pd.notna(movie['runtimeMinutes']) else 'No especificado'} minutos
                Rating: {movie['averageRating'] if pd.notna(movie['averageRating']) else 'Sin rating'}/10
                Votos: {movie['numVotes'] if pd.notna(movie['numVotes']) else 'Sin votos'}
                """.strip()
                
                knowledge_texts.append(text)
            
            logger.info(f"Base de conocimiento creada con {len(knowledge_texts)} películas")
            return knowledge_texts
            
        except Exception as e:
            logger.error(f"Error creando base de conocimiento: {e}")
            return []
