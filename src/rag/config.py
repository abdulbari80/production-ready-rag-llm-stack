from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()  # Loads the .env file

class Settings(BaseSettings):
    # hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
    data_dir: str = "data/text"
    chunk_size : int = 500
    chunk_overlap: int = 100
    faiss_index_dir: str = "vector_store/faiss_index"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    hfh_repo_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    top_k: int = 3
    llm_temperature: float = 0.20
    threshold: float = 0.40
    relative_factor : float = 0.80
    context_window_size: int = 2048

settings = Settings()

