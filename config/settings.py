from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from functools import lru_cache

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding='utf-8',extra="ignore"
    )

    #secrets nd models
    groq_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GROQ_API_KEY", "GROQ_API_KEYS"),
    )
    llm_model : str = Field(default="openai/gpt-oss-120b",alias="LLM_MODEL")
    grader_model : str = Field(default="", alias= "GRADER_MODEL")
    llm_temperature : float = Field(default=0.0,alias="LLM_TEMPERATURE")
    embedding_model : str = Field(
        default="ibm-granite/granite-embedding-107m-multilingual",
        alias="EMBEDDING_MODEL"
    )
    hf_token : str = Field(default="", alias="HF_TOKEN")
    hf_hub_offline : bool = Field(default=False, alias="HF_HUB_OFFLINE")


    retrieval_k : int = Field(default=4, alias="RETRIEVAL_k")
    max_revise_tries : int = Field(default=2,alias="MAX_REVISE_TRIES")
    max_rewrite_tries : int = Field(default=2,alias="MAX_REWRITE_TRIES")
    recursion_limit : int = Field(default=150,alias="RECURSION_LIMIT")


    data_dir : Path = Field(default=Path("data"),alias="DATA_DIR")
    index_dir : Path = Field(default=Path("storage/faiss_index"), alias= 'INDEX_DIR')


    chuck_size : int = Field(default=800,alias='CHUCK_SIZE')
    chuck_overlap : int = Field(default=120,alias="CHUCK_OVERLAP")

    host : str = Field(default="0.0.0.0", alias="HOST")
    port : int = Field(default=8000,alias='PORT')
    cors_origins : str = Field(default= "*",alias="CORS_ORIGINS")

    log_level : str =Field(default="INFO",alias="LOG_LEVEL")
    max_history_turns : int = Field(default= 6, alias="MAX_HISTORY_TURNS")

    @property
    def grader_model_name(self) -> str:
        """Fall back to the min model when no dedicatred grader is set"""
        return self.grader_model or self.llm_model

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(',') if o.strip()]

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()