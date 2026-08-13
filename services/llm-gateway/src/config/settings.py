from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    ollama_timeout: int = 120
    ollama_keep_alive: str = "10m"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )


settings = Settings()