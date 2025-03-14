import sys

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", cli_parse_args=True)

    TARANIS_DATASET_PATH: str = ""

    PREPROCESS_TOKENIZER: str = ""
    PREPROCESS_MAX_TOKENS: int

    SUMMARY_TEACHER_MODEL: str = "Mistral-Nemo-Instruct-2407"
    SUMMARY_TEACHER_ENDPOINT: str = "https://mistral-nemo-instruct-2407.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1"
    SUMMARY_MAX_TOKENS: int = 130
    SUMMARY_TEACHER_API_KEY: str = ""
    SUMMARY_MAX_LENGTH: int = 50
    SUMMARY_TEACHER_TEMPERATURE: float = 0.7
    SUMMARY_QUALITY_THRESHOLD: float = 0.6
    SUMMARY_REQUEST_WAIT_TIME: float = 2.0

    DEBUG: bool = False

    DB_PATH: str = "taranis_data_pipeline.db"
    TABLE_NAME: str = "results"

    @field_validator("TARANIS_DATASET_PATH", "PREPROCESS_TOKENIZER", "SUMMARY_TEACHER_API_KEY", "DB_PATH", "TABLE_NAME", mode="before")
    def check_non_empty_string(cls, v: str, info: ValidationInfo) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return v

    def __init__(self, **kwargs):
        if "pytest" in str(sys.argv):  # if run with pytest
            self.model_config["cli_parse_args"] = False
            super().__init__()
        else:
            super().__init__()


Config = Settings()
