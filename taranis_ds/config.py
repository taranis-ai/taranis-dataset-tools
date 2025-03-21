import sys

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


VALID_TASKS = ["preprocess", "summary", "cybersecurity_class"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", cli_parse_args=True)

    TASKS: list = VALID_TASKS
    TARANIS_DATASET_PATH: str = ""

    PREPROCESS_TOKENIZER: str = "facebook/bart-large-cnn"
    PREPROCESS_MAX_TOKENS: int = 1e5

    PROCESSED_DATASET_PATH: str = ""

    SUMMARY_MODEL: str = "Mistral-Nemo-Instruct-2407"
    SUMMARY_ENDPOINT: str = "https://mistral-nemo-instruct-2407.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1"
    SUMMARY_API_KEY: str = ""
    SUMMARY_MAX_LENGTH: int = 50
    SUMMARY_TEMPERATURE: float = 0.7
    SUMMARY_QUALITY_THRESHOLD: float = 0.6
    SUMMARY_MIN_WAIT_TIME: float = 0.06

    CYBERSEC_CLASS_MODEL: str = "Mixtral-8x7B-Instruct-v0.1"
    CYBERSEC_CLASS_ENDPOINT: str = "https://mixtral-8x7b-instruct-v01.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/chat/completions"
    CYBERSEC_CLASS_API_KEY: str = ""
    CYBERSEC_CLASS_TEMPERATURE: float = 0.7
    CYBERSEC_CLASS_MIN_WAIT_TIME: float = 0.06

    DEBUG: bool = False

    DB_PATH: str = "taranis_data_pipeline.db"

    @field_validator("DB_PATH", mode="before")
    def check_non_empty_string(cls, value: str, info: ValidationInfo) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return value

    @field_validator("TASKS")
    def check_valid_tasks(cls, value: str) -> str:
        if len(value) != len(set(value)):
            raise ValueError("TASKS must not contain duplicate elements")

        if any(item not in VALID_TASKS for item in value):
            raise ValueError(f"All TASKS must be one of {VALID_TASKS}.")
        return value

    def __init__(self, **kwargs):
        if "pytest" in str(sys.argv):  # if run with pytest
            self.model_config["cli_parse_args"] = False
        super().__init__()


Config = Settings()
