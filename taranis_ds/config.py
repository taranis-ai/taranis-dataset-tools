from pydantic import field_validator, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", cli_parse_args=True)

    SUMMARY_TEACHER_MODEL: str = "Mistral-Nemo-Instruct-2407"
    SUMMARY_TEACHER_ENDPOINT: str = "https://mistral-nemo-instruct-2407.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1"
    SUMMARY_MAX_TOKENS: int = 130
    SUMMARY_TEACHER_API_KEY: str = ""
    SUMMARY_MAX_LENGTH: int = 50

    @field_validator("SUMMARY_TEACHER_API_KEY", mode="before")
    def check_non_empty_string(cls, v: str, info: ValidationInfo) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return v


Config = Settings()
