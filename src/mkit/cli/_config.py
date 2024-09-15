from pathlib import Path

import pydantic_settings as ps


class BaseConfig(ps.BaseSettings):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, yaml_file="params.yaml")
    log_level: int | str = "INFO"
    log_file: Path | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[ps.BaseSettings],
        init_settings: ps.PydanticBaseSettingsSource,
        env_settings: ps.PydanticBaseSettingsSource,
        dotenv_settings: ps.PydanticBaseSettingsSource,
        file_secret_settings: ps.PydanticBaseSettingsSource,
    ) -> tuple[ps.PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            ps.YamlConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )
