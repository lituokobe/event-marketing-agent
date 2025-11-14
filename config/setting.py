from typing import List, Union
from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings

from config.db_setting import DBSetting


class Settings(BaseSettings, DBSetting):
    AI_MODEL_SERVICE_URL: AnyHttpUrl = 'http://localhost:5002'




settings = Settings()
