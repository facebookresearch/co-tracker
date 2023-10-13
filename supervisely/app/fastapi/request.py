from typing import List, Dict, Any
from pydantic import BaseModel, root_validator
from supervisely.api.api import Api


class Request(BaseModel):
    state: dict = {}
    context: dict = {}
    api: Api

    class Config:
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    @classmethod
    def from_request(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # https://lyz-code.github.io/blue-book/coding/python/pydantic/#initialize-attributes-at-object-creation
        _env_api = Api()
        api_token = values["context"].get("apiToken")
        if api_token is None:
            api_token = values["api_token"]
        values["api"] = Api(_env_api.server_address, api_token)
        return values
