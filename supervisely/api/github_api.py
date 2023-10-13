try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from supervisely.api.module_api import ApiField, ModuleApiBase


class GithubApi(ModuleApiBase):
    """
    """
    def get_account_info(self):
        """
        """
        response = self._api.post("github.user.info", {})
        return response.json()

    def create_repo(self, name, visibility: Literal["private", "public"] = "private"):
        """
        """
        response = self._api.post(
            "github.user.info", {"repository": name, "visibility": visibility}
        )
        return response.json()
