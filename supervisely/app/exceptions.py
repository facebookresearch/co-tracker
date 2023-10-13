try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class DialogWindowBase(Exception):
    def __init__(self, title, description, status):
        self.title = title
        self.description = description
        self.status = status
        super().__init__(self.get_message())

    def get_message(self):
        return f"{self.title}: {self.description}"

    def __str__(self):
        return self.get_message()


class DialogWindowError(DialogWindowBase):
    def __init__(self, title, description):
        super().__init__(title, description, "error")


class DialogWindowWarning(DialogWindowBase):
    def __init__(self, title, description):
        super().__init__(title, description, "warning")


# for compatibility
class DialogWindowMessage(DialogWindowError):
    def __init__(self, title, description):
        super().__init__(title, description, "info")


def show_dialog(
    title, description, status: Literal["info", "success", "warning", "error"] = "info"
):
    from supervisely.app import StateJson, DataJson

    StateJson()["slyAppShowDialog"] = True
    DataJson()["slyAppDialogTitle"] = title
    DataJson()["slyAppDialogMessage"] = description
    DataJson()["slyAppDialogStatus"] = status
    DataJson().send_changes()
    StateJson().send_changes()
