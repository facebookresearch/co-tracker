from supervisely.app.widgets import Widget, DynamicWidget
from supervisely.app.widgets_context import JinjaWidgets
from supervisely.app.fastapi.websocket import WebsocketManager
from supervisely.app.fastapi.utils import run_sync
from supervisely.app.content import DataJson, StateJson


class ReloadableArea(DynamicWidget):
    """Widget for dynamic content reloading. It allows to update the content of the widget without reloading the whole
    UI of the app. It can be used when the widgets are needed to be added or removed dynamically.

    :param content: a Widget to be set as content of the ReloadableArea. Defaults to None.
    :type content: Widget, optional
    :param widget_id: The id of the widget. Defaults to None.
    :type widget_id: str, optional

    :Methods:
    reload(): Reloads the widget in UI.
    set_content(): Replaces content of the ReloadableArea with new widget.

    :Usage example:

     .. code-block:: python
        from supervisely.app.widgets import ReloadableArea, Container, Button

        # Creating button, which will be added to widget at initialization.
        button_plus = Button("plus")
        buttons_container = Container(widgets=[button_plus])

        # Initializing ReloadableArea with buttons_container as content.
        reloadable_area = ReloadableArea(content=buttons_container)

        # Now we need to create new widget, after UI was initialized.
        button_minus = Button("minus")
        buttons_container._widgets.append(button_minus)

        # Widget of new button was added to container, but it will not appear in UI
        # until we reload the ReloadableArea.

        reloadable_area.reload()

        # Now the new button will appear in UI.
    """

    def __init__(self, content: Widget = None, widget_id: str = None):
        self._content = content
        super().__init__(widget_id=widget_id, file_path=__file__)

        script_path = "./sly/css/app/widgets/reloadable_area/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

    def get_json_data(self):
        return None

    def get_json_state(self):
        return None

    def set_content(self, content: Widget):
        """Replaces content of the ReloadableArea with new widget. Note: this method doesn't reload the widget.
        To reload the widget in UI use reload() function.

        :param content: new widget to be set as content
        :type content: Widget
        """
        self._content = content
        DataJson().send_changes()
        StateJson().send_changes()

    def reload(self):
        """Reloads the widget in UI."""
        DataJson().send_changes()
        StateJson().send_changes()

        html_content = self._content.to_html()
        run_sync(
            WebsocketManager().broadcast(
                {
                    "runAction": {
                        "action": f"rerender-template-{self.widget_id}",
                        "payload": {"template": html_content},
                    }
                }
            )
        )

    def hide(self):
        """Hides the content of the ReloadableArea."""
        self._content.hide()
        
    def show(self):
        """Shows the content of the ReloadableArea."""
        self._content.show()
