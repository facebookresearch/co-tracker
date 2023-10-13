from __future__ import annotations
from collections import namedtuple
from supervisely.app.widgets import Widget
from supervisely.app.content import DataJson, StateJson

from typing import List, Dict, Union, Optional, Callable

class Transfer(Widget):
    """Widget for transfering items between source and target lists.
    Important info: the widget operates with a single list of items (not with two lists of items).
    So, when initializing widget, you should pass a single list of items, which will be displayed in both lists.
    If you want to initialize widget with some items in the right (target) list, you can pass them in the
    `transferred_items` argument, which should containt keys of the items in the 'items' list, NOT NEW items.
    Otherwise, they won't be displayed. Likewise, if you want to change the current list of transferred items
    (which shown in the right list), you can use the 'set_transferred_items' method, which works exactly
    the same as the argument 'transferred_items' in the constructor:
    you should pass keys of the items in the 'items' list.

    :param items: A list of Transfer.Item objects or a list of strings (keys), which will be automatically
                transformed into Transfer.Item objects. Defaults to None.
    :type items: Union[List[Transfer.Item], List[str]], optional
    :param transferred_items: A list of keys of the items to be displayed in the right (target) list. This list
                            contains keys of the items in the 'items' list, not new items. Defaults to None.
    :type transferred_items: List[str], optional
    :param widget_id: The id of the widget. Defaults to None.
    :type widget_id: str, optional
    :param filterable: If True, the widget will have a filter input. Defaults to False.
    :type filterable: bool, optional
    :param filter_placeholder: The placeholder for the filter input. This placeholder will be displayed only if
                            the filterable argument is True. Defaults to None.
    :type filter_placeholder: str, optional
    :param titles: A list of titles for the source and target lists. If not specified, the default titles will be
                used ('Source' and 'Target'). Defaults to None.
    :type titles: List[str], optional
    :param button_texts: A list of texts for the buttons, which transfer items between the lists. If not
                        specified, the buttons will have only icons (> and <). Defaults to None.
    :type button_texts: List[str], optional
    :param left_checked: A list of keys of the items in the left (source) list, which should be checked at widget
                        initialization. Defaults to None.
    :type left_checked: List[str], optional
    :param right_checked: A list of keys of the items in the right (target) list, which should be checked at widget
                        initialization. Defaults to None.
    :type right_checked: List[str], optional
            
    :Methods:
    get_transferred_items(): returns the list of keys of the items, which are currently displayed in the
        right (target) list.
    get_untransferred_items(): returns the list of keys of the items, which are currently displayed in the
        left (source) list.
    @value_changed(func): decorator function, which can be used to handle the events, when the list of
        transferred items is changed (transferred items are moved from the left to the right list or vice versa).
        Function under this decorator should recieve one argument - the list of transferred items (keys of the
        items which are currently displayed in the right list).
    set_items(): sets the list of items to be displayed in the widget. Likewise in the class constructor,
        the list of items can either be a list of Transfer.Item objects or a list of strings.
    set_transferred_items(): sets the list of transferred items (keys of the items which should be displayed
        in the right list).
    add(): adds new items to the widget. Likewise in the class constructor, the list of items can either be
        a list of Transfer.Item objects or a list of strings. Must not contain items with the same keys as
        in the current list of items.
    remove(): removes items from the widget by their keys.
    get_items_keys(): returns the list of keys of the items, which are currently displayed in the widget.

    :Usage example:

     .. code-block:: python
        from supervisely.app.widgets import Transfer

        # Creating widget items with Transfer.Item objects.
        item1 = Transfer.Item(key="cat", label="Cat", disabled=True)
        item2 = Transfer.Item(key="dog", label="Dog")

        # Creating Transfer widget with the list of Transfer.Item objects. The item "dog" will be displayed
        # in the right (target) list at widget initialization.
        transfer = Transfer(items=[item1, item2], transferred_items=["dog"])

        # Setting new transferred items. The item "cat" will be displayed in the right (target) list.
        # Note: items that was in transferred_items list before, but not in the new list, will be moved
        # to untransferred items list (left).
        transfer.set_transferred_items(["cat"])
        print(transfer.get_transferred_items()) # ["cat"]
        print(transfer.get_untransferred_items()) # ["dog"]

        # Creating empty Transfer widget.
        transfer = Transfer()
        # Adding items (as strings) to the widget. The item "dog" will be displayed in the right (target) list.
        transfer.set(["cat", "dog", "mouse"], transferred_items=["dog"])
    """
    class Routes:
        VALUE_CHANGED = "value_changed"
        
    class Item:
        """
        Class for representing items in the Transfer widget.

        Each item has a key, label and disabled flag. Key is required and should be unique. Label is optional and will be
        displayed in the widget. If not specified, label will be equal to the key. Disabled flag is optional and if True,
        the item will be disabled and won't be transferable. Disabled flag is False by default.

        :param key: The key of the item which is mostly used for identifying the item (like value).
            It's important to note, that the keys of the items should be unique.
        :type key: str
        :param label: The label of the item, which will be displayed in the widget. If not specified, the key will be used
            as label. Defaults to None.
        :type label: Optional[str]
        :param disabled: If True, the item will be disabled and won't be transferable. Defaults to False.
        :type disabled: Optional[bool]

        :Usage example:

         .. code-block:: python

            disabled_item = Transfer.Item(key="dog", label="Dog", disabled=True)
            item_with_label = Transfer.Item(key="cat", label="Cat")
            simple_item = Transfer.Item(key="mouse")
        """

        def __init__(self, key: str, label: Optional[str] = None, disabled: Optional[bool] = False):
            self.key = key
            if not label:
                # If label is not specified, the key will be used as label.
                self.label = key
            else:
                self.label = label
            self.disabled = disabled
    
        def to_json(self):
            return {"key": self.key, "label": self.label, "disabled": self.disabled}
    
    def __init__(self, 
                 items: Optional[Union[List[Item], List[str]]] = None, 
                 transferred_items: Optional[List[str]] = None, 
                 widget_id: Optional[str] = None,
                 filterable: Optional[bool] = False, 
                 filter_placeholder: Optional[str] = None, 
                 titles: Optional[List[str]] = None,
                 button_texts: Optional[List[str]] = None, 
                 left_checked: Optional[List[str]] = None, 
                 right_checked: Optional[List[str]] = None):    
        
        self._changes_handled = False
        self._items = []
        self._transferred_items = []
        
        if items:
            self._items = self.__checked_items(items)
        
        if transferred_items:
            self._transferred_items = self.__checked_transferred_items(transferred_items)
        
        # If wrong items are specified, items won't be checked.
        self._left_checked = left_checked
        self._right_checked = right_checked
        
        self._filterable = filterable
        self._filter_placeholder = filter_placeholder
        
        self._titles = titles if titles is not None else ["Source", "Target"]
        
        self._button_texts = button_texts
        
        super().__init__(widget_id=widget_id, file_path=__file__)
    
    
    def __checked_items(self, items: Optional[Union[List[Item], List[str]]]) -> List[Transfer.Item]:
        """
        If the list of items is specified as a list of strings, they will be converted to Transfer.Item objects. List of
        Transfer items will be checked for uniqueness of the keys. If the keys of the items are not unique, an error will be
        raised.

        :param items: The list of items can either be a list of Transfer.Item objects or a list of strings, containing the
            keys for Transfer.Item objects to be created.
        :type items: Optional[Union[List[Item], List[str]]]

        :raises ValueError: If the keys of the items are not unique.

        :return: The list of Transfer.Item objects with unique keys.
        :rtype: List[Transfer.Item]
        """

        if isinstance(items[0], str):
            # If items are specified as strings, they will be converted to Transfer.Item objects.
            if len(set(items)) != len(items):
                # If the keys of the items are not unique, an error will be raised.
                raise ValueError("The keys of the items should be unique.")
            
            checked_items = [Transfer.Item(key=item) for item in items]
        else:
            # If items are specified as Transfer.Item objects, they will be checked for uniqueness.
            if len({item.key for item in items}) != len(items):
                # If the keys of the items are not unique, an error will be raised.
                raise ValueError("The keys of the items should be unique.")
            
            checked_items = items
        return checked_items
    
    def __checked_transferred_items(self, transferred_items: List[str]) -> List[str]:
        """
        If the self._items is specified, the list of transferred items will be checked for the keys of the items. Since
        transferred items are specified by keys of the items, each key of the transferred items should exist in the list of
        items. Otherwise, an error will be raised.

        :param transferred_items: List of keys of the items to be shown in the right (target) list.
        :type transferred_items: List[str]

        :raises ValueError: If transferred items are specified, but the list of items is not specified.
        :raises ValueError: If any of transferred items keys is not in the list of items.

        :return: List of transferred items (keys of the items which should be displayed in the right list).
        :rtype: List[str]
        """

        if not self._items:
            # If the list of items is not specified, the list of transferred items can't be specified since
            # the keys of the items should be specified in the list of transferred items.
            raise ValueError("The 'items' argument should be specified if "\
                "the 'transferred_items' argument is specified.")
        else:
            if not set(transferred_items).issubset([item.key for item in self._items]):
                # If any of the keys of the items specified in the list of transferred items is not
                # in the list of items, an error will be raised.
                raise ValueError("The 'transferred_items' argument should contain only "\
                    "the keys of the items specified in the 'items' argument.")
            else:
                return transferred_items
    
    def get_json_data(self) -> Dict[str, Union[List[Dict[str, Union[str, bool]]], None]]:
        """
        Returns the data of the widget in JSON format.

        Data will contain the list of items and the list of transferred items.

        :return: The data of the widget in JSON format: {"items": List[Dict[str, Any]], "transferred_items": List[str]}.
            "items" - the list of items in the widget in JSON format. Each item is represented as Transfer.Item object.
            "transferred_items" - the list of transferred items (keys of the items which should be displayed in the right list).
        :rtype: Dict[str, List[Dict[str, Union[str, bool]]]]
        """

        res = {
            "items": [],
        }
        if self._items:
            res["items"] = [item.to_json() for item in self._items]

        return res
    
    def get_json_state(self) -> Dict[str, List[str]]:
        """
        Returns the state of the widget in JSON format.

        State will contain the list of transferred items.

        :return: The state of the widget in JSON format: {"transferred_items": List[str]}. "transferred_items" - the list of
            transferred items (keys of the items which should be displayed in the right list).
        :rtype: Dict[str, List[str]]
        """

        transferred_items = self._transferred_items
        
        return {"transferred_items": transferred_items}
    
    def get_transferred_items(self) -> List[str]:        
        """
        Returns the list of transferred items.

        :return: List of transferred items (keys of the items which should be displayed in the right list).
        :rtype: List[str]
        """

        return StateJson()[self.widget_id]["transferred_items"]
    
    
    def get_untransferred_items(self) -> List[str]:
        """
        Returns the list of untransferred items.

        :return: List of untransferred items (keys of the items which should be displayed in the left list).
        :rtype: List[str]
        """

        return [item.key for item in self._items if item.key not in self.get_transferred_items()]
    
    
    def value_changed(self, func: Callable) -> Callable:
        """
        Decorates a function which will be called when the the items in right list are changed (moved in or out of the list).

        :param func: Function to be wrapped with the decorator. The function should have one argument which will contain
            namedtuple with the following fields: transferred_items, untransferred_items.
        :type func: Callable

        :return: Wrapped function.
        :rtype: Callable
        
        :Usage example:

         .. code-block:: python

            tr = Transfer(items=["item1", "item2", "item3"], transferred_items=["item1"])

            # Move "item2" from the left list to the right list. 
            @tr.value_changed
            def func(items):        
                print(items.transferred_items) # ["item1", "item2"]
                print(items.untransferred_items) # ["item3"]
        """

        
        route_path = self.get_route_path(Transfer.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            
            Items = namedtuple("Items", ["transferred_items", "untransferred_items"])
            
            res = Items(transferred_items=self.get_transferred_items(), untransferred_items=self.get_untransferred_items())
            
            func(res)

        return _click
    
    def set_items(self, items: Union[List[Transfer.Item], List[str]]):
        """
        Sets the list of items for the widget.

        If the list of items is specified as strings, they will be converted to Transfer.Item objects.
        Note: this method will REPLACE the current list of items with the new one. If you want to add new items to the current
        list, use .add() method.

        :param items: List of items to be set as the new list of items for the widget. Can be specified as Transfer.Item
            objects or as strings of the item keys.
        :type items: Union[List[Transfer.Item], List[str]]

        :Usage example:

         .. code-block:: python

            tr = Transfer(items=["cat", "dog"])
            print(tr.get_untransferred_items()) # ["cat", "dog"]
                
            tr.set(items=["bird", "mouse"])
            print(tr.get_untransferred_items()) # ["bird", "mouse"]

            # As you can see, the list of items was replaced with the new one.
        """
        
        if items:
            self._items = self.__checked_items(items)
        else:
            self._items = []

        self.update_data()
        DataJson().send_changes()

    def set_transferred_items(self, transferred_items: List[str]):
        """
        Sets the list of transferred items.
        The list should contain only the keys of the items specified in the list of items. Otherwise, an error will be raised.
        :param transferred_items: List of keys of the items which should be displayed in the right list.
        :type transferred_items: List[str]
        """

        self._transferred_items = self.__checked_transferred_items(transferred_items)
        self.update_state()
        StateJson().send_changes()
        
    def add(self, items: Union[List[Item], List[str]]):
        """
        Adds new items to the current list of items.

        If the list of items is specified as strings, Transfer.Item objects will be created from them. If the list of adding 
        items contains any items with the same key as the items in the current list, an error will be raised.

        :param items: List of items to be added to the current list of items. Can be specified as Transfer.Item objects or
            as strings of the item keys.
        :type items: Union[List[Item], List[str]]

        :raises ValueError: If the list of adding items contains any items with the same key as the items in the current list.

        :Usage example:

         .. code-block:: python

            tr = Transfer(items=["cat", "dog"])
            print(tr.get_untransferred_items()) # ["cat", "dog"]
                
            tr.add(items=["bird", "mouse"])
            print(tr.get_untransferred_items()) # ["cat", "dog", "bird", "mouse"]
        """

        items = self.__checked_items(items)
        
        if any([item.key in [item.key for item in self._items] for item in items]):
            raise ValueError("The 'items' argument should not contain any items with the same key as the items in the current list.")
        else:
            self._items.extend(items)
            self.update_data()
            DataJson().send_changes()
            
    def remove(self, items_keys: List[str]):
        """
        Removes items from the current list of items.

        The list of items to be removed should contain keys of the items which should be removed. If there are no items with 
        the specified keys in the current list, nothing will be removed and no error will be raised.

        :param items_keys: List of keys of the items to be removed.
        :type items_keys: List[str]
        """

        self._items = [item for item in self._items if item.key not in items_keys]
        self._transferred_items = [item for item in self._transferred_items if item not in items_keys]
        self.update_data()
        self.update_state()
        DataJson().send_changes()
        StateJson().send_changes()
    
    def get_items_keys(self) -> List[str]:
        """
        Returns the list of keys of the items.

        :return: List of keys of the items.
        :rtype: List[str]
        """

        return [item.key for item in self._items]
