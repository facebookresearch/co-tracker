# coding: utf-8

from enum import Enum


class StrEnum(Enum):
    r"""Allows to get the value of enum on string conversion.

    Method ``__str__`` is added to standard enum to provide a custom string representation.
    See additional stackoverflow `examples <https://stackoverflow.com/questions/24487405/enum-getting-value-of-enum-on-string-conversion>`_

    Example of default python enum::

        >>> from enum import Enum
        >>> class MyEnum(Enum):
        ONE = "one"
        TWO = "two"
        >>> MyEnum.ONE
        <MyEnum.ONE: 'one'>
        >>> print(MyEnum.ONE)
        MyEnum.ONE
    
    How to use of StrEnum::
    
        >>> import supervisely as sly
        >>> class MyStrEnum(sly.StrEnum):
        ONE = "one"
        TWO = "two"
        >>> MyStrEnum.ONE
        <MyStrEnum.ONE: 'one'>
        >>> str(MyStrEnum.ONE)
        'one'
        >>> print(MyStrEnum.ONE)
        one
        
    """

    def __str__(self):
        return str(self.value)

    @classmethod
    def has_value(cls, value):
        for possible_value in cls:
            if value == str(possible_value.value):
                return True
        return False

    @classmethod
    def values(cls):
        return [value.value for value in cls]


