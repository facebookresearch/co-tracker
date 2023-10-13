# coding: utf-8
from supervisely.collection.str_enum import StrEnum


# ['admin',  'developer', 'manager', 'reviewer', 'annotator', 'viewer']
class UserRoleName(StrEnum):
    ADMIN = 'admin'
    DEVELOPER = 'developer'
    MANAGER = 'manager'
    REVIEWER = 'reviewer'
    ANNOTATOR = 'annotator'
    VIEWER = 'viewer'
