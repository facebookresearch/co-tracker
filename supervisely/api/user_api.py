# coding: utf-8
"""create and manipulate already existing users in your team"""

# docs
from __future__ import annotations
from typing import Callable, Dict, List, NamedTuple, Optional, TYPE_CHECKING, Union

from collections import namedtuple

from supervisely.api.module_api import ApiField, ModuleApiBase, _get_single_item
from supervisely.task.progress import Progress

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame

from tqdm import tqdm


class UserInfo(NamedTuple):
    """ """

    id: int
    login: str
    role: str
    role_id: int
    name: str
    email: str
    logins: int
    disabled: bool
    last_login: str
    created_at: str
    updated_at: str


class UserApi(ModuleApiBase):
    """
    API for working with :class:`Users<supervisely.user.user.UserRoleName>`. :class:`UserApi<UserApi>` object is immutable.

    :param api: API connection to the server.
    :type api: Api
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervise.ly", token="4r47N...xaTatb")

        users = api.user.get_list() # api usage example
    """

    Membership = namedtuple("Membership", ["id", "name", "role_id", "role"])

    @staticmethod
    def info_sequence():
        """
        NamedTuple UserInfo information about User.

        :Example:

         .. code-block:: python

            UserInfo(id=8,
                     login='alex',
                     role=None,
                     role_id=None,
                     name=None,
                     email=None,
                     logins=20,
                     disabled=False,
                     last_login='2021-03-24T15:06:26.804Z',
                     created_at='2020-04-17T10:24:09.077Z',
                     updated_at='2021-03-24T15:13:01.148Z')
        """
        return [
            ApiField.ID,
            ApiField.LOGIN,
            ApiField.ROLE,
            ApiField.ROLE_ID,
            ApiField.NAME,
            ApiField.EMAIL,
            ApiField.LOGINS,
            ApiField.DISABLED,
            ApiField.LAST_LOGIN,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **UserInfo**.
        """
        return "UserInfo"

    def _convert_json_info(self, info: dict, skip_missing=True):
        res = super(UserApi, self)._convert_json_info(info, skip_missing=skip_missing)
        return UserInfo(**res._asdict())

    def get_info_by_id(self, id: int) -> UserInfo:
        """
        Get User information by ID.

        :param id: User ID in Supervisely.
        :type id: int
        :return: Information about User. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`UserInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            user_info = api.user.get_info_by_id(8)
            print(user_info)
            # Output: [
            #     8,
            #     "alex",
            #     null,
            #     null,
            #     null,
            #     null,
            #     20,
            #     false,
            #     "2021-03-24T15:06:26.804Z",
            #     "2020-04-17T10:24:09.077Z",
            #     "2021-03-24T15:13:01.148Z"
            # ]
        """
        return self._get_info_by_id(id, "users.info")

    def get_info_by_login(self, login: str) -> UserInfo:
        """
        Get User information by login.

        :param login: User login in Supervisely.
        :type login: str
        :return: Information about User. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`UserInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            user_info = api.user.get_info_by_login('alex')
            print(user_info)
            # Output: [
            #     8,
            #     "alex",
            #     null,
            #     null,
            #     null,
            #     null,
            #     20,
            #     false,
            #     "2021-03-24T15:06:26.804Z",
            #     "2020-04-17T10:24:09.077Z",
            #     "2021-03-24T15:13:01.148Z"
            # ]
        """
        filters = [{"field": ApiField.LOGIN, "operator": "=", "value": login}]
        items = self.get_list(filters)
        return _get_single_item(items)

    def get_member_info_by_login(self, team_id: int, login: str) -> UserInfo:
        """
        Get information about team member by Team ID and User login.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param login: User login in Supervisely.
        :type login: str
        :return: Information about User. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`UserInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            member_info = api.user.get_member_info_by_login(64, 'alex')
            print(member_info)
            # Output: [
            #     8,
            #     "alex",
            #     "manager",
            #     3,
            #     null,
            #     null,
            #     20,
            #     false,
            #     "2021-03-24T15:06:26.804Z",
            #     "2020-04-17T10:24:09.077Z",
            #     "2021-03-24T15:13:01.148Z"
            # ]
        """
        filters = [{"field": ApiField.LOGIN, "operator": "=", "value": login}]
        team_members = self.get_list_all_pages(
            "members.list",
            {ApiField.TEAM_ID: team_id, ApiField.FILTER: filters},
            convert_json_info_cb=self._api.user._convert_json_info,
        )
        return _get_single_item(team_members)

    def get_member_info_by_id(self, team_id: int, user_id: int) -> UserInfo:
        """
        Get information about team member by Team ID and User ID.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param user_id: User ID in Supervisely.
        :type user_id: int
        :return: Information about User. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`UserInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            member_info = api.user.get_member_info_by_id(64, 8)
            print(member_info)
            # Output: [
            #     8,
            #     "alex",
            #     "manager",
            #     3,
            #     null,
            #     null,
            #     20,
            #     false,
            #     "2021-03-24T15:06:26.804Z",
            #     "2020-04-17T10:24:09.077Z",
            #     "2021-03-24T15:13:01.148Z"
            # ]
        """
        filters = [{"field": ApiField.ID, "operator": "=", "value": user_id}]
        team_members = self.get_list_all_pages(
            "members.list",
            {ApiField.TEAM_ID: team_id, ApiField.FILTER: filters},
            convert_json_info_cb=self._api.user._convert_json_info,
        )
        return _get_single_item(team_members)

    def get_list(self, filters: List[Dict[str, str]] = None) -> List[UserInfo]:
        """
        Get list of information about Users.

        :param filters: List of params to sort output Users.
        :type filters: List[dict], optional
        :return: List of information about Users. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[UserInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            # Get list of Users with id = 8
            user_info = api.user.get_list(filters=[{'field': 'id', 'operator': '=', 'value': '8'}])
            print(user_info)
            # Output: [
            #     8,
            #     "alex",
            #     "manager",
            #     3,
            #     null,
            #     null,
            #     20,
            #     false,
            #     "2021-03-24T15:06:26.804Z",
            #     "2020-04-17T10:24:09.077Z",
            #     "2021-03-24T15:13:01.148Z"
            # ]
        """
        return self.get_list_all_pages("users.list", {ApiField.FILTER: filters or []})

    def create(
        self,
        login: str,
        password: str,
        is_restricted: Optional[bool] = False,
        name: Optional[str] = "",
        email: Optional[str] = "",
    ) -> UserInfo:
        """
        Creates new User with given login and password.

        :param login: New User login.
        :type login: str
        :param password: New User password.
        :type password: str
        :param is_restricted: If True, new User will have no access to Explore section, won't be able to create or switch Teams, and no personal Team will be created for this User during signup.
        :type is_restricted: bool, optional
        :param name: New User name.
        :type name: str, optional
        :param email: New User email.
        :type email: str, optional
        :return: Information about new User. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`UserInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            new_user_info = api.user.create('John', 'qwerty', is_restricted=True, name='John Wick', email='excomunicado@gmail.com')
            print(new_user_info)
            # Output: [
            #     274,
            #     "John",
            #     null,
            #     null,
            #     "John Wick",
            #     "excomunicado@gmail.com",
            #     0,
            #     false,
            #     null,
            #     "2021-03-24T16:20:03.110Z",
            #     "2021-03-24T16:20:03.110Z"
            # ]
        """
        response = self._api.post(
            "users.add",
            {
                ApiField.LOGIN: login,
                ApiField.PASSWORD: password,
                ApiField.IS_RESTRICTED: is_restricted,
                ApiField.NAME: name,
                ApiField.EMAIL: email,
            },
        )
        return self.get_info_by_id(response.json()[ApiField.USER_ID])

    def _set_disabled(self, id, disable):
        """
        Check status of the user with given id
        :param id: int
        :param disable: bool
        """
        self._api.post("users.disable", {ApiField.ID: id, ApiField.DISABLE: disable})

    def disable(self, id: int) -> None:
        """
        Disables User with the given ID.

        :param id: User ID in Supervisely.
        :type id: int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            user_id = 8
            api.user.disable(user_id)
        """
        self._set_disabled(id, True)

    def enable(self, id: int) -> None:
        """
        Enables User with the given ID.

        :param id: User ID in Supervisely.
        :type id: int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            user_id = 8
            api.user.enable(user_id)
        """
        self._set_disabled(id, False)

    def get_token(self, login):
        raise NotImplementedError()

    def get_teams(self, id: int) -> List[UserInfo]:
        """
        Get list with information about User Teams.

        :param id: User ID in Supervisely.
        :type id: int
        :return: List of teams in which the User with the given ID is located
        :rtype: :class:`List[UserInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            teams = api.user.get_teams(8)
            print(teams)
            # Output: [
            #     [
            #         9,
            #         "alex",
            #         1,
            #         "admin"
            #     ],
            #     [
            #         64,
            #         "test",
            #         3,
            #         "manager"
            #     ]
            # ]
        """
        response = self._api.post("users.info", {ApiField.ID: id})
        teams_json = response.json()[ApiField.TEAMS]
        teams = []
        for team in teams_json:
            member = self.Membership(
                id=team[ApiField.ID],
                name=team[ApiField.NAME],
                role_id=team[ApiField.ROLE_ID],
                role=team[ApiField.ROLE],
            )
            teams.append(member)
        return teams

    def add_to_team(self, user_id: int, team_id: int, role_id: int) -> None:
        """
        Invites User to Team with the given role.

        :param user_id: User ID in Supervisely.
        :type user_id: int
        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param role_id: Role ID in Supervisely.
        :type role_id: int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            user_id = 8
            team_id = 76
            role_id = 5
            api.user.add_to_team(user_id, team_id, role_id)
        """
        user = self.get_info_by_id(user_id)
        response = self._api.post(
            "members.add",
            {
                ApiField.LOGIN: user.login,
                ApiField.TEAM_ID: team_id,
                ApiField.ROLE_ID: role_id,
            },
        )

    def remove_from_team(self, user_id: int, team_id: int) -> None:
        """
        Removes User from Team.

        :param user_id: User ID in Supervisely.
        :type user_id: int
        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            user_id = 8
            team_id = 76
            api.user.remove_from_team(user_id, team_id)
        """
        response = self._api.post(
            "members.remove", {ApiField.ID: user_id, ApiField.TEAM_ID: team_id}
        )

    def update(
        self, id: int, password: Optional[str] = None, name: Optional[str] = None
    ) -> UserInfo:
        """
        Updates User info.

        :param id: User ID in Supervisely.
        :type id: int
        :param password: User password.
        :type password: str
        :param name: User name.
        :type name: str
        :return: New information about User. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`UserInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            user_info = api.user.update(8, name='Aleksey')
            print(user_info)
            # Output: [
            #     8,
            #     "alex",
            #     null,
            #     null,
            #     "Aleksey",
            #     null,
            #     21,
            #     false,
            #     "2021-03-25T08:06:03.498Z",
            #     "2020-04-17T10:24:09.077Z",
            #     "2021-03-25T08:37:17.257Z"
            # ]
        """
        data = {}
        if password is not None:
            data[ApiField.PASSWORD] = password
        if name is not None:
            data[ApiField.NAME] = name
        if len(data) == 0:
            return
        data[ApiField.ID] = id

        self._api.post("users.editInfo", data)
        return self.get_info_by_id(id)

    def change_team_role(self, user_id: int, team_id: int, role_id: int) -> None:
        """
        Changes User role in the given Team.

        :param user_id: User ID in Supervisely.
        :type user_id: int
        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param role_id: Role ID in Supervisely.
        :type role_id: int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            user_id = 8
            team_id = 64
            new_role_id = 2
            api.user.change_team_role(user_id, team_id, new_role_id)
        """
        response = self._api.post(
            "members.editInfo",
            {
                ApiField.ID: user_id,
                ApiField.TEAM_ID: team_id,
                ApiField.ROLE_ID: role_id,
            },
        )

    def get_team_members(self, team_id: int) -> List[UserInfo]:
        """
        Get list of information about Team Users.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :return: List of information about Team Users
        :rtype: :class:`List[UserInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            team_id = 9
            team_members = api.user.get_team_members(team_id)
        """
        team_members = self.get_list_all_pages(
            "members.list",
            {ApiField.TEAM_ID: team_id, ApiField.FILTER: []},
            convert_json_info_cb=self._api.user._convert_json_info,
        )
        return team_members

    def get_team_role(self, user_id: int, team_id: int) -> UserInfo:
        """
        Get Team role for given User and Team IDs.

        :param user_id: User ID in Supervisely.
        :type user_id: int
        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :return: Information about Team :class:`Role<supervisely.api.role_api.RoleApi`
        :rtype: :class:`UserInfo`
        :Usage example:

        .. code-block:: python

           import supervisely as sly

           os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
           os.environ['API_TOKEN'] = 'Your Supervisely API Token'
           api = sly.Api.from_env()

           user_id = 8
           team_id = 9
           team_role = api.user.get_team_role(user_id, team_id)
           print(team_role)
           # Output: [
           #     9,
           #     "alex",
           #     1,
           #     "admin"
           # ]
        """
        user_teams = self.get_teams(user_id)
        for member in user_teams:
            if member.id == team_id:
                return member
        return None

    def get_member_activity(
        self, team_id: int, user_id: int, progress_cb: Optional[Union[tqdm, Callable]] = None
    ) -> DataFrame:
        """
        Get User activity data.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param user_id: User ID in Supervisely.
        :type user_id: int
        :param progress_cb: Function to check progress.
        :type progress_cb: tqdm or callable, optional
        :return: Activity data as `pd.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
        :rtype: :class:`pd.DataFrame`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            activity = api.user.get_member_activity(64, 8)
            print(activity)
            # Output:
            #    userId               action                      date  ... jobId   tag tagId
            # 0        8        login_to_team  2021-03-13T08:57:26.832Z  ...  None  None  None
            # 1        8  annotation_duration  2021-03-02T13:16:23.833Z  ...  None  None  None
            # 2        8        login_to_team  2021-03-02T13:15:58.775Z  ...  None  None  None
            # 3        8        login_to_team  2021-02-06T09:47:22.999Z  ...  None  None  None
            # ................................................................................
            # 38       8     create_workspace  2021-01-04T12:25:37.916Z  ...  None  None  None
            # 39       8        login_to_team  2021-01-04T12:24:58.257Z  ...  None  None  None
            # 40       8        login_to_team  2021-01-04T12:23:43.056Z  ...  None  None  None
            # 41       8        login_to_team  2021-01-04T11:53:56.447Z  ...  None  None  None
            # [42 rows x 18 columns]
        """
        import pandas as pd

        activity = self._api.team.get_activity(
            team_id, filter_user_id=user_id, progress_cb=progress_cb
        )
        df = pd.DataFrame(activity)
        return df

    def add_to_team_by_login(self, user_login: str, team_id: int, role_id: int) -> Dict[str, int]:
        """
        Invite User to Team with given role by login.

        :param user_login: User login in Supervisely.
        :type user_login: str
        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param role_id: Role ID in Supervisely.
        :type role_id: int
        :return: Information about new User in Team
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            team_id = 13
            role_id = 2
            new_user_data = api.user.add_to_team_by_login('alex', team_id, role_id)
            print(new_user_data)
            # Output: {
            #     "userId": 8
            # }
        """
        response = self._api.post(
            "members.add",
            {
                ApiField.LOGIN: user_login,
                ApiField.TEAM_ID: team_id,
                ApiField.ROLE_ID: role_id,
            },
        )
        return response.json()

    def get_ssh_keys(self):
        """ """
        response = self._api.post("users.ssh-keys", {})
        return response.json()

    def get_my_info(self):
        response = self._api.post("users.me", {})
        return self._convert_json_info(response.json())
