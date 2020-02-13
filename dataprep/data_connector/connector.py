"""
This module contains the Connector class,
where every data fetching should begin with instantiating
the Connector class.
"""
from typing import Any, Dict, List, Optional

import pandas as pd
from jinja2 import Environment
from requests import Request, Response, Session

from .errors import RequestError
from ..errors import UnreachableError
from .implicit_database import ImplicitDatabase, ImplicitTable

from pathlib import Path
from json import load as jload

class Connector:
    """
    The main class of DataConnector.
    """

    impdb: ImplicitDatabase
    vars: Dict[str, Any]
    auth_params: Dict[str, Any]
    session: Session
    jenv: Environment

    def __init__(
        self,
        config_path: str,
        auth_params: Optional[Dict[str, Any]] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Connector

        parameters
        ----------
        config_path : str
            The path to the config. It can be hosted, e.g. "yelp", or from
            local filesystem, e.g. "file://./yelp"
        **kwargs : Dict[str, Any]
            Additional parameters
        """
        assert config_path.startswith("file://"), "Hosted config not implemented"
        # TODO download config file from server

        self.session = Session()
        self.impdb = ImplicitDatabase(config_path.lstrip("file://"))
        self.vars = kwargs
        self.auth_params = auth_params or {}
        self.jenv = Environment()
        self.config_path = config_path.lstrip("file://")

    def _fetch(
        self,
        table: ImplicitTable,
        auth_params: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> Response:
        method = table.method
        url = table.url
        req_data: Dict[str, Dict[str, Any]] = {
            "headers": {},
            "params": {},
            "cookies": {},
        }

        merged_vars = {**self.vars, **kwargs}
        if table.authorization is not None:
            table.authorization.build(req_data, auth_params or self.auth_params)

        for key in ["headers", "params", "cookies"]:
            if getattr(table, key) is not None:
                instantiated_fields = getattr(table, key).populate(
                    self.jenv, merged_vars
                )
                req_data[key].update(**instantiated_fields)
        if table.body is not None:
            # TODO: do we support binary body?
            instantiated_fields = table.body.populate(self.jenv, merged_vars)
            if table.body_ctype == "application/x-www-form-urlencoded":
                req_data["data"] = instantiated_fields
            elif table.body_ctype == "application/json":
                req_data["json"] = instantiated_fields
            else:
                raise UnreachableError

        resp: Response = self.session.send(  # type: ignore
            Request(
                method=method,
                url=url,
                headers=req_data["headers"],
                params=req_data["params"],
                json=req_data.get("json"),
                data=req_data.get("data"),
                cookies=req_data["cookies"],
            ).prepare()
        )

        if resp.status_code != 200:
            raise RequestError(status_code=resp.status_code, message=resp.text)

        return resp


    def query(
        self,
        table: str,
        auth_params: Optional[Dict[str, Any]] = None,
        **where: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Query the API to get a table.

        Parameters
        ----------
        table : str
            The table name.
        auth_params : Optional[Dict[str, Any]] = None
            The parameters for authentication. Usually the authentication parameters
            should be defined when instantiating the Connector. In case some tables have different
            authentication options, a different authentication parameter can be defined here.
            This parameter will override the one from Connector if passed.
        **where: Dict[str, Any]
            The additional parameters required for the query.

        Returns
        -------
            pd.DataFrame
        """
        assert table in self.impdb.tables, f"No such table {table} in {self.impdb.name}"

        itable = self.impdb.tables[table]

        resp = self._fetch(itable, auth_params, where)

        return itable.from_response(resp)

    @property
    def table_names(self) -> List[str]:
        """
        Return all the table names contained in this database.
        """
        return list(self.impdb.tables.keys())


    def show_schema(self, table_name: str) -> pd.DataFrame:
        # read config file
        path = Path(self.config_path)

        for table_config_path in path.iterdir():
            if not table_config_path.is_file():
                # ignore configs that are not file
                continue
            if table_config_path.suffix != ".json":
                # ifnote non json file
                continue

            if table_name != table_config_path.name.replace(".json", ''):
                continue

            #print(table_config_path.name)
            # parse json and fetch schemas
            with open(table_config_path) as f:
                table_config_content = jload(f)
                schema = table_config_content['response']['schema']
                new_schema_dict = {}
                for k in schema.keys():
                    new_schema_dict[k] = schema[k]['type']
                    #print("attribute name:", k, ", data type:", schema[k]['type'])
                return pd.DataFrame.from_dict(new_schema_dict, orient='index', columns=['data_type'])

        

    # def show_schema(self):
    #     res = self._request({"term": "hotpot", "location": "vancouver"})
    #     df = {}
    #     if self.config["response"]["ctype"] == "application/json":
    #         df = self._json(res)
    #     elif self.config["response"]["ctype"] == "application/xml":
    #         df = self._xml(res)
    #     for col in pd.DataFrame(df).columns:
    #         print(col)
