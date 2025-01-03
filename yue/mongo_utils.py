"""
This module contains feynman-specific functionality for interacting with MongoDB.

It's a relatively thin wrapper around pymongo, about which more is available here:
https://pymongo.readthedocs.io/en/stable/tutorial.html
"""

import logging
import os
from functools import lru_cache
from typing import Literal

from azure.core.exceptions import HttpResponseError
from azureml._restclient.models.error_response import ErrorResponseException
from pymongo import MongoClient
from azure.core.exceptions import ClientAuthenticationError

from yue.aml_utils import NotRunningInAMLError, get_secret_from_workspace
from yue.authentication import get_azure_creds
from yue.subscription import get_subscription_id_from_name

LOG = logging.getLogger(__name__)


def _get_cosmos_db_connection_string_from_azure_identity(
    subscription_id: str, resource_group: str, account_name: str, mode: Literal["r", "rw"] = "r"
) -> str:
    from azure.mgmt.cosmosdb import CosmosDBManagementClient

    if mode not in ("r", "rw"):
        raise ValueError(f"Illegal mode {mode}.")
    client = CosmosDBManagementClient(get_azure_creds(), subscription_id)
    print('resource group:',resource_group,'account_name:',account_name)
    cs_result = client.database_accounts.list_connection_strings(resource_group, account_name)
    cs_idx = {"r": 2, "rw": 0}[mode]
    return cs_result.connection_strings[cs_idx].connection_string


def _get_cosmos_db_connection_string_from_keyvault(keyvault_name: str, account_name: str) -> str:
    import azure.keyvault.secrets
    from utilities.authentication import get_azure_creds

    secret_client = azure.keyvault.secrets.SecretClient(
        vault_url=f"https://{keyvault_name}.vault.azure.net",
        credential=get_azure_creds(),
    )
    return secret_client.get_secret(f"{account_name}-connection-string").value


def _get_cosmos_db_connection_string_from_workspace(account_name: str) -> str:
    return get_secret_from_workspace(f"{account_name}-connection-string")

from typing import Union
@lru_cache
def get_cosmos_db_connection_string(
    subscription_name: str,
    resource_group: str,
    account_name: str,
    keyvault_name: str,
    subscription_id: Union[str, None] = None,
) -> str:
    """Gets Cosmos DB connection string.

    Args:
        subscription_name: the name of the Azure subscription (default: 'Molecular Dynamics')
        resource_group: the resource group name (default: 'shared_infrastructure')
        account_name: the Cosmos DB account name (default: 'msrmdscore')
        keyvault_name: the name of the Azure Key Vault that stores the connection string (default: 'msrmoldyn-vault')
        subscription_id: (optional) if provided, uses the ID instead of the subscription_name.

    Returns:
        the connection string.
    """
    print(
        "checking values: "
        + ", ".join(
            [
                f"{subscription_name=}",
                f"{resource_group=}",
                f"{account_name=}",
                f"{keyvault_name=}",
                f"{subscription_id=}",
            ]
        )
    )

    # Try retrieving from AML workspace.
    running_in_aml = False
    try:
        return _get_cosmos_db_connection_string_from_workspace(account_name)
    except (NotRunningInAMLError, ErrorResponseException) as e:
        # NotRunningInAMLError is raised if e.g., the code is run locally.
        # ErrorResponseException is raised if the code is run in Azure but AML token has expired
        # (Operation returned an invalid status code 'Unauthorized'), or secret not found in the keyvault.
        running_in_aml = isinstance(e, ErrorResponseException)
        LOG.debug(
            f"Failed to get connection string for account {account_name} from AML workspace due to error: {e}"
        )

    # Try retrieving from keyvault.
    try:
        return _get_cosmos_db_connection_string_from_keyvault(keyvault_name, account_name)
    except (ModuleNotFoundError, ClientAuthenticationError, HttpResponseError) as e:
        LOG.debug(
            f"Failed to get connection string for account {account_name} from keyvault {keyvault_name} due to error: {e}"
        )

    message = (
        f"Failed to get connection string for account {account_name} from keyvault {keyvault_name}.\n"
        "Please create a project-specific keyvault and store your connection string there.\n"
        "See: https://github.com/msr-ai4science/feynman/issues/10592"
    )

    if running_in_aml:
        # Don't use managed identity to access the connection string if running in Azure
        # to prevent flooding management plane requests.
        raise ValueError(message.replace("\n", " "))

    # Otherwise warn the user and proceed with identity-based access.
    LOG.warning("\n" + message + "\n")

    # Try retrieving through Azure identity.
    try:
        subscription_id = subscription_id or get_subscription_id_from_name(subscription_name)
        return _get_cosmos_db_connection_string_from_azure_identity(
            subscription_id,
            resource_group,
            account_name,
            mode=(
                # Read only if Azure ML online endpoints
                "r"
                if "AZUREML_MODEL_DIR" in os.environ
                else "rw"
            ),
        )
    except (ModuleNotFoundError, HttpResponseError) as e:
        LOG.debug(
            f"Failed to get connection string for account {account_name} using Azure identity due to error: {e}"
        )
        raise


@lru_cache
def get_mongo_client(**kwargs) -> MongoClient:
    """Gets a Mongo DB client for a Cosmos DB hosted in Azure.

    Keyword arguments may contain:
        subscription_name: name of the subscription (usually 'Molecular Dynamics')
        resource_group: name of the resource group (usually 'shared_infrastructure')
        account_name: database account name (e.g., 'msrmdscore')
        keyvault_name: name of the Azure key vault that stores the connection string.

    For kwargs, see `get_cosmos_db_connection_string` for more information.

    As a prerequisite, please install Azure CLI (https://docs.microsoft.com/en-us/cli/azure/)
    and sign in to azure by running `az login` in the terminal.

    Please see pymongo documentation to learn how to use the DB client:
    https://pymongo.readthedocs.io/en/stable/tutorial.html
    """
    return MongoClient(get_cosmos_db_connection_string(**kwargs))


