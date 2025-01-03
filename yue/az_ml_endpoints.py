import json
import logging
import os
import ssl
import time
import typing as ty
import urllib.request
from urllib.error import HTTPError

from typeguard import check_type

from yue.api_keys import (
    AZURE_LOG_S_ML_ENDPOINT_API_KEY,
    AZURE_RED_POT_ML_ENDPOINT_API_KEY,
)


def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if (
        allowed
        and not os.environ.get("PYTHONHTTPSVERIFY", "")
        and getattr(ssl, "_create_unverified_context", None)
    ):
        ssl._create_default_https_context = ssl._create_unverified_context


allowSelfSignedHttps(
    True
)  # this line is needed if you use self-signed certificate in your scoring service.


class AzureMLEndpointParams(ty.TypedDict):
    """
    A parameter set for use with AzureMLEndpoint.
    """

    url: str
    api_key: str
    deployment: str
    query_type: ty.Type
    max_batch_size: int
    retries: int
    enforce_success: bool


class AzureMLEndpoint:
    def __init__(
        self,
        url: str,
        api_key: str,
        deployment: str,
        query_type: ty.Type,
        max_batch_size: int,
        retries: int,
        enforce_success: bool,
    ):
        """
        A wrapper class to smooth inference calls to AzureML endpoints.

        url (str): the scoring URL of the API endpoint
        api_key (str): the primary/secondary key for the endpoint
        deployment (str): the deployment of the API endpoint
        query_type (Type): the expected type signature of the query
        max_batch_size (int): the maximum number of inputs allowed in a single inference call
        retries (int): the number of time the endpoint call will be retried before giving up
        enforce_success (bool): if True, an error will be thrown if all inference calls are not successful
        """
        self._url = url
        self._api_key = api_key
        self._deployment = deployment
        self._query_type = query_type
        self._max_batch_size = max_batch_size
        self._retries = retries
        self._enforce_sucess = enforce_success

    def query(
        self,
        data: ty.Dict[str, ty.Any],
    ) -> ty.Dict:
        """
        Query the endpoint.

        data (Dict[str, Any]): the body of the query to be made, JSON compatible
        """
        # type check data
        try:
            # returns None if pass, raises TypeError if fail
            check_type(data, self._query_type)
        except TypeError as e:
            raise ValueError(f"Data does not conform to expected type signature: {e}")

        # check that max_batch_size is obeyed where relevant
        for value in iter(data.values()):
            if isinstance(value, list):
                assert (
                    len(value) <= self._max_batch_size
                ), f"Max batch size is {self._max_batch_size}."

        body = str.encode(json.dumps(data))
        logging.debug(f"Body JSON for endpoint '{self._url}' is: '{json.dumps(data)}'.")

        # The azureml-model-deployment header will force the request to go to a specific deployment.
        # Remove this header to have the request observe the endpoint traffic rules.
        headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self._api_key),
            "azureml-model-deployment": self._deployment,
        }
        logging.debug(f"Headers for endpoint '{self._url}' are: '{headers}'.")

        req = urllib.request.Request(self._url, body, headers)

        # query endpoint
        attempts = 0
        results = None
        while True:
            try:
                logging.debug(f"Making request to Azure ML API endpoint '{self._url}'...")
                response = urllib.request.urlopen(req)
                logging.debug(f"Response recieved from Azure ML API endpoint '{self._url}'.")
                results = json.loads(response.read())
                logging.debug(f"Response from Azure ML API endpoint '{self._url}' is '{results}'.")
            except HTTPError as error:
                logging.error("The request failed with status code: " + str(error.code))
                # Print the headers - they include the request ID and the timestamp,
                # which are useful for debugging the failure.
                logging.error(error.info())
                logging.error(error.read().decode("utf8", "ignore"))

            if results:
                break
            else:
                attempts += 1
                if attempts > self._retries:
                    raise RuntimeError("Failed to retrieve a response from Azure ML API endpoint.")
                else:
                    logging.info(f"Retry number {attempts} of {self._retries}.")
                    time.sleep(min([5**attempts, 600]))

        # check results for successful inference
        for result in results:
            if result["status"] != "succeeded":
                if self._enforce_sucess:
                    raise RuntimeError(f"Inference failed: {result}")
                else:
                    logging.info(f"Inference failed: {result}")

        return results


log_s_params = AzureMLEndpointParams(
    url="https://graphomer-solubility.westus3.inference.ml.azure.com/score",
    api_key=AZURE_LOG_S_ML_ENDPOINT_API_KEY,
    deployment="crash2-pipelinev1",
    query_type=ty.Dict[ty.Literal["smiles"], ty.List[str]],
    max_batch_size=32,
    retries=3,
    enforce_success=True,
)


reduction_potential_params = AzureMLEndpointParams(
    url="https://graphormer-reduction.southcentralus.inference.ml.azure.com/score",
    api_key=AZURE_RED_POT_ML_ENDPOINT_API_KEY,
    deployment="emerald-pipelinev2",
    query_type=ty.Dict[ty.Literal["smiles"], ty.List[str]],
    max_batch_size=32,
    retries=3,
    enforce_success=True,
)
