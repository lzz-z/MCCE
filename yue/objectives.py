import logging
import os
import sys
from typing import Any, Callable, Dict, List

sys.path.append(os.path.join(os.environ["CONDA_PREFIX"], "share", "RDKit", "Contrib"))

import pandas as pd
from bunnet import init_bunnet
from rdkit.Chem import AllChem
from tdc import Oracle
sascorer = Oracle(name = 'SA')

from yue.az_ml_endpoints import (
    AzureMLEndpoint,
    AzureMLEndpointParams,
    log_s_params,
    reduction_potential_params,
)
from yue.globals import (
    EARTHSHOTS_AZURE_SUBSCRIPTION_NAME,
    EARTHSHOTS_AZURE_RESOURCE_GROUP,
    EARTHSHOTS_COSMOS_DB_ACCOUNT,
    EARTHSHOTS_AML_KEYVAULT_NAME,
    ORFB_MONGO_DATABASE_NAME,
)
from yue.mongo_utils import get_mongo_client
from yue.properties import PropertyModel, PropertiesDoc


class SmartsFilterset:
    """
    Identifies undesirable substructures in molecules using SMARTS queries.
    SMILES that contain an undesirable substructure are labeled `True`.
    """
    def __init__(self):
        self.smarts_filters = pd.read_csv("yue/smarts_filters.csv", index_col=False)

    def _apply_filters(self, smiles: str) -> bool:
        """
        Apply filters to a single SMILES.

        smiles (str): the SMILES to label
        """
        mol = AllChem.MolFromSmiles(smiles)
        for filter in self.smarts_filters.smarts:
            filter_pattern = AllChem.MolFromSmarts(filter)
            if mol.HasSubstructMatch(filter_pattern):
                return True
        return False

    def apply_filters(self, input_data: List[str]) -> List[bool]:
        """
        Apply SMARTS filters to a list of SMILES.
        
        input_data (List[str]): a set of SMILES to be labeled
        """
        return list(map(self._apply_filters, input_data))


class EndpointObserver:
    """
    Collects necessary methods to retrieve labels from an AML endpoint given fingerprints.

    endpoint_params (AzureMLEndpointaParams): parameter set for the target endpoint
    archive_func (Callable[[str, Any], None]): a function that, given a str and a result, stores the result in some way
    batch_size (int): the number of SMILES fed to the endpoint in a batch, max 32
    retries (int): the number of times to retry the endpoint before raising an error
    """
    def __init__(
        self,
        endpoint_params: AzureMLEndpointParams,
        archive_func: Callable[[str, Any], None],
        batch_size: int = 32,
        retries: int = 2,
    ):
        if batch_size > 32:
            raise ValueError("Endpoint supports a max batch size of 32.")

        self._archive_func = archive_func
        self._batch_size = batch_size

        self._target_endpoint = AzureMLEndpoint(**endpoint_params)
        self._target_endpoint._retries = retries
        self._target_endpoint._enforce_sucess = True

    def observe(
        self,
        input_data: List[str],
    ) -> List[float]:
        """
        The observer function handles queries to the AML endpoint.

        input_data (List[str]): a set of SMILES to be labeled
        """
        # query endpoint
        cursor = 0
        results = []
        while cursor < len(input_data):
            query_dict = {"smiles": input_data[cursor : cursor + self._batch_size]}

            # retrieve labels from the endpoint
            response = self._target_endpoint.query(query_dict)
            results += [result["prediction"] for result in response]
            for smiles, result in zip(input_data[cursor : cursor + self._batch_size], response):
                # archive results in PropertiesDoc collection
                self._archive_func(smiles, result["prediction"])

            logging.info("Collected batch of redox values.")
            cursor += self._batch_size
        return results


class SqueezePotential:
    """
    Provides linearly increasing loss for distance from `target_pot` within the bounds of [`min_pot`, `max_pot`].
    Provides extra-linearly increasing loss for distance outside `min_pot` + `max_pot`.

    target_pot (float): the target potential
    min_pot (float): the minimum potential in the linear range
    max_pot (float): the maximum potential in the linear range
    """
    def __init__(
        self,
        target_pot: float = -1.2,
        min_pot: float = -1.6,
        max_pot: float = -0.8,
    ):
        self.target_pot = target_pot
        self.min_pot = min_pot
        self.max_pot = max_pot

    def evaluate(self, pot: float) -> float:
        base_loss = abs(pot - self.target_pot)
        if pot < self.min_pot:
            additional_loss = (pot - self.min_pot) ** 2
        elif pot > self.max_pot:
            additional_loss = (pot - self.max_pot) ** 2
        else:
            additional_loss = 0
        return base_loss + additional_loss


def connect_bunnet() -> None:
    """
    connect bunnet with the earthshots orfb database
    """
    client = get_mongo_client(
        subscription_name=EARTHSHOTS_AZURE_SUBSCRIPTION_NAME,
        resource_group=EARTHSHOTS_AZURE_RESOURCE_GROUP,
        account_name=EARTHSHOTS_COSMOS_DB_ACCOUNT,
        keyvault_name=EARTHSHOTS_AML_KEYVAULT_NAME,
    )

    init_bunnet(
        database=client[ORFB_MONGO_DATABASE_NAME],
        document_models=[PropertiesDoc],  # type: ignore
    )
    print("Connected bunnet", flush=True)


def log_s_to_properties_db(smiles: str, log_s: float):
    """
    Adds a logS label generated by the endpoint to the mongo database as/under a PropertiesDoc.

    smiles (str): SMILES string
    log_s (float): the predicted logS solubility label
    """
    property_model: PropertyModel = PropertyModel(
        name="logS",
        value=log_s,
        units="M",
        method="ml",
        origin="logS_endpoint",
    )
    if properties_doc := PropertiesDoc.find_one(PropertiesDoc.smiles == smiles).run():  # type: ignore
        if properties_doc.properties.get("log_s"):
            properties_doc.update({"$push": {"log_s": property_model}})
        else:
            properties_doc.update({"$set": {"log_s": [property_model]}})
    else:
        new_properties_doc = PropertiesDoc(
            smiles=smiles,
            properties={"logS": [property_model]},# origin: log_s
        )
        new_properties_doc.insert()


def red_pot_to_properties_db(smiles: str, red_pot: float):
    """
    Adds a reduction potential label generated by the endpoint to the mongo database as/under a PropertiesDoc.

    smiles (str): SMILES string
    red_pot (float): the predicted reduction potential
    """
    property_model: PropertyModel = PropertyModel(
        name="reduction_potential",
        value=red_pot,
        units="V",
        method="ml",
        origin="reduction_potential_endpoint",
    )

    if properties_doc := PropertiesDoc.find_one(PropertiesDoc.smiles == smiles).run():  # type: ignore
        if properties_doc.properties.get("reduction_potential"):
            properties_doc.update({"$push": {"reduction_potential": property_model}})
        else:
            properties_doc.update({"$set": {"reduction_potential": [property_model]}})
    else:
        new_properties_doc = PropertiesDoc(
            smiles=smiles,
            properties={"reduction_potential": [property_model]},
        )
        new_properties_doc.insert()

    
def calculate_sa_score(smiles: str) -> float:
    return sascorer(smiles)
        
    
class AnolyteLabeler:
    """
    A set of classes for labeling ORFB anolyte candidates.
    """
    def __init__(self):
        connect_bunnet()

        self.filterset = SmartsFilterset()
        
        self.log_s_endpoint = EndpointObserver(
            endpoint_params=log_s_params,
            archive_func=log_s_to_properties_db,
        )
        
        self.red_pot_endpoint = EndpointObserver(
            endpoint_params=reduction_potential_params,
            archive_func=red_pot_to_properties_db,
        )
            
    def get_filter_results(self,smiles):
        filter_labels = self.filterset.apply_filters(smiles)
        filter_labels = [float(i) for i in filter_labels]
        return filter_labels

    def label_smiles(self, input_data: List[str]) -> List[Dict[str, float]]:
        """
        Label a list of SMILES strings.

        input_data (List[str]): a list of SMILES to label
        """
        filter_labels = self.filterset.apply_filters(input_data)
        log_s_labels = self.log_s_endpoint.observe(input_data)
        red_pot_labels = self.red_pot_endpoint.observe(input_data)
        sa_score_labels = calculate_sa_score(input_data)
        labels = {
            "smartsFilter":[float(i) for i in filter_labels] ,
            "logs": log_s_labels,
            "reductionPotential": red_pot_labels,
            "sa": sa_score_labels,
        }
        '''labels = [
            {
                "smarts_filter": float(filter_label),
                "log_s": log_s_label,
                "reduction_potential": red_pot_label,
                "sa_score": sa_score,
            }
            for filter_label, log_s_label, red_pot_label, sa_score
            in zip(filter_labels, log_s_labels, red_pot_labels, sa_score_labels)
        ]'''

        return labels