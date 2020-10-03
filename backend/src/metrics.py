from typing import List, Union
import numpy as np
import wefe
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel
from wefe.metrics.WEAT import WEAT
from wefe.datasets.datasets import load_weat
from gensim.models.keyedvectors import BaseKeyedVectors


def get_terms(term_type: str):
    word_sets = load_weat("en")
    
    if term_type in word_sets:
        return word_sets[term_type]
    else:
        raise ValueError("Term type is not recognized.")


def run_WEAT(model: BaseKeyedVectors, property_terms_1: List[str], property_terms_2: List[str], attribute_terms_1: List[str], attribute_terms_2: List[str]):
    emb_model = WordEmbeddingModel(model)

    weat = WEAT()
    
    query = Query([property_terms_1, property_terms_2], [attribute_terms_1, attribute_terms_2])

    result = weat.run_query(query, emb_model)

    return result["result"]


def get_metric(metric_name: str, model: BaseKeyedVectors, property_terms_1: List[str], property_terms_2: List[str], attribute_terms_1: List[str], attribute_terms_2: List[str]):
    if metric_name == "WEAT": 
        result = run_WEAT(model, property_terms_1, property_terms_2, attribute_terms_1, attribute_terms_2)
    else:
        raise ValueError("Metric is not recognized.")

    return result
