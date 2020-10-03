"""
The debiasing methods here originate from Lauscher et al. (2020). https://arxiv.org/abs/1909.06092
See also https://github.com/anlausch/DEBIE
"""

from typing import List, Dict
import numpy as np
from sklearn.decomposition import PCA
from gensim.models.keyedvectors import BaseKeyedVectors


def get_bias_direction_gbdd(equality_sets: List[tuple], vecs_norm: np.ndarray, vocab: Dict):
    Q = []
    for _,(f,m) in enumerate(equality_sets): 
        if f in vocab and m in vocab:
            Q.append(vecs_norm[vocab[f].index] - vecs_norm[vocab[m].index])
    Q = np.array(Q)
    u, sigma, v = np.linalg.svd(Q)
    v_b = v[0]  # top singular vector of Q
    return v_b


def get_pis(v_b: np.ndarray, vecs_norm: np.ndarray):
    dots = np.dot(vecs_norm, v_b)
    dots = np.reshape(dots, (len(vecs_norm), 1))
    v_b_tiled = np.tile(v_b, (len(vecs_norm), 1))
    pis = np.multiply(dots, v_b_tiled)
    return pis


def debias_direction_linear(v_b: np.ndarray, vecs_norm: np.ndarray):
    return vecs_norm - get_pis(v_b, vecs_norm)


def make_pairs(list_a: List[str],list_b: List[str]):
    pairs = [(a,b) for b in list_b for a in list_a]
    return pairs


def debias(model: BaseKeyedVectors, terms_1: List[str], terms_2: List[str]):
    pairs = make_pairs(terms_1, terms_2)  # create all possible word pairs
    
    v_b = get_bias_direction_gbdd(pairs, model.vectors_norm, model.vocab)
    utah = debias_direction_linear(v_b, model.vectors_norm)
    
    return utah


def debias_model(model: BaseKeyedVectors, method: str, terms_1: List[str], terms_2: List[str]):

    if method.lower() == "gbdd":
        vecs_gbdd = debias(model, terms_1, terms_2)
        model.vectors = vecs_gbdd
    else:
        raise ValueError("Debiasing method is not recognized.")

    return model