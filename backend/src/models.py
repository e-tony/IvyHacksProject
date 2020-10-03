from typing import List, Dict
import numpy as np
from sklearn.decomposition import PCA
from gensim.models.keyedvectors import BaseKeyedVectors


def get_3D_coordinates(model: BaseKeyedVectors, terms: List[list]):
    """
    Compute 3D coordinates for each word in a list of list of terms. 
    The coordinates are computed by reducing the vector space of the word embeddings 
    for the terms to 3 principal components via PCA. 
    """
    if len(terms) != 3 or type(terms[0]) is not list:
        raise ValueError("'terms' must have a length of 3.")
    
    # combine term lists 
    if terms == None:
        raise ValueError(f"'terms' cannot be of type: {type(terms)}")
    else:
        terms_1, terms_2, terms_3 = terms  # unroll terms
        terms = terms_1+terms_2+terms_3  # flatten

    # create an array of word vectors
    word_vectors = np.array([model[t] for t in terms])

    # compute the first 3 principal components
    three_dim = PCA().fit_transform(word_vectors)[:,:3]

    res = {}

    for term, (x,y,z) in zip(terms, three_dim):
        res[term] = [str(x),str(y),str(z)]

    return res