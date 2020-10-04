from typing import List, Union
import pathlib
import numpy as np
from gensim.models import fasttext as FT
import copy
import os
import json

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.responses import Response
from pydantic import BaseModel

from src.models import get_3D_coordinates
from src.debias_Lauscher2020 import debias_model
from src.metrics import get_terms, get_metric


# LOCAL_ROOT = pathlib.Path(os.path.abspath('')).parent


# fasttext = FT.load_facebook_vectors(LOCAL_ROOT / "models" / "cc.en.25.bin")
fasttext = FT.load_facebook_vectors("/app/models/cc.en.25.bin")
fasttext_deb = copy.deepcopy(fasttext)
fasttext_deb.init_sims(replace=True)
lookup = {"fastText": fasttext, "fastText debiased": fasttext_deb}


app = FastAPI(title="FastText model.", description="You can visualize, debias, and evaluate the fastText model here.", version="0.1.0")


@app.post("/get_3D_coordinates/{model_name}")
def get_coordinates(model_name: str, property_terms_1: List[str], property_terms_2: List[str], neutral_terms: List[str]):
    model = lookup[model_name]
    terms = [property_terms_1, property_terms_2, neutral_terms]
    coordinates = get_3D_coordinates(model, terms)
    return Response(json.dumps({"coordinates": coordinates}))


@app.post("/debias_model/{model_name}")
def run_debias(model_name: str, debiasing_method: str, terms_1: List[str], terms_2: List[str]):
    model = lookup[model_name+" debiased"]
    lookup[model_name+" debiased"] = debias_model(model, debiasing_method, terms_1, terms_2)

    # debias only once TODO
    return JSONResponse(status_code=200)


@app.post("/terms/{term}")
def terms(term: str):
    return Response(json.dumps({"terms": get_terms(term)}))


@app.post("/get_metric/{model_name}")
def evaluate_on_metric(model_name: str, metric_name: str, property_terms_1: List[str], property_terms_2: List[str], attribute_terms_1: List[str], attribute_terms_2: List[str]):
    model = lookup[model_name]
    result = get_metric(metric_name, model, property_terms_1, property_terms_2, attribute_terms_1, attribute_terms_2)
    return Response(json.dumps({"model_name": model_name, "metric_name": metric_name, "result": str(result)}))
