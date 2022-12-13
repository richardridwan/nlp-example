from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from typing import List
import pickle
import random

from spacy.util import minibatch, compounding
from spacy.training import Example
from spacy import load
from pathlib import Path

import spacy
import scispacy

nlp = spacy.load("en_core_sci_sm")

app = FastAPI()

class InputQuery(BaseModel):
    sentence: str

training_data = [
    ('Batuk berdahak, Mual dan Hidung Meler. Sudah lama batuk kering juga. Curiga nya sih Influenza, tapi bisa jadi Covid juga ga sih', {'entities': [(0, 14, 'SYMPTOM'), (16, 20, 'SYMPTOM'), (25, 31, 'ORGAN'), (50, 62, 'SYMPTOM'), (84, 93, 'DISEASE'), (110, 115, 'DISEASE')]}),
    ('Batuk, Pilek dan Demam', {'entities': [(0, 5, 'SYMPTOM'), (7, 12, 'SYMPTOM'), (17, 22, 'SYMPTOM')]}),
    ('badan lemas, batuk dan pilek', {'entities': [(0, 11, 'SYMPTOM'), (14, 18, 'SYMPTOM'), (23, 28, 'SYMPTOM')]}),
    ('Sakit kepala', {'entities': [(0, 12, 'SYMPTOM')]})
]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/train")
async def train():
    med_terms = training_data
    
    nlp = spacy.blank("id")
    nlp.add_pipe('ner')
    nlp.begin_training()

    ner = nlp.get_pipe("ner")

    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    for _, annotations in med_terms:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
            break

    #train the model using 
    with nlp.disable_pipes(*unaffected_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(20):
            print("Statring iteration " + str(itn))
            random.shuffle(med_terms)
            losses = {}
            for text, annotations in med_terms:
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example])
            print(losses)
    
    output_dir = Path('./output/med_terms_2022')
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)

    return "successfully trained the model!"

@app.post("/extraction")
async def extraction(query: InputQuery):
    model_dir = Path('./output/med_terms_2022')
    id_nlp = spacy.load(model_dir)  
    doc = id_nlp(query.sentence)

    return [(ent.text, ent.label_) for ent in doc.ents]