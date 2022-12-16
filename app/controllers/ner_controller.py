from fastapi import APIRouter
from app.schemas import common_schema
import spacy
import random

from spacy.training import Example
from pathlib import Path

router = APIRouter()
local_prefix = "/users/"

training_data = [
    ('Batuk berdahak, Mual dan Hidung Meler. Sudah lama batuk kering juga. Curiga nya sih Influenza, tapi bisa jadi Covid juga ga sih', {'entities': [(0, 14, 'SYMPTOM'), (16, 20, 'SYMPTOM'), (25, 31, 'ORGAN'), (50, 62, 'SYMPTOM'), (84, 93, 'DISEASE'), (110, 115, 'DISEASE')]}),
    ('Batuk, Pilek dan Demam', {'entities': [(0, 5, 'SYMPTOM'), (7, 12, 'SYMPTOM'), (17, 22, 'SYMPTOM')]}),
    ('badan lemas, batuk dan pilek', {'entities': [(0, 11, 'SYMPTOM'), (14, 18, 'SYMPTOM'), (23, 28, 'SYMPTOM')]}),
    ('Sakit kepala', {'entities': [(0, 12, 'SYMPTOM')]}),
    ('Kuping gatal', {'entities': [(0, 6, 'ORGAN'), (7, 12, 'SYMPTOM')]}),
    ('Semenjak 10 hari yang lalu, perut sakit', {'entities': [(0, 8, 'TIME'), (28, 39, 'SYMPTOM')]}),
    ('Nyeri perut dan nyeri tenggorokan', {'entities': [(0, 8, 'SYMPTOM'), (22, 33, 'ORGAN')]}),
    ('Panas 37 derajat celcius', {'entities': [(0, 5, 'SYMPTOM'), (9, 24, 'MEASUREMENT')]}),
]

class NerController():

    @router.post("/train", response_model=common_schema.BaseResponse)
    async def train(iteration: int = 20):
        med_terms = training_data
        
        #initiate new blank spacy ner file
        nlp = spacy.blank("id")
        nlp.add_pipe('ner')
        nlp.begin_training()

        ner = nlp.get_pipe("ner")

        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

        unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

        #join label entities with specified words
        for _, annotations in med_terms:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])
                break

        #train the model using 
        with nlp.disable_pipes(*unaffected_pipes):  # only train NER
            optimizer = nlp.begin_training()
            for itn in range(iteration):
                print("starting iteration " + str(itn))
                random.shuffle(med_terms)
                losses = {}
                for text, annotations in med_terms:
                    example = Example.from_dict(nlp.make_doc(text), annotations)
                    nlp.update([example])
                print(losses)
        
        output_dir = Path('./output/med_terms_2022')
        nlp.to_disk(output_dir)
        print("saved model to", output_dir)

        return {
            'message': "successfully trained the model!"
        }


@router.post("/extraction")
async def extraction(query: common_schema.InputQuery):
    model_dir = Path('./output/med_terms_2022')
    id_nlp = spacy.load(model_dir)
    doc = id_nlp(query.sentence)

    return {
        'message': "successfully extracted the model!",
        'data': [(ent.text, ent.label_) for ent in doc.ents]
    }