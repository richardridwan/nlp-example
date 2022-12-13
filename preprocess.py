import spacy
from spacy.training import Example
from spacy.tokens import DocBin

nlp = spacy.blank("id")
training_data = [
    ("2 hari Batuk berdahak", {"entities": [(7, 12, "SYMPTOM")]})
]
# the DocBin will store the example documents
db = DocBin()
for text, annotations in training_data:
    example = Example.from_dict(nlp.make_doc(text), annotations)
    db.add(example.reference)
db.to_disk("./train.spacy")