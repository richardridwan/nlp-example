import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Yesterday my body temperature was 37 degrees celcius, now it is 39 degrees.")

displacy.serve(doc, style="ent")

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)