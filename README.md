# Patient Symptom NER

NER API Example on how to train pre-processed datasets into a Medical Machine Learning Model using Spacy as the ML Framework with FastAPI for API.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install spacy.

```bash
pip install fastapi
pip install "uvicorn[standard]"
pip install spacy
```

## Usage

```python
uvicorn main:app --reload

...
INFO:     Finished server process [95060]
INFO:     Started server process [95148]
INFO:     Waiting for application startup.
INFO:     Application startup complete.

...
Go to http://127.0.0.1:8000/docs#/
Try endpoint /train until it says success
Try endpoint /extraction with your desired sentence
```


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)