Text based brand safety system
==============================

`word2vec` embeddings can be downloaded
[here](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download).

Python libaries used:
* scikit-learn
* gensim
* nltk
* numpy
* flask

Website classification
-----------

Training classifier:
```
./train_text_classifier.py
```

Classifier validation:
```
./train_text_classifier.py validate
```

Benchmark:
```
./test.py
```
This script uses test data from file `data/test_data.json`.


Running agents
-----------
Before running agents, website classifier
needs to be trained with `./train_text_classifier.py`.

Running single front agent:
```
./start_front_agent.py
```

Running worker agents:
```
./start_worker.py <front agent ip address>
```

Usage
-----------
Following commands returns vector of compability measurement for given webiste code:
```
curl -X POST localhost:5000/compat -d "<html>Some website code<\html>"
curl -X POST localhost:5000/compat -F "data=@index.
```
