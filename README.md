Text based brand safety system
==============================

`word2vec` embeddings can be downloaded
[here](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download).

Python libaries used:
* scikit-learn
* gensim
* nltk
* numpy

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
Following command returns vector of compability measurement for given webiste code:
```
curl -X POST localhost:5000/compat -b "<html>Some website code<\html>"
```
