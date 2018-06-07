Text based brand safety system
==============================

Word embeddings can be downloaded
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

Running agents
-----------
Running single front agent
```
python agents/front_agent.py
```

Usage
-----------
Following command returns vector of compability measurement for given webiste code:
```
curl -X POST localhost:8060/compat -b "<html>Some website code<\html>"
```
