Text based brand safety system
==============================

The system provides brand safety functionality --
answer queries with a html document as an argument.
The answer is vector of advertisement types cmpatibilities.
These values can be thresholded for discrimination of incompatible
ads or for selecting best ads for the given website.

Requirements
-----------

Python libaries:
* scikit-learn
* gensim
* nltk
* numpy
* flask

For website classification `word2vec` embeddings are needed.
They can be downloaded
[here](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download).

Website classification
-----------

### Training classifier ###
```
./train_text_classifier.py
```
This script will generate output file with trained model - `classifier_data`.

### Classifier validation ###
```
./train_text_classifier.py validate
```
This script require `classifier_data` file.

### Benchmark ###
```
./test.py
```
This script uses test data from file `data/test_data.json`,
and require `classifier_data` file.


Running agents
-----------
Before running agents, website classifier
needs to be trained with `./train_text_classifier.py`.

### Running single front agent ###
```
./start_front_agent.py
```
In current version, only one such agent can be running in the system.

### Running worker agents: ###
```
./start_worker.py <front agent ip address>
```
These agents can be deployed at any number on different machines
(note that one such agent allocates word vectors that take above 3GB of RAM).

Usage
-----------
Following commands examples return vector of compability measurement for given webiste code:
```
curl -X POST localhost:5000/compat -d "<html>Some website code<\html>"
curl -X POST localhost:5000/compat -d "Or just plain text"
curl -X POST localhost:5000/compat -F "data=@index.html"
```
