# lex-sims (for Portuguese only)
Our lexical similarity measures, as employed in http://propor2016.di.fc.ul.pt/wp-content/uploads/2016/07/ASSIN_2016_paper_4.pdf


# requirements

(I recommend Linux and the PyCharm IDE (Community edition): https://www.jetbrains.com/pycharm/)

python 2.7

pip install numpy scipy scikit-learn nltk web2py_utils pyter jaro_winkler

download Brown clusters from https://drive.google.com/open?id=15YLgY7lgXXieb3nJyYsnXu6sDmZOx5b-
and place next to .py files

- if using the web service:
pip install flask flask_restful


# usage guide

if just to detect the semantic similarity between 2 sentences (from 1 to 5):
- download previously trained models from https://drive.google.com/open?id=1oGFrKzschiPwWen0tzUBGz4IRcdnmNVc
- invoke following code:

```
# RUN ONCE
with open("cacheric/kridgePoly_proporPT", 'rb') as f1:
    clf1 = pickle.load(f1)

cria_dict('brown-clusters-pt.txt')

# RUN ALWAYS
encPair = unicode('TEXT: ' + sys.argv[1] + '\nHYPOTHESIS:' + sys.argv[2])
cria_tf_idf([encPair])
featVec = similarity_features(encPair, 1, 1, 1, 1, '', [], [])

print (str(clf1.predict(np.array(featVec).reshape(1, -1))[0]))
```
or
- use the web service described in function start_service of file assin-eval_l2f.py

if training a new model:
- follow the main function of assin-eval_l2f.py

or

- for a set of pairs of sentences (corpusPairs), obtain a set of feature sets (trainset) by:
```
pairs = [unicode('TEXT: ' + sentenceA + '\nHYPOTHESIS:' + sentenceB) for (sentenceA, sentenceB) in corpusPairs]

cria_dict('brown-clusters-pt.txt')
cria_tf_idf(pairs, clusterfilename='tfidf_YOUR_DATASET_ID_HERE')

trainset = [similarity_features(d, 1, 1, 1, 1, '', [], []) for d in pairs]
```

function similarity_features is defined in MSC.py and describes all features computed for a given pair. 


# hints for errors

some features may require the instalation of nltk specific data (such as stopwords) by https://www.nltk.org/data.html

