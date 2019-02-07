# lex-sims (for Portuguese only)
Our lexical similarity measures, as employed in http://propor2016.di.fc.ul.pt/wp-content/uploads/2016/07/ASSIN_2016_paper_4.pdf


# requirements

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

if training a new model:
- follow the main function of assin-eval_l2f.py
