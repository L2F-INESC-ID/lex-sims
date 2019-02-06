# -*- coding: utf-8 -*-
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier

from string_features import *
from text_manipulation import *
from representation import *
from rte_features import *
from ta_sum import *
from numeric_features import *
from sklearn import datasets
from sklearn import model_selection
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_selection import RFECV
from sklearn.svm import *
from sklearn import metrics
import numpy as np
import os, sys, codecs
# import matplotlib.pyplot as plt
# from myutils import *

# lcnttot = 0

def similarity_features(par, baseline=None, rte=None, ta_sum=None, numeric=None,
                        outfnameprefx='', plines=[], plinesSemsim=[]):
    # global lcnttot
    # lcnttot += 1
    # print str(lcnttot) + " ",

    parbak=par
    sim_features = []

    # dump features in svm light format
    #
    # fname = outfnameprefx + '.txt'
    # with open(fname, "a") as text_file:
    #     for i in range(0,len(sim_features)):
    #         text_file.write(str(i) + ':' + str(sim_features[i]) + '\t')
    #
    #     partowrite = parbak.replace('TEXT: ', '').replace(' HYPOTHESIS: ','\t')
    #     text_file.write(partowrite.encode('utf-8'))

    # old, (still) for msrp
    # li = parbak.replace('TEXT: ', '').replace(' HYPOTHESIS: ','\t').split('\t')
    # s1 = preprocSentence(li[0].encode('utf-8'))
    # s2 = preprocSentence(li[1][:-2].encode('utf-8'))

    # new, for OLI
    # if plines or plinesSemsim:
    #     li = parbak.replace('TEXT: ', '').replace('HYPOTHESIS: ', '\t').split('\t')
    #     s1 = preprocSentence(li[0].replace('\r', '').replace('\n', '').encode('utf-8'))
    #     s2 = preprocSentence(li[1].replace('\r', '').replace('\n', '').encode('utf-8'))
    #
    # # SEMILAR from CSV
    # if plines:
    #     matchfound=0
    #     for exline in plines:
    #         if preprocSentence(exline[0].encode('utf-8')) == s1 \
    #                 and preprocSentence(exline[1].encode('utf-8')) == s2:
    #             matchfound = 1
    #             for i in exline[2]:
    #                 fval = float(i)
    #                 if i == 'NaN':
    #                     fval = float(0)
    #
    #                 sim_features.append(fval)
    #
    #             break
    #
    #     if not matchfound:
    #         print 'fault: ' + parbak
    #
    #     if len(sim_features) != 5:
    #         print 'wrong match: ' + parbak
    #
    # # SEMSIM from CSV
    # if plinesSemsim:
    #     matchfound=0
    #     for exline in plinesSemsim:
    #         if preprocSentence(exline[0].encode('utf-8')) == s1 \
    #                 and preprocSentence(exline[1].encode('utf-8'))[:-1] == s2:
    #             matchfound = 1
    #             for i in exline[2]:
    #                 fval = float(i)
    #                 if i == 'NaN':
    #                     fval = float(0)
    #
    #                 sim_features.append(fval)
    #
    #             break
    #
    #     if not matchfound:
    #         print 'fault semsim: ' + parbak

    par = remove_ponctuation(par)
    text, hypo = split_par(par)

    text_trigram = character_trigram(text)
    hypo_trigram = character_trigram(hypo)
    lexical_trigram = String_features(text_trigram, hypo_trigram)
    rte_trigram = RTE_features(text_trigram,hypo_trigram)
    ta_e_sum_trigram = TA_SUM(text_trigram, hypo_trigram)

    text_cluster = cluster(text)
    hypo_cluster = cluster(hypo)
    lexical_cluster = String_features(text_cluster, hypo_cluster)
    rte_cluster = RTE_features(text_cluster,hypo_cluster)
    ta_e_sum_cluster = TA_SUM(text_cluster, hypo_cluster)

    text_lower_case = lower_case(text)
    hypo_lower_case = lower_case(hypo)
    lexical_lower_case = String_features(text_lower_case, hypo_lower_case)
    rte_lower_case = RTE_features(text_lower_case,hypo_lower_case)
    ta_e_sum_lower_case = TA_SUM(text_lower_case, hypo_lower_case)
    num_lower_case = Numeric_features(text_lower_case, hypo_lower_case)

    text_stem_lowered = stem_lowered(text)
    hypo_stem_lowered = stem_lowered(hypo)
    lexical_stem_lowered = String_features(text_stem_lowered, hypo_stem_lowered)
    rte_stem_lowered = RTE_features(text_stem_lowered,hypo_stem_lowered)
    ta_e_sum_stem_lowered = TA_SUM(text_stem_lowered, hypo_stem_lowered)
    num_stem_lowered = Numeric_features(text_stem_lowered, hypo_stem_lowered)

    text_metaphone = go_double_metaphone(text)
    hypo_metaphone = go_double_metaphone(hypo)
    lexical_metaphone = String_features(text_metaphone, hypo_metaphone)
    rte_metaphone = RTE_features(text_metaphone,hypo_metaphone)
    ta_e_sum_metaphone = TA_SUM(text_metaphone, hypo_metaphone)

    lexical = String_features(text, hypo)
    rte = RTE_features(text,hypo)
    ta_e_sum = TA_SUM(text, hypo)
    num = Numeric_features(text, hypo)

    if baseline is not None:
        sim_features.append(lexical.lcs)
        sim_features.append(lexical_lower_case.lcs)
        sim_features.append(lexical_stem_lowered.lcs)
        sim_features.append(lexical_cluster.lcs)
        sim_features.append(lexical_metaphone.lcs)

        sim_features.append(lexical.edit_distance)
        sim_features.append(lexical_lower_case.edit_distance)
        sim_features.append(lexical_stem_lowered.edit_distance)
        sim_features.append(lexical_cluster.edit_distance)
        sim_features.append(lexical_metaphone.edit_distance)

        sim_features.append(lexical.cosine)
        sim_features.append(lexical_lower_case.cosine)
        sim_features.append(lexical_stem_lowered.cosine)
        sim_features.append(lexical_cluster.cosine)
        sim_features.append(lexical_metaphone.cosine)
        sim_features.append(lexical_trigram.cosine)

        sim_features.append(lexical.len)
        sim_features.append(lexical_lower_case.len)
        sim_features.append(lexical_stem_lowered.len)
        sim_features.append(lexical_cluster.len)
        sim_features.append(lexical_metaphone.len)

        sim_features.append(lexical.minimo)
        sim_features.append(lexical_lower_case.minimo)
        sim_features.append(lexical_stem_lowered.minimo)
        sim_features.append(lexical_cluster.minimo)
        sim_features.append(lexical_metaphone.minimo)

        sim_features.append(lexical.maximo)
        sim_features.append(lexical_lower_case.maximo)
        sim_features.append(lexical_stem_lowered.maximo)
        sim_features.append(lexical_cluster.maximo)
        sim_features.append(lexical_metaphone.maximo)

        sim_features.append(lexical.jaccard)
        sim_features.append(lexical_lower_case.jaccard)
        sim_features.append(lexical_stem_lowered.jaccard)
        sim_features.append(lexical_cluster.jaccard)
        sim_features.append(lexical_metaphone.jaccard)
        sim_features.append(lexical_trigram.jaccard)

        sim_features.append(lexical.tfidf)
        sim_features.append(lexical_lower_case.tfidf)
        sim_features.append(lexical_stem_lowered.tfidf)

    if rte is not None:
        sim_features.append(rte.overlap('ne'))
        sim_features.append(rte_lower_case.overlap('ne'))
        sim_features.append(rte_stem_lowered.overlap('ne'))
        sim_features.append(rte_cluster.overlap('ne'))
        sim_features.append(rte_metaphone.overlap('ne'))
        sim_features.append(rte_trigram.overlap('ne'))

        sim_features.append(rte._racio_neg)
        sim_features.append(rte_lower_case._racio_neg)
        sim_features.append(rte_stem_lowered._racio_neg)
        sim_features.append(rte_cluster._racio_neg)
        sim_features.append(rte_metaphone._racio_neg)
        sim_features.append(rte_trigram._racio_neg)

        sim_features.append(rte._racio_modal)
        sim_features.append(rte_lower_case._racio_modal)
        sim_features.append(rte_stem_lowered._racio_modal)
        sim_features.append(rte_cluster._racio_modal)
        sim_features.append(rte_metaphone._racio_modal)
        sim_features.append(rte_trigram._racio_modal)

    if ta_sum is not None:
        sim_features.append(ta_e_sum.ter)
        sim_features.append(ta_e_sum_lower_case.ter)
        sim_features.append(ta_e_sum_stem_lowered.ter)
        sim_features.append(ta_e_sum_cluster.ter)
        sim_features.append(ta_e_sum_metaphone.ter)

        sim_features.append(ta_e_sum.ncd)
        sim_features.append(ta_e_sum_lower_case.ncd)
        sim_features.append(ta_e_sum_stem_lowered.ncd)
        sim_features.append(ta_e_sum_cluster.ncd)
        sim_features.append(ta_e_sum_metaphone.ncd)

        sim_features.append(ta_e_sum.met)
        sim_features.append(ta_e_sum_lower_case.met)
        sim_features.append(ta_e_sum_stem_lowered.met)
        sim_features.append(ta_e_sum_cluster.met)
        sim_features.append(ta_e_sum_metaphone.met)

        sim_features.append(ta_e_sum.rouge_n)
        sim_features.append(ta_e_sum_lower_case.rouge_n)
        sim_features.append(ta_e_sum_stem_lowered.rouge_n)
        sim_features.append(ta_e_sum_cluster.rouge_n)
        sim_features.append(ta_e_sum_metaphone.rouge_n)

        sim_features.append(ta_e_sum.rouge_l)
        sim_features.append(ta_e_sum_lower_case.rouge_l)
        sim_features.append(ta_e_sum_stem_lowered.rouge_l)
        sim_features.append(ta_e_sum_cluster.rouge_l)
        sim_features.append(ta_e_sum_metaphone.rouge_l)

        sim_features.append(ta_e_sum.rouge_s)
        sim_features.append(ta_e_sum_lower_case.rouge_s)
        sim_features.append(ta_e_sum_stem_lowered.rouge_s)
        sim_features.append(ta_e_sum_cluster.rouge_s)
        sim_features.append(ta_e_sum_metaphone.rouge_s)

    if numeric is not None:
        sim_features.append(num.num)
        sim_features.append(num_lower_case.num)
        sim_features.append(num_stem_lowered.num)

    #return [s1, s2, sim_features]
    return sim_features

class MCSR(object):

    def MCSR_nofold(self, baseline=None, rte=None, ta_sum=None, numeric=None):

        # filename="metrics/MCSR/nofold/all/"+string_metrics+".txt"
        # if not os.path.exists(os.path.dirname(filename)):
        #     os.makedirs(os.path.dirname(filename))
        # file_metrics = open(filename, "w")

        # categories=['nao','sim'],
        mydata_train = datasets.load_files("msr_paraphrase_train", description=None,
            load_content=True, shuffle=True, encoding="utf-8", decode_error='strict', random_state=0)

        # categories=['nao','sim'],
        mydata_test = datasets.load_files("msr_paraphrase_test", description=None,
            load_content=True, shuffle=True, encoding="utf-8", decode_error='strict', random_state=0)


        # cache load (both)
        #
        clusterfilename='train_featSemilarSemsimCONLL03'
        with open("cacheric/" + clusterfilename, 'rb') as f1:
            featuresets_docs = pickle.load(f1)

        clusterfilename='test_featSemilarSemsimCONLL03'
        with open("cacheric/" + clusterfilename, 'rb') as f1:
            testes_featured = pickle.load(f1)


        # cache save (both)
        #

        # cria_dict('yelpac-c1000-m25') # yelpac-c1000-m25 brown-rcv1.clean.tokenized-CoNLL03.txt-c3200-freq1.txt'
        # cria_tf_idf(mydata_train.data)
        #
        # plines = getFeatFromCSV('../msr_paraphrase_train_semilarTo0ExcDep.csv', 0)
        # plinesSemsim = getFeatFromCSV('../msr_paraphrase_train.txt_semsimAll.csv', 0)
        # print 'train'
        # featuresets_docs = [similarity_features(d, baseline, rte, ta_sum, numeric, 'train', plines, plinesSemsim) for d in mydata_train.data]
        #
        # # cache save
        # clusterfilename='train_featSemilarSemsim'
        # if not os.path.isfile("cacheric/" + clusterfilename):
        #     print  "caching into: cacheric/" + clusterfilename
        #
        #     if not os.path.exists("cacheric"):
        #         os.makedirs("cacheric")
        #
        #     with open("cacheric/" + clusterfilename, 'wb') as f1:
        #         pickle.dump(featuresets_docs, f1)

        # sys.exit(0)

        # global lcnttot
        # lcnttot = 0

        # plines = getFeatFromCSV('../msr_paraphrase_test_semilarTo0ExcDep.csv', 0)
        # plinesSemsim = getFeatFromCSV('../msr_paraphrase_test.txt_semsimAll.csv', 0)
        # print 'test'
        # testes_featured = [similarity_features(a, baseline, rte, ta_sum, numeric, 'test', plines, plinesSemsim) for a in mydata_test.data]
        #
        # # cache save
        # clusterfilename='test_featSemilarSemsim'
        # if not os.path.isfile("cacheric/" + clusterfilename):
        #     print  "caching into: cacheric/" + clusterfilename
        #
        #     if not os.path.exists("cacheric"):
        #         os.makedirs("cacheric")
        #
        #     with open("cacheric/" + clusterfilename, 'wb') as f1:
        #         pickle.dump(testes_featured, f1)

        # sys.exit(0)

        # get feature vectors from pickle
        featuresets_docs2 = []
        for i in featuresets_docs:
            featuresets_docs2.append(i[2][:6] + [i[2][7]] + i[2][9:])
            # i[2][:5] + i[2][10:] --> Ric + Semilar
            # i[2][:7] + [i[2][8]] + i[2][10:] -> Ric + Semilar + Semsim (only s/ WN)
            # i[2][:6] + [i[2][7]] + i[2][9:] -> Ric + Semilar + Semsim (only c/ WN)

        testes_featured2 = []
        for i in testes_featured:
            testes_featured2.append(i[2][:6] + [i[2][7]] + i[2][9:])

        print len(featuresets_docs2[0])
        print len(testes_featured2[0])

        # evals
        teste_vec = np.array(testes_featured2)

        print 'fitting'
        clf = SVC(kernel='poly').fit(featuresets_docs2, mydata_train.target)

        print 'pred'
        y_pred = clf.predict(teste_vec)

        acc = clf.score(teste_vec, mydata_test.target)
        accstr = "Accuracy:\t %0.3f (+/- %0.3f)\n" % (acc.mean(), acc.std() * 2)
        # file_metrics.write(accstr)
        print accstr

        f1 = metrics.f1_score(y_pred, mydata_test.target)
        f1str = "F1:\t %0.3f (+/- %0.3f)\n" % (f1.mean(), f1.std() * 2)
        # file_metrics.write(f1str)
        print f1str

        # file_metrics.close()


def mean_scores(X, y, clf, skf):
    cm = np.zeros(len(np.unique(y)) ** 2)
    for i, (train, test) in enumerate(skf):
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        cm += metrics.confusion_matrix(y[test], y_pred).flatten()

    return compute_measures(*cm / skf.n_folds)


def compute_measures(tp, fp, fn, tn):
    """Computes effectiveness measures given a confusion matrix."""
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    fmeasure = 2 * (specificity * sensitivity) / (specificity + sensitivity)

    prec = tp / (tp + fp + 0.0)
    print 'precision: \t\t' + str(prec)
    acc = (tp + tn + 0.0) / (tp + tn + fp + fn + 0.0)
    print 'accuracy: \t\t' + str(acc)
    reca = tp / (tp + fn + 0.0)
    print 'recall: \t\t' + str(reca)
    f = (2 * prec * reca) / (prec + reca + 0.0)
    print 'f:\t\t\t\t' + str(f)
    print

    return sensitivity, specificity, fmeasure


#if __name__ == "__main__":
def exps():
    #exp1
    clusterfilename='oli_3pairsPerEx_91featPerPair_yelp_ex'
    print clusterfilename
    with open("cacheric/" + clusterfilename, 'rb') as f1:
        featuresets_docs = pickle.load(f1)

    clusterfilename='oli_3pairsPerEx_91featPerPair_yelp_targets'
    with open("cacheric/" + clusterfilename, 'rb') as f1:
        targets = pickle.load(f1)

    clf = LinearSVC()

    print len(featuresets_docs)
    kf = KFold(len(featuresets_docs), n_folds=10)
    print mean_scores(np.array(featuresets_docs), np.array(targets), clf, kf)

    scores = model_selection.cross_val_score(clf, np.array(featuresets_docs), np.array(targets), cv=10, n_jobs=-1)     #, scoring='f1')
    print("Acc: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = model_selection.cross_val_score(clf, np.array(featuresets_docs), np.array(targets), cv=10, n_jobs=-1, scoring='f1')
    print("F1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    sys.exit(0)



    mydata_train = datasets.load_files("oli2", description=None,
            load_content=True, shuffle=True, encoding="utf-8", decode_error='strict', random_state=0)

    cria_dict('brown-rcv1.clean.tokenized-CoNLL03.txt-c3200-freq1-600k.txt') # yelpac-c1000-m25 brown-rcv1.clean.tokenized-CoNLL03.txt-c3200-freq1-200k.txt brown-rcv1.clean.tokenized-CoNLL03.txt-c3200-freq1-600k.txt
    cria_tf_idf(mydata_train.data)

    lcnttot=0

    featuresets_docs = []
    targets = []
    done = []
    for i, d in enumerate(mydata_train.data):
        file1 = mydata_train.filenames[i].replace('ab', '').replace('a', '')
        if file1 in done:
            continue

        done.append(file1)
        file2 = file1 + 'a'
        file3 = file2 + 'ab'

        inslist = similarity_features(mydata_train.data[np.where(mydata_train.filenames == file1)[0][0]], 1, 1, 1, 1, '', [], []) \
                  + similarity_features(mydata_train.data[np.where(mydata_train.filenames == file2)[0][0]], 1, 1, 1, 1, '', [], []) \
                  + similarity_features(mydata_train.data[np.where(mydata_train.filenames == file3)[0][0]], 1, 1, 1, 1, '', [], [])

        lcnttot += 1
        print str(lcnttot) + " ",

        featuresets_docs.append(inslist)
        targets.append(mydata_train.target[np.where(mydata_train.filenames == file1)[0][0]])


    clusterfilename='oli_3pairsPerEx_91featPerPair_conll_ex'
    if not os.path.isfile("cacheric/" + clusterfilename):
        print  "caching into: cacheric/" + clusterfilename

        if not os.path.exists("cacheric"):
            os.makedirs("cacheric")

        with open("cacheric/" + clusterfilename, 'wb') as f1:
            pickle.dump(featuresets_docs, f1)

    clusterfilename='oli_3pairsPerEx_91featPerPair_conll_targets'
    if not os.path.isfile("cacheric/" + clusterfilename):
        print  "caching into: cacheric/" + clusterfilename

        if not os.path.exists("cacheric"):
            os.makedirs("cacheric")

        with open("cacheric/" + clusterfilename, 'wb') as f1:
            pickle.dump(targets, f1)

    sys.exit(1)

    # v1
    msc = MCSR()
    msc.MCSR_nofold("baseline", "rte", "ta_sum", "numeric")
    sys.exit(1)


    # v2
    mydata_train = datasets.load_files("msr_paraphrase_train", description=None,
                                       load_content=True, shuffle=True, encoding="utf-8",
                                       decode_error='strict', random_state=0)

    mydata_test = datasets.load_files("msr_paraphrase_test", description=None,
                                      load_content=True, shuffle=True, encoding="utf-8",
                                      decode_error='strict', random_state=0)

    # yelpac-c1000-m25 brown-rcv1.clean.tokenized-CoNLL03.txt-c3200-freq1.txt'
    cria_dict('brown-rcv1.clean.tokenized-CoNLL03.txt-c3200-freq1.txt')
    cria_tf_idf(mydata_train.data)

    # cache load (both)
    #
    clusterfilename='train_featSemilarAll'
    with open("cacheric/" + clusterfilename, 'rb') as f1:
        featuresets_docs = pickle.load(f1)

    clusterfilename='test_featSemilarAll'
    with open("cacheric/" + clusterfilename, 'rb') as f1:
        testes_featured = pickle.load(f1)

    teste_vec = np.array(testes_featured)


    print 'fitting'
    clf = SVC(kernel='poly').fit(featuresets_docs, mydata_train.target)

    print 'pred'
    y_pred = clf.predict(teste_vec)

    acc = clf.score(teste_vec, mydata_test.target)
    accstr = "Accuracy:\t %0.3f (+/- %0.3f)\n" % (acc.mean(), acc.std() * 2)
    print accstr

    f1 = metrics.f1_score(y_pred, mydata_test.target)
    f1str = "F1:\t %0.3f (+/- %0.3f)\n" % (f1.mean(), f1.std() * 2)
    print f1str

    sys.exit(0)

    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    # for SGDClassifier()
    # parameters = {
    #     'vect__max_df': (0.5, 0.75, 1.0),
    #     #'vect__max_features': (None, 5000, 10000, 50000),
    #     'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #     #'tfidf__use_idf': (True, False),
    #     #'tfidf__norm': ('l1', 'l2'),
    #     'clf__alpha': (0.00001, 0.000001),
    #     'clf__penalty': ('l2', 'elasticnet'),
    #     #'clf__n_iter': (10, 50, 80),
    # }

    # tuned_parameters = {'kernel': ['poly', 'rbf', 'linear', 'sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4],
    #                     'C': [1, 10, 100, 1000], 'coef0': [0, 1], 'degree': [2, 3]}
    #
    # grid_search = GridSearchCV(SVC(), tuned_parameters, n_jobs=2, verbose=1, cv=5)
    #
    # print("Performing grid search...")
    # grid_search.fit(featuresets_docs, mydata_train.target)
    #
    # print("Best score: %0.3f" % grid_search.best_score_)
    # print("Best parameters set:")
    # best_parameters = grid_search.best_estimator_.get_params()
    # for param_name in sorted(tuned_parameters.keys()):
    #     print("\t%s: %r" % (param_name, best_parameters[param_name]))


    # cache best clf
    # clusterfilename='svcPolyGrid'
    # if not os.path.isfile("cacheric/" + clusterfilename):
    #     print  "caching into: cacheric/" + clusterfilename
    #
    #     if not os.path.exists("cacheric"):
    #         os.makedirs("cacheric")
    #
    #     with open("cacheric/" + clusterfilename, 'wb') as f1:
    #         pickle.dump(grid_search, f1)


    # print 'pred'
    # y_pred = grid_search.predict(teste_vec)
    #
    # acc2 = grid_search.score(teste_vec, mydata_test.target)
    # accstr2 = "Accuracy:\t %0.3f (+/- %0.3f)\n" % (acc2.mean(), acc2.std() * 2)
    # print accstr2
    #
    # f12 = metrics.f1_score(y_pred, mydata_test.target)
    # f1str2 = "F1:\t %0.3f (+/- %0.3f)\n" % (f12.mean(), f12.std() * 2)
    # print f1str2



    # msc = MCSR()
    # msc.MCSR_nofold("baseline", "rte", "ta_sum", "numeric")

    # msc.MCSR_nofold(None, None, "ta_sum", "numeric")
    # msc.MCSR_nofold(None, "rte", None, "numeric")
    # msc.MCSR_nofold(None, "rte", "ta_sum", None)
    # msc.MCSR_nofold("baseline", None, None, "numeric")
    # msc.MCSR_nofold("baseline", None, "ta_sum", None)
    # msc.MCSR_nofold("baseline", "rte", None, None)
    # msc.MCSR_nofold(None, None, None, "numeric")
    # msc.MCSR_nofold(None, None, "ta_sum", None)
    # msc.MCSR_nofold(None, "rte", None, None)
    # msc.MCSR_nofold("baseline", None, None, None)
    # msc.MCSR_nofold( None, "rte", "ta_sum", "numeric")
    # msc.MCSR_nofold("baseline", None, "ta_sum", "numeric")
    # msc.MCSR_nofold("baseline", "rte", None, "numeric")
    # msc.MCSR_nofold("baseline", "rte", "ta_sum", None)


