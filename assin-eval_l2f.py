# -*- coding: utf-8 -*-

'''
Script to evaluate system performance on the ASSIN shared task data.

Author: Erick Fonseca
'''

from __future__ import division, print_function

import argparse, traceback, pickle, os, sys
from xml.etree import cElementTree as ET
import numpy as np
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from scipy.stats import pearsonr
from MSC import *
from sklearn import model_selection
from sklearn.kernel_ridge import KernelRidge
from sklearn import svm
from flask import Flask
from flask_restful import Resource, Api, reqparse

class Pair(object):
    '''
    Class representing a pair of texts from SICK or RTE.
    It is meant to be used as an abstract representation for both.
    '''
    def __init__(self, t, h, id_, entailment, similarity):
        '''
        :param entailment: boolean
        :param attribs: extra attributes to be written to the XML
        '''
        self.t = t
        self.h = h
        self.id = id_
        self.entailment = entailment
        self.similarity = similarity
    
def read_xml(filename, lang=''):
    '''
    Read an RTE XML file and return a list of Pair objects.
    '''
    pairs = []
    tree = ET.parse(filename)
    root = tree.getroot()
    
    for xml_pair in root.iter('pair'):
        t = xml_pair.find('t').text
        h = xml_pair.find('h').text
        attribs = dict(xml_pair.items())
        id_ = int(attribs['id'])

        if lang == 'ptbr':
            id_ += 3001
        
        if 'entailment' in attribs:
            ent_string = attribs['entailment'].lower()
            
            if ent_string == 'none':
                ent_value = 0
            elif ent_string == 'entailment':
                ent_value = 1
            elif ent_string == 'paraphrase':
                ent_value = 2
            else:
                msg = 'Unexpected value for attribute "entailment" at pair {}: {}'
                raise ValueError(msg.format(id_, ent_value))
                        
        else:
            ent_value = None
            
        if 'similarity' in attribs:
            similarity = float(attribs['similarity']) 
        else:
            similarity = None
        
        if similarity is None and ent_value is None:
            msg = 'Missing both entailment and similarity values for pair {}'.format(id_)
            # raise ValueError(msg)
            # print (msg)
        
        pair = Pair(t, h, id_, ent_value, similarity)
        pairs.append(pair)
    
    return pairs

def eval_rte(pairs_gold, pairs_sys):
    '''
    Evaluate the RTE output of the system against a gold score. 
    Results are printed to stdout.
    '''
    # check if there is an entailment value
    if pairs_sys[0].entailment is None:
        print()
        print('No RTE output to evaluate')
        return
    
    gold_values = np.array([p.entailment for p in pairs_gold])
    sys_values = np.array([p.entailment for p in pairs_sys])
    macro_f1 = f1_score(gold_values, sys_values, average='macro')
    accuracy = (gold_values == sys_values).sum() / len(gold_values)
    
    print()
    print('RTE evaluation')
    print('Accuracy\tMacro F1')
    print('--------\t--------')
    print('{:8.2%}\t{:8.2f}'.format(accuracy, macro_f1))

def eval_similarity(pairs_gold, pairs_sys):
    '''
    Evaluate the semantic similarity output of the system against a gold score. 
    Results are printed to stdout.
    '''
    # check if there is an entailment value
    if pairs_sys[0].similarity is None:
        print()
        print('No similarity output to evaluate')
        return
    
    gold_values = np.array([p.similarity for p in pairs_gold])
    sys_values = np.array([p.similarity for p in pairs_sys])
    pearson = pearsonr(gold_values, sys_values)[0]
    absolute_diff = gold_values - sys_values
    mse = (absolute_diff ** 2).mean()
    
    print()
    print('Similarity evaluation')
    print('Pearson\t\tMean Squared Error')
    print('-------\t\t------------------')
    print('{:7.2f}\t\t{:18.2f}'.format(pearson, mse))


def cachesave(fname, data):
    # cache save
    clusterfilename=fname
    if not os.path.isfile("cacheric/" + clusterfilename):
        print ("caching into: cacheric/" + clusterfilename)

        if not os.path.exists("cacheric"):
            os.makedirs("cacheric")

        with open("cacheric/" + clusterfilename, 'wb') as f1:
            pickle.dump(data, f1)


def runProporEval(gold_file, system_file):
    pairs_gold = read_xml(gold_file, '')
    pairs_sys = read_xml(system_file, '')

    eval_rte(pairs_gold, pairs_sys)
    eval_similarity(pairs_gold, pairs_sys)


def getFeaturesPropor(trainfnameList, isGold):
    """
    :param trainfnameList: list of file names, for files containing training examples/pairs in XML

    (names only! relative to current path; no absolute paths!)
    :return:
    """

    pairsAll = []
    trainsetAll = []
    simall = []
    entall = []
    for trainfile in trainfnameList:
        print ('Processing ' + trainfile)

        # if trainfile.endswith('.xml'):
        #     print ('xml')
        pairs_gold = read_xml(trainfile)

        if isGold:
            sim_values = np.array([p.similarity for p in pairs_gold])
            simall = np.concatenate([simall, sim_values])

            ent_values = np.array([p.entailment for p in pairs_gold])
            entall = np.concatenate([entall, ent_values])

        if not os.path.isfile("cacheric/" + trainfile):
            pairs_goldRic = [unicode('TEXT: ' + p.h + '\nHYPOTHESIS:' + p.t) for p in pairs_gold]

            cria_dict('brown-clusters-pt.txt')
            cria_tf_idf(pairs_goldRic, clusterfilename='tfidf_' + trainfile)

            print ('Calculating features for ' + str(len(pairs_goldRic)) + ' samples...')
            trainset = [similarity_features(d, 1, 1, 1, 1, '', [], []) for d in pairs_goldRic]
            cachesave(trainfile, trainset)
        else:
            print ('loaded')
            with open("cacheric/" + trainfile, 'rb') as f1:
                trainset = pickle.load(f1)

        trainsetAll += trainset
        pairsAll += pairs_gold

    return pairsAll, trainsetAll, simall, entall


def trainAndTestCV(trainfnameList, expIDstr='', foldN=5):
    pairsAll, trainsetAll, simall, entall = getFeaturesPropor(trainfnameList, 1)

    print ('Using ' + str(len(trainsetAll)) + ' samples to build models')
    print ('with ' + str(len(simall)) + ' similarity values')
    print ('and ' + str(len(entall)) + ' entailment values')

    cacheFname = 'kridgePoly_' + expIDstr + str(foldN) + 'fCV'
    if not os.path.isfile("cacheric/" + cacheFname):
        clf1 = KernelRidge(kernel='poly')
        simscores = model_selection.cross_val_predict(clf1, np.array(trainsetAll), np.array(simall), cv=foldN, n_jobs=-1)
        cachesave(cacheFname, simscores)
    else:
        with open("cacheric/" + cacheFname, 'rb') as f1:
            clf1 = pickle.load(f1)

    cacheFname = 'svcPoly_' + expIDstr + str(foldN) + 'fCV'
    if not os.path.isfile("cacheric/" + cacheFname):
        clf2 = OneVsRestClassifier(SVC(kernel='poly'))
        entscores = model_selection.cross_val_predict(clf2, trainsetAll, entall, cv=foldN, n_jobs=-1)
        cachesave(cacheFname, entscores)
    else:
        with open("cacheric/" + cacheFname, 'rb') as f1:
            clf2 = pickle.load(f1)

    # TODO: move to separate function; the following 'return' is OK
    print ('Building SYS XML')

    # gold
    strxml = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<entailment-corpus>\n'
    j = 0
    for i in range(0, len(pairsAll)):
        j += 1

        ent_value = ''
        ent_string = pairsAll[i].entailment
        if ent_string == 0:
            ent_value = 'none'
        elif ent_string == 1:
            ent_value = 'entailment'
        elif ent_string == 2:
            ent_value = 'paraphrase'

        strxml += '\t<pair entailment="' + ent_value \
                  + '" id="' + str(pairsAll[i].id) \
                  + '" similarity="' + str(pairsAll[i].similarity) + '">\n'
        strxml += '\t\t<t>' + pairsAll[i].t + '</t>\n'
        strxml += '\t\t<h>' + pairsAll[i].h + '</h>\n'
        strxml += '\t</pair>\n'
    strxml += '</entailment-corpus>\n'
    print ('gold ' + str(j))

    newTrainSysFname = str(foldN) + 'fCV.' + expIDstr + '.gold'
    with open(newTrainSysFname, "w") as text_file:
        text_file.write(strxml.encode('UTF-8').replace('&', '&amp;'))

    # sys
    strxml = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<entailment-corpus>\n'
    j = 0
    for i in range(0, len(pairsAll)):
        j += 1

        ent_string = entscores[i]

        if ent_string == 0:
            ent_value = 'none'
        elif ent_string == 1:
            ent_value = 'entailment'
        elif ent_string == 2:
            ent_value = 'paraphrase'

        strxml += '\t<pair entailment="' + ent_value \
                  + '" id="' + str(pairsAll[i].id) \
                  + '" similarity="' + str(simscores[i]) + '">\n'
        strxml += '\t\t<t>' + pairsAll[i].t + '</t>\n'
        strxml += '\t\t<h>' + pairsAll[i].h + '</h>\n'
        strxml += '\t</pair>\n'
    strxml += '</entailment-corpus>\n'
    print ('sys ' + str(j))

    newTestSysFname = str(foldN) + 'fCV.' + expIDstr + '.sys'
    with open(newTestSysFname, "w") as text_file:
        text_file.write(strxml.encode('UTF-8').replace('&', '&amp;'))

    runProporEval(newTrainSysFname, newTestSysFname)

    return pairsAll, simscores, entscores


def trainAndTest(trainfnameList, testSysFname, testGoldFname, expIDstr=''):
    #
    #                   train
    #
    pairsAll, trainsetAll, simall, entall = getFeaturesPropor(trainfnameList, 1)

    print ('Using ' + str(len(trainsetAll)) + ' samples to build models')
    print ('with ' + str(len(simall)) + ' similarity values')
    print ('and ' + str(len(entall)) + ' entailment values')

    cacheFname = 'kridgePoly_' + expIDstr
    if not os.path.isfile("cacheric/" + cacheFname):
        clf1 = KernelRidge(kernel='poly').fit(np.array(trainsetAll), np.array(simall))
        cachesave(cacheFname, clf1)
    else:
        with open("cacheric/" + cacheFname, 'rb') as f1:
            clf1 = pickle.load(f1)

    cacheFname = 'svcPoly_' + expIDstr
    if not os.path.isfile("cacheric/" + cacheFname):
        clf2 = OneVsRestClassifier(SVC(kernel='poly')).fit(trainsetAll, entall)
        cachesave(cacheFname, clf2)
    else:
        with open("cacheric/" + cacheFname, 'rb') as f1:
            clf2 = pickle.load(f1)

    #
    #                   test
    #
    pairs_sys, testset, emptyList1, emptyList2 = getFeaturesPropor([testSysFname], 0)

    print ('Predicting from ' + str(len(testset)) + ' samples')
    teste_vec = np.array(testset)
    sim_pred = clf1.predict(teste_vec)

    # TODO: use predicted similarity values on entailment classification...
    # print (str(len(testset[0])))
    # bootstrap_testset = np.array([testset[p].append(sim_pred[p]) for p in range(0, len(testset))])
    # print (str(len(bootstrap_testset[0])))
    # teste_vec = np.array(bootstrap_testset)

    ent_pred = clf2.predict(teste_vec)

    # TODO: move to separate function; the following 'return' is OK
    print ('Building SYS XML')
    strxml = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<entailment-corpus>\n'
    for i in range(0, len(pairs_sys)):

        ent_value = ''
        ent_string = ent_pred[i]
        if ent_string == 0:
            ent_value = 'none'
        elif ent_string == 1:
            ent_value = 'entailment'
        elif ent_string == 2:
            ent_value = 'paraphrase'

        strxml += '\t<pair entailment="' + ent_value + '" id="' + str(pairs_sys[i].id) + '" similarity="' + str(sim_pred[i]) + '">\n'
        strxml += '\t\t<t>' + pairs_sys[i].t + '</t>\n'
        strxml += '\t\t<h>' + pairs_sys[i].h + '</h>\n'
        strxml += '\t</pair>\n'
    strxml += '</entailment-corpus>\n'

    newTestSysFname = testSysFname + '.' + expIDstr + '.sys'
    with open(newTestSysFname, "w") as text_file:
        text_file.write(strxml.encode('UTF-8').replace('&', '&amp;'))

    runProporEval(testGoldFname, newTestSysFname)

    return pairs_sys, sim_pred, ent_pred


def start_service():

    app = Flask(__name__)
    api = Api(app)

    # RUN ONCE
    with open("cacheric/kridgePoly_proporPT", 'rb') as f1:
        clf1 = pickle.load(f1)

    cria_dict('brown-clusters-pt.txt')

    # RUN ALWAYS
    class ParaphraseService(Resource):
        def post(self):
            parser = reqparse.RequestParser()
            parser.add_argument('sentence1', required=True)
            parser.add_argument('sentence2', required=True)
            args = parser.parse_args()

            encPair = unicode('TEXT: ' + args['sentence1'] + '\nHYPOTHESIS:' + args['sentence2'])
            cria_tf_idf([encPair])
            featVec = similarity_features(encPair, 1, 1, 1, 1, '', [], [])

            classifierValue = str(clf1.predict(np.array(featVec).reshape(1, -1))[0])

            return {'sentence1': args['sentence1'], 'sentence2': args['sentence2'], 'value': classifierValue.encode('utf-8')}

    api.add_resource(ParaphraseService, '/')

    app.run(host='172.16.254.34',port=21000,debug=False)


if __name__ == '__main__':
#    # RUN ONCE
#    with open("cacheric/kridgePoly_proporPT", 'rb') as f1:
#        clf1 = pickle.load(f1)
#
#    cria_dict('brown-clusters-pt.txt')

    # RUN ALWAYS
#    encPair = unicode('TEXT: ' + sys.argv[1] + '\nHYPOTHESIS:' + sys.argv[2])
#    cria_tf_idf([encPair])
#    featVec = similarity_features(encPair, 1, 1, 1, 1, '', [], [])

#    print (str(clf1.predict(np.array(featVec).reshape(1, -1))[0]))

    # app.run(host='172.16.254.34',port=21000,debug=False)

    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument('gold_file', help='Gold file')
    # parser.add_argument('system_file', help='File produced by a system')
    # args = parser.parse_args()

    # pairs_gold = read_xml(args.gold_file)
    # pairs_sys = read_xml(args.system_file)
    #
    # eval_rte(pairs_gold, pairs_sys)
    # eval_similarity(pairs_gold, pairs_sys)

    #
    #   previous -> original Propor code
    #
    #   following -> L2F; each line is an experiment
    #

    trainAndTest(['assin-ptpt-train.xml'],
              'assin-ptbr-test.xml', 'assin-ptbr-test_gold.xml', expIDstr='ptBR')

    # # run 1: as is
    # trainAndTest(['assin-ptpt-train.xml', 'assin-ptbr-train.xml'],
    #           'assin-ptbr-test.xml', 'assin-ptbr-test_gold.xml', expIDstr='proporBR')

    # # run 2: more data
    # trainAndTest(['assin-ptpt-train.xml', 'assin-ptbr-train.xml', 'bingtranslatedSICK.xml'],
    #           'assin-ptbr-test.xml', 'assin-ptbr-test_gold.xml', expIDstr='allBR')

    # trainAndTestCV(['assin-ptpt-train.xml'], expIDstr='ptpt', foldN=5)
