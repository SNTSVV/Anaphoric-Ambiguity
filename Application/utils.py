# ignore warnings
import warnings
warnings.filterwarnings('ignore')
# import basic libraries to read from docx and xml files
import os, time
from xml.dom import minidom
# import nlp libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import verbnet as vn
import spacy
from spacy.tokens import Span

import en_core_web_sm
# import neuralcoref
from benepar.spacy_plugin import BeneparComponent
import benepar
from spacy.matcher import Matcher, PhraseMatcher
import stanza
from stanza.server import CoreNLPClient
# import utils libraries
from tqdm.notebook import tqdm
import importlib, sys, re, copy, pickle, random, json, ast, csv
# import arrays and dataframes libraries
import pandas as pd
import numpy as np
# import visualisation libraries
from matplotlib import pyplot as plt
from statistics import mean
# import machine learning algorithms
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
# from IPython.display import Image  
import pydotplus
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier, RidgeClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit, RandomizedSearchCV, cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import cohen_kappa_score, fbeta_score, make_scorer, confusion_matrix, classification_report, mean_squared_error
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.base import clone
from datetime import datetime
from nimbusml.linear_model import AveragedPerceptronBinaryClassifier
import scikitplot as skplt
from timeit import default_timer as timer

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

## for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing

## for deep learning
from tensorflow.keras import layers, preprocessing as kprocessing
from tensorflow.keras import backend as K
import torch
## for bert language model
import transformers
import joblib
from bertutils import *
from anaphora_utils import *    

tqdm.pandas()
nltk.download('wordnet')
benepar.download('benepar_en3')
stanza.download('en')
nltk.download('verbnet')


def applynlp(string,nlp):
    tr=np.nan
    try:
        tr=nlp(string)
    except:
        print(string)
    return tr        

def startCoreNLPClient():
    client = CoreNLPClient(
        annotators=['tokenize','ssplit', 'pos', 'lemma', 'coref'], 
        memory='4G', 
        endpoint='http://localhost:9000',
        be_quiet=True)
#     print(client)
    client.start()
    return client

# get coreferences from input document using CoreNLP

def getCorefsCoreNLP(document):

    mychains = list()

    chains = document.corefChain

    for chain in chains:

        mychain = list()

        # Loop through every mention of this chain

        for mention in chain.mention:

            # Get the sentence in which this mention is located, and get the words which are part of this mention

            # (we can have more than one word, for example, a mention can be a pronoun like "he", but also a compound noun like "His wife Michelle")

            words_list = document.sentence[mention.sentenceIndex].token[mention.beginIndex:mention.endIndex]

            #build a string out of the words of this mention

            ment_word = ' '.join([x.word for x in words_list])

            mychain.append(ment_word)

        mychains.append(mychain)

    for chain in mychains:

        print(' <-> '.join(chain))

# save text file

def saveFile(txt, filename):

    text_file = open(filename, "w")

    text_file.write(txt)

    text_file.close()

# get antecedents candidates from the coreNLP coreference resolver

def get_mentions(spacyDoc,coreClient):

    coreDoc=coreClient.annotate(spacyDoc.text)

    mentions=[]

    for m in coreDoc.mentionsForCoref:

        mentions.append(spacyDoc[m.startIndex:m.endIndex])

    return mentions

# get filtred antecedents candidates from the coreNLP coreference resolver that comes before the pronoun in parameters
def get_mentions_p(twosents,p):
    mentions=get_mentions(twosents,client)
    return [m for m in mentions if not ((m.start >= p.i) or 
                                        (len(m)==1 and m[0].pos_=="PUNCT") 
                                        or (m.root.pos_ in ["VERB","CCONJ","ADP","PUNCT"]))]
# F1

def distance(noun_phrase, pronoun):
    if noun_phrase.doc==pronoun.doc:
        if noun_phrase.end < pronoun.i:

            return pronoun.i-noun_phrase.end

        if noun_phrase.start > pronoun.i:

            return noun_phrase.start-pronoun.i
    else:
        return abs(pronoun.i+len(noun_phrase.doc)-noun_phrase.end)
    return pronoun.i - noun_phrase.root.i

# get the number of a noun phrase

def number_np(np):
    for t in np:
        if t.tag_=="CC":
            return 2
    if np.root.tag_ == "NNS":

        return 2

    if np.root.tag_ == "NN":

        return 1

    return 0

# get the number of a pronoun

def number_pr(pr,nlp1):

    ft=nlp1(pr).sentences[0].words[0].feats
    if not ft:
        return 0

    if "Number=Sing" in ft:

        return 1

    if "Number=Plur" in ft:

        return 2

    return 0

# F3

def sameNumber(np,pr,nlp1):

    return number_np(np)==number_pr(pr.text,nlp1)

# get gender

def gender(pr,nlp1):

    ft=nlp1(pr).sentences[0].words[0].feats

    if not ft: return 0

    if "Gender=Masc" in ft:

        return 1

    if "Gender=Fem" in ft:

        return 2

    return 0

# F2

def sameGender(np,pr,nlp1):

    pG=gender(pr.text,nlp1)

    npG=gender(np.root.text,nlp1)

    return pG==npG

# F4

def isPerson(token):

    for syn in wn.synsets(token.text):

        if 'person' in syn.lexname():

            return True

    return token.ent_type_=="PERSON"

# F5

def isDefinite(head):

    d=["the","this","that","these","those"]

    for child in head.children:

        if child.tag_ == "PRP$":

            return True

        if child.text.lower() in d:

            return True

    return False

# F6

def isPrepositional(token):

    try:
        pS=token._.parent._.parse_string

        if pS.split(" ")[0].equals("(PP"):

            return True
    except Exception as e:
        return False
    

#     if pS.contains("PP"):

#         return True

    return False  

# F7

def isSubject(token):

    if "subj" in token.dep_:

        return True

    return False

# F8

def isDirectObject(token):

    if "dobj" in token.dep_:

        return True

    return False

# F9

def isIndirectObject(token):

    if "iobj" in token.dep_:

        return True

    return False

# F10

def sameRole(antroot,anaphor):

    if antroot.dep_==anaphor.dep_:

        return True

    if antroot.dep_[-3:]==anaphor.dep_[-3:]:

        return True

    return False

# F11

def repeatedPatterns(antroot,anaphor):

    lant=getNtokens(antroot)

    lana=getNtokens(anaphor)

    for t1 in lant:

        for t2 in lana:

            if t1!=t2 and t1.text==t2.text:

                return True

    return False

# get n neighbour tokens

def getNtokens(token,n=3):

    lt=[]

    ix=token.i

    for i in range(ix-3 if ix-3>0 else 0,ix+n+1 if ix+n+1<len(token.doc) else len(token.doc)):

        if i==ix: continue

        lt.append(token.doc[i])

    return lt

# F12

def sameHead(antroot,anaphor):

    return antroot.head==anaphor.head

# check if a verb requires animate agent in VerbNet


def isNextVerbAnimate(anaphor):

    for i in range(1,len(anaphor.doc)-anaphor.i):

        if anaphor.nbor(i).pos_=="VERB":

            return isVerbAnimate(anaphor.nbor(i).lemma_)

    return None

def isVerbAnimate(v):

    for vc in vn.classids(v):

        if "Agent" in str(vn.themroles(vn.vnclass(vc))) and "animate" in str(vn.themroles(vn.vnclass(vc))):

            return True

    return False

# check if a noun is animate

def isAnimacy(t):

    if isPerson(t):

        return True

    animate=['person','animal']

    if not wn.synsets(t.text): return False

    s=wn.synsets(t.text)[0]

    if str(s).split("'")[1].split(".")[0] in animate:

            return True

    while s.hypernyms():

        s=s.hypernyms()[0]

        if str(s).split("'")[1].split(".")[0] in animate:

            return True

    return False

# F14

def sentenceRecency(antroot,anaphor):

    return antroot.sent==anaphor.sent

# F15

def DistSimilarity(antroot,anaphor):

    return antroot.similarity(anaphor)

# F16

def position(anaphor):

    if anaphor.is_sent_start: 

        return "first"

    if (anaphor.i+2)==anaphor.sent.end:

        return "last"

    return "middle"

# F17

def nextPOS(anaphor):

    if anaphor.i==len(anaphor.doc)-1:
        return "."
    return anaphor.nbor().pos_
    

# F18

def hasClause(doc):

    clauselist=["csubj","ccomp","xcomp","advcl","acl"]

    for t in doc:

        if t.dep_ in clauselist:

            return True

    return False

def hasClauseinContext(sents):

    for sent in sents:
        if hasClause(sent):
            return True
    return False

# F19

def relativePositionAnaphor(anaphor):

    return (anaphor.i-anaphor.sent.start)/len(anaphor.sent)

# get the first preceding noun to an input token

def firstPrecedingNoun(token):

    for i in range(1,token.i):

        if token.nbor(-i).pos_=="NOUN":

            return token.nbor(-i).text

    return None

# F20

def fPN(token):

    for i in range(1,token.i):

        if token.nbor(-i).pos_=="NOUN":

            return token.nbor(-i)

    return None

def relativePositionToNoun(anaphor):

    noun=fPN(anaphor)

    if noun!=None:

        if noun.sent==anaphor.sent:

            return (anaphor.i-noun.i)/len(anaphor.sent)

        else:
            return (anaphor.i+len(noun.doc)-noun.i)/(len(anaphor.doc)+len(noun.doc))

    return -1

# F21

def relativePositionSentence(sents,anaphor):

    return sents.index(anaphor.doc)+1/len(sents)

# F22

def headLemma(anaphor):

    return anaphor.head.lemma_

# F23

def headPOS(anaphor):

    return anaphor.head.pos_

# F24

def precedingPOS(anaphor):

    return anaphor.nbor(-1).pos_
# F25

# def nextPOS(anaphor):

#     return anaphor.nbor(1).pos_

# F26

def numPrecedingNPsSent(anaphor):

    return len([x for x in anaphor.sent.noun_chunks if x.end<=anaphor.i])

# F27

def numPrecedingNPsPara(anaphor,sents):
    num=0
    for sent in sents:
        if anaphor in sent:
            num+= len([x for x in anaphor.doc.noun_chunks if x.end<=anaphor.i])
        else: 
            num+= len(list(sent.noun_chunks))

    return num

# F28

def numSucceedingNPsSent(anaphor):

    return len([x for x in anaphor.doc.noun_chunks if x.start>anaphor.i])

# F29

def numNPsInSent(anaphor):

    return len(list(anaphor.sent.noun_chunks))

# F30

def numNPsInPara(sents):
    num=0
    for sent in sents:
        num+= len(list(sent.noun_chunks))
    return num
# F31

def numSucceedingAdjSent(anaphor):

    return len([x for x in anaphor.sent if x.i>anaphor.i and x.pos_=="ADJ"])

# F32

def precedingVerb(anaphor):

    for i in range(1,anaphor.i):

        if anaphor.nbor(-i).pos_=="VERB":

            return anaphor.nbor(-i).lemma_

    return None

# F33

def nextAdj(anaphor):

    for i in range(1,len(anaphor.doc)-anaphor.i):

        if anaphor.nbor(i).pos_=="ADJ":

            return anaphor.nbor(i).lemma_

    return None

# F34

def nextVerb(anaphor):

    for i in range(1,len(anaphor.doc)-anaphor.i):

        if anaphor.nbor(i).pos_=="VERB":

            return anaphor.nbor(i).lemma_

    return None

# F35

def numSucceedingComp(anaphor):

    return len([x for x in anaphor.sent if x.i>anaphor.i and x.dep_=="mark"])

# F36

def numWordsB4Comp(anaphor):

    for i in range(1,len(anaphor.doc)-anaphor.i):

        if anaphor.nbor(i).dep_=="mark":

            return i

    return None

# F37

def isAdjB4NP(anaphor):

    nounphrase=None

    for np in anaphor.doc.noun_chunks:

        if np.start>anaphor.i:

            nounphrase=np

            break

    if nounphrase:

        for i in range(anaphor.i+1,nounphrase.start):

            if anaphor.doc[i].pos_=="ADJ":

                return True

        return False

    return None
# F38

def numWordsB4Inf(anaphor):

    for i in range(1,len(anaphor.doc)-anaphor.i):

        if anaphor.nbor(i).tag_=="VB":

            return i

    return None

# F39

def numWordsB4Prep(anaphor):

    for i in range(1,len(anaphor.doc)-anaphor.i):

        if anaphor.nbor(i).dep_=="prep":

            return i

    return None

# F40

def numWordsB4Ing(anaphor):

    for i in range(1,len(anaphor.doc)-anaphor.i):

        if anaphor.nbor(i).pos_=="VERB" and anaphor.nbor(i).text[-3:]=="ing":

            return i

    return None

# F41

def isCompB4NP(anaphor):

    nounphrase=None

    for np in anaphor.doc.noun_chunks:

        if np.start>anaphor.i:

            nounphrase=np

            break

    if nounphrase:

        for i in range(anaphor.i+1,nounphrase.start):

            if anaphor.doc[i].dep_=="mark":

                return True

    return False

# F42

def isprecedingPrep(anaphor):
    if anaphor.i==0: 
        return False
    return anaphor.nbor(-1).dep_=="prep"

# F43

def precedingWord(anaphor):
    if anaphor.i==0: 
        return ""
    return anaphor.nbor(-1).lemma_

# F44

def nextWord(anaphor):
    if anaphor.i==len(anaphor.doc)-1:
        return "."
    return anaphor.nbor(1).lemma_

# F45
def numPunctuations(anaphor):
    return len([x for x in anaphor.sent if x.pos_=="PUNCT"])

# F45

def numPunctuations(anaphor):

    return len([x for x in anaphor.sent if x.pos_=="PUNCT"])

# F46

def followedByToVerb(anaphor):
    doc=anaphor.doc
    for i in range(anaphor.i+1,len(doc)):
        if doc[i].tag_ == "TO":
            if i+1==len(doc):
                return False
            if doc[i+1].pos_ == "VERB":
                return True
    return False

# F47
def samePrecedingFollownigVerbs(anaphor):
    doc=anaphor.doc
    for i in range(anaphor.i+1,len(doc)):
        if doc[i].tag_ == "TO":
            if i+1==len(doc):
                return False
            if doc[i+1].pos_ == "VERB":
                return doc[0:].root.lemma_==doc[i+1].lemma_
    return False


def extract_LF(Data,nlp1):
    Data['Distance']=Data.apply(lambda x: distance( x['Candidate Antecedent'], x['Pronoun']), axis=1)
    Data['sameGender']=Data.apply(lambda x: sameGender( x['Candidate Antecedent'], x['Pronoun'],nlp1), axis=1)
    Data['sameNumber']=Data.apply(lambda x: sameNumber( x['Candidate Antecedent'], x['Pronoun'],nlp1), axis=1)
    Data['isPerson']=Data.apply(lambda x: isPerson( x['Candidate Antecedent'].root), axis=1)
    Data['isDefinite']=Data.apply(lambda x: isDefinite( x['Candidate Antecedent'].root), axis=1)
    Data['isPrepositional']=Data.apply(lambda x: isPrepositional( x['Candidate Antecedent'].root), axis=1)
    Data['isSubject']=Data.apply(lambda x: isSubject( x['Candidate Antecedent'].root), axis=1)
    Data['isDirectObject']=Data.apply(lambda x: isDirectObject( x['Candidate Antecedent'].root), axis=1)
    Data['isIndirectObject']=Data.apply(lambda x: isIndirectObject( x['Candidate Antecedent'].root), axis=1)
    Data['sameRole']=Data.apply(lambda x: sameRole( x['Candidate Antecedent'].root , x['Pronoun']), axis=1)
    Data['repeatedPatterns']=Data.apply(lambda x: repeatedPatterns( x['Candidate Antecedent'].root , x['Pronoun']), axis=1)
    Data['sameHead']=Data.apply(lambda x: sameHead( x['Candidate Antecedent'].root , x['Pronoun']), axis=1)
    Data['DistSimilarity']=Data.apply(lambda x: DistSimilarity(x['Candidate Antecedent'].root ,x['Pronoun']), axis=1)
    Data['isNextVerbAnimate']=Data.apply(lambda x: isNextVerbAnimate(x['Pronoun']), axis=1)
    Data['sentenceRecency']=Data.apply(lambda x: True, axis=1)
    Data['position']=Data.apply(lambda x: position(x['Pronoun']), axis=1)
    Data['nextPOS']=Data.apply(lambda x: nextPOS(x['Pronoun']), axis=1)
    Data['hasClause']=Data.apply(lambda x: hasClause(x['Context']), axis=1)
    Data['relativePositionAnaphor']=Data.apply(lambda x: relativePositionAnaphor(x['Pronoun']), axis=1)
    Data['firstPrecedingNoun']=Data.apply(lambda x: firstPrecedingNoun(x['Pronoun']), axis=1)
    Data['relativePositionToNoun']=Data.apply(lambda x: relativePositionToNoun(x['Pronoun']), axis=1)
    Data['relativePositionSentence']=Data.apply(lambda x: 1, axis=1)
    Data['headLemma']=Data.apply(lambda x: headLemma(x['Pronoun']), axis=1)
    Data['headPOS']=Data.apply(lambda x: headPOS(x['Pronoun']), axis=1)
    Data['precedingPOS']=Data.apply(lambda x: precedingPOS(x['Pronoun']), axis=1)
    Data['nextPOS']=Data.apply(lambda x: nextPOS(x['Pronoun']), axis=1)
    Data['numPrecedingNPsSent']=Data.apply(lambda x: numPrecedingNPsSent(x['Pronoun']), axis=1)
    Data['numPrecedingNPsPara']=Data.apply(lambda x: numPrecedingNPsPara(x['Pronoun'],x['Context'].sents), axis=1)
    Data['numSucceedingNPsSent']=Data.apply(lambda x: numSucceedingNPsSent(x['Pronoun']), axis=1)
    Data['numNPsInSent']=Data.apply(lambda x: numNPsInSent(x['Pronoun']), axis=1)
    Data['numNPsInPara']=Data.apply(lambda x: numNPsInPara(x['Context'].sents), axis=1)
    Data['numSucceedingAdjSent']=Data.apply(lambda x: numSucceedingAdjSent(x['Pronoun']), axis=1)
    Data['precedingVerb']=Data.apply(lambda x: precedingVerb(x['Pronoun']), axis=1)
    Data['nextAdj']=Data.apply(lambda x: nextAdj(x['Pronoun']), axis=1)
    Data['nextVerb']=Data.apply(lambda x: nextVerb(x['Pronoun']), axis=1)
    Data['numSucceedingComp']=Data.apply(lambda x: numSucceedingComp(x['Pronoun']), axis=1)
    Data['numWordsB4Comp']=Data.apply(lambda x: numWordsB4Comp(x['Pronoun']), axis=1)
    Data['isAdjB4NP']=Data.apply(lambda x: isAdjB4NP(x['Pronoun']), axis=1)
    Data['numWordsB4Inf']=Data.apply(lambda x: numWordsB4Inf(x['Pronoun']), axis=1)
    Data['numWordsB4Prep']=Data.apply(lambda x: numWordsB4Prep(x['Pronoun']), axis=1)
    Data['numWordsB4Ing']=Data.apply(lambda x: numWordsB4Ing(x['Pronoun']), axis=1)
    Data['isCompB4NP']=Data.apply(lambda x: isCompB4NP(x['Pronoun']), axis=1)
    Data['isprecedingPrep']=Data.apply(lambda x: isprecedingPrep(x['Pronoun']), axis=1)
    Data['precedingWord']=Data.apply(lambda x: precedingWord(x['Pronoun']), axis=1)
    Data['nextWord']=Data.apply(lambda x: nextWord(x['Pronoun']), axis=1)
    Data['numPunctuations']=Data.apply(lambda x: numPunctuations(x['Pronoun']), axis=1)
    return Data

def getprediction(test,y_pred,idx,Wy,Set):
    di={i:[] for i in idx[test].unique()}
    dd={i:"" for i in idx[test].unique()}
    for i,p in zip(idx[test],y_pred):
        di[i].append(p)
    for k,v in di.items():
        QNY=[0,0,0]
        Yw=0
        for l in v:
            l=l.tolist()
            m=max(l)
            QNY[l.index(m)]+=1
            if l[2]<Wy and l[2]>0: 
                Yw+=1
        if (QNY[2]==1 and QNY[0]==0) or (QNY[2]==1 and Yw==0):
            #Not ambiguous
            dd[k]="Unambiguous"
        else:
            #Ambiguous
            dd[k]="Ambiguous"
    newdf = Set[Set['Id'].isin(idx[test])][["Id","Context","Pronoun"]]
    newdf.drop_duplicates(inplace=True)
    newdf['result']=newdf.apply(lambda x: dd[x['Id']] ,axis=1)
    return newdf

def getResolution(y_proba,X_test,gt3,theta=0.5):
    li=[]
    for Id in X_test.Id.unique():
        preds=proba2pred(y_proba[X_test[X_test.Id==Id].index])
        if 1 in preds.values:
            if list(preds.values).count(1)==1:
                li.append([Id,gt3["Candidate Antecedent"][preds[preds==1].index[0]]])
            else:
                if theta!=0:
                    above_theta=[]
                    for i in preds[preds==1].index:
                        #print(y_proba[i][1],preds[i],i)
                        if y_proba[i][1]>=theta:
                            above_theta.append(i)
                    if len(above_theta)!=1:
                        li.append([Id,None])
                    elif preds[above_theta[0]]==1:
                        li.append([Id,gt3["Candidate Antecedent"][above_theta[0]]])
                    else:
                        li.append([Id,None])
                else:
                    maxs=[]
                    mx=y_proba[preds[preds==1].index].apply(lambda x: x[1]).max()
                    for i,p in zip(preds[preds==1].index,y_proba[preds[preds==1].index]):
                        if p[1]==mx:
                            maxs.append(i)
                    if len(maxs)>1:
                        trm=[]
                        for i in range(0,len(maxs)-1):
                            for j in range(i+1,len(maxs)):
                                if gt3["Candidate Antecedent"][maxs[i]]==gt3["Candidate Antecedent"][maxs[j]]:
                                    trm.append(i)
                        maxs=np.delete(maxs,trm)
                        if len(maxs)>1:
                            li.append([Id,None])
                        elif preds[maxs[0]]==1:
                            li.append([Id,gt3["Candidate Antecedent"][maxs[0]]])
                        else:
                            li.append([Id,None])
                    elif preds[maxs[0]]==1:
                        li.append([Id,gt3["Candidate Antecedent"][maxs[0]]])
                    else:
                        li.append([Id,None])
        else:
            li.append([Id,None])
    return pd.DataFrame(li,columns=["Id","Predicted"])


def ensembleprobaN(ypred1,ypred2,theta=0):
    ypred=[]
    for l1,l2 in zip(ypred1,ypred2):
        m1=np.max(l1)
        m2=np.max(l2)
        d1=np.where(l1==m1)[0][0]
        d2=np.where(l2==m2)[0][0]
        if d1==0 or d2==0: 
            ypred.append([np.max([m1,m2]),0,0])
            continue
        elif theta!=0:
            if d1!=d2 and abs(m1-m2)<theta:
                ypred.append([np.max([m1,m2]),0,0])
                continue
        else:
            e=[0,0,0]
            e[d1]+=m1
            e[d2]+=m2
            ypred.append(e)
    return np.array(ypred)

# get previous sentence

def prevSent(sent):

    if list(sent.doc.sents).index(sent)==0:

        return None

    return list(sent.doc.sents)[list(sent.doc.sents).index(sent)-1]

# get next sentence

def nextSent(sent):

    if list(sent.doc.sents).index(sent)+1==len(list(sent.doc.sents)):

        return None

    return list(sent.doc.sents)[list(sent.doc.sents).index(sent)+1]

# get previous i sentences

def previSents(sent,i):

    if i==0: return [None]

    sentslist=list(sent.doc.sents)

    sentidx=sentslist.index(sent)

    if sentidx<i:

        return [None]

    return sent.doc[sentslist[sentidx].start-i:sent.start-1]

# get anaphors in the input sentence

def getAnaphor(sent):

    anaphors=[]

    rt=[]

    for token in sent:

        if token.tag_.startswith("PRP"):#and str(doc[i]).lower()=="it":

            anaphors.append(token)

    if not anaphors:

        return (False, rt)

    for t in anaphors:

        if CONTEXE==1:

            rt.append(([sent.start,sent.end],t.i))

            continue

        startidx=previSents(sent,CONTEXE-1)

        if not startidx[0]: startidx=sent.start

        rt.append(([startidx.start,sent.end],t.i))

    return (True,rt)

# preprocess a document by adding EOL (.) when there isn't in the end of a sentence
def pp(string):
    if not string: return string
    if string[-1] not in [".","!","?"]:
        if string[-1].isalnum():
            return string+"."
        return string[:-1]+"."
    return string
# save object function
def saveObj(ob,filename):
    filehandler = open(filename, 'wb') 
    pickle.dump(ob, filehandler)
    filehandler.close()
# load object function
def loadObj(filename):
    filehandler = open(filename, 'rb') 
    obj=pickle.load(filehandler)    
    filehandler.close()
    return obj

# find all pronouns in the input sentence

def findPronouns(sent,pronouns):

    tokens=[]

    for t in sent:

        if "PRP" in t.tag_ and t.text.lower() in pronouns and t not in tokens:

            tokens.append(t)

    return tokens

def findPronounsv2(sent,pronouns,nlpcoref):

    sentDoc=nlpcoref(sent)

    tokens=[]

    for p in pronouns:

        for t in sentDoc:

            if "PRP" in t.tag_ and t.text==p and t not in tokens:

                tokens.append(t)

                break

    return sentDoc, tokens

# find the pronouns between the xml tags in the input string sentence

def getpr(sent):

    if "<referential>" in sent:

        return (sent.split("<referential>")[1]).split("</referential>")[0]

    if "<referential id=a>" in sent:

        return (sent.split("<referential id=a>")[1]).split("</referential>")[0]

    if "<referential id=b>" in sent:

        return (sent.split("<referential id=b>")[1]).split("</referential>")[0]

    return None

# remove the xml tags in the input string sentence

def removeTags(sent):

    if "<referential>" in sent:

        return sent.replace("<referential>",'',1).replace("</referential>",'',1)

    if "<referential id=a>" in sent:

        return sent.replace("<referential id=a>",'',1).replace("</referential>",'',1)

    if "<referential id=b>" in sent:

        return sent.replace("<referential id=b>",'',1).replace("</referential>",'',1)

    return None

# get all nouns and noun phrases before the input pronoun in the context (multiple sentences)
def getNPsFromContext(sents,p):
    if len(sents)==1:
        return getNPs(sents[0],p)
    nps=[]
    for sent in sents:
        if p.doc==sent:
            nps.extend(getNPs(sent,p))
        else:
            nps.extend(getAllNPsFromSent(sent))
    return nps

# get all nouns and noun phrases before the input pronoun in the input sentence
def getNPs(sent,p,include_nouns=False):

    nps=[]

    npstr=[]

    chunks = list(sent.noun_chunks)

    for i in range(len(chunks)):
        np=chunks[i]
        
        if np.end<=p.i:

            if len(np)==1:

                if np[0].pos_ not in ["NOUN","PROPN"]:

                    continue

            if np.text.lower() in npstr:

                for x in nps:

                    if x.text.lower() == np.text.lower():

                        nps.remove(x)

                npstr.remove(np.text.lower())

            nps.append(np)

            npstr.append(np.text.lower())
            
            if i < len(chunks)-1:
                np1=chunks[i+1]
                if np1.start-np.end==1:
                    if sent.doc[np.end].tag_=="CC":
                        newnp = sent.doc[np.start:np1.end]
                        if newnp.text.lower() in npstr:

                            for x in nps:

                                if x.text.lower() == newnp.text.lower():

                                    nps.remove(x)

                            npstr.remove(newnp.text.lower())

                        nps.append(newnp)

                        npstr.append(newnp.text.lower())

    if include_nouns:
        for t in sent:

            if t.i<p.i and "subj" in t.dep_ and t.pos_=="NOUN": # to revisit

                if t.text.lower() in npstr:

                    for x in nps:

                        if x.text.lower() == t.text.lower():

                            nps.remove(x)

                    npstr.remove(t.text.lower())

                npstr.append(t.text.lower())

                nps.append(sent[t.i:t.i+1])

    return nps


# get all nouns and noun phrases from the input sentence
def getAllNPsFromSent(sent,include_nouns=False):

    nps=[]

    npstr=[]

    chunks = list(sent.noun_chunks)

    for i in range(len(chunks)):
        np=chunks[i]
        if len(np)==1:

            if np[0].pos_!="NOUN":
                continue

        if np.text.lower() in npstr:

            for x in nps:

                if x.text.lower() == np.text.lower():

                    nps.remove(x)

            npstr.remove(np.text.lower())

        nps.append(np)

        npstr.append(np.text.lower())
        
        if i < len(chunks)-1:
            np1=chunks[i+1]
            if np1.start-np.end==1:
                if sent.doc[np.end].tag_=="CC":
                    newnp = sent[np.start:np1.end]
                    if newnp.text.lower() in npstr:

                        for x in nps:

                            if x.text.lower() == newnp.text.lower():

                                nps.remove(x)

                        npstr.remove(newnp.text.lower())

                    nps.append(newnp)

                    npstr.append(newnp.text.lower())

    if include_nouns:
        for t in sent:

            if "subj" in t.dep_ and t.pos_=="NOUN": # to revisit

                if t.text.lower() in npstr:

                    for x in nps:

                        if x.text.lower() == t.text.lower():

                            nps.remove(x)

                    npstr.remove(t.text.lower())

                npstr.append(t.text.lower())

                nps.append(sent[t.i:t.i+1])

    return nps   

# keep only the alphanumeric characters in a string

def stripStr(string):

    return ''.join(e for e in string if e.isalnum())

def stripStr2(string):
    string=string.replace('\xa0',' ')

    return ''.join(e for e in string if e.isalnum() or e ==' ')

# check if a noun/nounphrase is the correct answer of an sentence_id using the annotation file

def isAnswer(sent_id,np,ranswers,nlp):

    if sent_id in list(ranswers.sent_id):

        answer=ranswers.resolution[ranswers.sent_id == sent_id].iloc[0]

        if " and " in answer.lower():

            if " and " not in np.text.lower():

                return "No"

        if " and " in np.text.lower():

            if " and " not in answer.lower():

                return "No"

        if stripStr(answer) in stripStr(np.text) or nlp(answer)[0:].root.text.lower()==np.root.text.lower():
            return "Yes"
        else:
            return "No"

    return "Maybe"

def isAnswer_2(sent_id,ranswers,nlp):

    if sent_id in list(ranswers.sent_id):

        return "Unambiguous"

    return "Ambiguous"

def Answer(sent_id,ranswers):

    if sent_id in list(ranswers.sent_id):

        answer=ranswers.resolution[ranswers.sent_id == sent_id].iloc[0]

        return answer

    return "AMBIGUOUS"

def preprocess(s):
    return s.replace(' ','').lower().strip()

# check if a noun/nounphrase is the correct answer of an sentence_id using the annotation file of Conll dataset

def isConllAnswer(ents,np):

    for answer in ents:

        if stripStr(answer.text) in stripStr(np.text) or preprocess(answer.text) in preprocess(np.text) :
            
            return True

        try:
             if answer.root.text.lower()==np.root.text.lower():
                return True
        except Exception as e:
            pass

    return False

# transform pronouns and candidate antecedents to indexes to save them

def prepare_data_to_save(dataframe):

    dataframe['Pronoun'] = dataframe.apply(lambda x: x['Pronoun'].i, axis=1)

    dataframe['Candidate Antecedent'] = dataframe.apply(
        lambda x:
        [x['Candidate Antecedent'].start, x['Candidate Antecedent'].end],
        axis=1)

    return dataframe

def prepare_data_to_savev2(dataframe):

    dataframe['Pronoun']=dataframe.apply(lambda x: x['Pronoun'].i, axis=1)

    dataframe['CandidateAntecedent']=dataframe.apply(lambda x: [x['CandidateAntecedent'].start,x['CandidateAntecedent'].end], axis=1)

    return dataframe

# transform pronouns and candidate antecedents from indexes to Tokens and Spans after loading them

def loadSavedData(dataframe):

    dataframe['Pronoun']=dataframe.apply(lambda x: x['Context'][x['Pronoun']], axis=1)

    dataframe['Candidate Antecedent']=dataframe.apply(lambda x: x['Context'][x['Candidate Antecedent'][0]:x['Candidate Antecedent'][1]], axis=1)

    return dataframe

def loadSavedDatav2(dataframe):

    dataframe['Pronoun']=dataframe.apply(lambda x: x['Context'][x['Pronoun']], axis=1)

    dataframe['CandidateAntecedent']=dataframe.apply(lambda x: x['Context'][x['CandidateAntecedent'][0]:x['CandidateAntecedent'][1]], axis=1)

    return dataframe

def printScores(ytest,ypred):

    print("Accuracy:",metrics.accuracy_score(ytest, ypred))

    print("Precision:",metrics.precision_score(ytest, ypred))

    print("Recall:",metrics.recall_score(ytest, ypred))

def heatmap(x, y, **kwargs):
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors) 

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size], 
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')

    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False) # Hide grid
        ax.set_facecolor('white') # Make background white
        ax.set_xticks([]) # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right() # Show vertical ticks on the right 


def corrplot(data, size_scale=500, marker='s'):
    corr = pd.melt(data.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    heatmap(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale
    )

def isNeuralCorefAnswer(sent,np,pr):
    if not sent._.has_coref:
        return None
    res=None
    for clr in sent._.coref_clusters:
        if sent[pr.i:pr.i+1] in clr:
            for itm in clr:
                if len(itm)==1:
                    if itm[0].pos_=="NOUN": 
                        res=itm
                elif len(itm)>1: 
                    res=itm
    if res:
        return stripStr(res.text) in stripStr(np.text) or res.root.text.lower()==np.root.text.lower()
    return None
# get coreferences from input document using CoreNLP
def getCorefsCoreNLP(document):
    mychains = list()
    chains = document.corefChain
    for chain in chains:
        mychain = list()
        # Loop through every mention of this chain
        for mention in chain.mention:
            # Get the sentence in which this mention is located, and get the words which are part of this mention
            # (we can have more than one word, for example, a mention can be a pronoun like "he", but also a compound noun like "His wife Michelle")
            words_list = document.sentence[mention.sentenceIndex].token[mention.beginIndex:mention.endIndex]
            #build a string out of the words of this mention
            ment_word = ' '.join([x.word for x in words_list])
            mychain.append(ment_word)
        mychains.append(mychain)
    return mychains

def getAntecedentCoreNLP(document,pronoun):
    mychains = list()
    chains = document.corefChain
    for chain in chains:
        mychain = list()
        # Loop through every mention of this chain
        found=False
        for mention in chain.mention:
            # Get the sentence in which this mention is located, and get the words which are part of this mention
            # (we can have more than one word, for example, a mention can be a pronoun like "he", but also a compound noun like "His wife Michelle")
            words_list = document.sentence[mention.sentenceIndex].token[mention.beginIndex:mention.endIndex]
            #build a string out of the words of this mention
            ment_word = ' '.join([x.word for x in words_list])
            if ment_word == pronoun:
                found=True
                continue
            mychain.append(ment_word)
        if found:
            return mychain
        mychains.append(mychain)
    return []

class coreference_tagged_sentence:

    def __init__(self, r_c_matrix):

        self.part_number = r_c_matrix[0][0]

        self.words = []

        self.links = []               # list of all links in this sentence

        self.chain_hash = {}          # hash of all links in this sentence

        chain_start_hash = {}



        for r_i, r in enumerate(r_c_matrix):

            assert self.part_number == r[0], "all rows should contain the same part number"

            self.words.append(r[1])

            # process encoded chain

            encoded_chain=r[2]

            bits = encoded_chain.split("|")


            ids = []

            for i in range(0, len(bits)):

                id = bits[i].replace("(", "")

                id = id.replace(")", "")

                ids.append(id)



            assert len(ids) == len(bits), "the length of ids and bits should be the same"



            for i in range(0, len(bits)):

                if(bits[i].startswith("(")):

                    if(ids[i] not in chain_start_hash):

                        chain_start_hash[ids[i]] = []

                    chain_start_hash[ids[i]].append(r_i)


                if(bits[i].endswith(")")):


                    if(ids[i] not in chain_start_hash):

                        print (chain_start_hash)

                        raise Exception("problem, found link end without a start")


                    try:

                        a_link = link(ids[i], chain_start_hash[ids[i]].pop(), r_i)


                        self.links.append(a_link)


                        if(ids[i] not in self.chain_hash):

                            self.chain_hash[ids[i]] = []


                        self.chain_hash[ids[i]].append(a_link)

                    except:

                        sys.stderr.write("WARNING: dropping link with id [%s]" % (ids[i]))

     

        for k, v in chain_start_hash.items():

            if( len(v) != 0):

                raise Exception("all the lists in the start hash should be empty")


        self.links.sort()


    def __repr__(self):


        coref_tagged_words = []

        coref_tagged_words.extend(self.words)


        # make words sgml safe

        for i in range(0, len(coref_tagged_words)):

            coref_tagged_words[i] = make_sgml_safe(coref_tagged_words[i])

            

        for a_link in self.links:

            coref_tagged_words[a_link.start] = """<C=%s>%s""" % (a_link.id, coref_tagged_words[a_link.start])

            coref_tagged_words[a_link.end] = "%s</C>" % (coref_tagged_words[a_link.end])


        return "%s" % (" ".join(coref_tagged_words))



class link:

    def __init__(self, id, start, end):

        self.id = id

        self.start = start

        self.end = end

    def __repr__(self):

        return "<link: %s:%s:%s>" % (self.id, self.start, self.end)

    def __eq__(self, other):

        return ((self.start, self.end) ==

                (other.start, other.end))

    def __lt__(self, other):

        if(self.start != other.start):

            return (self.start < other.start)

        else:

            return (self.end < other.end)

def make_sgml_safe(s, reverse=False, keep_turn=True):

    """ return a version of the string that can be put in an sgml document


    This means changing angle brackets and ampersands to '-LAB-',

    '-RAB-', and '-AMP-'.  Needed for creating ``.name`` and

    ``.coref`` files.


    If keep_turn is set, <TURN> in the input is turned into [TURN], not turned into -LAB-TURN-RAB-


    """


    if not reverse and keep_turn:

        s = s.replace("<TURN>", "[TURN]")


    for f, r in [("<", "-LAB-"),

                 (">", "-RAB-"),

                 ("&", "-AMP-")]:

        if reverse:

            r, f = f, r

        s = s.replace(f, r)


    return s

def expand_document_id(document_id, language):

    abbr_language=""

    if language == "english":

        abbr_language = "en"

    file_bit=document_id[-4:]

    genre_bit, source_bit, ignore = document_id.split("/", 2)

    constant="%s@on" % (abbr_language)

    return "%s@%s@%s@%s@%s" % (document_id, file_bit, source_bit, genre_bit, constant)

# get all nouns and noun phrases from the input sentence

def getNPsfromsent(sent):

    nps=[]

    npstr=[]

    chunks = list(sent.noun_chunks)
    
    for i in range(len(chunks)):

        np = chunks[i]

#         if len(np)==1:

#             if np[0].pos_ not in ["NOUN","PROPN"]:

#                 continue

        if np.text.lower() in npstr:

            for x in nps:

                if x.text.lower() == np.text.lower():

                    nps.remove(x)

            npstr.remove(np.text.lower())

        nps.append(np)

        npstr.append(np.text.lower())

        if i < len(chunks)-1:
            np1=chunks[i+1]
            if np1.start-np.end==1:
                if sent[np.end].tag_=="CC":
                    newnp = sent[np.start:np1.end]
                    if newnp.text.lower() in npstr:

                        for x in nps:

                            if x.text.lower() == newnp.text.lower():

                                nps.remove(x)

                        npstr.remove(newnp.text.lower())

                    nps.append(newnp)

                    npstr.append(newnp.text.lower())

    for t in sent:

        if "subj" in t.dep_ or t.pos_ in ["NOUN","PROPN"] or "PRP" in t.tag_: # to revisit

            if t.text.lower() in npstr:

                for x in nps:

                    if x.text.lower() == t.text.lower():

                        nps.remove(x)

                npstr.remove(t.text.lower())

            npstr.append(t.text.lower())

            nps.append(sent[t.i:t.i+1])

    return nps



def matches(doc,terms,nlp):

    matcher = PhraseMatcher(nlp.vocab)

    patterns = [nlp.make_doc(text) for text in terms]

    matcher.add("TerminologyList", None, *patterns)

    matches = matcher(doc)

    spans=[]

    spanstxt={}

    for match_id, start, end in matches:

        if doc[start:end].text in spanstxt:

            spans.remove(doc[spanstxt[doc[start:end].text][0]:spanstxt[doc[start:end].text][1]])

            spanstxt.pop(doc[start:end].text)

        for spantxt in list(spanstxt):

            if doc[start:end].text.replace(" ","") == spantxt.replace(" ",""):

                spans.remove(doc[spanstxt[spantxt][0]:spanstxt[spantxt][1]])

                spanstxt.pop(spantxt)

        spans.append(doc[start:end])

        spanstxt[doc[start:end].text]=[start,end]

    return spans
def getEnt(sent,nlp,DEBUG=False):
    osent=sent.strip()
    ids=[]
    ents={}
    nps={}
    pronouns={}
    while "<C=" in sent:
        Id=sent.split("<C=")[1].split(">")[0]
        ent=sent.split("<C=")[1].split(">")[1].split("<")[0].strip()
        sent=re.sub(r"<C=[0-9]+>","",sent,1)
        sent=sent.replace("</C>","",1)
        if not ent: continue
        ids.append(Id)
        ents[ent]=Id
    sent=nlp(sent)  
    ms=matches(sent,list(ents),nlp)
    if DEBUG:
        if len(ms)==0:
            print(ents,sent)
    if len(ms)==len(ents):
        for match, ent in zip(ms,ents):
            nps[match]=ents[ent]
    else:
        for span in ms:
            if hasIndex(ents,span.text):
                nps[span]=ents[span.text]
            else:
                if DEBUG:
                    print(sent,span,ents)    
    return sent, ids, nps, getPronounsFromEnts(nps)

def getPronounsFromEnts(ents):

    prs={}

    for ent in ents:

        if len(ent)==1:

            if "PRP" in ent[0].tag_:

                prs[ent[0]]=ents[ent]

    return prs

def removePronounsFromEnts(ents):

    for ent in list(ents):

        if len(ent)==1:

            if "PRP" in ent[0].tag_:

                ents.pop(ent)

    return ents

def getItems(dic,val):

    items={}

    for k,v in dic.items():

        if v==val: items[k]=v

    return items

def hasIndex(arr,ix):

    try:

        t=arr[ix]

        return True

    except:

        return False

def classify(clf,X_train,X_test,y_train,y_test):
    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print(classification_report(y_test,y_pred))

def iscorefanswer(sent,pr,CA):
    a=neuralcorefresolve(sent,pr)
    ca=CA.text
    if a:
        if stripStr(a.text) in stripStr(ca) or a.root.text.lower()==CA.root.text.lower():
            return "Yes"
        else:
            return "No"
    return "Maybe"

def neuralcorefresolve(sent,pr):
    ss=sent
    if ss._.has_coref:
        for v in ss._.coref_clusters:
            for p in v:
                if p[0]==pr:
                    return v[0]

def classification_report_with_accuracy_score(y_true, y_pred):
    print(classification_report(y_true, y_pred)) # print classification report
    return metrics.accuracy_score(y_true, y_pred) # return accuracy score


def scoringArgMax(clf, x, y,Class,MultiClass=False):
    ypred=clf.predict(x)
    cr=classification_report(y,[np.argmax(pred) for pred in ypred],output_dict=True)
    if MultiClass:
        keys=list(cr.keys())
        scores=[]
        classes=[]
        for i in range(0,len(keys)-3):
            precision=cr[keys[i]]['precision']
            recall=cr[keys[i]]['recall']
            accuracy=cr['accuracy']
            scores.append([float("{:.3f}".format(precision)), float("{:.3f}".format(recall)), float("{:.3f}".format(accuracy))])
            classes.append(keys[i])
        return scores, classes
    else:
        precision = cr[str(Class)]['precision']
        recall = cr[str(Class)]['recall']
        accuracy = cr['accuracy']
        return [float("{:.3f}".format(precision)), float("{:.3f}".format(recall)), float("{:.3f}".format(accuracy))],str(Class)

def scoring(clf, x, y,Class,MultiClass=False):
    ypred=clf.predict(x)
    cr=classification_report(y,ypred,output_dict=True)
    if MultiClass:
        keys=list(cr.keys())
        scores=[]
        classes=[]
        for i in range(0,len(keys)-3):
            precision=cr[keys[i]]['precision']
            recall=cr[keys[i]]['recall']
            accuracy=cr['accuracy']
            scores.append([float("{:.3f}".format(precision)), float("{:.3f}".format(recall)), float("{:.3f}".format(accuracy))])
            classes.append(keys[i])
        return scores, classes
    else:
        precision = cr[str(Class)]['precision']
        recall = cr[str(Class)]['recall']
        accuracy = cr['accuracy']
        return [float("{:.3f}".format(precision)), float("{:.3f}".format(recall)), float("{:.3f}".format(accuracy))],str(Class)

def ScoringFB(clf, x, y,Class,MultiClass=False,B=1):
    ypred=clf.predict(x)
    cr=classification_report(y,ypred,output_dict=True)
    if MultiClass:
        keys=list(cr.keys())
        scores=[]
        classes=[]
        for i in range(0,len(keys)-3):
            precision=cr[keys[i]]['precision']
            recall=cr[keys[i]]['recall']
            FB=(1+B*B)*precision*recall/(B*B*precision+recall)
            scores.append([float("{:.3f}".format(precision)), float("{:.3f}".format(recall)), float("{:.3f}".format(FB))])
            classes.append(keys[i])
        return scores
    else:
        precision = cr[str(Class)]['precision']
        recall = cr[str(Class)]['recall']
        FB=(1+B*B)*precision*recall/(B*B*precision+recall)
        return [float("{:.3f}".format(precision)), float("{:.3f}".format(recall)), float("{:.3f}".format(FB))]

def biScoringFB(clf, x, y,B=1):
    ypred=clf.predict(x.drop("Id",axis=1))
    tp,fp,tn,fn=0,0,0,0
    ypredx=y.copy()
    for p,i in zip(ypred,y.index):
        ypredx[i]=p

    for Id in x.Id.unique():
        idx=x[x.Id==Id].index
        if 0 in list(y[idx]):
            if 0 in list(ypredx[idx]):
                tp+=1
            else:
                fn+=1
        else:
            if 0 in list(ypredx[idx]):
                fp+=1
            else:
                tn+=1

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    FB=((1+B*B)*tp)/((1+B*B)*tp+B*B*fn+fp)
    return [float("{:.3f}".format(precision)), float("{:.3f}".format(recall)), float("{:.3f}".format(FB))]


def evalclf(clf, x,y,sampler=None,cv=10,log=False,Train_score=False,Class="Maybe",MultiClass=False):
    cv_scores = [["",0.,0.,0.],["",0.,0.,0.],["",0.,0.,0.]]
    cv_score = [0.,0.,0.]
    train_scores = 0.
    test_scores = 0.
    model=clf

    skf = StratifiedKFold(n_splits=cv, shuffle=True)

    for train_idx, test_idx in skf.split(x,y):
        clf=clone(model)
        if sampler:
            try:
                xfold_train, yfold_train = sampler.fit_resample(x.iloc[train_idx],y.iloc[train_idx])
            except:
                xfold_train, yfold_train = x.iloc[list(train_idx)],y.iloc[list(train_idx)]
        else:
            xfold_train, yfold_train = x.iloc[list(train_idx)],y.iloc[list(train_idx)]
        clf.fit(xfold_train, yfold_train)

        if Train_score:
            train_scores, classes = scoring(clf, xfold_train, yfold_train,Class,MultiClass)
            if log:
                print("Train AUPRC:",train_scores, "Test AUPRC:",test_scores)

        test_scores, classes = scoringF1(clf, x.iloc[list(test_idx)], y.iloc[list(test_idx)],Class,MultiClass)
    
        # for i in range(0,len(classes)):
        #     cv_scores[i][0] =""
        #     cv_scores[i][1] =0
        #     cv_scores[i][2] =0
        #     cv_scores[i][3] =0
        if MultiClass:
            for i in range(0,len(classes)):
                cv_scores[i][0]=classes[i]
                cv_scores[i][1] += test_scores[i][0]
                cv_scores[i][2] += test_scores[i][1]
                cv_scores[i][3] += test_scores[i][2]

        else:
            cv_score[0] += test_scores[0]
            cv_score[1] += test_scores[1]
            cv_score[2] += test_scores[2]
    if MultiClass:
        return [(l[0],[float("{:.3f}".format(s/cv)) for s in l[1:]]) for l in cv_scores]        
    else:
        return [(float("{:.3f}".format(s/cv)),float("{:.3f}".format(np.std(s)))) for s in cv_score]

def newevalCV(clf,Xids,Yids, x,y,sampler=None,cv=10,log=False,Train_score=False,Class="Maybe",MultiClass=False,B=1,bi=False):
    cv_score = [0.,0.,0.]
    model=clf

    skf = StratifiedKFold(n_splits=cv, shuffle=True)

    for train_idx, test_idx in skf.split(Xids,Yids):
        clf=clone(model)
        train_idx=x[x.Id.isin(Xids[train_idx])].index
        test_idx=x[x.Id.isin(Xids[test_idx])].index
        if sampler:
            try:
                xfold_train, yfold_train = sampler.fit_resample(x.drop("Id",axis=1).loc[train_idx],y.loc[train_idx])
            except:
                xfold_train, yfold_train = x.drop("Id",axis=1).loc[train_idx],y.loc[train_idx]
        else:
            xfold_train, yfold_train = x.drop("Id",axis=1).loc[train_idx],y.loc[train_idx]
        clf.fit(xfold_train, yfold_train)

       
        if bi:
            test_scores = biScoringFB(clf, x.loc[test_idx], y.loc[test_idx],B=B)
        else:
            test_scores = ScoringFB(clf, x.loc[test_idx].drop("Id",axis=1), y.loc[test_idx],Class=Class,B=B)

    

        cv_score[0] += test_scores[0]
        cv_score[1] += test_scores[1]
        cv_score[2] += test_scores[2]

    return [float("{:.3f}".format(s/cv)) for s in cv_score]

def groupTTsplit(x,y,group,ratio,random_state=42):
    Yid=[]
    for Id in group:
        if 0 in y[x[x.Id==Id].index].unique():
            Yid.append(0)
        else:
            Yid.append(1)
    X_train, X_test, y_train, y_test = train_test_split(group,Yid,test_size=ratio,random_state=random_state,stratify=Yid)

    return x[x.Id.isin(X_train)].index.values.astype(int),x[x.Id.isin(X_test)].index.values.astype(int)

def allmodels(models,X,y,use_over_under=False,Class="0"):
    for m in models:
        clf=models[m]
        print("=====",clf)
        score = evalclf(clf, X, y, Class=Class)
        print("PRA scores:",str(score[0])+"\t"+str(score[1])+"\t"+str(score[2]))
        if use_over_under:

            score,c = evalclf(clf, X, y, RandomOverSampler(),Class=Class)
            print("Random over-sampling:",score)

            # Logistic regression score with SMOTE

            score,c = evalclf(clf, X, y, SMOTE(),Class=Class)
            print("SMOTE over-sampling:",score)

            # Logistic regression score with ADASYN

            score,c = evalclf(clf, X, y, ADASYN(),Class=Class)
            print("ADASYN over-sampling:",score)

            # Logistic regression score with Random Under Sampling

            score,c = evalclf(clf, X, y, RandomUnderSampler(),Class=Class)
            print("Random under-sampling:",score)

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = 50#len(next(iter(word2vec.values())))

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = 50#len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text

def ensembleproba(ypred1,ypred2,theta=0):
    ypred=[]
    for l1,l2 in zip(ypred1,ypred2):
        if l1[0]>l1[1] and l2[0]>l2[1]:
            ypred.append("Ambiguous")
            continue
        if l1[1]> l1[0] and l2[1]>l2[0]:
            ypred.append("Unambiguous")
            continue
        if (l1[0]>l1[1] and l1[0]>l2[1]) or (l2[0]>l2[1] and l2[0]>l1[1]):
            ypred.append("Ambiguous")
            continue
        if (l1[1]>l1[0] and l1[1]>l2[0]) or (l2[1]>l2[0] and l2[1]>l1[0]):
            ypred.append("Unambiguous")
            continue
        ypred.append("Ambiguous")
    return ypred 

def ensembleprobamulti(ypred1,ypred2,Class=0,theta=0):
    ypred=[]
    for l1,l2 in zip(ypred1,ypred2):
        m1=np.max(l1)
        m2=np.max(l2)
        d1=np.where(l1==m1)[0][0]
        d2=np.where(l2==m2)[0][0]
        if d1==Class or d2==Class:
            ypred.append(Class)
            continue
        if  d1==d2:
            ypred.append(d1)
            continue
        if m1>m2:
            ypred.append(d1)
            continue
        if m2>m1:
            ypred.append(d2)
            continue
        ypred.append(0)
    return ypred   

def ensemblescoring(clf1, x1, clf2, x2, y,Class):
    ypred1=clf1.predict_proba(x1)
    ypred2=clf2.predict_proba(x2)
    return ensemblescoringF1(ypred1,ypred2,y,Class)

def ensemblescoringF1(ypred1,ypred2, y,Class):
    ypred=ensembleprobamulti(ypred1,ypred2,Class)
    cr=classification_report(y,ypred,output_dict=True)
    precision = cr[str(Class)]['precision']
    recall = cr[str(Class)]['recall']
    F1 = 2*precision*recall/(precision+recall)
    return [float("{:.3f}".format(precision)), float("{:.3f}".format(recall)), float("{:.3f}".format(F1))]

def ensemblescoringA(ypred1,ypred2, y,Class):
    ypred=ensembleprobamulti(ypred1,ypred2,Class)
    cr=classification_report(y,ypred,output_dict=True)
    precision = cr[str(Class)]['precision']
    recall = cr[str(Class)]['recall']
    accuracy = cr['accuracy']
    return [float("{:.3f}".format(precision)), float("{:.3f}".format(recall)), float("{:.3f}".format(accuracy))]


def ensembleval(clf1, x1,clf2,x2, y,cv=10,Class=0):
    cv_score = [0.,0.,0.]
    test_scores = [0.,0.,0.]

    skf = StratifiedKFold(n_splits=cv, shuffle=False)

    models=[clf1,clf2]
    for train_idx, test_idx in skf.split(x1,y):

        clf1=clone(models[0])
        clf2=clone(models[1])

        x1_train, x2_train, y_train = x1.iloc[train_idx], x2.iloc[train_idx],y.iloc[train_idx]
        x1_test, x2_test, y_test = x1.iloc[test_idx], x2.iloc[test_idx],y.iloc[test_idx]
        
        clf1.fit(x1_train, y_train)
        clf2.fit(x2_train, y_train)

        test_scores = ensemblescoring(clf1, x1_test, clf2, x2_test, y_test,Class)

        cv_score[0] += test_scores[0]
        cv_score[1] += test_scores[1]
        cv_score[2] += test_scores[2]

    return [float("{:.3f}".format(s/cv)) for s in cv_score]

def ensemblevalTT(clf1, x1_train,x1_test,clf2,x2_train,x2_test, y_train,y_test,Class=0):
    test_scores = [0.,0.,0.]
    models=[clf1,clf2]
    clf1=clone(models[0])
    clf2=clone(models[1])    
    clf1.fit(x1_train, y_train)
    clf2.fit(x2_train, y_train)
    test_scores = ensemblescoring(clf1, x1_test, clf2, x2_test, y_test,Class)
    return [float("{:.3f}".format(s)) for s in test_scores]

def ensemblevalparallel(clf1, x1,clf2,x2, y,cv=10,Class=0):
    
    skf = StratifiedKFold(n_splits=cv, shuffle=False)

    models=[clf1,clf2]
    imap=[]
    for train_idx, test_idx in skf.split(x1,y):
        imap.append((train_idx, test_idx,clf1,x1,clf2,x2,y,models,Class))

    with multiprocessing.Pool() as pool:
        cv_score = pool.starmap(ensembleparellel, imap)


    precisions=[s[0] for s in cv_score]
    recalls=[s[1] for s in cv_score]
    accuracies=[s[2] for s in cv_score]

    return [float("{:.3f}".format(s)) for s in [mean(precisions),mean(recalls),mean(accuracies)]]


def ensembleparellel(t):
    (train_idx, test_idx,clf1,x1,clf2,x2,y,models,Class) = t
    clf1=models[0]
    clf2=models[1]

    x1_train, x2_train, y_train = x1.iloc[train_idx], x2.iloc[train_idx],y.iloc[train_idx]
    x1_test, x2_test, y_test = x1.iloc[test_idx], x2.iloc[test_idx],y.iloc[test_idx]
    
    clf1.fit(x1_train, y_train)
    clf2.fit(x2_train, y_train)

    return ensemblescoring(clf1, x1_test, clf2, x2_test, y_test,Class)

    

def weval(clf, x,y,sampler=None,cv=10,Train_score=False,Class="Ambiguous", MultiClass=False):
    cv_scores = [["",0.,0.,0.],["",0.,0.,0.],["",0.,0.,0.]]
    cv_score = [0.,0.,0.]
    train_scores = 0.
    test_scores = 0.
    
    skf = StratifiedKFold(n_splits=cv, random_state=0, shuffle=False)

    for train_idx, test_idx in skf.split(x,y):
        
        X_train, X_test, y_train, y_test = np.array(x)[train_idx], np.array(x)[test_idx], y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)

        if Train_score:
            train_scores, classes = scoring(clf, X_train, y_train,Class,MultiClass)
            print("Train AUPRC:",train_scores, "Test AUPRC:",test_scores)

        test_scores, classes = scoring(clf, X_test, y_test, Class, MultiClass)

        cv_score[0] += test_scores[0]
        cv_score[1] += test_scores[1]
        cv_score[2] += test_scores[2]

    return [float("{:.3f}".format(s/cv)) for s in cv_score]



##### BERT ######


def bert_cv(model,x,y,cv=10, Train_score=False, Class="Ambiguous", verbose=True,epochs=1,batch_size=64):
    X_train=[0,1,2]
    X_test=[0,1,2]
    y=encode_label(y)
    cv_score = np.array([0.,0.,0.])
    cv_train = np.array([0.,0.,0.])
    train_scores = 0.
    test_scores = 0.
    skf = StratifiedKFold(n_splits=cv, random_state=0, shuffle=False)
    i=1
    for train_idx, test_idx in skf.split(x[0],y):
        X_train[0],X_train[1],X_train[2]=x[0][train_idx],x[1][train_idx],x[2][train_idx]
        X_test[0],X_test[1],X_test[2]=x[0][test_idx],x[1][test_idx],x[2][test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # model= getBertModel(len(x[0][0]),y)
        model, predicted_prob, predicted=fit_bert_classif(X_train, y_train, X_test, encode_y=False, 
                                                    model=model, epochs=epochs, batch_size=batch_size,verbose=False)

        if Train_score:
            train_scores, classes = scoringArgMax(model, X_train, y_train, Class)
            cv_train+=np.array(train_scores)
            if verbose:
                print("iteration",i)
                print("Train scores === precision",train_scores[0],"recall",train_scores[1],"accuracy",train_scores[2])

        cr=classification_report(y_test,predicted,output_dict=True)
        precision=cr[str(Class)]['precision']
        recall=cr[str(Class)]['recall']
        accuracy=cr['accuracy']
        test_score=np.array([float("{:.3f}".format(precision)), float("{:.3f}".format(recall)), float("{:.3f}".format(accuracy))])
        print("Test scores === precision",test_score[0],"recall",test_score[1],"accuracy",test_score[2])
        cv_score+=test_score
        i+=1
    if Train_score:
        final_train_scores=[float("{:.3f}".format(s/cv)) for s in cv_train]
    final_test_scores=[float("{:.3f}".format(s/cv)) for s in cv_score]
    if verbose:
        if Train_score:
            print("AVG Train scores with",str(cv)+"fold CV: precision",final_train_scores[0],"recall",final_train_scores[1],"accuracy",final_train_scores[2])
        print("AVG Test scores with",str(cv)+"fold CV: precision",final_test_scores[0],"recall",final_test_scores[1],"accuracy",final_test_scores[2])
    return final_train_scores, final_test_scores

def getBertModel(shape, y, nlp,verbose=False):
    ## inputs
    idx = layers.Input((shape), dtype="int32", name="input_idx")
    masks = layers.Input((shape), dtype="int32", name="input_masks")
    ## pre-trained bert with config   
    bert_out = nlp(idx, attention_mask=masks)[0]
    ## fine-tuning
    x = layers.GlobalAveragePooling1D()(bert_out)
    x = layers.Dense(64, activation="relu")(x)
    y_out = layers.Dense(len(np.unique(y)), activation='softmax')(x)
    ## compile
    model = models.Model([idx, masks], y_out)
    for layer in model.layers[:3]:
        layer.trainable = False
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if verbose:
        model.summary()
    return model


def encode_label(y_train,verbose=True):
    dic_y_mapping = {n:label for n,label in enumerate(np.unique(y_train))}
    inverse_dic = {v:k for k,v in dic_y_mapping.items()}
    print(inverse_dic)
    y_train = np.array( [inverse_dic[y] for y in y_train] )
    return y_train

def tokenize(txt,tokenizer):
    return " ".join(tokenizer.tokenize(re.sub('r[^\w\s]+\n','', str(txt).lower().strip())))

def resetmodels():
    return {
    'dt' : DecisionTreeClassifier(random_state=0),
    'mlp': MLPClassifier(max_iter=100,random_state=0),
    'knn': KNeighborsClassifier(),
    'lr' : LogisticRegression(penalty='l2', random_state=0),
    'gnb': GaussianNB(),
    'rf' : RandomForestClassifier(random_state=0),
    'svm': SVC(random_state=0,probability=True)}

def getpredictionScores(test,y_pred,idx,Wy,Set,Class="Ambiguous",B=2,f=3):
    di={i:[] for i in idx[test].unique()}
    dd={i:"" for i in idx[test].unique()}
    for i,p in zip(idx[test],y_pred):
        di[i].append(p)
    for k,v in di.items():
        QNY=[0,0,0]
        Yw=0
        for l in v:
            l=l.tolist()
            m=max(l)
            QNY[l.index(m)]+=1
            if l[2]<Wy and l[2]>0: 
                Yw+=1
        if (QNY[2]==1 and QNY[0]==0) or (QNY[2]==1 and Yw==0):
            #Not ambiguous
            dd[k]=1
        else:
            #Ambiguous
            dd[k]=0
    newdf = Set[Set['Id'].isin(idx[test])][["Id", "Answer"]]
    newdf['result']=newdf.apply(lambda x: dd[x['Id']] ,axis=1)
    cr=classification_report(newdf['Answer'],newdf['result'],output_dict=True)
    precision = cr[Class]['precision']
    recall = cr[Class]['recall']
    fb = (1+B*B)*precision*recall/(B*B*precision+recall)
    return float("{:.3f}".format(precision)), float("{:.3f}".format(recall)), float("{:.3f}".format(fb))

## Resolution 

def proba2pred(proba):
    pred=proba.copy()
    for p,i in zip(proba,proba.index):
        pred[i]=np.argmax(p)
    return pred

def mindex(y_pred,y_test):
    return pd.Series(y_pred.tolist(), index =y_test.index)

def resolve(y_proba,X_test,y_test,gt3,theta=0.5):
    ci=[0,0]
    for Id in X_test.Id.unique():
        if 0 not in y_test[X_test[X_test.Id==Id].index].unique():
            preds=proba2pred(y_proba[X_test[X_test.Id==Id].index])
            if 1 in preds.values:
                if list(preds.values).count(1)==1:
                    if y_test[preds[preds==1].index[0]]==1:
                        ci[0]+=1
                    else:
                        ci[1]+=1
                else:
                    if theta!=0:
                        above_theta=[]
                        for i in preds[preds==1].index:
                            #print(y_proba[i][1],preds[i],i)
                            if y_proba[i][1]>=theta:
                                above_theta.append(i)
                        if len(above_theta)!=1:
                            ci[1]+=1
                        elif preds[above_theta[0]]==1:
                            ci[0]+=1
                        else:
                            ci[1]+=1
                    else:
                        maxs=[]
                        mx=y_proba[preds[preds==1].index].apply(lambda x: x[1]).max()
                        for i,p in zip(preds[preds==1].index,y_proba[preds[preds==1].index]):
                            if p[1]==mx:
                                maxs.append(i)
                        if len(maxs)>1:
                            trm=[]
                            for i in range(0,len(maxs)-1):
                                for j in range(i+1,len(maxs)):
                                    if gt3["Candidate Antecedent"][maxs[i]]==gt3["Candidate Antecedent"][maxs[j]]:
                                        trm.append(i)
                            maxs=np.delete(maxs,trm)
                            if len(maxs)>1:
                                ci[1]+=1
                            elif preds[maxs[0]]==1:
                                ci[0]+=1
                            else:
                                ci[1]+=1
                        elif preds[maxs[0]]==1:
                            ci[0]+=1
                        else:
                            ci[1]+=1
            else:
                ci[1]+=1
    return ci[0]/(ci[0]+ci[1])


def ensembleprobas(y1,y2):
    y=[]
    for p1,p2 in zip(y1,y2):
        y.append(list(np.maximum(p1,p2)))
    return np.array(y)


def baselineResolve(xx,context=0):
    ci=[0,0]
    for Id in xx.Id.unique():
        CAs=xx[xx.Id==Id]["Candidate Antecedent"].unique()
        if context==0:
            if len(CAs)>1:
                ci[1]+=1
            elif xx[xx.Id==Id][xx["Candidate Antecedent"]==CAs[0]].Answer.unique()[0]==1:
                ci[0]+=1
            else:
                ci[1]+=1
        elif context==1:
            c=xx[xx.Id==Id].Context.unique()[0]
            sents=sent_tokenize(c.text)
            CIS=[]
            for CA in CAs:
                if CA.text in sents[-1]:
                    CIS.append(CA)
            if len(CIS)!=1:
                ci[1]+=1
            elif xx[xx.Id==Id][xx["Candidate Antecedent"]==CIS[0]].Answer.unique()[0]==1:
                ci[0]+=1
            else:
                ci[1]+=1
    return ci[0]/(ci[0]+ci[1])

def getantecedents(token):
    antecedents=[]
    for cluster in token._.coref_clusters:
        for span in cluster:
            if token in span:
                continue
            antecedents.append(span)
    return antecedents

def getAntsCoreNLP(context,pronoun):
    document = client.annotate(context)
    return getAntecedentCoreNLP(document,pronoun)

def toText(ls):
    ls1=[]
    for s in ls:
        ls1.append(s.text)
    return ls1

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'