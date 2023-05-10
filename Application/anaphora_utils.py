# ignore warnings
import warnings
warnings.filterwarnings('ignore')
# import basic libraries to read from docx and xml files
import os
import time
from xml.dom import minidom
# import nlp libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import verbnet as vn
import spacy
from spacy.tokens import Span
# import neuralcoref
from benepar.spacy_plugin import BeneparComponent
from spacy.matcher import PhraseMatcher
import stanza
from stanza.server import CoreNLPClient
# import utils libraries
from tqdm.auto import tqdm
import pickle
import copy
import re
import sys
# import arrays and dataframes libraries
import pandas as pd
import numpy as np
# import visualisation libraries
from matplotlib import pyplot as plt
# import machine learning algorithms
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.tree import export_graphviz
from six import StringIO  
import pydotplus
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV



def startCoreNLPClient():
    client = CoreNLPClient(
        annotators=['tokenize','ssplit', 'pos', 'lemma', 'ner','coref'], 
        memory='4G', 
        endpoint='http://localhost:9001',
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

    if noun_phrase.start < pronoun.i:

        return pronoun.i-noun_phrase.end

    if noun_phrase.start > pronoun.i:

        return noun_phrase.start-pronoun.i

    return -1

# get the number of a noun phrase

def number_np(np):

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

#     try:

#         if token.label_.equals("PERSON"):

#             return True

#     except:

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

    if token._.parent:
	    pS=token._.parent._.parse_string

	    if pS.split(" ")[0]=="(PP":

	        return True

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

    try:
        return anaphor.nbor().pos_
    except:
        return "."

# F18

def hasClause(doc):

    clauselist=["csubj","ccomp","xcomp","advcl","acl"]

    for t in doc:

        if t.dep_ in clauselist:

            return True

    return False

# F19

def relativePositionAnaphor(anaphor):

    return (anaphor.i-anaphor.sent.start)/len(anaphor.sent)

# get the first preceding noun to an input token

def firstPrecedingNoun(token):

    for i in range(1,token.i):

        if token.nbor(-i).pos_=="NOUN":

            return token.nbor(-i)

    return None

# F20

def relativePositionToNoun(anaphor):

    noun=firstPrecedingNoun(anaphor)

    if noun!=None:

        if noun.sent==anaphor.sent:

            return (anaphor.i-noun.i)/len(anaphor.sent)

    return -1

# F21

def relativePositionSentence(sent):

    return list(sent.doc.sents).index(sent)+1/len(list(sent.doc.sents))

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

def numPrecedingNPsPara(anaphor):

    return len([x for x in anaphor.doc.noun_chunks if x.end<=anaphor.i])

# F28

def numSucceedingNPsSent(anaphor):

    return len([x for x in anaphor.doc.noun_chunks if x.start>anaphor.i])

# F29

def numNPsInSent(anaphor):

    return len(list(anaphor.sent.noun_chunks))

# F30

def numNPsInPara(anaphor):

    return len(list(anaphor.doc.noun_chunks))

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

            try:
                if anaphor.nbor(i).pos_=="ADJ":

                    return True
            except:
                    return False

    return False

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

            try:
                if anaphor.nbor(i).dep_=="mark":

                    return True
            except:
                return False

    return False

# F42

def isprecedingPrep(anaphor):

    return anaphor.nbor(-1).dep_=="prep"

# F43

def precedingWord(anaphor):

    return anaphor.nbor(-1).lemma_

# F44

def nextWord(anaphor):

    try:
        return anaphor.nbor(1).lemma_
    except:
        return "."

# F45
def numPunctuations(anaphor):
    return len([x for x in anaphor.sent if x.pos_=="PUNCT"])

# F45

def numPunctuations(anaphor):

    return len([x for x in anaphor.sent if x.pos_=="PUNCT"])
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
# load object function
def loadObj(filename):
    filehandler = open(filename, 'rb') 
    return pickle.load(filehandler)    

# find all pronouns in the input sentence

def findPronouns(sent,pronouns,nlpcoref):

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
# get all nouns and noun phrases before the input pronoun in the input sentence

def getNPs(sent,p):

    nps=[]

    npstr=[]

    for np in sent.noun_chunks:

        if np.end<p.i:

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

# keep only the alphanumeric characters in a string

def stripStr(string):

    return ''.join(e for e in string if e.isalnum())

# check if a noun/nounphrase is the correct answer of an sentence_id using the annotation file

def isAnswer(sent_id,np):

    if sent_id in list(ranswers.sent_id):

        answer=ranswers.resolution[ranswers.sent_id == sent_id].iloc[0]

        return stripStr(answer) in stripStr(np.text) or nlp(answer)[0:].root.text.lower()==np.root.text.lower()

    return None

# transform pronouns and candidate antecedents to indexes to save them

def prepare_data_to_save(dataframe):

    dataframe['Pronoun']=dataframe.apply(lambda x: x['Pronoun'].i, axis=1)

    dataframe['CandidateAntecedent']=dataframe.apply(lambda x: [x['CandidateAntecedent'].start,x['CandidateAntecedent'].end], axis=1)

    return dataframe

# transform pronouns and candidate antecedents from indexes to Tokens and Spans after loading them

def loadSavedData(dataframe):

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

# get all nouns and noun phrases before the input pronoun in the input sentence

def getNPsfromsent(sent):

    nps=[]

    npstr=[]

    for np in sent.noun_chunks:

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



def matches(doc,terms):

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
def getEnt(sent):
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
    ms=matches(sent,list(ents))
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
                print(sent,span,ents)

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



