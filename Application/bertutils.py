import pandas as pd
import numpy as np
import ast
from torch.utils.data import Dataset
import torch
from transformers import Adafactor, get_linear_schedule_with_warmup, AdamW
from typing import Tuple
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizerFast, BertForTokenClassification 
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from itertools import groupby
from operator import itemgetter
import ast
from numpy import exp
import importlib, sys
from nltk.tokenize import WhitespaceTokenizer

def getTokensSpans(s):
  span_generator = WhitespaceTokenizer().span_tokenize(s)
  spans = [span for span in span_generator]
  return spans

# Utility functions


def transform_strtab_inttab(s):
  s=str(s)
  return [int(x) for x in s[1:-1].split(',') if len(x)>0]


def compute_chunks(tab):
  init_chunck  = 0
  chunks = []
  for i, k in enumerate(tab[:-1]):
    if k+1 != tab[i+1]:
      chunks.append(tab[init_chunck: i+1])
      init_chunck = i+1
  chunks.append(tab[init_chunck:])
  return chunks



def predict_spans(truncated_predictions, cdf,fast_tokenizer,T=0):
  predited_spans = []
  mean=-sys.maxsize 
  for i in range(len(cdf)):
    span=[]
    nmean=-sys.maxsize-1
    aboveT=0
    for x in continuousSubArrays(np.where(np.argmax(truncated_predictions[i],1)==1)[0]):
        if softmax(truncated_predictions[i])[:,1][x].mean()>nmean:
            nmean=softmax(truncated_predictions[i])[:,1][x].mean()
            if nmean>T:
              aboveT+=1
            span=(np.array(x, dtype=np.int64),)
    #nmean=softmax(truncated_predictions[i])[:,1][span].mean()
    if T!=0:
        if nmean<T:
            predited_spans.append([])
            continue
    if aboveT>1:
      predited_spans.append([])
      continue
    text = cdf.iloc[i]['context']
    #print(text)
    #discminative_preds = np.argmax(truncated_predictions[i],1)
    #print(len(discminative_preds))
    one_idx = span#np.where(discminative_preds == 1)
    one_idx = one_idx[0]
    tokens_info = fast_tokenizer(text, return_offsets_mapping=True)
    # print(len(tokens_info['input_ids']))
    # print(len(tokens_info['offset_mapping']))
    #print("************************")
    yoyo = [tokens_info['offset_mapping'][k] for k in one_idx if k<len(tokens_info['offset_mapping'])]
    pred = []
    for p in yoyo:
      pred += list(range(p[0],p[1]))
    predited_spans.append(pred)
  return predited_spans


def predict_spans_T(truncated_predictions, cdf,fast_tokenizer,T):
  predited_spans = []
  for i in range(len(cdf)):
    text = cdf.iloc[i]['context']
    #print(text)
    discminative_preds =  softmax(truncated_predictions[i])[:,1]
    #print(len(discminative_preds))
    one_idx = np.where(discminative_preds > T)
    one_idx = one_idx[0]
    tokens_info = fast_tokenizer(text, return_offsets_mapping=True)
    # print(len(tokens_info['input_ids']))
    # print(len(tokens_info['offset_mapping']))
    #print("************************")
    yoyo = [tokens_info['offset_mapping'][k] for k in one_idx if k<len(tokens_info['offset_mapping'])]
    pred = []
    for p in yoyo:
      pred += list(range(p[0],p[1]))
    predited_spans.append(pred)
  return predited_spans

def closest(mylist,mynumber):
    c=min(mylist, key=lambda x:abs(x-mynumber))
    d=abs(c-mynumber)
    return d

def continuousSubArrays(L):
    sas=[]
    sa=[]
    for i,j in zip(range(0,len(L)-1),range(1,len(L))):
            if L[i]==L[j]-1 or L[i]==L[j]-2:
                if L[i] not in sa:
                    sa.append(L[i])
                if L[j] not in sa:
                    sa.append(L[j])
            else:
                if sa:
                    sas.append(sa)
                    sa=[]
    if sa:
        sas.append(sa)
    return sas              



def get_tokenclassification_annotation(item, fast_tokenizer, train=True, debug=False,max_length=128):
  text = item['context']+" [SEP] "+item['pronoun']
  tokens_info = fast_tokenizer(text, padding='max_length', max_length=max_length, truncation=True, return_offsets_mapping = True)
  # padding='max_length', max_length=max_length, truncation=True,
  if debug:
    print(f'Original Text:\n\t {text}')
  if train:
    spans = transform_strtab_inttab(item['schars'])
    chunks = compute_chunks(spans)
    if debug:
      print("Antecedent text")
      for chunk in chunks:
        print('\t',text[chunk[0]:chunk[-1]])
    tokenized_text = fast_tokenizer.convert_ids_to_tokens(tokens_info['input_ids'])
    if debug:
      print(f'Tokenized Text:\n\t {tokenized_text}')
    offsets_mapping = tokens_info['offset_mapping']
    annot = []
    for k in spans:
      for x, (j, v) in enumerate(offsets_mapping[1:-1]):
        if j <= k and k < v  :
          annot.append(x)
    annot = set(annot)
    annotation = [0 for x in  offsets_mapping[1:-1]]
    if debug:
      print("Antecedent words")
    for x in annot:
      annotation[x] = 1
      if debug:
        print('\t',tokenized_text[1:-1][x], "||||index: ", x)
    return {'input_ids':tokens_info['input_ids'],
            'labels':[0] + annotation + [0],
            'attention_mask': tokens_info['attention_mask']
    }
  return {'input_ids':tokens_info['input_ids'],
            'attention_mask': tokens_info['attention_mask']
    }


def get_sentclassification_annotation(item, fast_tokenizer, train=True, debug=False,max_length=128):
  text = item['context']+" [SEP] "+item['pronoun']
  tokens_info = fast_tokenizer(text, padding='max_length', max_length=max_length, truncation=True, return_offsets_mapping = True)
  # padding='max_length', max_length=max_length, truncation=True,
  if debug:
    print(f'Original Text:\n\t {text}')
  if train:
    tokenized_text = fast_tokenizer.convert_ids_to_tokens(tokens_info['input_ids'])
    offsets_mapping = tokens_info['offset_mapping']
    annot = 0 if item['combined']=="Ambiguous" else 1
    return {'input_ids':tokens_info['input_ids'],
            'labels':annot,
            'attention_mask': tokens_info['attention_mask']
    }
  return {'input_ids':tokens_info['input_ids'],
            'attention_mask': tokens_info['attention_mask']
    }


def encode(data):
  labels=[]
  for d in data:
    if d=="Ambiguous":
      labels.append(1)
    else:
      labels.append(0)
  return labels

def concat(li1,li2):
  li=[]
  for i,j in zip(li1,li2):
    li.append(i+" [SEP] "+j)
  return li

class SeqClassificationData(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class SpanDetectionData(Dataset):
  def __init__(self, df, tokenizer, train=True):
    self.df = df
    self.tokenizer = tokenizer
    self.train = train
  
  def __len__(self,):
    return len(self.df)
  
  def __getitem__(self, idx):
    item = get_tokenclassification_annotation(self.df.iloc[idx], self.tokenizer, self.train)
    return item


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def compute_metricsX(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def nbrSpansT(preds,thresh):
  nbr_of_spans=0
  data=list(np.where(softmax(preds.predictions[0])[:,1]>thresh)[0])
  for k, g in groupby(enumerate(data), lambda ix : ix[0] - ix[1]):
      nbr_of_spans+=1
  return nbr_of_spans

def nbrSpans(preds):
  nbr_of_spans=0
  for k, g in groupby(enumerate(preds), lambda ix : ix[0] - ix[1]):
      nbr_of_spans+=1
  return nbr_of_spans

def nbrSpansX(preds,x=2):
  if not preds:
    return 0
  nbr_of_spans=1
  previ=preds[0]
  for i in preds:
    if i-previ>x:
      nbr_of_spans+=1
    previ=i
  return nbr_of_spans

def processPred(predictions,train_data,df,fast_tokenizer,T=0.9):
    truncated_predictions = []
    for k in range(len(train_data)):
      inputs_ids = train_data[k]['input_ids']
      truncated_predictions.append(predictions.predictions[k][:len(inputs_ids)])
    predicted_spans = predict_spans(truncated_predictions, df,fast_tokenizer,T)
    return truncated_predictions,predicted_spans

def is_slice_in_list(s,l):
    if not s:
      return False
    len_s = len(s) #so we don't recompute length of s on every iteration
    return any(s == l[i:len_s+i] for i in range(len(l) - len_s+1))

def getMetrics(tp,tn,fp,fn,beta=2):
  precision=tp/(tp+fp)
  recall=tp/(tp+fn)
  accuracy=(tp+tn)/(tp+tn+fp+fn)

  fbeta=(1+beta*beta)*precision*recall/(beta*beta*precision+recall) if precision+recall!=0 else 0
  return precision,recall,fbeta,accuracy

def printMetrics(tp,tn,fp,fn,beta=2):
  m=getMetrics(tp,tn,fp,fn,beta)
  precision=m[0]
  recall=m[1]
  fbeta=m[2]
  accuracy=m[3]

  print("precision=",precision)
  print("recall=",recall)
  print("f2-score=",fbeta)
  print("accuracy=",accuracy)

def evaluate(predicted_spans,tests):
  tp=0
  tn=0
  fp=0
  fn=0
  for p,t in zip(predicted_spans,tests):
    if not p and not t:
      tp+=1
    elif not p and t:
      fp+=1
    elif not t and p:
      fn+=1
    else:
    # elif is_slice_in_list(p,t):
      tn+=1
  printMetrics(tp,tn,fp,fn)

def evaluateThresh(truncated_predictions,tests,threshold):
  tp=0
  tn=0
  fp=0
  fn=0
  for p,t in zip(truncated_predictions,tests):
    p=list(np.where(softmax(p)[:,1]>threshold)[0])
    if not p and not t:
      tp+=1
    elif is_slice_in_list(p,t):
      tn+=1
    elif not p and t:
      fp+=1
    elif not t and p:
      fn+=1
  printMetrics(tp,tn,fp,fn)

def evaluateS(predicted_spans,tests,X=2):
  tp=0
  tn=0
  fp=0
  fn=0
  for p,t in zip(predicted_spans,tests):
    nbrS=nbrSpansX(p,x=X)
    if (nbrS==0 or nbrS>1) and not t:
      tp+=1
    elif (nbrS==0 or nbrS>1) and t:
      fp+=1
    elif not t and (nbrS==1):
      fn+=1
    # elif is_slice_in_list(p,t):
    else:
      tn+=1
  printMetrics(tp,tn,fp,fn)

def evaluateST(predicted_spans,tests,X=2,threshold=0.9):
  tp=0
  tn=0
  fp=0
  fn=0
  for p,t in zip(predicted_spans,tests):
    p=list(np.where(softmax(p)[:,1]>threshold)[0])
    nbrS=nbrSpansX(p,x=X)
    if (nbrS==0 or nbrS>1) and not t:
      tp+=1
    elif (nbrS==0 or nbrS>1) and t:
      fp+=1
    elif not t and (nbrS==1):
      fn+=1
    else:
    # elif is_slice_in_list(p,t):
      tn+=1
  printMetrics(tp,tn,fp,fn)
  return fn+tn


def findWspans(s,ids):
    spans=[]
    lastspan=""
    prevend=-1
    Spans=getTokensSpans(s)
    for start,end in Spans:
      for i in ids:
        if s[int(i)]==' ':
          continue
        if int(i) in range(start,end+1):
          if prevend==start-1:
            lastspan+=" "+s[start:end]
            prevend=end
            break
          elif prevend==-1:
            lastspan+=s[start:end]
            prevend=end
            break
          else:
            spans.append(lastspan)
            lastspan=s[start:end]
            prevend=end
            break
    spans.append(lastspan)
    return spans  

def findspans(s,ids):
    spans=[]
    lastspan=""
    previ=-1
    for i in ids:
      i=int(i)
      if previ==-1 or i==previ+1:
          lastspan+=s[i]
          previ=i
          continue
      elif i==previ+2:
          lastspan+=" "+s[i]
          previ=i
          continue
      else:
          if lastspan not in spans:
              if lastspan:
                spans.append(lastspan)
          lastspan=s[i]
          previ=i
          continue
    if lastspan not in spans:
        if lastspan:
          spans.append(lastspan)
    return spans

def softmax(vector):
  li=[]
  for element in vector:
    se=exp(element).sum()
    ne=exp(element)/se
    li.append(ne)
  return np.array(li)

def bert_text_preparation(text, tokenizer):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return tokenized_text, tokens_tensor, segments_tensors
    
def get_hidden_states(tokens_tensor, segments_tensors, model):    
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]
    
    return hidden_states
def get_token_embeddings(hidden_states):
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)
    return token_embeddings

def get_word_emb(text,tokenizer,model):
    tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text,tokenizer)
    hidden_states = get_hidden_states(tokens_tensor, segments_tensors, model)
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]
    return list_token_embeddings

def get_word_emb_ml(text,tokenizer,model,mode="concate"):
    tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text,tokenizer)
    hidden_states = get_hidden_states(tokens_tensor, segments_tensors, model)
    token_embeddings = get_token_embeddings(hidden_states)
    if mode == "concate":
        return concat4(token_embeddings)
    elif mode == "sum":
        return sum4(token_embeddings)
    else:
        print("invalid mode option")

def get_sent_emb(text, tokenizer, model):
    tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(
        text, tokenizer)
    hidden_states = get_hidden_states(tokens_tensor, segments_tensors, model)
    token_vecs = hidden_states[-2][0]
    # Calculate the average of all token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding

def concat4(token_embeddings):
    token_vecs_cat = []
    # `token_embeddings` is a [22 x 12 x 768] tensor.
    for token in token_embeddings:
        # `token` is a [12 x 768] tensor
        # Concatenate the vectors (that is, append them together) from the last
        # four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3,072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]),
                            dim=0)

        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec)
    #     print ('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))
    return token_vecs_cat

def sum4(token_embeddings):
    token_vecs_sum = []
    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)
    return token_vecs_sum

def get_4layers_emb(text, tokenizer, model,concat=True):
    tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(
        text, tokenizer)
    hidden_states = get_hidden_states(tokens_tensor, segments_tensors, model)
    token_embeddings = get_token_embeddings(hidden_states)
    token_vecs=np.nan
    if concat:
        token_vecs=concat4(token_embeddings)
    else:
        token_vecs=sum4(token_embeddings)
    # Calculate the average of all token vectors.
    sentence_embedding = torch.mean(torch.stack(token_vecs), dim=0)
    return sentence_embedding

def hashsent1(c,p):
    return c[:p.i].text+" "+p.text+"#1 "+c[p.i+1:].text
def hashsent2(c,p):
    return c[:p.i].text+" "+p.text+"#2 "+c[p.i+1:].text
def hashdouble(c,p,ca):
    if p.i>ca.start:
        return c[:ca.start].text+" "+ca.text+"#2 "+c[ca.end:p.i].text+" "+p.text+"#1 "+c[p.i+1:].text
    if p.i<ca.start:
        return c[:p.i].text+" "+p.text+"#1 "+c[p.i+1:ca.start].text+" "+ca.text+"#2 "+c[ca.end:].text
    print(p, 'in', ca)