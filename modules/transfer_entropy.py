import numpy as np
import pandas as pd
from math import log2
import matplotlib.pyplot as plt
import random



def Median_Symbolise_3bins(Timeseries):
  Median = np.median(Timeseries)
  Q1 = np.quantile(Timeseries,1/3)
  Q3 = np.quantile(Timeseries,2/3)
  score_labels = [0,1,2]
  binned = pd.cut(Timeseries,[(min(Timeseries)-1),Q1,Q3,(max(Timeseries)+1)],labels=score_labels)
  return (binned.astype(int))

def takensEmbedding(data, delay, dimension):
  "This function returns the Takens embedding of data with delay into dimension, delay*dimension must be < len(data)"
  if delay * dimension > len(data):
    raise NameError('Delay times dimension exceed length of data!')
  embeddedData = np.array([data[0:len(data) - delay * dimension]])
  for i in range(1, dimension):
    embeddedData = np.append(embeddedData, [data[i * delay:len(data) - delay * (dimension - i)]], axis=0)
  return embeddedData;


def CondEntropy(X,Y):
  numpy_data = np.array([X,Y])
  df = pd.DataFrame(data=numpy_data,index=["X", "Y"]).T
  T = len(X)
  data_crosstab = pd.crosstab(df['X'],
                              df['Y'], 
                                margins = False) #access columns first
  uniqueX, CountX = np.unique(X, return_counts=True)
  uniqueY, CountY = np.unique(Y, return_counts=True)
  HXY = 0
  for i in range(len(uniqueY)):
    HXy = 0
    for j in range(len(uniqueX)):
      pxy = data_crosstab[uniqueY[i]][uniqueX[j]]/sum(data_crosstab[uniqueY[i]])
      if pxy!=0: HXy -= pxy*log2(pxy)
    py = CountY[i]/T
    HXY += py*HXy
  return HXY

def getPvalue(surrogate,value):
  surr = np.append(value,surrogate)
  sorted = np.sort(surr)
  index  = np.where(sorted == value)[0][0]
  Pvalue = (1+index)/len(sorted)
  return Pvalue

## shuffle x keeping yz dynamics
def Permutate(x,y,z):
  ZZ = z.copy()
  ZZ.append(y)
  dynamics = joint(ZZ)
  numpy_data = np.array([x,dynamics])
  df = pd.DataFrame(data=numpy_data,index=["X", "YZ"]).T
  grouped = df.groupby(df.YZ)
  d={}
  for i in range(len(np.unique(dynamics))):
    d["df{}".format(i)] = grouped.get_group(i)
    d["df{}".format(i)].index = d["df{}".format(i)].sample(frac=1).index.tolist()
  frames = pd.concat(d, axis=0).sum(axis=1, level=0)
  frames = frames.reset_index(level=0, drop=True)
  result = frames.sort_index()
  return np.array(result['X']),np.array(y),np.array(z)

def joint(L):
  numpy_data = np.array(L)
  df = pd.DataFrame(data=numpy_data).T
  df=df.applymap(str)
  df['joint'] = df.values.sum(axis=1)
  df.joint=df.joint.astype('category').cat.codes
  return np.array(df["joint"])


## Conditional transfer entropy from Y to X conditioned on Z
def CompTE_emb(X, Y, Z, X_lags, Y_lags, Z_lags, iterations):
  max_emb = 1+max([X_lags, Y_lags, Z_lags])
  X_emb = takensEmbedding(X, 1, max_emb)
  Y_emb = takensEmbedding(Y, 1, max_emb)
  Z_emb = takensEmbedding(Z, 1, max_emb)
  X_t = X_emb[-1:][0]
  X_his = joint(X_emb[-1 - X_lags:-1])
  Z_his = joint(Z_emb[-1 - Z_lags:-1])
  Y_his = joint(Y_emb[-1 - Y_lags:-1])

  X_Z = joint([X_his, Z_his])
  X_Y_Z = joint([X_his, Z_his, Y_his])
  TranEnt = CondEntropy(X_t, X_Z) - CondEntropy(X_t, X_Y_Z)
  null_dis = []
  for i in range(iterations):
    null_Y_his = Permutate(Y_his, X_his, [Z_his])[0]
    null_X_Y_Z = joint([X_his, Z_his, null_Y_his])
    null_TranEnt = CondEntropy(X_t, X_Z) - CondEntropy(X_t, null_X_Y_Z)
    null_dis.append(null_TranEnt)
  return TranEnt, np.percentile(null_dis, 95), 1 - getPvalue(null_dis, TranEnt)

def ComputeH1H2_lags(PB,CR,PT,NT,iterations):
  resultsdf  = pd.DataFrame(columns = ["Source","Target","TE","Quantile","p_val","lags"])

  #Crimes to negative number
  onelag = CompTE_emb(NT, CR, PB, 1, 1, 1, iterations)
  y_lag2 = CompTE_emb(NT, CR, PB, 1, 2, 1, iterations)
  z_lag2 = CompTE_emb(NT, CR, PB, 1, 1, 2, iterations)
  resultsdf.loc[len(resultsdf.index)] = ["Crimes", "negative Tweets", onelag[0], onelag[1], onelag[2],
                                         "target1 source1 condition1"]
  resultsdf.loc[len(resultsdf.index)] = ["Crimes", "negative Tweets", y_lag2[0], y_lag2[1], y_lag2[2],
                                         "target1 source2 condition1"]
  resultsdf.loc[len(resultsdf.index)] = ["Crimes", "negative Tweets", z_lag2[0], z_lag2[1], z_lag2[2],
                                         "target1 source1 condition2"]


  #Crimes to positive number
  onelag = CompTE_emb(PT, CR, PB, 1, 1, 1, iterations)
  y_lag2 = CompTE_emb(PT, CR, PB, 1, 2, 1, iterations)
  z_lag2 = CompTE_emb(PT, CR, PB, 1, 1, 2, iterations)
  resultsdf.loc[len(resultsdf.index)] = ["Crimes", "Positive Tweets", onelag[0], onelag[1], onelag[2],
                                         "target1 source1 condition1"]
  resultsdf.loc[len(resultsdf.index)] = ["Crimes", "Positive Tweets", y_lag2[0], y_lag2[1], y_lag2[2],
                                         "target1 source2 condition1"]
  resultsdf.loc[len(resultsdf.index)] = ["Crimes", "Positive Tweets", z_lag2[0], z_lag2[1], z_lag2[2],
                                         "target1 source1 condition2"]

  #PB to negative number
  onelag = CompTE_emb(NT, PB, CR, 1, 1, 1, iterations)
  y_lag2 = CompTE_emb(NT, PB, CR, 1, 2, 1, iterations)
  z_lag2 = CompTE_emb(NT, PB, CR, 1, 1, 2, iterations)
  resultsdf.loc[len(resultsdf.index)] = ["Police brut", "Negative Tweets", onelag[0], onelag[1], onelag[2],
                                         "target1 source1 condition1"]
  resultsdf.loc[len(resultsdf.index)] = ["Police brut", "Negative Tweets", y_lag2[0], y_lag2[1], y_lag2[2],
                                         "target1 source2 condition1"]
  resultsdf.loc[len(resultsdf.index)] = ["Police brut", "Negative Tweets", z_lag2[0], z_lag2[1], z_lag2[2],
                                         "target1 source1 condition2"]

  return resultsdf
