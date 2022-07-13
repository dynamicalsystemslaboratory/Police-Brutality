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

def transition_matrix(series):
    numpy_data = np.array(series)
    df = pd.DataFrame(data=numpy_data).T
    df=df.applymap(str)
    df['joint'] = df.values.sum(axis=1)
    transitions = np.array(df['joint'])
    df2 = pd.DataFrame(transitions)
    
    df2['shift'] = df2[0].shift(-1)
    df2['count'] = 1
    trans_mat = df2.groupby([0, 'shift']).count().unstack().fillna(0)
    return(trans_mat)


def Direction(s, index1, index2):
  # index starts from zero
  # doesnt work if symbolze are negative as the - will be counted as the first term
  tranMat = transition_matrix(s)
  states = np.unique(s)
  l = len(s)
  df = tranMat["count"]
  columns = df.columns
  outputdf = pd.DataFrame(columns=np.append(["state"], columns))
  listofrows = []
  for state in states:
    sublistofrows = []
    for i, col in enumerate(columns):
      if col[index1] == str(state): sublistofrows.append(i)
    listofrows.append(sublistofrows)
    row = np.array(str(state))
    rowval = np.array(df.iloc[sublistofrows].sum().values.tolist())
    outputdf.loc[len(outputdf.index)] = np.append(row, rowval).tolist()
  outputdf.set_index('state')
  outputdf = outputdf.astype(float)

  listofcolumns = []
  listofstates = []
  for state in states:
    listofstates.append(str(state))
    sublistofcolumns = []
    for i, col in enumerate(columns):
      if col[index2] == str(state): sublistofcolumns.append(i)
    listofcolumns.append(sublistofcolumns)
    outputdf[str(state)] = outputdf[np.take(np.array(columns), sublistofcolumns)].sum(axis=1)
  output = outputdf[listofstates].div(outputdf[listofstates].sum(axis=1), axis=0)
  return output



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


##Test the hypothesis code
def ComputeH1H2(PB,CR,PT,NT,iterations):
  lx = 1
  ly = 1
  lz = np.array([1,1])
  resultsdf  = pd.DataFrame(columns = ["Sender","Receiver","p_val","TE","Quantile"])

  #Crimes to negative number
  True_TE_CRNT = CondTransEnt(X =NT,Y =CR,Z =[PB],lagx = lx,lagy = ly,lagz =lz)
  surrogateValues = []
  for index in range(iterations):
    Series = Permutate(CR, NT, [PB])
    surrogateValues.append(CondTransEnt(Series[1], Series[0], Series[2], lagx=lx, lagy=ly, lagz=lz))
  nintyfifth = np.percentile(surrogateValues, 95)
  resultsdf.loc[len(resultsdf.index)] = ["Crimes","negative Tweets",round(1 - getPvalue(surrogateValues, True_TE_CRNT), 6),True_TE_CRNT,nintyfifth]

  #Crimes to positive number
  True_TE_CRPT = CondTransEnt(X =PT,Y =CR,Z =[PB],lagx = lx,lagy = ly,lagz =lz)
  surrogateValues = []
  for index in range(iterations):
    Series = Permutate(CR, PT, [PB])
    surrogateValues.append(CondTransEnt(Series[1], Series[0], Series[2], lagx=lx, lagy=ly, lagz=lz))
  nintyfifth = np.percentile(surrogateValues, 95)
  resultsdf.loc[len(resultsdf.index)] = ["Crimes","Positive Tweets",round(1 - getPvalue(surrogateValues, True_TE_CRPT), 6),True_TE_CRPT,nintyfifth]

  #PB to negative number
  True_TE_PBNT = CondTransEnt(X =NT,Y =PB,Z =[CR],lagx = lx,lagy = ly,lagz =lz)
  surrogateValues = []
  for index in range(iterations):
    Series = Permutate(PB, NT, [CR])
    surrogateValues.append(CondTransEnt(Series[1], Series[0], Series[2], lagx=lx, lagy=ly, lagz=lz))
  nintyfifth = np.percentile(surrogateValues, 95)
  resultsdf.loc[len(resultsdf.index)] = ["Police brutality","negative Tweets",round(1 - getPvalue(surrogateValues, True_TE_PBNT), 6),True_TE_PBNT,nintyfifth]
  return resultsdf



## Conditional transfer entropy from Y to X conditioned on Z with lag=lag
## x and y are series, Z is a list of series. lagx and lag y are numbers and lagz is a list of lagz corresponding to the list of conditions
def CondTransEnt(X,Y,Z,lagx,lagy,lagz):
  L = len(X)
  maxx = max([lagx,lagy,max(lagz)])
  Xt = X[maxx:L]
  Xlag = X[maxx-lagx:(L-lagx)]
  Ylag = Y[maxx-lagy:(L-lagy)]
  Zlag = []
  for i, z in enumerate(Z):
    Zlag.append(z[maxx-lagz[i]:(L-lagz[i])])
  Zlag.append(Xlag)
  xzlag = joint(Zlag)
  Zlag.append(Ylag)
  xyzlag = joint(Zlag)
  TYXZ = CondEntropy(Xt,xzlag)-CondEntropy(Xt,xyzlag)
  return TYXZ
