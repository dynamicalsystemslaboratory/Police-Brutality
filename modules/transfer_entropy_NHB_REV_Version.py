import numpy as np
import pandas as pd
from math import log2
import matplotlib.pyplot as plt
import random
# from statsmodels.tsa.seasonal import MSTL
#
# def detrend_and_remove_seasonal_effects(time_series):
#   fakedate = np.arange(len(time_series))
#   lintrend_co = np.polyfit(fakedate, time_series, deg=1)
#   lintrend = lintrend_co[0] * fakedate + lintrend_co[1]
#   detrended_times_series = time_series - lintrend
#   seasonsdf = MSTL(detrended_times_series, periods=(7, 30, 365),
#                    stl_kwargs={"seasonal_deg": 0, "trend_deg": 1}).fit().seasonal
#   return detrended_times_series - seasonsdf["seasonal_7"] - seasonsdf["seasonal_30"] - seasonsdf["seasonal_365"]
#
# def detrend_and_remove_seasonal_effects_MSTL(time_series):
#   component = MSTL(time_series, periods=(7, 30, 365),
#                    stl_kwargs={"seasonal_deg": 0, "trend_deg": 1}).fit()
#   seasonsdf = component.seasonal
#   trend = component.trend
#   return time_series - seasonsdf["seasonal_7"] - seasonsdf["seasonal_30"] - seasonsdf["seasonal_365"] - trend
#
#
# def detrend_and_remove_seasonal_effects_MSTL_hourly(time_series):
#   component = MSTL(time_series, periods=(24, 168),
#                    stl_kwargs={"seasonal_deg": 0, "trend_deg": 1}).fit()
#   seasonsdf = component.seasonal
#   trend = component.trend
#   return time_series - seasonsdf["seasonal_24"] - seasonsdf["seasonal_168"] - trend


def either_or_symbolise(Timeseries):
  arr = np.array(Timeseries)
  arr[arr > 0] = 1
  return arr

def Change_Symbolise(Timeseries):
  return  (np.diff(Timeseries) > 0).astype(int)

def Median_Symbolise(Timeseries):
  Median = np.median(Timeseries)
  return (Timeseries < Median).astype(int)

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
    
    # create a new column with data shifted one space (initial state)
    df2['shift'] = df2[0].shift(-1)
    # add a count column (for group by function)
    df2['count'] = 1
    # groupby and then unstack, fill the zeros
    trans_mat = df2.groupby([0, 'shift']).count().unstack().fillna(0)
    # normalise by occurences and save values to get transition matrix
    #trans_mat_v2 = trans_mat.div(trans_mat.sum(axis=1), axis=0).values
    #trans_mat_v2 = trans_mat.div(trans_mat.sum(axis=1), axis=0)

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


def TransEnt(X,Y):
  #TE from Y to X
  #For now only works with lag = 1 since xt is not constructed to be always t-1
  lag = 1
  L = len(X)
  Xt = X[lag:L]
  Xlag = X[0:(L-lag)]
  Ylag = Y[0:(L-lag)]
  xylag = joint([Xlag,Ylag])
  TYX = CondEntropy(Xt,Xlag)-CondEntropy(Xt,xylag)
  return TYX

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

def CondTranEntRe(senders, receivers, fixed, lagsX, lagsY, lagsZ, iterations):
  # This function takes a dictionary of senders a list of receivers and a list fixed series and will return
  # the p-value from all of conbination of senders receiver and fixed with different lags with the surrogate being generated with a fixed number, iteration
  senderspd = pd.DataFrame(data=senders)
  receiverspd = pd.DataFrame(data=receivers)
  fixedpd = pd.DataFrame(data=fixed)
  results = pd.DataFrame(columns=['Sender', 'Reciever', 'Fixed', 'Lagx','Lagy','Lagz','p_value'])
  for sender in senderspd:
    for reciver in receiverspd:
      for fixd in fixedpd:
        if (sender != reciver) and (sender != fixd) and (fixd != reciver):

          fromseries = np.array(senderspd[sender].values)
          toseries = np.array(receiverspd[reciver].values)
          conditionedonseries = np.array(fixedpd[fixd].values)
          for lx in lagsX:
            for ly in lagsY:
              for lz in lagsZ:
                trueConTE = CondTransEnt(toseries, fromseries, conditionedonseries, lx,ly,lz)
                surrogateValues = []
                for i in range(iterations):
                  Series = Permutate(fromseries, toseries, conditionedonseries)
                  surrogateValues.append(CondTransEnt(Series[1], Series[0], Series[2], lx,ly,lz))
                fifth = np.percentile(surrogateValues, 5)
                nintyfifth = np.percentile(surrogateValues, 95)
                plt.axvline(fifth, color='r', linewidth=1)
                plt.axvline(nintyfifth, color='r', linewidth=1)
                plt.axvline(trueConTE, color='k', linestyle='dashed', linewidth=1)
                plt.hist(surrogateValues, bins='auto')
                plt.title(sender + '-->' + reciver + '|' + fixd + ' LagXYZ =' + str(lx)+str(ly)+str(lz) + ' p_value = ' + str(
                  round(1 - getPvalue(surrogateValues, trueConTE), 4)))
                plt.show()
                results.loc[len(results.index)] = [sender, reciver, fixd, lx,ly,lz,
                                               round(1 - getPvalue(surrogateValues, trueConTE), 4)]
  return results


## shuffle both and combine
def CondMutualInfoHisNet(lista,iterations):
  quantilespd = pd.DataFrame(columns = [*lista])
  MIpd = pd.DataFrame(columns = [*lista])
  seriespd = pd.DataFrame(data=lista)
  df = pd.DataFrame(columns=[*lista])
  for i, sender in enumerate(lista):
    for receiver in lista:
      if sender != receiver:
        fixed = [*lista].copy()
        fixed.remove(sender)
        fixed.remove(receiver)

        fromseries = np.array(seriespd[sender].values).tolist()
        toseries = np.array(seriespd[receiver].values).tolist()
        conditionedonseries = np.array(seriespd[fixed].values).T.tolist()
        TrueMI = CondMutualInfoHis(X=toseries, Y=fromseries, Z=conditionedonseries)
        surrogateValues = []
        for index in range(iterations):
          random.shuffle(fromseries)
          surrogateValues.append(CondMutualInfoHis(X=toseries, Y=fromseries, Z=conditionedonseries))

        fromseries = np.array(seriespd[sender].values).tolist()
        toseries = np.array(seriespd[receiver].values).tolist()
        conditionedonseries = np.array(seriespd[fixed].values).T.tolist()
        for index in range(iterations):
          random.shuffle(toseries)
          surrogateValues.append(CondMutualInfoHis(X=toseries, Y=fromseries, Z=conditionedonseries))

        fifth = np.percentile(surrogateValues, 5)
        nintyfifth = np.percentile(surrogateValues, 95)
        plt.axvline(fifth, color='r', linewidth=1)
        plt.axvline(nintyfifth, color='r', linewidth=1)
        plt.axvline(TrueMI, color='k', linestyle='dashed', linewidth=1)
        plt.hist(surrogateValues, bins='auto')
        plt.title(sender + " & " + receiver +" " +str(
          round(1 - getPvalue(surrogateValues, TrueMI), 4)))
        plt.show()

        df.at[i, receiver] = round(1 - getPvalue(surrogateValues, TrueMI), 6)
        MIpd.at[i, receiver] = TrueMI
        quantilespd.at[i, receiver] = nintyfifth
  return df, MIpd, quantilespd


## Shuffle both==>
# def CondMutualInfoHisNet(lista,iterations):
#   quantilespd = pd.DataFrame(columns = [*lista])
#   MIpd = pd.DataFrame(columns = [*lista])
#   seriespd = pd.DataFrame(data=lista)
#   df = pd.DataFrame(columns=[*lista])
#   for i, sender in enumerate(lista):
#     for receiver in lista:
#       if sender != receiver:
#         fixed = [*lista].copy()
#         fixed.remove(sender)
#         fixed.remove(receiver)
#         fromseries = np.array(seriespd[sender].values).tolist()
#         toseries = np.array(seriespd[receiver].values).tolist()
#         conditionedonseries = np.array(seriespd[fixed].values).T.tolist()
#         TrueMI = CondMutualInfoHis(X=toseries, Y=fromseries, Z=conditionedonseries)
#         surrogateValues = []
#         for index in range(iterations):
#           random.shuffle(fromseries)
#           random.shuffle(toseries)
#           surrogateValues.append(CondMutualInfoHis(X=toseries, Y=fromseries, Z=conditionedonseries))
#         fifth = np.percentile(surrogateValues, 5)
#         nintyfifth = np.percentile(surrogateValues, 95)
#         plt.axvline(fifth, color='r', linewidth=1)
#         plt.axvline(nintyfifth, color='r', linewidth=1)
#         plt.axvline(TrueMI, color='k', linestyle='dashed', linewidth=1)
#         plt.hist(surrogateValues, bins='auto')
#         plt.title(sender + " & " + receiver +" " +str(
#           round(1 - getPvalue(surrogateValues, TrueMI), 4)))
#         plt.show()
#
#         df.at[i, receiver] = round(1 - getPvalue(surrogateValues, TrueMI), 6)
#         MIpd.at[i, receiver] = TrueMI
#         quantilespd.at[i, receiver] = nintyfifth
#   return df, MIpd, quantilespd


## this was done to study a specific case
def fournodeonelinkpval(X,Y,Z1,Z2,ly,lz1,lz2,iterations):
  toseries = X
  fromseries = Y
  conditionedonseries = [Z1,Z2]
  TrueTE = CondTransEnt(X=toseries, Y=fromseries, Z=conditionedonseries, lagx=1, lagy=ly, lagz=[lz1,lz2])
  surrogateValues = []
  for index in range(iterations):
    Series = Permutate(fromseries, toseries, conditionedonseries)
    surrogateValues.append(CondTransEnt(Series[1], Series[0], Series[2], lagx=1, lagy=ly, lagz=[lz1,lz2]))
  return  1-getPvalue(surrogateValues, TrueTE)


##Test the hypothesis link code
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

  # PB to negative number
  True_TE_NTPB = CondTransEnt(X=PB, Y=NT, Z=[CR], lagx=lx, lagy=ly, lagz=lz)
  surrogateValues = []
  for index in range(iterations):
    Series = Permutate(NT, PB, [CR])
    surrogateValues.append(CondTransEnt(Series[1], Series[0], Series[2], lagx=lx, lagy=ly, lagz=lz))
  nintyfifth = np.percentile(surrogateValues, 95)
  resultsdf.loc[len(resultsdf.index)] = ["negative Tweets","Police brutality",
                                         round(1 - getPvalue(surrogateValues, True_TE_NTPB), 6), True_TE_NTPB,
                                         nintyfifth]
  return resultsdf



# list is a dictionary with name and iteration is the number of shufffle
def CondTranEntNet(lista,iterations, lagy, lagz):
  lx = 1
  ly = lagy
  lz = lagz*np.ones(len(lista)-2, dtype = int)
  df  = pd.DataFrame(columns = [*lista])
  quantilespd = pd.DataFrame(columns = [*lista])
  TEpd = pd.DataFrame(columns = [*lista])
  seriespd = pd.DataFrame(data=lista)
  for i, sender in enumerate(lista):
    for receiver in lista:
      if sender != receiver:
        fixed = [*lista].copy()
        fixed.remove(sender)
        fixed.remove(receiver)
        fromseries = np.array(seriespd[sender].values).tolist()
        toseries = np.array(seriespd[receiver].values).tolist()
        conditionedonseries = np.array(seriespd[fixed].values).T.tolist()
        TrueTE = CondTransEnt(X =toseries,Y =fromseries,Z =conditionedonseries,lagx = lx,lagy = ly,lagz =lz)
        surrogateValues = []
        for index in range(iterations):
          Series = Permutate(fromseries, toseries, conditionedonseries)
          surrogateValues.append(CondTransEnt(Series[1], Series[0], Series[2],lagx = lx,lagy = ly,lagz = lz))
        
        fifth = np.percentile(surrogateValues, 5)
        nintyfifth = np.percentile(surrogateValues, 95)
        # plt.axvline(fifth, color='r', linewidth=1)
        # plt.axvline(nintyfifth, color='r', linewidth=1)
        # plt.axvline(TrueTE, color='k', linestyle='dashed', linewidth=1)
        # plt.hist(surrogateValues, bins='auto')
        # plt.title(sender + '-->' + receiver + ' LagXYZ =' + str(lx)+str(ly)+str(lz) + ' p_value = ' + str(
        #   round(1 - getPvalue(surrogateValues, TrueTE), 4)))
        # plt.show()
        df.at[i,receiver] =  round(1 - getPvalue(surrogateValues, TrueTE), 6)
        TEpd.at[i, receiver] = TrueTE
        quantilespd.at[i, receiver] = nintyfifth
  return df, TEpd, quantilespd

def CondMutualInfo(X,Y,Z):
  ## compute the mutual information between x and y conditioned on multiple Zs
  bigz = joint(Z)
  YZ = []
  YZ.append(Y)
  for z in Z:
    YZ.append(z)
  YZ = joint(YZ)
  IYXZ = CondEntropy(X,bigz)-CondEntropy(X,YZ)
  return IYXZ


#test for conditional independence between X and Y condition on their past and the past of Z
# H(Xt|Xt-1,Yt-1,Zt-1)-H(Xt|Yt,Xt-1,Yt-1,Zt-1)
def CondMutualInfoHis(X,Y,Z):
  L = len(X)
  Xlag = X[:(L-1)]
  Ylag = Y[:(L-1)]
  Xt = X[1:]
  Yt = Y[1:]
  Zlag = []
  for i, z in enumerate(Z):
    Zlag.append(z[:(L-1)])
  Zlag.append(Xlag)
  Zlag.append(Ylag)
  past = joint(Zlag)
  Zlag.append(Yt)
  yandpast = joint(Zlag)
  return CondEntropy(Xt, past) - CondEntropy(Xt, yandpast)


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

def CondMutualInfotNet(lista,iterations):
  df  = pd.DataFrame(columns = [*lista])
  seriespd = pd.DataFrame(data=lista)
  for i, sender in enumerate(lista):
    for receiver in lista:
      if sender != receiver:
        fixed = [*lista].copy()
        fixed.remove(sender)
        fixed.remove(receiver)
        fromseries = np.array(seriespd[sender].values).tolist()
        toseries = np.array(seriespd[receiver].values).tolist()
        conditionedonseries = np.array(seriespd[fixed].values).T.tolist()
        TrueMI = CondMutualInfo(X =toseries,Y =fromseries,Z =conditionedonseries)
        surrogateValues = []
        for index in range(iterations):
          random.shuffle(fromseries)
          surrogateValues.append(CondMutualInfo(X =toseries,Y =fromseries,Z =conditionedonseries))
        
        fifth = np.percentile(surrogateValues, 5)
        nintyfifth = np.percentile(surrogateValues, 95)
        plt.axvline(fifth, color='r', linewidth=1)
        plt.axvline(nintyfifth, color='r', linewidth=1)
        plt.axvline(TrueMI, color='k', linestyle='dashed', linewidth=1)
        plt.hist(surrogateValues, bins='auto')
        plt.title(sender + '-->' + receiver + ' p_value = ' + str(
          round(1 - getPvalue(surrogateValues, TrueMI), 4)))
        plt.show()
        print(surrogateValues)
        
        df.at[i,receiver] =  round(1 - getPvalue(surrogateValues, TrueMI), 4)
  return df

## shuffle x
def PermutatePairwise(x,y):
  numpy_data = np.array([x,y])
  df = pd.DataFrame(data=numpy_data,index=["X", "Y"]).T
  grouped = df.groupby(df.Y)
  d={}
  for i in range(len(np.unique(y))):
    d["df{}".format(i)] = grouped.get_group(i)
    d["df{}".format(i)].index = d["df{}".format(i)].sample(frac=1).index.tolist()
  frames = pd.concat(d, axis=0).sum(axis=1, level=0)
  frames = frames.reset_index(level=0, drop=True)
  result = frames.sort_index()
  return np.array(result['X']),np.array(y)

def TransferEntropypval(X,Y,iterations):
  # y to x
  TrueTE = TransEnt(X,Y)
  surrogateValues = []
  for index in range(iterations):
    Series = PermutatePairwise(Y, X)
    surrogateValues.append(TransEnt(Series[1], Series[0]))

  fifth = np.percentile(surrogateValues, 5)
  nintyfifth = np.percentile(surrogateValues, 95)
  plt.axvline(fifth, color='r', linewidth=1)
  plt.axvline(nintyfifth, color='r', linewidth=1)
  plt.axvline(TrueTE, color='k', linestyle='dashed', linewidth=1)
  plt.hist(surrogateValues, bins='auto')
  plt.title(str(round(1 - getPvalue(surrogateValues, TrueTE), 6)))
  plt.show()

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

  # PB to positive number
  onelag = CompTE_emb(PT, PB, CR, 1, 1, 1, iterations)
  y_lag2 = CompTE_emb(PT, PB, CR, 1, 2, 1, iterations)
  z_lag2 = CompTE_emb(PT, PB, CR, 1, 1, 2, iterations)
  resultsdf.loc[len(resultsdf.index)] = ["Police brut", "Positive Tweets", onelag[0], onelag[1], onelag[2],
                                         "target1 source1 condition1"]
  resultsdf.loc[len(resultsdf.index)] = ["Police brut", "Positive Tweets", y_lag2[0], y_lag2[1], y_lag2[2],
                                         "target1 source2 condition1"]
  resultsdf.loc[len(resultsdf.index)] = ["Police brut", "Positive Tweets", z_lag2[0], z_lag2[1], z_lag2[2],
                                         "target1 source1 condition2"]
  return resultsdf