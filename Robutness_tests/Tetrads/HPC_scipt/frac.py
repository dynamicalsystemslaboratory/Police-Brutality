import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from math import log2
from multiprocessing import Process

def Change_Symbolise(Timeseries):
    return (np.diff(Timeseries) > 0).astype(int)


def Median_Symbolise(Timeseries):
    Median = np.median(Timeseries)
    return (Timeseries < Median).astype(int)


def Median_Symbolise_3bins(Timeseries):
    Q1 = np.quantile(Timeseries, 1 / 3)
    Q3 = np.quantile(Timeseries, 2 / 3)
    score_labels = [0, 1, 2]
    binned = pd.cut(Timeseries, [(min(Timeseries) - 1), Q1, Q3, (max(Timeseries) + 1)], labels=score_labels)
    return (binned.astype(int))


def CondEntropy(X, Y):
    numpy_data = np.array([X, Y])
    df = pd.DataFrame(data=numpy_data, index=["X", "Y"]).T
    T = len(X)
    data_crosstab = pd.crosstab(df['X'],
                                df['Y'],
                                margins=False)  # access columns first
    uniqueX, CountX = np.unique(X, return_counts=True)
    uniqueY, CountY = np.unique(Y, return_counts=True)
    # py = CountY/T
    HXY = 0
    for i in range(len(uniqueY)):
        HXy = 0
        for j in range(len(uniqueX)):
            pxy = data_crosstab[uniqueY[i]][uniqueX[j]] / sum(data_crosstab[uniqueY[i]])
            if pxy != 0: HXy -= pxy * log2(pxy)
        py = CountY[i] / T
        HXY += py * HXy
    return HXY


## not modifified with new joint or different lags
def TransEnt(X, Y, lag):
    L = len(X)
    Xt = X[lag:L]
    Xlag = X[0:(L - lag)]
    Ylag = Y[0:(L - lag)]
    xylag = joint(Xlag, Ylag)
    TYX = CondEntropy(Xt, Xlag) - CondEntropy(Xt, xylag)
    return TYX


def getPvalue(surrogate, value):
    surr = np.append(value, surrogate)
    sorted = np.sort(surr)
    index = np.where(sorted == value)[0][0]
    Pvalue = (1 + index) / len(sorted)
    return Pvalue


## shuffle x keeping yz dynamics
def Permutate(x, y, z):
    ZZ = z.copy()
    ZZ.append(y)
    dynamics = joint(ZZ)
    numpy_data = np.array([x, dynamics])
    df = pd.DataFrame(data=numpy_data, index=["X", "YZ"]).T
    grouped = df.groupby(df.YZ)
    d = {}
    for i in range(len(np.unique(dynamics))):
        d["df{}".format(i)] = grouped.get_group(i)
        d["df{}".format(i)].index = d["df{}".format(i)].sample(frac=1).index.tolist()
    frames = pd.concat(d, axis=0).sum(axis=1, level=0)
    frames = frames.reset_index(level=0, drop=True)
    result = frames.sort_index()
    return np.array(result['X']), np.array(y), np.array(z)


def joint(L):
    numpy_data = np.array(L)
    df = pd.DataFrame(data=numpy_data).T
    df = df.applymap(str)
    df['joint'] = df.values.sum(axis=1)
    df.joint = df.joint.astype('category').cat.codes
    return np.array(df["joint"])



# list is a dictionary with name and iteration is the number of shufffle
def CondTranEntNet(lista, iterations):
    lx = 1
    ly = 1
    lz = np.ones(len(lista) - 2, dtype=int)
    #df = pd.DataFrame(columns=[*lista])
    df = pd.DataFrame(columns=["Sender", "receiver", "pval"])
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
                TrueTE = CondTransEnt(X=toseries, Y=fromseries, Z=conditionedonseries, lagx=lx, lagy=ly, lagz=lz)
                surrogateValues = []
                for index in range(iterations):
                    Series = Permutate(fromseries, toseries, conditionedonseries)
                    surrogateValues.append(CondTransEnt(Series[1], Series[0], Series[2], lagx=lx, lagy=ly, lagz=lz))
                #df.at[i, receiver] = round(1 - getPvalue(surrogateValues, TrueTE), 4)
                df.loc[len(df.index)] = [sender,receiver, 1 - getPvalue(surrogateValues, TrueTE)]
    return df


## Conditional transfer entropy from Y to X conditioned on Z with lag=lag
## x and y are series, Z is a list of series. lagx and lag y are numbers and lagz is a list of lagz corresponding to the list of conditions
def CondTransEnt(X, Y, Z, lagx, lagy, lagz):
    L = len(X)
    maxx = max([lagx, lagy, max(lagz)])
    Xt = X[maxx:L]
    Xlag = X[maxx - lagx:(L - lagx)]
    Ylag = Y[maxx - lagy:(L - lagy)]
    Zlag = []
    for i, z in enumerate(Z):
        Zlag.append(z[maxx - lagz[i]:(L - lagz[i])])
    Zlag.append(Xlag)
    xzlag = joint(Zlag)
    Zlag.append(Ylag)
    xyzlag = joint(Zlag)
    TYXZ = CondEntropy(Xt, xzlag) - CondEntropy(Xt, xyzlag)
    return TYXZ



def FractionOut(index):
    np.random.seed(index)
    FracNetworkoutout = pd.DataFrame(columns= ["Sender", "receiver", "pval"])
    Frac_sync = Median_Symbolise_3bins(np.random.normal(0, 0.35, length))
    Frac_d = {"Frac_sync": Frac_sync,
         "NewProQ_PolBrut_MainPapers_sa_det_lagged1": NewProQ_PolBrut_MainPapers_sa_det_lagged1,
         "FractionPositive_sa_det": FractionPositive_sa_det, "FractionNegative_sa_det": FractionNegative_sa_det}
    FracNetworkoutout = FracNetworkoutout.append(CondTranEntNet(lista  = Frac_d,iterations = iteration_number))

    Frac_sync = Median_Symbolise_3bins(np.random.normal(0, 250, length))
    Frac_d = {"Composite_Crimes_sa_det": Composite_Crimes_sa_det,
         "Frac_sync": Frac_sync,
         "FractionPositive_sa_det": FractionPositive_sa_det, "FractionNegative_sa_det": FractionNegative_sa_det}
    FracNetworkoutout = FracNetworkoutout.append(CondTranEntNet(lista=Frac_d, iterations=iteration_number))

    Frac_sync = Median_Symbolise_3bins(np.random.normal(0, 0.05, length))
    Frac_d = {"Composite_Crimes_sa_det": Composite_Crimes_sa_det,
         "NewProQ_PolBrut_MainPapers_sa_det_lagged1": NewProQ_PolBrut_MainPapers_sa_det_lagged1,
         "Frac_sync": Frac_sync, "FractionNegative_sa_det": FractionNegative_sa_det}
    FracNetworkoutout = FracNetworkoutout.append(CondTranEntNet(lista=Frac_d, iterations=iteration_number))

    Frac_sync = Median_Symbolise_3bins(np.random.normal(0, 0.07, length))
    Frac_d = {"Composite_Crimes_sa_det": Composite_Crimes_sa_det,
         "NewProQ_PolBrut_MainPapers_sa_det_lagged1": NewProQ_PolBrut_MainPapers_sa_det_lagged1,
         "FractionPositive_sa_det": FractionPositive_sa_det, "Frac_sync": Frac_sync}
    FracNetworkoutout = FracNetworkoutout.append(CondTranEntNet(lista=Frac_d, iterations=iteration_number))
    FracNetworkoutout.to_csv("Frac{}.csv".format(index))




df = pd.read_csv("../../../Data/Seasonally_Adjusted_and_detrended_Patched.csv")

listofcolumnnames = {"Composite_Crimes_sa_det", "NewProQ_PolBrut_MainPapers_sa_det_lagged1",
                     "Pntweets_sa_det", "Nntweets_sa_det",
                     "medianPtweets_sa_det", "medianntweets_sa_det",
                     "Sent_sa_det", "PandNtweets_sa_det",
                     "FractionPositive_sa_det", "FractionNegative_sa_det"}

for col in listofcolumnnames:
    df[col] = df[col].astype(float)
    nonsymbolizedseries = np.array(df[col])
    locals()[col] = Median_Symbolise_3bins(nonsymbolizedseries)

iteration_number = 20000
Number_of_sync_series = 100
length = 3726

def main():
    for i in range(Number_of_sync_series):
        Process(target=FractionOut, args=(i,)).start()



if __name__ == '__main__':
    main()

