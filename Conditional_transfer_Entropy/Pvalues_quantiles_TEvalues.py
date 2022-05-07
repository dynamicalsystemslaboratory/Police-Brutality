import pandas as pd
import numpy as np
import transfer_entropy as TE

df = pd.read_csv("../Data/Seasonally_Adjusted_and_detrended_Patched.csv")
listofcolumnnames = {"Composite_Crimes_sa_det","NewProQ_PolBrut_MainPapers_sa_det_lagged1",
     "Pntweets_sa_det","Nntweets_sa_det",
     "medianPtweets_sa_det", "medianntweets_sa_det",
     "FractionPositive_sa_det","FractionNegative_sa_det",
     "felonies_sa_det","ProQ_crimeNYC_lagged1_sa_det"}

for col in listofcolumnnames:
  df[col] = df[col].astype(float)
  nonsymbolizedseries = np.array(df[col])
  locals()[col]  = TE.Median_Symbolise_3bins(nonsymbolizedseries)

Surrogate_iteration_Number = 2  ## this was set to 20 000 in the paper
## Manuscript Tables
NumberTetrad = {"Composite_Crimes_sa_det": Composite_Crimes_sa_det,"NewProQ_PolBrut_MainPapers_sa_det_lagged1":NewProQ_PolBrut_MainPapers_sa_det_lagged1,
     "Pntweets_sa_det":Pntweets_sa_det,"Nntweets_sa_det":Nntweets_sa_det}

MedianTetrad = {"Composite_Crimes_sa_det": Composite_Crimes_sa_det,"NewProQ_PolBrut_MainPapers_sa_det_lagged1":NewProQ_PolBrut_MainPapers_sa_det_lagged1,
     "medianPtweets_sa_det":medianPtweets_sa_det,"medianntweets_sa_det":medianntweets_sa_det}

FractionTetrad = {"Composite_Crimes_sa_det": Composite_Crimes_sa_det,"NewProQ_PolBrut_MainPapers_sa_det_lagged1":NewProQ_PolBrut_MainPapers_sa_det_lagged1,
     "FractionPositive_sa_det":FractionPositive_sa_det,"FractionNegative_sa_det":FractionNegative_sa_det}

## Supplement/ Raw Crimes analysis
NumberTetrad_raw_crimes = {"felonies_sa_det": felonies_sa_det,"NewProQ_PolBrut_MainPapers_sa_det_lagged1":NewProQ_PolBrut_MainPapers_sa_det_lagged1,
     "Pntweets_sa_det":Pntweets_sa_det,"Nntweets_sa_det":Nntweets_sa_det}

MedianTetrad_raw_crimes = {"felonies_sa_det": felonies_sa_det,"NewProQ_PolBrut_MainPapers_sa_det_lagged1":NewProQ_PolBrut_MainPapers_sa_det_lagged1,
     "medianPtweets_sa_det":medianPtweets_sa_det,"medianntweets_sa_det":medianntweets_sa_det}

FractionTetrad_raw_crimes = {"felonies_sa_det": felonies_sa_det, "NewProQ_PolBrut_MainPapers_sa_det_lagged1":NewProQ_PolBrut_MainPapers_sa_det_lagged1,
     "FractionPositive_sa_det":FractionPositive_sa_det,"FractionNegative_sa_det":FractionNegative_sa_det}

## Supplement/ Media on crimes
NumberTetrad_media_Crimes = {"ProQ_crimeNYC_lagged1_sa_det": ProQ_crimeNYC_lagged1_sa_det,"NewProQ_PolBrut_MainPapers_sa_det_lagged1":NewProQ_PolBrut_MainPapers_sa_det_lagged1,
     "Pntweets_sa_det":Pntweets_sa_det,"Nntweets_sa_det":Nntweets_sa_det}

MedianTetrad_media_crimes = {"ProQ_crimeNYC_lagged1_sa_det": ProQ_crimeNYC_lagged1_sa_det,"NewProQ_PolBrut_MainPapers_sa_det_lagged1":NewProQ_PolBrut_MainPapers_sa_det_lagged1,
     "medianPtweets_sa_det":medianPtweets_sa_det,"medianntweets_sa_det":medianntweets_sa_det}

FractionTetrad_media_crimes = {"ProQ_crimeNYC_lagged1_sa_det": ProQ_crimeNYC_lagged1_sa_det, "NewProQ_PolBrut_MainPapers_sa_det_lagged1":NewProQ_PolBrut_MainPapers_sa_det_lagged1,
     "FractionPositive_sa_det":FractionPositive_sa_det,"FractionNegative_sa_det":FractionNegative_sa_det}


List_of_Tetrads = [NumberTetrad,MedianTetrad,FractionTetrad,
                   NumberTetrad_raw_crimes,MedianTetrad_raw_crimes,FractionTetrad_raw_crimes,
                   NumberTetrad_media_Crimes,MedianTetrad_media_crimes,FractionTetrad_media_crimes]
List_of_Tetrads_names = ["NumberTetrad","MedianTetrad","FractionTetrad",
                        "NumberTetrad_raw_crimes","MedianTetrad_raw_crimes","FractionTetrad_raw_crimes",
                        "NumberTetrad_media_Crimes","MedianTetrad_media_crimes","FractionTetrad_media_crimes"]

##iterate over this loop to get all the conditional transfer entropy analysis done in the main manuscript and in the supplement
for index, item in enumerate(List_of_Tetrads):
    pval, TE_val, Quantiles = TE.CondTranEntNet(lista  = item, iterations = Surrogate_iteration_Number, lagy=1, lagz=1)
    tetradname = List_of_Tetrads_names[index]
    pval.to_csv("{}pvalues.csv".format(tetradname))
    TE_val.to_csv("{}TEvalues.csv".format(tetradname))
    Quantiles.to_csv("{}Quantiles.csv".format(tetradname))
