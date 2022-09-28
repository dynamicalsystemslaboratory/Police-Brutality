This readme file will help you replicate the results in the paper titled: "Understanding the role of media and crimes in the formation of public opinion towards the police"

There are two main directories and 4 notebooks:

1-Data

2-modules

3-Convergent cross mapping

4-High-resolution analysis in the wake of George Floyd’s murder

5-Transfer Entropy

6-Partial Correlation

7-Memory of target variables
 ---

1- Data:
- "Times_series_sa_det.csv" contains the daily raw time series and detrended and seasonally adjusted time series; and 
- "Floyd_period_Minutes_PBandNeg.csv" contains time series of MPB and NT at a resolution of one minute; and
- "Negative_tweets_Embedding_optimization.csv" contains the predictive power of the NT time series for different embedding dimensions; and
- "Positive_tweets_Embedding_optimization.csv" contains the predictive power of the PT time series for different embedding dimensions.

2- modules
- "EDM.py" is a module that includes the convergent cross mapping functions; and
- "transfer_entropy.py" is a module that includes several functions, such as the computation of conditional transfer entropy, the process of symbolization, and the estimation of the transition probabilities;


3-Convergent cross mapping
is the script that produces the convergence cross mapping plots in the supplement and displays them within the notebook.

4-High-resolution analysis in the wake of George Floyd’s murder.ipynb
is the script that produces the statistics of the analysis in the wake of George Floyd’s murder and displays the results within the notebook.

5-Transfer Entropy
is the script that computes the transfer entropy values and test statistics and displays them within the notebook.

6-Partial Correlation
is the script that computes the partial correlation in the supplement and displays them within the notebook.

7-Memory of target variables
is the script that computes the entropy of the target variables conditioned on their history and displays them within the notebook.
