This readme file will help you replicate the results in the paper titled: "Understanding the role of media and crimes on public opinion towards the police"

There are three main directories:

1-Data

2-Conditional_transfer_Entropy

3-Robutness_tests

1- Data:
- "Raw_time_series.csv" contains the rawtime series before detrending and seasonal adjustment; and 
- "Seasonally_Adjusted_and_detrended_Patched.csv" contains data after removing the trend and the seasonal effects


2-Conditional_transfer_Entropy:
- "transfer_entropy.py" is a module that includes several functions, such as the calculation of conditional transfer entropy, the process of symbolization, and the estimation of the transition matrices;
- "Pvalues_quantiles_TEvalues.py" is the script that generates p-values, transfer entropy values, and quantiles corresponding to the 9 tetrads studied in the paper and the supplement as 27 csv files; and
- "TransitionMatrices.ipynb" is the script that computes the transition matrices in the supplement and displays them within the notebook.


3-Robutness_tests:

-The subdirectory "Tetrads" contains a code for parallelized computing that was used to do the first robustness test (the scripts "frac.py", "med.py", and "numb.py" generate p-values of all the links in 100 tetrads each containing one randomly generated time series; the seed for the random-number generator is also provided in the code for replication of the results; and the.batch files are included as a sample); and

-The subdirectory "Pentads" contains a code for parallelized computing that was used to do the second robustness test (the scripts "frac.py", "med.py", and "numb.py" generate p-values of all the links in 100 pentads each containing 1 randomly generated time series; the seed for the random-number generator is also provided in the code for replication of the results; and  the .batch files are included as a sample).

Python dependencies:
Pandas V 1.1.3
Numpy V 1.21.5
