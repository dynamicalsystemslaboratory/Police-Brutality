import warnings
import numpy as np
from scipy.ndimage import label as sclabel
from scipy.spatial import distance
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt


def ConvergentCrossMapping(data, lib_column, target_column, lib=None, ref=None,
                           m=2, tau=1, tp=0, lib_sizes=None, n_samples=100,
                           n_neighbors=None, tw=0):
    """Python implementation of the convergent cross mapping algorithm.

    Python version of the Convergent Cross Mapping Algorithm (CCM) [1]_
    intended to reproduce the results of the original CCM implementation
    provided by the R package rEDM [2]_. If the predicted variable is
    predicted using its own records, the algorithm is equivalent to the
    Simplex method [3]_.

    Notes
    -----
        This implementation samples libraries with replacement while reference
        states are sampled without replacement. The code is developed for
        legibility and faithful reproduction of the original implementation,
        not for performance. This code is distributed under GNU GPL 3 license
        without any warranty.

    Parameters
    ----------
    data : array
        Two dimensional array of inputs. Axis 0 (rows) is time and Axis 1
        (columns) are variables.
    lib_column : int
        Column index in `data  of the variable from which cross-mapping is
        performed (predictor variable). From a causal perspective, `lib_column`
        points to the response variable which is embedded according to the
        embedding parameters `m` and `tau`.
    target_column : int
        Column index in `data` of the variable to which cross-mapping is
        performed (predicted variable). If same as `lib_column`, simplex
        self-predictions are performed [3]_. From a causal perspective,
        `target_column` points to the driving variable.
    lib : array
        Two dimensional array specifying segments of indices to be used as
        library points for identifying dynamical neighbors of reference points.
        First column are starting indices (included) and second columns are
        ending indices (not included). This follows python indexing convention
        where the first elements has index 0. By default, lib is None and the
        entire embedding serves as library excluding the reference point with
        an eventual Theiler window `tw`.
    ref : array
        Two dimensional array specifying segments of indices to be used as
        reference points for predictions. Same format as `lib`. Predictions are
        made at the references indices + `tp`. By default, all point in the
        embedding serve as potential reference points. Actual reference points
        depends on the prediction horizon `tp`.
    m : int
        Embedding dimension used for Takens state space reconstruction [4]_.
        Default is 2.
    tau : int
        Embedding delay used for Takens state space reconstruction [4]_.
        Default is 1.
    tp : int
        Time to prediction. Predictions are done on reference states + `tp`.
        Default is zero for instantaneous mapping. Could be either positive or
        negative. Cross-mapping skills are usually reported against a range
        of `tp` to distinguish causal relationships from synchrony [5]_.
    lib_sizes : array
        One dimensional array containing library sizes. Library sizes greater
        than the embedding length are automatically truncated.
    n_samples : int
        Specify the number of bootstrapped libraries for each library size.
        Libraries are bootstrapped randomly with replacement. Default is 100.
    n_neighbors : int
        Number of nearest neighbors. If None, default number of neighbors is
        set to `m` + 1.
    tw : int
        Theiler window specifying the time exclusion radius [6]_. For a
        reference state defined at t_ref, dynamical neighbors cannot be sampled
        in the library if they are indexed within [t_ref - tw, t_ref + tw].
        Default is 0.

    Returns
    -------
    x_array : array
        Array of forecast with n_samples columns and of length equal to
        the length of x_true.
    x_true : array
        1d array of the observed true values.


    References
    ----------
    .. [1] Sugihara, G., May, R., Ye, H., Hsieh, C. -h., Deyle, E., Fogarty, M.
    and Munch, S.: Detecting Causality in Complex Ecosystems, Science,
    338(6106), 496–500, doi:10.1126/science.1227079, 2012.
    .. [2] Ye, H., Clark, A., Deyle, E., Keyes, O. and Sugihara, G.: rEDM:
    Applications of Empirical Dynamic Modeling from Time Series. [online]
    Available from:
    https://cran.r-project.org/web/packages/rEDM/index.html, 2016.
    .. [3] Sugihara, G. and May, R. M.: Nonlinear forecasting as a way of
    distinguishing chaos from measurement error in time series, Nature,
    344(6268), 734–741, doi:10.1038/344734a0, 1990.
    .. [4]Takens, F.: Detecting strange attractors in turbulence, Lecture Notes
     in Mathematics, Berlin Springer Verlag, 898, 366,
     doi:10.1007/BFb0091924, 1981.
    .. [5] Ye, H., Deyle, E. R., Gilarranz, L. J. and Sugihara, G.:
    Distinguishing time-delayed causal interactions using convergent cross
    mapping, Scientific Reports, 5, 14750, doi:10.1038/srep14750, 2015.
    .. [6] Theiler, J.: Spurious dimension from correlation algorithms applied
    to limited time-series data, Phys Rev A Gen Phys, 34(3), 2427–2432, 1986.
    """

    if lib_sizes is None:
        lib_sizes = np.arange(10, min(len(data), 101), 10)

    if n_neighbors is None:
        n_neighbors = m + 1

    y = data[:, lib_column]
    x = data[:, target_column]

    # embedding of the response y
    my = np.zeros((len(y), m))
    for i in range(m):
        my[:, i] = np.roll(y, i * tau)

    # embedding window
    w = (m - 1) * tau

    # original time index
    ix = np.arange(len(my))
    # valid index for potential reference and library states
    ix_valid = np.arange(len(my))[
               w - ((w + tp) if w + tp < 0 else 0):-tp if tp > 0 else None]

    if ref is None:
        ix_ref = ix_valid
    else:
        ix_ref_sel = np.array([], dtype=int)
        for start, stop in ref:
            a = np.arange(start, stop)
            ix_ref_sel = np.concatenate((ix_ref_sel, a))
        ix_ref = ix_valid[np.isin(ix_valid, ix_ref_sel)]

    if lib is None:
        ix_lib = ix_valid
    else:
        ix_lib_sel = np.array([], dtype=int)
        for start, stop in lib:
            a = np.arange(start, stop)
            ix_lib_sel = np.concatenate((ix_lib_sel, a))
        ix_lib = ix_valid[np.isin(ix_valid, ix_lib_sel)]

    # index of predicted points
    ix_pred = ix[np.isin(ix, ix_ref + tp)]

    x_true = x[ix_pred]

    # precomputed distances
    # same as sklearn dist_mx = pairwise_distances(my)
    dist_mx = distance.squareform(distance.pdist(my))
    dist_mx[dist_mx == 0] = 1e-10  # get rid of tie distances

    k = 0
    for n_lib in lib_sizes:

        x_array = np.zeros((len(x_true), n_samples))

        for p, _ in enumerate(range(n_samples)):

            x_pred = np.zeros(len(x_true))

            for i, y_ref in enumerate(my[ix_ref]):
                t_ref = ix_ref[i]
                exclude = np.arange(t_ref - tw, t_ref + tw + 1)
                ix_lib_t = ix_lib[~np.isin(ix_lib, exclude)]
                # Random choice of library
                library = np.random.choice(ix_lib_t, min(n_lib, len(ix_lib_t)),
                                           replace=False)
                # Distances to t_ref
                dist = dist_mx[t_ref][library]
                # Neighbors to t_ref
                neighbors = library[
                    np.argpartition(dist, n_neighbors)[:n_neighbors]]
                dist_neighbors = dist_mx[t_ref][neighbors]
                # Weights computation
                weight = np.exp(-dist_neighbors / np.min(dist_neighbors))
                # Cross-prediction
                x_pred[i] = np.sum(weight * x[neighbors + tp]) / weight.sum()

            x_array[:, p] = x_pred

            k += 1

    return x_array, x_true


def getPvalue(surrogate,value):
    surr = np.append(value,surrogate)
    sorted = np.sort(surr)
    index  = np.where(sorted == value)[0][0]
    Pvalue = (1+index)/len(sorted)
    return Pvalue


def get_rho_diff(source, target, m, tau):
    data = np.array([source, target]).T
    target_predicted, true_source = ConvergentCrossMapping(data=data, lib_column=0, target_column=1,
                                                               m=m, tau=tau, lib_sizes=[len(source)], n_samples=1)
    rho_target_hat_using_source = stats.pearsonr(target_predicted.T[0], true_source)[0]

    source_predicted, true_target = ConvergentCrossMapping(data=data, lib_column=1, target_column=0,
                                                               m=m, tau=tau, lib_sizes=[len(source)], n_samples=1)
    rho_source_hat_using_target = stats.pearsonr(source_predicted.T[0], true_target)[0]
    rho_diff = rho_source_hat_using_target - rho_target_hat_using_source

    return rho_diff


def Get_two_rhos(source, target, tp,m=2, tau=1):
    data = np.array([source, target]).T
    target_predicted, true_source = ConvergentCrossMapping(data=data, lib_column=0, target_column=1, tp = tp,
                                                               m=m, tau=tau, lib_sizes=[len(source)], n_samples=1)
    rho_target_hat_using_source = stats.pearsonr(target_predicted.T[0], true_source)[0]

    source_predicted, true_target = ConvergentCrossMapping(data=data, lib_column=1, target_column=0, tp = tp,
                                                               m=m, tau=tau, lib_sizes=[len(source)], n_samples=1)
    rho_source_hat_using_target = stats.pearsonr(source_predicted.T[0], true_target)[0]
    return rho_source_hat_using_target, rho_target_hat_using_source

def Get_CCM_p_val(source, target, itetations, m=10, tau=1):
    ## H0: source is not the source
    ## H1: source is the source
    true_rho_diff = get_rho_diff(source =source , target=target, m=m ,tau=tau)
    surro_source = source.copy()
    surro_target = target.copy()
    surrogateValues = []
    for index in range(itetations):
        np.random.shuffle(surro_source)
        np.random.shuffle(surro_target)
        surrogateValues.append(get_rho_diff(source =surro_source , target=surro_target, m=m ,tau=tau))
    return 1 - getPvalue(surrogateValues, true_rho_diff), true_rho_diff

def lib_corr(lib,true):
    rhos = []
    for i in range(lib.shape[1]):
        ## loop over the number of samples
        rhos.append(stats.pearsonr(lib[:,i],true)[0])
    return rhos

def Get_CCM_Plot_df(source, target, m_source=2, m_target=2, tau_source=1, tau_target=1, nsamp=20):
    ##Make sure to change nsmap for short time series
    data = np.array([source, target]).T
    l = len(source)
    lib_size = np.linspace(l/10,l,10, dtype= int)-(max(m_source,m_target)-1)*max(tau_target,tau_source)
    lib_size = lib_size[lib_size > (max(m_source,m_target)-1)*max(tau_target,tau_source)]
    df_source_hat_knowing_target = pd.DataFrame()
    df_target_hat_knowing_source = pd.DataFrame()
    for lib in lib_size:
        # target_predicted, true_source = ConvergentCrossMapping(data=data, lib_column=0, target_column=1,
        #         #                                                            m=m_source, tau=tau_source, lib_sizes=[lib], n_samples=nsamp)
        source_predicted, true_target = ConvergentCrossMapping(data=data, lib_column=1, target_column=0,
                                                                   m=m_target, tau=tau_target, lib_sizes=[lib], n_samples=nsamp)
        df_source_hat_knowing_target[str(lib)] = lib_corr(source_predicted,true_target)
       # df_target_hat_knowing_source[str(lib)] = lib_corr(target_predicted,true_source)
    return df_source_hat_knowing_target#, df_target_hat_knowing_source


def get_scientitic_report_model_1(l,taw = 0):
    ## ref[5]
    x = np.zeros([l])
    y = np.zeros([l])
    x[0] = 0.2
    y[0] = 0.4
    for i in range(1,taw):
        x[i] = x[i-1]*(3.78-3.78*x[i-1]-0.07*y[i-1])
        y[i] = y[i-1]*(3.77 -3.77*y[i-1]-0.08*x[i-1])
    for i in range(taw,l):
        x[i] = x[i-1]*(3.78-3.78*x[i-1]-0.07*y[i-1])
        y[i] = y[i-1]*(3.77 -3.77*y[i-1]-0.08*x[i-1-taw])
    return x,y

def Get_delayed_rhos_df(source, target, max_range = 8, m_source=2, m_target=2, tau_source=1, tau_target=1, nsamp=20, lib = 1000):
    ## Make sure to change nsmap for short time series
    ## tau_target is the optimal tau for the target time seires and so on...
    data = np.array([source, target]).T
    tps = np.arange(-max_range,max_range+1,1)
    df_source_hat_knowing_target = pd.DataFrame()
    df_target_hat_knowing_source = pd.DataFrame()
    for tp in tps:
        target_predicted, true_source = ConvergentCrossMapping(data=data, lib_column=0, target_column=1,tp=tp,
                                                                   m=m_source, tau=tau_source, lib_sizes=[lib], n_samples=nsamp)
        source_predicted, true_target = ConvergentCrossMapping(data=data, lib_column=1, target_column=0,tp=tp,
                                                                   m=m_target, tau=tau_target, lib_sizes=[lib], n_samples=nsamp)
        df_source_hat_knowing_target[str(tp)] = lib_corr(source_predicted,true_target)
        df_target_hat_knowing_source[str(tp)] = lib_corr(target_predicted,true_source)
    return df_source_hat_knowing_target, df_target_hat_knowing_source

def CCM_plot(df_source_hat_knowing_target,source_name, target_name, title):
    
    plt.rcParams.update({
      'font.size' : 12,
      "lines.linewidth": 1,
      "font.family":"serif",
    })
    
    source_color = "b"
    target_color = "r"
    
    l_source_hat =  [int(i) for i in df_source_hat_knowing_target.columns]
    mean_rho_source_hat = [np.mean(df_source_hat_knowing_target[i]) for i in df_source_hat_knowing_target.columns]
    std_rho_source_hat = [np.std(df_source_hat_knowing_target[i]) for i in df_source_hat_knowing_target.columns]
    
    # l_target_hat =  [int(i) for i in df_target_hat_knowing_source.columns]
    # mean_rho_target_hat = [np.mean(df_target_hat_knowing_source[i]) for i in df_target_hat_knowing_source.columns]
    # std_rho_target_hat = [np.std(df_target_hat_knowing_source[i]) for i in df_target_hat_knowing_source.columns]
    
    
    fig,(ax)  = plt.subplots(1, 1,
                                sharey='row',figsize=(5,5))
    ax.set_ylim([0,1])
    ax.set_ylabel(r'$\rho$')
    ax.set_xlabel("lag")
    #plt.yticks([0,250,500,750,1000])
    #plt.xticks([0,500,1000,1500,2000,2500,3000,3500,4000])
    #fig.savefig("percentilespos.png", dpi=150,bbox_inches="tight")
    
    
    ax.errorbar(l_source_hat, mean_rho_source_hat, std_rho_source_hat, color=source_color, marker='o',capsize=5)
    ax.plot(l_source_hat, mean_rho_source_hat,color=source_color, label = r'$\hat{}|M{}$'.format(source_name,target_name))

    # ax.errorbar(l_target_hat, mean_rho_target_hat, std_rho_target_hat, color=target_color, marker='o',capsize=5)
    # ax.plot(l_target_hat, mean_rho_target_hat,color=target_color, label = r'$\hat{}|M{}$'.format(target_name,source_name))
    plt.legend()
    plt.title(title)
    plt.show()