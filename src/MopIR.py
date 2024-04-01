import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def lin_interp(spectra, low = 400, high = 4000):
    '''
    Linear interpolation on each spectrum
    
    Parameters:
    -----
    spectra : {pandas.Series, pandas.DataFrame}
        FTIR spectrum/spectra
        
    low : int, default = 400
        the lower limit of the interpolation range
    
    high : int, default = 4000
        the upper limit of the interpolation range
     
    Return
    ------
    spectra :  {pandas.Series, pandas.DataFrame}
        Interpolated FTIR data in wavelength range of 400-4000
        a pandas series/dataframe that index/columns represent the interpolated wavenumber
    '''

    def _check_type(spectra):
        
        if not isinstance(spectra, pd.DataFrame) and not isinstance(spectra, pd.Series):
            raise TypeError(f" 'spectra' should be a 'pandas.DataFrame' or 'pandas.Series' ")
        
        return spectra

    spectra = _check_type(spectra)
    
    ## define the standard range of FTIR data
    interpolation_range = [i for i in range(low, high+1)]
    
    def _helper(y,x,x_input):
            f = interpolate.interp1d(x, y, fill_value='extrapolate')
            return(f(x_input))
    
    if isinstance(spectra, pd.Series):
        '''
        a single spectrum is represented by a pandas series, where index = measured wavenumber, and values = absorbance
        '''
        measured_k = spectra.index.to_numpy(dtype=np.int64)
        f = interpolate.interp1d(x=measured_k, y=spectra.values, fill_value='extrapolate')
        spectra = pd.Series(f(interpolation_range), index=interpolation_range)
    
    if isinstance(spectra, pd.DataFrame):
        '''
        multiple spectra are represented by a pandas dataframe, where index = sample ID, columns = measured wavenumber, and entries = absorbance
        '''
        measured_k = spectra.columns.to_numpy(dtype=np.int64)
        sample_ID = spectra.index
        
        
        spectra = pd.DataFrame(np.apply_along_axis(_helper, axis=1, arr=spectra.values, x = measured_k, x_input = interpolation_range),
                     index=sample_ID, columns=interpolation_range)
        
        
    
    return(spectra)


def denoise(spectra,
            baseline = True,
            als_params={"lam":1000,
                       "p":0.01,
                       "niter":10},
            sg_params={"window_length":31,
                      "polyorder":7,
                      "deriv":0},
            normalization = "vector"):
    
    '''
    Implement baseline correction, smoothing and normalization(optinal) on each spectrum
    
    Parameters:
    -----
    spectra : {pandas.Series, pandas.DataFrame}
        FTIR spectrum/spectra

    baseline: bool, default = True
        if True, the baseline correction will be implemented
    
    als_params : dict
        parameter dictionary for Asymmetric Least Square method
    
    sg_params : dict
        parameter dictionary for SG-filter
    
    normalization : {'vector', 'min_max', 'snv', 'None'}, \
        default = 'vector'
        normalization method used to normalize each spectrum.
        
    Return
    ------
    spectra : {pandas.Series, pandas.DataFrame}
        Denoised FTIR spectrum/spectra
    '''

    def _check_type(spectra):
        
        if not isinstance(spectra, pd.DataFrame) and not isinstance(spectra, pd.Series):
            raise TypeError(f" 'spectra' should be a 'pandas.DataFrame' or 'pandas.Series' ")
        
        return spectra

    spectra = _check_type(spectra)
    
    if isinstance(spectra, pd.Series):
        '''
        a single spectrum is represented by a pandas series, where index = measured wavenumber, and values = measurement
        '''
        index = spectra.index
        
        if baseline:
        ## baseline correction
            spectra = _als_baseline_correction(spectra, **als_params)
        
        ## smoothing
        spectra = pd.Series(savgol_filter(spectra, **sg_params), index=index)
        
        ## normalization
        spectra = _normalization(spectra, method=normalization)
        
        return pd.Series(spectra, index=index)
    
    if isinstance(spectra, pd.DataFrame):
        '''
        multiple spectra are represented by a pandas dataframe, where index = spectrum ID, columns = measured wavenumber, and entries = measurements
        '''
        index = spectra.index
        columns = spectra.columns
        
        ## baseline correction
        if baseline:
            spectra = spectra.apply(_als_baseline_correction, axis=1, **als_params) ## row-wise
        
        ## smoothing
        spectra = savgol_filter(spectra, **sg_params) # output = np.array
        
        ## normalization
        spectra = _normalization(spectra, method=normalization)
        
        return pd.DataFrame(spectra,
                           index=index,
                           columns=columns)

def _als_baseline_correction(y, lam = 1000, p = 0.01, niter=10):
    '''
    Use Asymmetric Least Square method to implement baseline correction on each spectrum
    
    Parameters:
    y: {pandas.Series, ndarray}
        a one-dimensional feature vector (or a single spectrum)
    
    lam: float, default = 0
        the parameter used to adjust the balance between fitness and smoothness, the higher the lam, the higher the smoothness
    
    p: float, default = 0.01
        the parameter used to set weights asymmetrically, p is rrecommended to set between 0.001 and 0.1
    
    niter: int, default = 10
        number of iteration
    -----
    
    Return
    ------
    y : {pandas.Series, ndarray}
        The type of output is the same as the input.
        A one-dimensional feature vector (or a single spectrum) subtracted by its baseline
    '''
    
    def _baseline_als(y, lam, p, niter=10):
        L = len(y)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for i in tqdm(range(niter)):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        return z
    
    z = _baseline_als(y=y, lam=lam, p=p, niter=niter)
    
    return (y-z)


def _normalization(spectra,
                   method = "vector"):
    
    '''
    Normalization of spectra
    
    Parameters:
    -----
    spectra : {ndarray}
        FTIR spectrum/spectra
    
    method : {'vector', 'min_max', 'snv', None}, \
        default = 'vector'
        normalization method used to normalize each spectrum. 
        If method == None, then input spectrum/spectra will NOT be normalized
        
    Return
    ------
    spectra : {ndarray}
        Normalized FTIR spectrum/spectra
    
    '''
    def _check_method(method):
        if method not in ("vector", "min_max", "snv", None):
            raise ValueError(f"{method} is not a valid normalization method")
        
        return method
    
    method = _check_method(method)
    
    if spectra.ndim == 1:
        '''
        a single spectrum is represented by a pandas series, where index = measured wavenumber, and values = measurements
        '''
        if method == "vector":
            norm = np.linalg.norm(spectra, ord=None)
            spectra = spectra/norm
            
        if method == "min_max":
            spectra = (spectra - spectra.min())/(spectra.max() - spectra.min())
        
        if method == "snv":
            spectra = (spectra - spectra.mean())/(spectra.std())
            
        if method == None:
            spectra = spectra
        
        return(spectra)
    
    if spectra.ndim == 2:
        '''
        multiple spectra are represented by a pandas dataframe, where index = spectrum ID, columns = measured wavenumber, and entries = measurements
        '''
        
        if method == "vector":
            norms = np.linalg.norm(spectra, axis=1, ord=None)
            spectra = spectra/norms.reshape(-1,1)
            
        if method == "min_max":
            spectra = (spectra - spectra.min(axis=1).reshape(-1,1))/((spectra.max(axis=1)-spectra.min(axis=1)).reshape(-1,1))
            
        if method == "snv":
            spectra = (spectra - spectra.mean(axis=1).reshape(-1,1))/(spectra.std(axis=1).reshape(-1,1))
        
        if method == None:
            spectra = spectra
        
        
        return(spectra)

def var_threshold(spectra, threshold=None):
    
    '''
    Apply variance threshold method on spectra
    
    Parameters:
    -----
    spectra : {pandas.DataFrame}
        Raw or preprocessed spectra
    
    threshold : float, default = None
        threshold value for variance threshold feature selection,
        
        
    
    Return
    ------
    spectra : {pandas.Series} if `threshold = None`, else {pandas.DataFrame}
        A pandas.Series of variance or each feature, 
        or a pandas.DataFrame with reduced features whose variances do not meet the threshold
        
    '''
    
    def _check_type(spectra):
        
        if not isinstance(spectra, pd.DataFrame):
            raise TypeError(f" 'spectra' should be a 'pandas.DataFrame' ")
        
        return spectra
    
    spectra = _check_type(spectra)
    
    if threshold is None:
        return spectra.var()
    
    else:
        removal = spectra.var()[spectra.var() < threshold].index
        spectra = spectra.drop(removal, axis=1)
        return spectra

def dim_reduction(spectra, n_components=2):
    '''
    Apply Principal Component Analysis on spectra
    
    Parameters:
    -----
    spectra : {pandas.DataFrame}
        Raw or preprocessed spectra
    
    n_components : int, default = 2
        Number of components to keep
        
        
    
    Return
    ------
    res :  tuple
        A tuple consists of a pandas.DataFrame with columns equal to principal components, 
        and a numpy.array with explained variance of each principal component
        
    '''
    
    def _check_type(spectra):
        
        if not isinstance(spectra, pd.DataFrame):
            raise TypeError(f" 'spectra' should be a 'pandas.DataFrame' ")
            
        return spectra
            
    spectra = _check_type(spectra)
    
    ## create PCA objects
    pca = PCA(n_components=n_components)
    ## create StandardScaler objects
    scaler = StandardScaler()
    
    ## Scaling and transformation
    spectra_scaled = pd.DataFrame(scaler.fit_transform(spectra), 
                                  index = spectra.index, 
                                  columns=spectra.columns)
    
    spectra_pca = pd.DataFrame(pca.fit_transform(spectra_scaled),
                               index = spectra.index,
                               columns = ['PC' + str(i) for i in range(1, n_components+1)])
        
    
    ## explained_variance_ratio
    evr = pca.explained_variance_ratio_
    
    
    return(spectra_pca, evr)
