import numpy as np
import bisect as bs
from box_plot_stats import box_plot_stats
    

## calculate parameters for phi relevance function
def phi_ctrl_pts(
    
    ## arguments / inputs
    y,                    ## response variable y
    method = "auto",      ## relevance method ("auto" or "manual")
    xtrm_type = "both",   ## distribution focus ("high", "low", "both")
    coef = 1.5,           ## coefficient for box plot
    ctrl_pts = None       ## input for "manual" rel method
    ):
    
    """ 
    generates the parameters required for the 'phi()' function, specifies the 
    regions of interest or 'relevance' in the response variable y, the notion 
    of relevance can be associated with rarity
    
    controls how the relevance parameters are calculated by selecting between 
    two methods, either "auto" or "manual"
    
    the "auto" method calls the function 'phi_extremes()' and calculates the 
    relevance parameters by the values beyond the interquartile range
    
    the "manual" method calls the function 'phi_range()' and determines the 
    relevance parameters by user specification (the use of a domain expert 
    is recommended for utilizing this method)
    
    returns a dictionary containing 3 items "method", "num_pts", "ctrl_pts": 
    1) the "method" item contains a chartacter string simply indicating the 
    method used calculate the relevance parameters (control points) either 
    "auto" or "manual"
    
    2) the "num_pts" item contains a positive integer simply indicating the 
    number of relevance parameters returned, typically 3
    
    3) the "ctrl_pts" item contains an array indicating the regions of 
    interest in the response variable y and their corresponding relevance 
    values mapped to either 0 or 1, expressed as [y, 0, 1]
    
    ref:
    
    Branco, P., Ribeiro, R., Torgo, L. (2017).
    Package 'UBL'. The Comprehensive R Archive Network (CRAN).
    https://cran.r-project.org/web/packages/UBL/UBL.pdf.
    
    Ribeiro, R. (2011). Utility-Based Regression.
    (PhD Dissertation, Dept. Computer Science, 
    Faculty of Sciences, University of Porto).
    """
    
    ## quality check for response variable 'y'
    if any(y == None) or isinstance(y, (int, float, complex)):
        raise ValueError("response variable 'y' must be specified and numeric")
    
    ## quality check for user specified method
    if method in ["auto", "manual"] is False:
        raise ValueError("method must be either: 'auto' or 'manual' ")
    
    ## quality check for xtrm_type
    if xtrm_type in ["high", "low", "both"] is False:
        raise ValueError("xtrm_type must be either: 'high' or 'low' or 'both' ")
    
    ## conduct 'extremes' method (default)
    if method == "auto":
        phi_params = phi_extremes(y, xtrm_type, coef)
    
    ## conduct 'range' method
    if method == "manual":
        phi_params = phi_range(ctrl_pts)
    
    ## return phi relevance parameters dictionary
    return phi_params

## calculates phi parameters for statistically extreme values
def phi_extremes(y, xtrm_type, coef):
    
    """ 
    assigns relevance to the most extreme values in the distribution of response 
    variable y according to the box plot stats generated from 'box_plot_stat()'
    """
    
    ## create 'ctrl_pts' variable
    ctrl_pts = []
    
    ## calculates statistically extreme values by
    ## box plot stats in the response variable y
    ## (see function 'boxplot_stats()' for details)
    bx_plt_st = box_plot_stats(y, coef)
    
    ## calculate range of the response variable y
    rng = [y.min(), y.max()]
    
    ## adjust low
    if xtrm_type in ["both", "low"] and any(bx_plt_st["xtrms"]
    < bx_plt_st["stats"][0]):
        ctrl_pts.extend([bx_plt_st["stats"][0], 1, 0])
   
    ## min
    else:
        ctrl_pts.extend([rng[0], 0, 0])
    
    ## median
    if bx_plt_st["stats"][2] != rng[0]:
        ctrl_pts.extend([bx_plt_st["stats"][2], 0, 0])
    
    ## adjust high
    if xtrm_type in ["both", "high"] and any(bx_plt_st["xtrms"]
    > bx_plt_st["stats"][4]):
        ctrl_pts.extend([bx_plt_st["stats"][4], 1, 0])
    
    ## max
    else:
        if bx_plt_st["stats"][2] != rng[1]:
            ctrl_pts.extend([rng[1], 0, 0])
    
    ## store phi relevance parameter dictionary
    phi_params = {}
    phi_params["method"] = "auto"
    phi_params["num_pts"] = round(len(ctrl_pts) / 3)
    phi_params["ctrl_pts"] = ctrl_pts
    
    ## return dictionary
    return phi_params

## calculates phi parameters for user specified range
def phi_range(ctrl_pts):
    
    """
    assigns relevance to values in the response variable y according to user 
    specification, when specifying relevant regions use matrix format [x, y, m]
    
    x is an array of relevant values in the response variable y, y is an array 
    of values mapped to 1 or 0, and m is typically an array of zeros
    
    m is the phi derivative adjusted afterward by the phi relevance function to 
    interpolate a smooth and continous monotonically increasing function
    
    example:
    [[15, 1, 0],
    [30, 0, 0],
    [55, 1, 0]]
    """
    
    ## convert 'ctrl_pts' to numpy 2d array (matrix)
    ctrl_pts = np.array(ctrl_pts)
    
    ## quality control checks for user specified phi relevance values
    if np.isnan(ctrl_pts).any() or np.size(ctrl_pts, axis = 1) > 3 or np.size(
    ctrl_pts, axis = 1) < 2 or not isinstance(ctrl_pts, np.ndarray):
        raise ValueError("ctrl_pts must be given as a matrix in the form: [x, y, m]" 
              "or [x, y]")
    
    elif (ctrl_pts[1: ,[1, ]] > 1).any() or (ctrl_pts[1: ,[1, ]] < 0).any():
        raise ValueError("phi relevance function only maps values: [0, 1]")
    
    ## store number of control points
    else:
        dx = ctrl_pts[1:,[0,]] - ctrl_pts[0:-1,[0,]]
    
    ## quality control check for dx
    if np.isnan(dx).any() or dx.any() == 0:
        raise ValueError("x must strictly increase (not na)")
    
    ## sort control points from lowest to highest
    else:
        ctrl_pts = ctrl_pts[np.argsort(ctrl_pts[:,0])]
    
    ## calculate for two column user specified control points [x, y]
    if np.size(ctrl_pts, axis = 1) == 2:
        
        ## monotone hermite spline method by fritsch & carlson (monoH.FC)
        dx = ctrl_pts[1:,[0,]] - ctrl_pts[0:-1,[0,]]
        dy = ctrl_pts[1:,[1,]] - ctrl_pts[0:-1,[1,]]
        sx = dy / dx
        
        ## calculate constant extrapolation
        m = np.divide(sx[1:] + sx[0:-1], 2)
        m = np.array(sx).ravel().tolist()
        m.insert(0, 0)
        m.insert(len(sx), 0)
        
        ## add calculated column 'm' to user specified control points 
        ## from [x, y] to [x, y, m] and store in 'ctrl_pts'
        ctrl_pts = np.insert(ctrl_pts, 2, m, axis = 1)
    
    ## store phi relevance parameter dictionary
    phi_params = {}
    phi_params["method"] = "manual"
    phi_params["num_pts"] = np.size(ctrl_pts, axis = 0)
    phi_params["ctrl_pts"] = np.array(ctrl_pts).ravel().tolist()
    
    ## return dictionary
    return phi_params



## calculate the phi relevance function
def phi(
    
    ## arguments / inputs
    y,        ## reponse variable y
    ctrl_pts  ## params from the 'ctrl_pts()' function
    
    ):
    
    """
    generates a monotonic piecewise cubic spline from a sorted list (ascending)
    of the response variable y in order to determine which observations exceed 
    a given threshold ('rel_thres' argument in the main 'smogn()' function)
    
    returns an array of length n (number of observations in the training set) of 
    the phi relevance values corresponding to each observation in y to determine
    whether or not an given observation in y is considered 'normal' or 'rare'
    
    the 'normal' observations get placed into a majority class subset or 'bin' 
    (normal bin) and are under-sampled, while the 'rare' observations get placed 
    into seperate minority class subset (rare bin) where they are over-sampled
    
    the original implementation was as an R foreign function call to C and later 
    adapted to Fortran 90, but was implemented here in Python for the purposes
    of consistency and maintainability
    
    ref:
    
    Branco, P., Ribeiro, R., Torgo, L. (2017). 
    Package 'UBL'. The Comprehensive R Archive Network (CRAN).
    https://cran.r-project.org/web/packages/UBL/UBL.pdf.
    
    Fritsch, F., Carlson, R. (1980).
    Monotone Piecewise Cubic Interpolation.
    SIAM Journal on Numerical Analysis, 17(2):238-246.
    https://doi.org/10.1137/0717021.
    
    Ribeiro, R. (2011). Utility-Based Regression.
    (PhD Dissertation, Dept. Computer Science, 
    Faculty of Sciences, University of Porto).
    """
    
    ## assign variables
    y = y                            ## reponse variable y
    n = len(y)                       ## number of points in y
    num_pts = ctrl_pts["num_pts"]    ## number of control points
    ctrl_pts = ctrl_pts["ctrl_pts"]  ## control points
    
    ## reindex y
    # y = y.reset_index(drop = True)
    
    ## initialize phi relevance function
    y_phi = phi_init(y, n, num_pts, ctrl_pts)
    
    ## return phi values
    return y_phi

## pre-process control points and calculate phi values
def phi_init(y, n, num_pts, ctrl_pts):
    
    ## construct control point arrays
    x = []
    y_rel = []
    m = []
    
    for i in range(num_pts):
        x.append(ctrl_pts[3 * i])
        y_rel.append(ctrl_pts[3 * i + 1])
        m.append(ctrl_pts[3 * i + 2])
    
    ## calculate auxilary coefficients for 'pchip_slope_mono_fc()'
    h = []
    delta = []
    
    for i in range(num_pts - 1):
        h.append(x[i + 1] - x[i])
        delta.append((y_rel[i + 1] - y_rel[i]) / h[i])
    
    ## conduct monotone piecewise cubic interpolation
    m_adj = pchip_slope_mono_fc(m, delta, num_pts)
    
    ## assign variables for 'pchip_val()'
    a = y_rel
    b = m_adj
    
    ## calculate auxilary coefficients for 'pchip_val()'
    c = []
    d = []
    
    for i in range(num_pts - 1):
        c.append((3 * delta[i] - 2 * m_adj[i] - m_adj[i + 1]) / h[i])
        d.append((m_adj[i] - 2 * delta[i] + m_adj[i + 1]) / (h[i] * h[i]))
    
    ## calculate phi values
    y_phi = [None] * n
    
    for i in range(n):
        y_phi[i] = pchip_val(y[i], x, a, b, c, d, num_pts)
    
    ## return phi values to the higher function 'phi()'
    return y_phi

## calculate slopes for shape preserving hermite cubic polynomials
def pchip_slope_mono_fc(m, delta, num_pts):
    
    for k in range(num_pts - 1):
        sk = delta[k]
        k1 = k + 1
        
        if abs(sk) == 0:
            m[k] = m[k1] = 0
        
        else:
            alpha = m[k] / sk
            beta = m[k1] / sk
            
            if abs(m[k]) != 0 and alpha < 0:
                m[k] = -m[k]
                alpha = m[k] / sk
            
            if abs(m[k1]) != 0 and beta < 0:
                m[k1] = -m[k1]
                beta = m[k1] / sk
            
            ## pre-process for monotoncity check
            m_2ab3 = 2 * alpha + beta - 3
            m_a2b3 = alpha + 2 * beta - 3
            
            ## check for monotoncity
            if m_2ab3 > 0 and m_a2b3 > 0 and alpha * (
                m_2ab3 + m_a2b3) < (m_2ab3 * m_2ab3):
                
                ## fix slopes if outside of monotoncity
                taus = 3 * sk / np.sqrt(alpha * alpha + beta * beta)
                m[k] = taus * alpha
                m[k1] = taus * beta
    
    ## return adjusted slopes m
    return m

## calculate phi values based on monotone piecewise cubic interpolation
def pchip_val(y, x, a, b, c, d, num_pts):
    
    ## find interval that contains or is nearest to y
    i = bs.bisect(
        
        a = x,  ## array of relevance values
        x = y   ## single observation in y
        ) - 1   ## minus 1 to match index position
    
    ## calculate phi values
    if i == num_pts - 1:
        y_val = a[i] + b[i] * (y - x[i])
    
    elif i < 0:
        y_val = 1
    
    else:
        s = y - x[i]
        y_val = a[i] + s * (b[i] + s * (c[i] + s * d[i]))
    
    ## return phi values to the higher function 'phi_init()'
    return y_val