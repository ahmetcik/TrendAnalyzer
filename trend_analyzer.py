import numpy as np
from sklearn.linear_model import LinearRegression


class GaussianRegression(object):
    """Class that determines a model based on a linear combination of Gaussians."""
    def __init__(self, n_gauss=5):
        self.n_gauss = n_gauss
        
    def get_sigma(self, X):
        # get maximum range in descriptor space
        range_max = (X.max(0) - X.min(0)).max()
        
        # Propose sigma of Gauss such that 2 sigma fit n_div into range.
        sigma = range_max / (2 * self.n_gauss)
        return sigma     
    
    def get_gaussian_kernel(self, X1, X2, sigma=1.):
        X_cart_diff = X1[:, np.newaxis, :] - X2
        return np.exp(-0.5 * np.linalg.norm(X_cart_diff, axis=2)**2 / sigma**2)
        
    def fit(self, X, y, **kwargs):

        # standardize
        self.mean = X.mean(0)
        self.std = X.std(0)
        X = (X - self.mean) / self.std 
        
        # Determine grid for placing Gaussians
        X_min, X_max = X.min(0), X.max(0)
        self.X_grid = np.linspace(X_min, X_max, self.n_gauss)
        
        # determine sigma of Gaussians such that 2 sigma fit n_gauss-1 into range
        range_max = (X_max - X_min).max()
        
        # determine sigma from self.n_gauss and ranges in descriptor space
        self.sigma = 0.5 * range_max * (self.n_gauss - 1)
        
        # determin Gaussian kernel matrix (non-square)
        K = self.get_gaussian_kernel(X, self.X_grid, sigma=self.sigma)
        
        # perform linear regression
        self.reg = LinearRegression()
        self.reg.fit(K, y)        
        
    def predict(self, X):
        X = (X - self.mean) / self.std 
        K = self.get_gaussian_kernel(X, self.X_grid, sigma=self.sigma)
        return self.reg.predict(K)

class TrendCluster(object):
    """A clustering algorithm that clusters time series based on
    local linear coefficients. Whenever a coefficient changes
    and this change is noticable comapred to others, a new 
    cluster/trend starts. 
    
    Iterate over every data point. At each data point, linear 
    regression is performed using the data point and previous
    l-1 data points. The outcoming model coefficient is assigned 
    to the data point of the iteration.
    
    Parameters
    ----------
    epsilon = float, between 0 and 1
        Only data points with coefficients that are separated from
        other coefficients by a margin that is larger than epsilon*c_range 
        are considered as separator of clusters. c_range is given by 
        difference between max and min coefficient among all data points.
    
    l: int
        Number of data points used for linear fit. 
    
    n_seed: int
        Number of grid points to search for maximum coefficient margin.
        
    min_cluster: float,
        Percentage of number of samples which must be fulfilled by 
        a cluster to be considered as cluster, i.e. small clusters
        are ignored and get assined nan as label.
    """
    
    def __init__(self, epsilon=0.1, l=5, n_seed=1000, min_cluster=None):
        self.epsilon = epsilon
        self.l = l
        self.n_seed = n_seed
        self.min_cluster = min_cluster
        self.is_set_min_cluster = False

    def set_min_cluster(self, n_samples):
        """Init min_cluster."""
        
        if self.is_set_min_cluster:
            return
        
        if self.min_cluster is None:
            self.min_cluster = int(n_samples * 0.1)
        elif not isinstance(self.min_cluster, (float, int)):
            raise TypeError("min_cluster needs to be int or float.")
        elif self.min_cluster < 0. or self.min_cluster >= 1.:
            raise ValueError("min_cluster needs to be between 0 and 1.")
        else:
            self.min_cluster = int(n_samples * self.min_cluster)
        
        self.is_set_min_cluster = True
          
    def get_maximum_seprator(self, x):
        """"""
        x_min, x_max = x.min(), x.max()
        x_lin = np.linspace(x_min, x_max, self.n_seed+2)[1:-1]
        margins = []

        for xi in x_lin:
            x_lower = x[x<xi].max()
            x_upper = x[x>xi].min()
            margin = x_upper - x_lower
            margins.append(margin)

        i_max = np.argmax(margins)
        margin_max = margins[i_max]
        if margin_max < self.epsilon * (x_max - x_min):
            return None
        
        x_max = x_lin[i_max]
        return x_max

    def get_isolations(self, x,):
        n_samples = x.size
        
        # init
        y = np.zeros(n_samples)
        
        x_max_separator = self.get_maximum_seprator(x)
        if x_max_separator is None:
            return y
        
        mask_lower = x < x_max_separator
        mask_upper = ~mask_lower
        
        if mask_lower.sum() < n_samples / 2:
            mask_isolation = mask_lower
        else:
            mask_isolation = ~mask_lower

        
        y[mask_isolation] = 1.
        return y

    def get_trend_change(self, x):
        y = self.get_isolations(x)
        diff = y[1:] - y[:-1]
        return np.where(diff > 0)[0]
        
    def fit(self, x, y):
        self.set_min_cluster(y.size)
        
        y = (y-y.mean()) / y.std()
        n = len(x)
        X = np.zeros((n, self.l))
        
        for i in range(self.l, n+1):
            x2 = x[i - self.l: i]
            y2 = y[i - self.l: i]
            x2 = np.transpose([x2, np.ones(self.l)])
            c = np.linalg.lstsq(x2, y2, rcond=-1)[0][0]
            
            i_x = range(i - self.l, i)
            i_y = list(reversed(range(self.l)))

            X[i_x, i_y] = c
        
        # Currently, only 1-d implemented. Use first dimension, ignore rest
        c = X[:, 0]
        indices_tend_change = self.get_trend_change(c)
        indices =  np.append(indices_tend_change + 1, n)
        self.labels_ = np.zeros(n)
        i_label = 1
        for i in range(len(indices) - 1):
            i1, i2 = indices[i], indices[i+1]
            if i2 - i1 >= self.min_cluster:
                self.labels_[i1: i2] = i_label
                i_label += 1
            else:
                self.labels_[i1: i2] = np.nan
        return X    
    
class TrendAnalyzer(object):
    """Algorithm that assigns high score, i.e. 1, to a time series if
    there is somehwere locally a trend and low score, i.e. 0, if there 
    is nowhere a trend. The algorithm first clusters the series. Then,
    it assigns to every cluster a trend score. This score is determied
    via evaluating how good a model fits the data of the cluster compared
    to the variance in the cluster, i.e. 1 - (mean squared error) / variance.
    The default regressor GaussianRegression fits a linear combination of
    Gaussians. Choosing a small number of Gaussians will give smoother/
    less wiggly models. The maximum score over all clusters determines the 
    overall score of the series.
    """
    
    def __init__(self, cluster_algorithm=TrendCluster, regressor=GaussianRegression):
        
        self.cluster_algorithm = cluster_algorithm
        self.regressor = regressor
    
    def fit(self, x, y):
        
        #TODO: pass kwargs
        reg = self.regressor()
        cl = self.cluster_algorithm()
        
        # get clusters
        cl.fit(x, y)
        self.cluster_labels = cl.labels_
        
        # init
        self.y_pred = np.empty(y.size)
        self.y_pred[:] = np.nan
        self.trend_measures = []
        
        
        for label in np.unique(self.cluster_labels):
            if np.isnan(label):
                continue
            mask = self.cluster_labels == label
            X_cluster = x[mask].reshape(-1, 1)
            y_cluster = y[mask]
            
            reg.fit(X_cluster, y_cluster)
            y_cluster_pred = reg.predict(X_cluster)
            self.y_pred[mask] = y_cluster_pred
            
            # calc mean squarred error and standard deviation
            mse = np.linalg.norm(y_cluster_pred - y_cluster)**2 / y_cluster.size
            var = y_cluster.var()
            
            # Determine trend measure as 1 - mse / var.
            # If measure is 1, this means there is a significant trend
            # If measure is 0, this means there is no trend.
            # Note that measure can become negative, which means also no trend
            trend_measure = 1 - mse / var
            self.trend_measures.append(trend_measure)
        
        trend_measure_max = max(self.trend_measures) 
        return trend_measure_max

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n_split = 5
    n_samples = 1000
    x = np.linspace(0, n_split, n_samples)
    xs = np.split(x, n_split)
    y = np.concatenate([np.sin(xs[i] + i+2) for i in range(5)])
    y *= np.sin(x*2)
    y += 0.1 * (np.random.rand(n_samples) - 0.5)

    ta = TrendAnalyzer()
    score = ta.fit(x, y)

    print(score)
    # 0.996

    labels = [label for label in np.unique(ta.cluster_labels) if not np.isnan(label)]
    for label in labels:
        mask = ta.cluster_labels == label
        plt.scatter(x[mask], y[mask])
        plt.plot(x[mask], ta.y_pred[mask], 'k')   

