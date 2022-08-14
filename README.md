# TrendAnalyzer
Algorithms to cluster time series and quantify if some cluster has a strong trend.

The used classes are TrendCluster and TrendAnalyzer.


TrendCluster is a clustering algorithm that clusters time series based on local linear coefficients. Whenever a coefficient changes and this change is noticable compared to other changes in the data set, a new cluster/trend starts. The algorithm iterates over every data point. At each data point, linear regression is performed using the data point and previous l-1 data points. The outcoming model coefficient is assigned to the data point of the iteration.

TrendAnalyzer is an algorithm that assigns high score, i.e. 1, to a time series if there is somehwere locally a trend and low score, i.e. 0, if there is nowhere a trend. The algorithm first clusters the series (default TrendCluster). Then, it assigns to every cluster a trend score. This score is determied via evaluating how good a model fits the data of the cluster compared to the variance in the cluster, i.e. 1 - (mean squared error) / variance. The default regressor GaussianRegression fits a linear combination of Gaussians. Choosing a small number of Gaussians will give smoother/less wiggly models. The maximum score over all clusters determines the overall score of the series.
    
The module as well as the method needs to be further developed. The method has conceptual as well as numerical drawbacks, currently. However, it might provide some starting point or idea for a better algorithm and implementation.

# Example
Generate a series by attaching (non-smoothly) sinus curves to each other. The example demonstrates how clusters are identified and a high score of above 99% for trend character is found.
```py
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

# 
print(score)
# 0.996

labels = [label for label in np.unique(ta.cluster_labels) if not np.isnan(label)]
for label in labels:
    mask = ta.cluster_labels == label
    plt.scatter(x[mask], y[mask])
    plt.plot(x[mask], ta.y_pred[mask], 'k')
plt.show()
```

<img src="https://github.com/ahmetcik/TrendAnalyzer/blob/main/Example.png" width="50%">
