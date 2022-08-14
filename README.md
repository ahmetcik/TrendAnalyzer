# TrendAnalyzer
Algorithms to cluster time series and quantify if some cluster has strong trend.

# Example
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

print(score)
# 0.996

labels = [label for label in np.unique(ta.cluster_labels) if not np.isnan(label)]
for label in labels:
    mask = ta.cluster_labels == label
    plt.scatter(x[mask], y[mask])
    plt.plot(x[mask], ta.y_pred[mask], 'k')
plt.show()
```

<img src="https://github.com/ahmetcik/TrendAnalyzer/blob/main/Example.png" width="60%">
