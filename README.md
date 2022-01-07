# shrinkage-estimation-of-large-dimensional-covariance-matrices
Implementation of Ledoit and Wolf (2003)

Sept. 20, 2009

Abstract

One of the most important considerations in a Markowitz mean variance optimization (or for that matter, any portfolio selection model) is the estimation of the covariance matrix. Practitioners typically take recent returns data and calculate covariances from this sample. Unfortunately, sample covariance data will produce a positive definite matrix but the actual (unobservable) matrix may indeed not be positive definite (hence, not invertible). From the Glivenko-Cantelli theorem, the empirical distribution of a set of IID variables tends to the true distribution as observations go to infinity (i.e. maximum likelihood in asymptotic statistics). For short sampling periods however, the benchmark estimators have high inefficiency although they exhibit low bias (close to the true, unobservable value in the location-dispersion framework). Stein showed weighted averages of the sample with a constant estimator (which has very high bias but no inefficiency) can outperform the standalone sample estimator. In this paper, three shrinkage estimators are calculated along with the sample covariance matrix alone (without shrinkage) to assess performance.
