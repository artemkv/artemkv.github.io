# Unsupervised Learning: Clustering

## References

- [Stanford CS229: Machine Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)


## Clustering

- Labeling the data is very costly and time-consuming process, it has to be done manually and requires expertise
- Clustering allows identifying similarities/differences in the input data and automatically bucket the data samples into groups (clusters) based on those similarities/differences
- In essence, we discover hidden categories in the input data
- The scientific way to say "hidden category" is "latent variable": a variable that is not directly observed but rather inferred from other variables that are observed
- Latent features are omnipresent in psychological tests, where different questions are often just the same question stated in different forms
- You still have to "come up with names for those categories". Meaning you have to look at the clusters to figure out the nature of those differences/similarities that were discovered by the algorithm
- When most of your data samples are very similar, clustering can also be helpful to identify the samples that are "unlike" others
- As a practical example, you can analyze multiple brain scans and identify some that don't look like others and so may raise concern
- Another classical example is market segmentation: finding different categories of users
- Clustering can be helpful in combination with supervised learning methods too. When using k-Nearest Neighbors, instead of looking for the closest points in the whole dataset (slow), we could split the data into clusters upfront, then map the given data point to one of the clusters and only look for the closest points in that cluster (faster)
- As with supervised learning, clustering only works well when there are actual clusters in the data. Donut-shaped or x-shaped data is hard to cluster
- Depending on what your data looks like, you might choose an algorithm that works best with that shape, see reference here: https://scikit-learn.org/stable/modules/clustering.html
- Clustering can be very useful for anomaly detection (i.e. identify the samples that don't look like others), which has a great practical application in medicine

### Steps in cluster analysis

- Feature selection/extraction
- Choose a clustering algorithm that is the best applicable
- Choose a proximity measure (which function do you use to measure how close the data points are)
- Validating the clustering
- Interpreting the results (give labels to the clusters etc.)

### Feature selection

- **Feature selection** is for filtering irrelevant or redundant features from your dataset
- **Variance thresholds** remove features whose values don't change much from observation to observation. These features provide little value (gender in case the most of the participants are men)
- **Correlation thresholds** remove features that are highly correlated with others. These features provide redundant information (distance in feet and meters)

### Feature extraction

- Feature extraction is for creating a new, smaller set of features that stills captures most of the useful information
- **Principal component analysis (PCA)** creates linear combinations of the original features

### Cluster validation

- **Compactness** measures how close are the objects within the same cluster
- **Separation** measures how well-separated a cluster is from other clusters
- **Connectivity** reflects the extent to which items that are placed in the same cluster are also considered their nearest neighbors in the data space
- **The silhouette coefficient** - defines compactness based on the pairwise distances between all elements in the cluster, and separation based on pairwise distances between all points in the cluster and all points in the closest other cluster. It ranges from −1 to +1, where a high value indicates that the object is well-matched to its own cluster and poorly matched to neighboring clusters
- **Dunn index** represents the ratio of the smallest distance between observations not in the same cluster to the largest intra-cluster distance. The Dunn Index has a value between zero and infinity, and should be maximized


## Algorithms

### KMeans

- Cluster books together to identify anonymous authors
- Cluster movies together to recommend similar items on Netflix
- Cluster the customers together based on the purchases they made for better ads targeting
- We group datapoints together in `K` clusters based on how close they are to one another
- We start by randomly choosing `K` datapoints (centroids) and assigning every remaining point in the dataset to the closest centroid
- Once we know which points belong to which group, we move the centroids to the center of the group
- Once we moved the centroids, some points might become closer to the centroids from the different group
- So we repeat the process until the centroids stop moving (and it can be proven that KMeans converges)
- Depending on where we start we might end up with different clusters of data as a result (and KMeans can get stuck in the local minimum)
- So to find the optimal arrangement, we should run the algorithm multiple times with different starting points
- Among 2 different groupings, the one that has a smaller average distance to the centroids across all data points is better
- Choosing the right value for the `K` is a mix of art and science, often chosen by hand, depending on the nature and the purpose of the task
- Sometimes we have research that suggest the number of clusters, like 3 types of learners on educational site
- In absence of a research, the common strategy to find the good value for the `K` is an elbow method
- You try increasing values for `K` `(1, 2, 3, 4...)` and every time measure the average distance of the points to the centroids of the corresponding cluster. You plot the chart "`avg. distance by K`" which often looks like an elbow, showing that at some specific value of K the average distance stops decreasing significantly
- Since every decision depends on measuring the average distance, the features have to be scaled. Using larger or smaller units for the features squeezes or stretches the plot in that dimension, which affects the clustering decisions dramatically
- **Normalizing (Max-Min Scaling)** rescales the values into a range of [0,1]. Normalizing the data is sensitive to outliers
- **Standardizing (Z-Score Scaling)** rescales data to have a mean of 0 and a standard deviation of 1. Since the values are unbound, it's not sensitive to outliers
- The weakness of KMeans clustering is that it is always trying to find clusters that are circular (because it is based on the distance to the center). It fails badly when the clusters have complex shapes (o, x, and even ||). It even fails on ellipses
- For categorical data, the distance does not make sense (genre 1 is no closer to genre 2 than to genre 152). Since KMeans is based on Euclidean distance, it cannot be used on categorical data

### Hierarchical clustering

- Start by putting every data point in its own cluster
- For every cluster, find the closest cluster and group them together (but without losing the information about the previous grouping)
- Continue until we end up with only one cluster, the root of the cluster tree
- Cut the upper `n` levels to obtain `K` clusters
- To find out the distance, different algorithms can be used
- **Single Link** looks at 2 points in 2 different clusters that are the closest. It works well if there is a good distance between clusters, otherwise one cluster often "eats" most of the data
- **Complete Link** looks at 2 points in 2 different clusters that are the farthest. So even if clusters almost touch each other in one of the points, we don't let them converge unless they are truly close. This produces more compact clusters, however, we are still looking at only 2 points
- **Average Link** uses the average distance between points of two clusters
- **Wards Method** uses variance. Variance is a sum of squares of distances to the central point. Wards method first finds the central point between 2 clusters and calculates the variance in respect of that point using points in both clusters. Then it subtracts the variances of both clusters in respect to their respective central points
- You have to choose the algorithm that works best given the shape of your data. You can experiment or look at the visual comparison (https://scikit-learn.org/stable/modules/clustering.html#different-linkage-type-ward-complete-average-and-single-linkage)
- Dendrograms help to visualize the tree of hierarchical clusters, which is super-useful in case the data is highly dimensional and the clustering is not immediately obvious by just looking at the plot
- The weakness of the hierarchical clustering is the sensitivity to noise and outliers, plus it is very computationally intense

### DBSCAN Clustering

- Density-based clustering groups the points that are densely packed together, and leaves out all the rest, considering it a noise
- DBSCAN analyzes every data point. For each point, it looks if there are other data points at a given distance from the starting point. If it manages to find the minimum required number of points, it approves them as a cluster. The current data point is considered a **core point**
- The point that does not have a minimum required data points at a given minimum distance, but is included in the cluster thanks to being close to a core point, is called a **border point**
- DBSCAN does not take the number of clusters as input parameter, which simplifies life a lot
- It is also very flexible to the shapes of the clusters
- It is very good at ignoring the noise
- The weakness of the density-based clustering is that it fails to find the clusters at varying distances. That can be overcome by using HDBSCAN which applies hierarchical clustering technique to the DBSCAN clustering
- Besides clustering, DBSCAN is also useful to detect abnormalities, if you set the parameters to group most of the datapoints in just 1 cluster

### Gaussian Mixture Model

- GMM is an example of Expectation-Maximization (EM) algorithms (an approach for maximum likelihood estimation in the presence of latent variables)
- With GMM, we assume that all the samples are coming from `k` gaussian (normal) distributions that are mixed together
- So every cluster is going to be described using mean and standard deviation
- The probability of the sample to come from any of `k` distributions is also a random variable `Z` that has a multinomial distribution with unknown parameter `fi` (the latent variable)
- More formally, we assume that each sample `x(i)` was generated by randomly first choosing `z(i)` from `{1, . . . , k}`, and then drawing from one of `k` Gaussians depending on `z(i)`
- We are trying to figure out which sample belongs to which of `k` normal distribution, and the parameters those normal distributions
- GMM uses **soft clustering**: every point belongs to every cluster, but with different "level of membership" in each cluster
- Step 1. Start by initializing `k` gaussian distributions (using, for example, KMeans)
- Step 2 (E Step). Soft-cluster the data by calculating a probability of each sample to be coming from each of the distributions (using Bayes rule, i.e. by calculating `P(z(i) = j|X = x(i)`)
- Step 3 (M Step). Re-estimate parameters of distributions based on soft clustering done at step 2 (find new mean and standard deviation, by maximizing the likelihood)
- Step 4. Evaluate the model using log-likelihood `L(X,Y;theta)`
- Repeat from step 2 until we are satisfied
- The model for GMM is very similar to GDA (see Supervised learning), with the key difference that you have the latent variable `z(i)` instead of known label `y(i)`
- GMM converges (but can get stuck in local minimum)
- GMM can be derived as a specific case of generic EM algorithms, using Jensen's inequality
- In EM we are oscillating between calculating the probability of a datapoint to come from a distribution with given parameters (E step) and finding the most likely parameters of distributions assuming the datapoints came from certain distributions (M step)
- Why do we do that? Formally, our end goal is to maximize the joint likelihood of `X` and `Z` with respect of `theta` `[L(X,Y;theta) = P(X=x,X=z;theta)]`, `theta` being parameters of a joint distribution, i.e. all the values of `fi`, `mu` and `sigma`
- Turns out, this is a hard problem to solve directly
- However, this likelihood expression can be massaged to take a form of expectation, which allows us to use Jensen's inequality to find a lower bound function `J = f(E[X])`, that depends on `theta` and `Q = P(z(i) = j|X = x(i))`. Maximizing `J` with respect of `Q` is equal to E step and maximizing `J` with respect of `theta` is equal to M step. Doing this iteratively brings you to the local maximum of the likelihood
- GMM is very sensitive to initial values, and it seems that to pick correct initial values you kind of need to know where the clusters are, or at least have some intuition
- As usual, GMM only works if every cluster contains the data that is indeed normally distributed
- In practice, algorithm seem to be extremely powerful in many areas: extract background on the video (HOW???), extract moving parts on the video (HOW???), recognizing the fingerprints (HOW???), generating Shakespeare play (HOW???), compose music (HOW???) etc.
- It can also be useful in anomaly detection: when you have a new point, you can estimate how likely it is to see that point in any of the distributions

### Factor Analysis

- GMM is a good model when `m >> n`, where `n` is a number of dimensions and `m` is a number of examples
- However, with `m <=n`, computing the likelihood for the Gaussian density function produces 0/0
- Factor analysis produces a better model when `m ~ n` or even `m << n`
- Factor analysis is a statistical method used to describe variability among observed, correlated variables in terms of a potentially lower number of unobserved variables called factors
- In GMM, the latent random variable `Z` is discrete, one-dimensinal, and a multinomial
- In Factor Analysis, the latent random variable `Z` is continuous, multi-dimensional, and normally distributed
- So `d`-dimensional variable `Z` here represents some `d` forces of nature that drive the value of `X` (or `d` latent variables, or factors)
- `p(x|z)` is modeled as `some mu + lambda*Z + error`
- Error (noise) is normally distributed, independently for each `d`
- Essentially, in this model, we take a multi-dimensional data, map it onto `d` dimensional plane and assume some Gaussian noise
- Just like in GMM, we fit the model parameters to the data using Expectation-Maximization algorithm
- Factor Analysis allows spotting anomalies, data points that are very unlikely under the given distributions
