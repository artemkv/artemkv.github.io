# Unsupervised Learning: Dimensionality Reduction
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## References

- [Stanford CS229: Machine Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)


## Dimensionality reduction

- You have multiple features, and you would like to condense it down to a fewer number of features, but retain all the useful information
- This can be useful for visualization, since you cannot visualize more than 3 dimensions
- This can be useful before applying some other ML algorithms that don't work well with too many features (or to speed it up)
- The method is based on the assumption that multiple directly measured features may in fact represent a small number of latent features + noise. So if we could calculate the values for those latent features, we could discard the rest without losing any information
- **Latent features** are features that aren't explicitly in your dataset
- Intuition: latent features are omnipresent in psychological tests, where features like "I get tired of meeting new people" and "I recharge my energy by being on my own" are actually related to just one latent feature "I am an introvert"
- One way to condense the dataset is to simply select only some initial features hoping that they represent the latent feature sufficiently. This, however, means discarding a lot of useful information
- Another way is to somehow combine the feature values to obtain a single value of a latent feature


## Principal component analysis (PCA)

- Converts a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called **principal components**
- **Principal components** are linear combinations of the original features in a dataset that aim to retain the most information in the original data
- Principal components are basically calculated values for latent features
- Principal components have to capture the largest amount of variance left in the data (minimize the information loss)
- Principal components have to be orthogonal
- If there are too many features and PCA is too slow, you could use random projections instead. While PCA tries to optimize the principal components, random projections just picks some random axes and projects the features on these. On a dataset with huge number of features this actually works quite well
- PCA requires data to be preprocessed: zero-centered and have the same scale
- PCA is useful for visualization (but consider using it together with T-SNE)
- PCA also allows getting rid of the noise in the data and to reduce amount of storage needed to store it
- PCA can be used to reduce number of features and combat overfitting, but this method should not be overused
- When applied to a dataset with thousands of features, 50 is a typical number of features to which you would try to reduce the original dataset
- You might want to try PCA when you want to find "a similar datapoints", for example, to find a photo similar to your photo (maybe to find similar faces). Points that are pretty far in multiple dimensions may get much closer in reduced dimensions
- In particular, when working with text, using one-hot encoding results in a huge number of dimensions. PCA can help to significantly reduce the number of features in this case. We also hope that words like "study" and "learn" that are normally 2 orthogonal dimensions may reduce to just one, revealing some latent variable like "words about education"
- This will allow detecting documents having "learn" and "study" as similar (As a side note, document similarity is roughly a cos of an angle between 2 feature vectors)

### Intuitive geometrical explanation

- If 2 dimensions represent a single latent feature, they will be highly correlated, except for the noise
- As an extreme example, think about the distances between cities in miles and km, measured using some inaccurate method, like by counting steps
- The stronger the correlation, the more will the data be stretched along a line approaching 45 degrees (perfect correlation is, in fact, a line at 45-degree angle)
- Now imagine projecting this 2-dimensional set of samples onto that line. The distances between point projections will perfectly represent the original distances between points in 2d space
- We can keep that line as a new (1-dimensional) axis and drop the second dimension completely, considering it noise
- When having more dimensions, all projections have to be orthogonal, otherwise they will be again correlated. Only when projections are orthogonal, the distance between points along one of the axes has zero influence on the distance along any other axes
- In practice, you can rotate original axes until one of them matches the abovementioned line
- Every time you rotate the axes, re-calculate every column as a projection to the new set of axes
- Calculate new variance for every column and sort columns by variance, high to low
- Eventually you may discover that first `N'` column will account for 95% (98%, 99%) of the total variance. All the remaining columns can be considered noise and dropped
- For the purpose of visualization, you simply keep the first 2 or 3 with most variance and forcefully drop the rest

```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([('pca', PCA()),
                 ('tree', DecisionTreeClassifier())])
pipe.fit(iris.data, iris.target)
pipe.predict(newdata)
```


## Independent component analysis (ICA)

- ICA allows separating data coming from different sources that was mixed together
- The classical definition is a "cocktail party problem": we record `n` speakers speaking simultaneously using `n` different microphones placed in the room in different places
- Because each microphone is a different distance from each of the speakers, it records a different combination of the speakers' voices
- More formally, every microphone is recording a linear combination of signals coming from all speakers. Meaning, at any given time, `x = As`, where:
- `x` is an `n`-dimensional vector with the acoustic readings recorded by each of `n` microphones
- `s` is an `n`-dimensional vector with the sounds uttered by each of `n` speakers
- `A` is an unknown **mixing matrix**
- Our goal is to find `W`, an **unmixing matrix**, or "inverse matrix", so that given our microphone recordings `x`, we could recover the source sound by computing `s = Wx`
- The most tricky part is to derive an expression for likelihood
- You pick a distribution on `s(i)` in a form of CDF, sigma works well for a human voice (as a default choice)
- Since all the speakers are independent, the distribution for vector `s` is a product of individual distributions of `s(i)`
- You then derive the CDF for `x`, assuming `x = As` and `s = Wx`
- Once it's done, `W` can be easily found by maximizing log likelihood using the (stochastic) gradient ascent
- ICA doesn't work on Gaussian data, but as long as the sources are non-Gaussian, there are only 3 sources of ambiguity:
- 1). The order of recovered sources as produced by ICA is not guaranteed
- 2). The sign of recovered signals is not guaranteed. For the sound, it doesn't matter
- 3). The scaling factor of recovered signals is not guaranteed. For the sound, it means the result can be louder or quieter
- The reason it doesn't work for Gaussian is that Gaussian is rotationally symmetric, which would introduce yet another source of ambiguity: you could rotate original data to any angle and this would make it impossible to recover


## T-SNE

- t-SNE is a tool to visualize high-dimensional data
- t-SNE is non-linear probability-based technique
- Simply put, similar to PCA, but tries to keep objects that were distant distant by sacrificing linearity of transformation
- Meaning with PCA you may lose a dimension, but still can, to some degree, imagine how the data looked in higher dimension. With t-SNE it's not possible at all, because it bends the space
- But when plotted, gives much better separation between points comparing to PCA
- On the other hand, unlike PCA, you cannot use t-SNE for classification, at least directly, because you cannot transform a new, unknown data point to a t-SNE embedding (With PCA you do "`pca.fit_transform`" on training set and "`pca.transform`" on test set)
- There are indirect methods. For example, you could train some other model to predict the t-SNE embedding based on original data, but it is not the purpose of t-SNE
- t-SNE is very slow on >50 dimensions
- So it is highly recommended using another dimensionality reduction method like PCA to reduce the number of dimensions to a reasonable amount (e.g. 50)
