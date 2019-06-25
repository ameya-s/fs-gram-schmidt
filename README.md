fs-gram-schmidt 
===============================
Feature Ranking and Selection Using Gram Schmidt Orthogonalisation  

**fs-gram-schmidt** is an open-source feature selection algorithm in Python. It is built upon scientific computing packages Numpy and Scipy.
It is based on the paper [**Ranking a Random Feature for Variable and Feature Selection** (Stoppiglia et al. 2003)](http://www.jmlr.org/papers/v3/stoppiglia03a.html) published in Journal of Machine Learning Research


## Installing fs-gram-schmidt 
### Prerequisites:
Python 2.7 *or Python 3*

Pandas

NumPy

SciPy

### Steps:
After you download/clone fs-gram-schmidt, 

For Linux users, you can install the repository by the following command:

    python setup.py install

For Windows users, you can also install the repository by the following command:

    setup.py install


## Examples

```
from fs-gram-schmidt.function import GSO

# feature_df : n*m pandas dataframe (preprocessed : should be scaled and works best when outliers are removed)
# target : n*1 pandas series of target values corresponding to feature_df
# risk : Predefined threshold of risk (0 < risk < 1)

ranked_features, feature_selection_risk = GSO.rank_features(train_X, target, risk=0.05)

# ranked_features : List of ranked features ordered by their relevancy(descending) and selection risk (ascending)
# feature_selection_risk : Mapping of nth feature in ranked_features with it's selection risk 
```

## Contact
Ameya Dahale

E-mail: 100ameya@gmail.com
