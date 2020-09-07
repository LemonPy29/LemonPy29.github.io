---
layout: page
title: Understanding Gradient Boosting
nav: false
---
<link rel="stylesheet" href="/assets/css/main.css"/>

## Introduction

If you're inside the machine learning world, it's for sure you have  heard about gradient boosting algorithms such as *xgboost* [^1] or *lightgbm* [^2] .  Indeed, gradient boosting techniques have seen a lot of success in a variety of inference and prediction tasks, particularly when dealing with tabular data. Although, in theory, g.b. can be applied to any black box that make predictions , the only case I've seen it correspond to classification and regressor trees. 

In the next article we'll cover the theory behind g.b and see a working example using the `sickit-learn.tree` to pick our base models. 

## Gradient Boosting and Gradient Descent

In order to understand the intuition behind gradient boosting, first we'll take a look at our familiar algorithm gradient descent. Given a loss function \\(L\\) and set of the learnable parameters \\(\Theta=(\theta_1, \ldots, \theta_n)\\),  we try to push $L$ to its local minima by substracting to each parameter a fraction of its gradient
\\[
\theta_j := \theta_j - \nu\dfrac{\partial L(y, \hat{y}(\Theta, x))}{\partial \theta_j} \tag{1.1}
\\]
Here \\(y\\) represents the ground truth and \\(\hat{y}\\) the prediction that depends on the parameters \\(\Theta\\) and the data (or a data batch) \\(x\\). Nowadays the minus term is much more fancier, but we're interested in this simpler version. The aim of gradient boosting is to minimize a loss function too, but in a slightly different way. 

Before we go deeper let me set up a dataset to work with

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt

seed = 1301
```

This is a famous one on kaggle: *The titanic dataset*. I will not go into the details, the only thing you need to now about it is that contains information (features) of the passengers of the Titanic and the aim is to predict if the passenger survived or not. 

```python
# modified version of the classic titanic data set from kaggle
# more about it here https://www.kaggle.com/heptapod/titanic

url_data = "https://storage.googleapis.com/kagglesdsdata/datasets%2F1275%2F2286%2Ftrain_and_test2.csv?GoogleAccessId=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com&Expires=1599677734&Signature=iYOv1Xp1F6iKrriHIaaG%2BIn3TBaDo2ECAKsCmU%2BQ7GH4RT37IFy3CKOwxH82%2BAAZLIUA4Iq1gITOGIWC9U7U%2BMLjsl8Owo9AsGkmSkx5qgEtHE0GB7j9bhyw6%2FYlS4X7QtsrGdcslfkIqGi47cDgo9Nv6CgDVh80LaobyqzDhnc8YCAEIZFB2Re7ch6KsEbalI0N6bmH%2BstJfPqoPl9zT4wf2UB0pjm9WVSuP6RTVWnS8incJbF5LMUZfmQdC7gcLIX2zI4dvSfpOibYHvvbJzHOL6Rdun6fBUp%2FkJL3Tnfv9LX%2B2%2BzTnASFl1spy5F9cFtvO3wIpz0RVmlBUtQCrg%3D%3D"

data = pd.read_csv(url_data)
```

Suppose you start with a base model that may or not have learnable parameters. For the purpose of this example, suppose also the model can either be trained or used to predict, but you're not allowed to make any changes on the internals of it and maybe you don't even know how the model works. As I pointed in the introduction, let's use trees.

First prepare and split the data

```python
def split_data(data, target, drop, test_size=0.2, seed=seed):
    # short method to prepare and split
    data_drop_nan = data.dropna()
    return train_test_split(data_drop_nan.drop(target + drop, axis=1),
                            data_drop_nan[target],
                            test_size=test_size,
                            random_state=seed)

drop = ['Passengerid'] 
target = ['2urvived']
X_train, X_test, y_train, y_test = split_data(data, target, drop)
```

We train with almost default arguments, just set the `max_depth`, and we'll use the \\(F_1\\) score to measure the performance

```python
base_learner = DecisionTreeClassifier(max_depth=3, random_state=seed)
base_learner.fit(X_train, y_train)
y_pred = base_learner.predict(X_test)

print(f"f1 score: {f1_score(y_test, y_pred):.2f}")
```
`f1 score: 0.63`

So we've trained but obviously wonder if we can improve these results under the rules. Why not try gradient descent? At first glance that could sound a bit silly, where are the parameters?   

Taking a closer look to the equation `(1.1)`, if we want to compute those gradients, by the chain rule, first we need to compute the gradients respect to \\(\hat{y}\\) 
\\[
\dfrac{\partial L(y, \hat{y})}{\partial \theta_j} = \dfrac{\partial L(y,\hat{y})}{\partial{\hat{y}}}\dfrac{\partial\hat{y}}{\partial\theta_j} \tag{1.2}
\\]
I pointed that to emphasize that we can differentiate the loss respect to the predictions and maybe we can try to lower the loss using a kind of gradient descent as `(1.3)`
\\[
\hat{y}(x_i) := \hat{y}(x_i) -\nu\dfrac{\partial L(y,\hat{y})}{\partial\hat{y}} \bigg\rvert _{\hat{y}=\hat{y}(x_i)}  \tag{1.3}
\\]
Here we're taking each train example (for \\(i=1,\ldots,m\\) if you have \\(m\\) data points),  predicting, computing the gradients respect to the predictions and updating the prediction according to the same gradient descent rule.

 Note that the spirit remains the same: push the loss to its minimum by changing the variables to go in the opposite direction of the gradients.  Although this should boost the performance, we still have to deal with a big detail in `(1.3)`. At train time there is no problem at all, we have all the ingredients needed. But what will happen at inference time? Of course the base learner will predict with no problem at all but recall that its predictions were poor. The improvements have been made over the predictions themselves. Can we apply the same technique? Sadly, we can't. The labels are no longer available so the gradients can't be used. 














[^1]: [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) 
[^2]:  [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)


