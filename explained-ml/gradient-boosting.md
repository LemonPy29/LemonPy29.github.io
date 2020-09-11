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

In order to understand the intuition behind gradient boosting, first we'll take a look at our familiar algorithm gradient descent. Given a loss function \\(L\\) and set of the learnable parameters \\(\Theta=(\theta_1, \ldots, \theta_n)\\),  we try to push \\(L\\) to its local minima by substracting to each parameter a fraction of its gradient
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
<br>

</br>

This is a famous one on kaggle: *The titanic dataset*. I will not go into the details, the only thing you need to now about it is that contains information (features) of the passengers of the Titanic and the aim is to predict if the passenger survived or not. In short, it's a binary classification problem. 

```python
# modified version of the classic titanic data set from kaggle
# download at https://www.kaggle.com/heptapod/titanic
path = 'titanic_mod.csv'
data = pd.read_csv(path)
```

Suppose you start with a base model that may or not have learnable parameters. For the purpose of this example, suppose also the model can either be trained or used to predict, but you're not allowed to make any changes on the internals of it and maybe you don't even know how the model works. As I pointed in the introduction, let's use trees.

First prepare and split the data

```python
def split_data(data, target, drop, test_size=0.2, seed=seed):
    # short method to prepare and split
    data_wo_nan = data.dropna()
    flat_target = data_wo_nan[target].values.ravel()
    return train_test_split(data_wo_nan.drop(target + drop, axis=1),
                            flat_target,
                            test_size=test_size,
                            random_state=seed)

drop = ['Passengerid'] 
target = ['Survived']
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

The model is trained, but you wonder if you can improve its results under the above constrains. Why not try gradient descent? Is it possible? 

The first step to do it so, is to pick a loss to minimize. If you have never seen or study gradient boosting, it could be kind of confusing because in gradient descent you optimize the loss modifying a single model. Here, as we stated, the model can't be changed. Don't worry though, things will make sense as we go forward. 

We'll choose the known binary cross-entropy or deviance loss

```python
def binary_loss():
    def func(y, p): 
        return -2.0 * np.mean(y * np.log(p/(1-p)) - np.logaddexp(0.0, np.log(p/(1-p))))
    def gradient(y, p):
        return (p-y)/(p*(1-p))
    func.gradient = gradient
    return func
```

Taking a closer look to the equation `(1.1)`, if we want to compute those gradients, by the chain rule, first we need to compute the gradients respect to \\(\hat{y}\\) 
\\[
\dfrac{\partial L(y, \hat{y})}{\partial \theta_j} = \dfrac{\partial L(y,\hat{y})}{\partial{\hat{y}}}\dfrac{\partial\hat{y}}{\partial\theta_j} \tag{1.2}
\\]
I pointed that to emphasize that we can differentiate the loss respect to the predictions and maybe we can try to lower the loss using a kind of gradient descent as `(1.3)`
\\[
\hat{y}(x_i) := \hat{y}(x_i) -\nu\dfrac{\partial L(y,\hat{y})}{\partial\hat{y}} \bigg\rvert _{\hat{y}=\hat{y}(x_i)}  \tag{1.3}
\\]
Here we're taking each train example (for \\(i=1,\ldots,m\\) if you have \\(m\\) data points),  predicting, computing the gradients respect to the predictions and updating the prediction according to the same gradient descent rule. Note that the spirit remains the same: push the loss to its minimum by changing the variables to go in the opposite direction of the gradients.  

Although this should boost the performance, we still have to deal with a big detail in `(1.3)`. To compute the gradients, the true label of each data point is needed. At train time, sure, we have them, but at inference time we don't and they probability don't exist. Don't worry though, there is a solution to this issue, but before we dive into that, let me give you a hint

\\[
\hat{y} = \hat{y} + \nu \left(-\dfrac{\partial L(y,\hat{y})}{\partial\hat{y}}\right) \rightsquigarrow \hat{y} = \hat{y} + \nu\cdot g(x) \tag{1.4}
\\]
\\[
g(x) \sim -\dfrac{\partial L(y,\hat{y})}{\partial\hat{y}} 
\\]

(From now on, I'm not going to add the data point sub-index) Equation `(1.4)` is indicating us to replace the gradients for something similar that only depends on the data at hand and not on the labels. Take a moment to think about it. It's just another machine learning problem: find a function or map from the data to a target, but this time the target is the gradient vector. Considering this, a natural solution is to fit a new model on the gradients and then replace them with the predictions.

Indeed, here's the gist of g.b. **Instead of modifying the same model parameters, we train another model on the loss gradients and procced to update the predictions (not parameters),  by adding to them the new model predictions of minus the gradients.**

## Fitting the gradients

Before we start coding let me show you just one step of the algorithm in equations. Let's define

\\[
\hat{y} = \hat{y}^{(0)} = f^{(0)}(x) \hspace{.2cm};\hspace{.2cm} r^{(0)} = -\dfrac{\partial L(y,\hat{y})}{\partial\hat{y}} \bigg\rvert _{\hat{y}=\hat{y}^{(0)}}  \tag{1.5}
\\]

Here \\(f^{(0)}\\) is our base learner. As stated before, fit a new model \\(f^{(1)}\\) on \\(r^{(0)}\\). The components of \\(r^{(0)}\\) are called the pseudo-residuals. Update the predictions \\(\hat{y}\\) as follows

\\[
    \hat{y} = \hat{y} + \nu\cdot \hat{y}^{(1)} \tag{1.6}
\\]

where \\(y^{(1)} = f^{(1)}(r^{(0)})\\). We'll repeat this process until some stopping criterion is hit or for simplicity, as in this example, just harcode the number of iterations (`boosting rounds`). 

Let's see some code. Start with a base class and define methods related to the base learner
```python
class gradient_booster:
    def __init__(self, loss, lr, **tree_config):
        self.lr = lr
        self.loss = loss 
        self.learners = [] 
        self.tree_config = tree_config
    
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    def _fit_base(self, X, y):
        base_learner = DecisionTreeClassifier(**self.tree_config, random_state=seed)
        base_learner.fit(X, y)
        self.learners.append(base_learner)
    
    def _predict_base(self, X):
        return self.learners[0].predict_proba(X)[:,1]
```
And finally, the boosting

```python
#inside gb class
    def fit(self, X, y, boosting_rounds):
            self.loss_history = []
            self._fit_base(X, y)
            prbs = self._predict_base(X)
            predictions = prbs

            for _ in range(boosting_rounds):
                target = -self.loss.gradient(y, prbs)
                current_model = DecisionTreeRegressor(**self.tree_config, random_state=seed) 
                current_model.fit(X, target)
                self.learners.append(current_model)
                predictions += self.lr * current_model.predict(X)
                prbs = self.sigmoid(predictions) 
                self.loss_history.append(loss(y, prbs))
```
A couple of comments:
* In this particular case the loss has a restricted domain, \\((0,1)\\). In theory, the residuals could take any real value so a sigmoid function is applied before passing the residuals to the loss. 
* As we are dealing with a classification problem, the base learner must be a classification algorithm. In the other hand, the residuals are a continuous target, therefore, the next learners are all regressors. 

The prediction method follows the same pattern
```python
#inside gb class
    def predict_proba(self, X):
            predictions = self._predict_base(X)
            for m in self.learners[1:]:
                predictions += self.lr * m.predict(X)
            return self.sigmoid(predictions)
```
Let's use our boosting algorithm
```python
booster = gradient_booster(loss=binary_loss(), lr=0.01, max_depth=3)
booster.fit(X_train, y_train, 50)

fig, ax = plt.subplots(1,1)
ax.plot(booster.loss_history)
ax.set_xlabel('Boosting Rounds')
ax.set_ylabel('Binary Loss')
```

<figure>  
   <img src="img/gb-loss.png"/>
   <figcaption>
       <b>Fig 1.</b> Loss function along boosting rounds
    </figcaption>
</figure>



Good, the loss is indeed decresing. Although for any real world validation you should look at the test loss, the graph let us know things are working as expected. What about the score?

```python
y_prob = booster.predict_proba(X_test)
y_pred = 1 * (y_prob>0.5)
print(f"f1 score: {f1_score(y_test,y_pred):.2f}")
```
`f1 score: 0.71`

## Final words

Although modern API's, as those mentioned in the introduction, are much more complex, I hope this article gives you a good understanding of what it is one of the most used machine learning techniques nowadays, but more than that I hope it gives a better intuition   


[^1]: [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) 
[^2]:  [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)


