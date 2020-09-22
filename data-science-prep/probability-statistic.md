---
layout: page
title: Probability and Statistic Questions
nav: false
---
<link rel="stylesheet" href="/assets/css/main.css"/>

### Conditional Expectation [Robinhood]
*Statment*: Say \\( X \\) and \\( Y \\) are independent and uniformly distributed on \\( (0, 1) \\). What is the expected value of \\( X \\), given that \\( X > Y \\)?

Using the definition of conditional expectation:

\\[
\textbf{E}(X|X>Y) = \dfrac{1}{\textbf{P}(X>Y)}\int_{(X>Y)}Xd\textbf{P} 
\\]

First compute the probability \\( \textbf{P}(X>Y) \\). There is two ways: 
  * Intuitively, the fact \\( X, Y\\) are independent and uniform distributed implies
    \\[ 
    \textbf{P}(X>Y) = \textbf{P}(X\leq Y) 
    \\]
    As those sets complement each other, their probability is \\( 1/2 \\)
  * Or analytically
    \\[ 
    \textbf{P}(X>Y) = \int_{0}^{1} \textbf{P}(Y<x)f_X(x) dx = \int_{0}^{1} xdx = \frac{1}{2}
    \\]
    
To compute the integral we condition too
\\[ \int_{(X>Y)}Xd\textbf{P} = \int_{0}^{1} \int_{X>y} X f_Y(y)d\textbf{P} dy = \int_{0}^{1} \int_{y}^{1} x f_X(x) f_Y(y)dxdy  = \frac{1}{3}\\]

So we conclude 
\\[ 
\textbf{E}(X|X>Y) = 2/3 
\\] 
Here is a verification 

```python
from pyro.distributions import Uniform
from torch import Tensor

def expected(samples):
    X = Uniform(0,1)
    Y = Uniform(0,1)
    over = []
    for _ in range(samples):
        x, y = X.sample().item(), Y.sample().item()
        if x > y: over.append(x)
    return Tensor(over).mean()

expected(100000)
```
`0.6667`

### Favorite Show [Disney]

*Statment*: Alice and Bob are choosing their top 3 shows from a list of 50 shows. Assume that they choose independently of one another. Being relatively new to Hulu, assume also that they choose randomly within the 50 shows. What is the expected number of shows they have in common, and what is the probability that they do not have any shows in common?

Suppose \\( X \\) and \\( Y \\) represents the shows choosen by Alice and Bob respectively. Let \\( s = (s_1, s_2, s_3) \\) be a possible show trio. As the shows are pick randomly we have 
\\[
\textbf{P}(X = s) = \textbf{P}(Y = s) = \frac{1}{\alpha} \hspace{.5cm} ; \hspace{.5cm} \alpha = \binom{50}{3}
\\]

Let \\( Z = (X == Y) \\) be the shows in common. If one person has already picked the shows, that is conditioning respecto to \\( X \\) or \\( Y \\), then we're trying to determine how many elements with a special feature can be found in a sample, whitout replacement, of a finite population. That's exactly the description of  a hypergeometric distribution. 

\\[
\textbf{P}(Z = k|X = s) = \frac{1}{\alpha}\binom{50}{k}\binom{47}{3-k}
\\]

So the distribution of \\( Z \\) is given by

\\[
P(Z = k)  = \sum_{s} \textbf{P}(Z = k|X = s) \cdot \textbf{P}(X = s) = \frac{1}\{\alpha} \sum_{s} \textbf{P}(Z = k|X = s)
\\] 

The probability inside the sum doesn't depend on the trio, so

\\[ 
\textbf{P}(Z = k) = \textbf{P}(Z = k| X = s)
\\]

With the distribution at hand you can computed everything that was asked. Here is a code implementation

```python
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import hypergeom

def common_shows(n_samples, n_shows=50):    
    z = []
    shows = [i for i in range(n_shows)]
    for _ in range(n_samples): 
        x = np.random.choice(shows, 3, replace=False)
        y = np.random.choice(shows, 3, replace=False)
        z.append(len(np.intersect1d(x, y)))
    return z
``` 

 Comparison against the reference distribution
 
```python
domain = [0, 1, 2, 3]

# distributions
reference = hypergeom(50, 3, 3)
z = common_shows(5000)

# cdf
reference.cdf(domain)
ecdf = ECDF(z)(domain)

print('Reference': reference_cdf)
print('Empirical': ecdf)
``` 
`Reference: [0.82729592 0.9927551  0.99994898 1.]<br/>  
Empirical: [0.8266 0.9922 1.     1.    ]`
