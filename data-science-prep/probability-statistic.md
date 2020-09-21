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

Suppose \\( X \\) and \\( Y \\) represents the shows choosen by Alice and Bob respectively. Let \\( s = (s_1, s_2, s_3) \\) be the possible show trios. As the order doesn't matter, we have
\\[
\textbf{P}(X = s) = \textbf{P}(Y = s) = \alpha = \frac{1}{\binom{N}{k}}
\\]
