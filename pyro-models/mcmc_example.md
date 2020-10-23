---
layout: page
title: Preliminaries
nav: false
---
<link rel="stylesheet" href="/assets/css/main.css"/>

# A minimal MCMC example

Many times one can estimate a statistic from a given distribution $X$, drawing
random i.i.d samples from it. For example, the mean

\\[
\frac{1}{n}\sum_i f(X_i) \sim \textbf{E}(f(X))
\\]

The law of the large numbers, ensure we'll get a decent estimation if enough
samples are drawn. This method works well until we can no longer sample from
the distribution. This can sound a bit harsh, but it's a pretty common
situation in bayesian inference when we try to compute the posterior
distribution. Even in simple situations the posterior can't be calculated.
Consider the model

```python
import pyro
import pyro.distributions as dist

def model(): mu = pyro.sample("mu", dist.HalfCauchy(scale=1)) 
    return pyro.sample("obs", dist.Normal(mu, 1)) 
```

Imagine we condition this model on some data. We can always compute the bayes
numerator, our problem is the evidence


\\[
\textbf{P}(X) = \int_0^{\infty}
\frac{1}{1+\mu^2}\exp\left(-\frac{1}{2}\sum_{i=1}^{n} (mu - x_i)^2\right)
\\]

Which is an intractable integral. This is just to estimate one parameter, so we
can imagine what could happen when there is a bunch of parameters. 

### What is MCMC?

Markov Chain Monte Carlo is a group of algorithms which allow us to approximate
distributions when you can't directly sample from them but you can evaluate its
density up to a constant. The most common one is the Hastie-Metropolis
algorithm. 

We have proposal distribution $q(x,y)$ from which we'll draw our next candidate

1. Suppose we are at $X_n = x$, so we sample $y$ from $q(x, y)$
2. Compute 
\\[
\alpha(x,y) = min{1, \frac{q(y,x)pi(y)}{q(x,y)pi(x)}}
\\]
