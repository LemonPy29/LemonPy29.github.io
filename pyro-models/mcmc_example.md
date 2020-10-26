---
layout: page
title: Minimal Understanding of MCMC
nav: false
---
<link rel="stylesheet" href="/assets/css/main.css"/>


Many times one can estimate a statistic from a given distribution \\(X\\),
drawing random i.i.d samples from it. For example, the mean

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

def model(): 
    mu = pyro.sample("mu", dist.HalfCauchy(scale=1)) 
    return pyro.sample("obs", dist.Normal(mu, 1)) 
```

Imagine we condition this model on some data. We can always compute the bayes
numerator, our problem is the evidence


\\[
\textbf{P}(X) = \int_0^{\infty}
\frac{1}{1+\mu^2}\exp\left(-\frac{1}{2}\sum_{i=1}^{n} (\mu - x_i)^2\right)
\\]

Which is an intractable integral. This is just to estimate one parameter, so we
can imagine what could happen when there is a bunch of parameters. 

## What is MCMC?

Markov Chain Monte Carlo is a group of algorithms which allow us to approximate
distributions when you can't directly sample from them but you can evaluate its
density up to a constant. The most common one is the Hastie-Metropolis
algorithm. 

We have proposal distribution \\(q(x,y)\\) from which we'll draw our next
candidate

1. Suppose we are at \\(X_n = x\\), so we sample \\(y\\) from \\(q(x, y)\))
2. Compute 
\\[
\alpha(x,y) = \min\left(1, \frac{q(y,x)\pi(y)}{q(x,y)\pi(x)\right)
\\]
3. Accept the proposal with probability \\(\alpha(x,y)\\). If accepted, set
\\(X_{n+1} = y\\) else \\(X_n=x\\)

Iterating this procedure, we end up with an Markov chain \\(X_n\\) that has
\\(\pi\\) as a stationary distribution. Why? We'll go into the details, but 
first, let us show a toy example which can help us to understand how the 
transition matrix looks like.

## Toy example

Let's compute the entries of the transition matrix. If
\\(i != j\\) the probability of jump from \\(i\\) to \\(j\\) can be interpreted
as picking \\(j\\) and the accept it. As those process are independent we have

\\[
p_{ij}=\textbf{P}[X_{n+1}=i|X_n=j]=q(i,j)\alpha(i,j)
\\]

The whole matrix can be constructed with this code

```python
def compute_transition_matrix(pi, q, size):
    alpha = lambda x, y: \
        min(1, (pi(y) * q(y,x)) / (pi(x) * q(x,y))) if x != y else 0
    fn = lambda x, y: q(x, y) * alpha(x, y)  
    g = np.indices((size, size))
    P = np.vectorize(fn, otypes=[np.float64])(g[0], g[1])
    np.fill_diagonal(P, 1 - P.sum(axis=1))
    return P
```

Suppose our we'll trying to sample from \\(bin(n, p)\\) (state space
\\(S=\{0,\ldots, n\}\\).

```python

class binom_mcmc:
    def __init__(self, n, p):
        self.n = n
	self.dist = binom(n, p)
		  
    def _pi(self, *args):
	return self.dist.pmf(*args)
				       
    @property
    def pi(self):
	return self._pi(range(self.n + 1))
	     
    def transition_matrix(self, q=None):
	if not q: q = self.unif_q
        return compute_transition_matrix(self._pi, q, self.n + 1)
		     
    def unif_q(self, *args):
	return 1 / (self.n + 1)
```

Let's verify if \\(\pi\\) is a stationary distribution 

```python
bm = binom_mcmc(3, .4)

v = bm.transition_matrix()
assert np.allclose(v @ P, v)

```
We can check with a different proposal

```python
def binom_q(value, *args):
    return binom(3, .7).pmf(value)

Q = bm.transition_matrix(binom_q)
assert np.allclose(v @ Q, v)

```
## Why the chain converge?


