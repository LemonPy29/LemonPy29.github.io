---
layout: page
title: Preliminaries
nav: false
---
<link rel="stylesheet" href="/assets/css/main.css"/>

The first chapter covers a lot of topics

## Random Sums

### Pizza Orders

This is a test 

```python
def pizza_orders(mu, p):
    orders = int(dist.Poisson(mu).sample().item())
    correct_orders = 0
    for order in range(int(orders)):
        correct_orders += pyro.sample('correct_order', dist.Bernoulli(p))
    return correct_orders
```


<p align="center"><img src="/pyro-adventures/tex/6ac09cad3b60b5b2394b883654c56d45.svg?invert_in_darkmode&sanitize=true" align=middle width=71.56533615pt height=33.4857765pt/></p>

