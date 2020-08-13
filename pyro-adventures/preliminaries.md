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

```math
SE = \frac{\sigma}{\sqrt{n}}
```
