---
layout: page
title: Pyro Models
nav: true
---
This series/section is about the construction and understanding of probabilistic models using Pyro: an API sitting on top of Pytorch for probabilistic programming that includes lots of helpful methods and classes to model most of stochastic situations. 

What do I mean with probabilistic models? Modern machine learning techniques are kind of probabilistic (when you pick initial weights and biases distributions for example) but I rather consider them deterministic because in general they doesn't account for the variability and uncertainty of the data. One of the finest examples I think can illustrate the difference is the versus between ordinary linear regression and Bayesian linear regression. 

Although probabilistic models and the more classical statistical methods have been left behind a little bit due to the explosive development of the machine and deep learning world, they remain pretty useful in a bunch of situations. I'm not trying to draw a line between both: they of them share many similarities and one complement the other. In fact, many models combines both approaches.  

### Index
* [Basic Examples of Stochastic Processes](pyro-models/basic-examples.html)
* [A minimal understanding of MCMC](pyro-models/mcmc_example.html)
