---
layout: page
title: Probability and Statistic Questions
nav: false
---
<link rel="stylesheet" href="/assets/css/main.css"/>

### Conditional Expectation [Robinhood]
*Statment*: Say X and Y are independent and uniformly distributed on (0, 1). What is the expected value of X, given that X > Y?

Using the definition of conditional expectation:

\\[
\textbf{E}(X|X>Y) = \dfrac{1}{\textbf{P}(X>Y)}\int_{(X>Y)}Xd\textbf{P} 
\\]

First compute the probability \\( \textbf{P}(X>Y) \\). There is two ways: 
  * Intuitively, the fact \\( X, Y\\) are independent and uniform distributed implies
    \\[ \textbf{P}(X>Y) = \textbf{P}(X\leq Y) ]\\ 
  * Or analytically
    \\[ 
    \textbf{P}(X>Y) = \int_{0}^{1} \textbf{P}(Y<x)f_X(x) dx = \int_{0}^{1} xdx = \frac{1}{2}
    \\]
    
To compute the integral we condition too
\\[ \int_{(X>Y)}Xd\textbf{P} = \int_{0}^{1} \int_{X>y} X f_Y(y)d\textbf{P} dy \\]
\\[ \hspace{.5cm} = \int_{0}^{1} \int_{y}^{1} x f_X(x) f_Y(y)dxdy \\]
\\[ \hspace{.5cm} = \int_{0}^{1} \left( \frac{1}{2} - \frac{x^2}{2} \right)dxdy \\]
