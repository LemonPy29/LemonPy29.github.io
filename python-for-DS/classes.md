---
layout: page
title: Something about classes
nav: false
---
<link rel="stylesheet" href="/assets/css/main.css"/>

Most of the library code we daily use come in form of classes. For example, any
time we create a data frame, a numpy array or a pytorch `nn.Module`we're
working with an instance, a particular example, of a class. Roughly speaking,
we can see classes as templates or abstractions which can carry states and
methods. Similar to functions, they can be useful tools to avoid duplicate
code. 


## When should we use classes?

In the following code we have several functions that take similar or almost the
same arguments

```python
def pdf(x, mu, sig):
    ...

def cdf(x, mu, sig):
    ...

def moment(p, mu, sig):
    ...
```

This pattern, of repeating code too much, it's a common sign that we are going
on the wrong path and it's probably better to do something about it an try to
generalize or wrap that code. In this case a class it's a good solution,
because, among other things, we no longer need to type the arguments again and
again, and can simply define those as class attributes. 

```python
class Normal:
    def __init__(self, mu, sig):
        self.mu = mu
	self.sig = sig

    def pdf(self, x):
        ...

    def cdf(self, x):
        ...

    def moment(self, p):
        ...
```

Now you can access `mu` and `sigma` as instance attributes and there is no
longer the need to type those arguments for every function signature. Not only
that, in this case the class help us to write more cleaner and declarative
code. 

So, in general, whenever we have functions sharing signatures, it may be a good
idea to create a class around them. 

Sometimes, we should go further and think about our specific case. For
instance, look at the next code

```python
polynomial = lambda x: x**2 - x + 2

def degree(polynomial, *args, **kwargs):
    ...

def roots(polynomial, *args, **kwargs):
    ...
```
Following the above pattern, we may end up writing 

```python
class Polynomial:
    def __init__(self, fn):
        self.fn = fn

    # then use self.fn in several places

fn = lambda x: x**2 - x + 2
p = Polynomial(fn)    
```

However, in this way the class end up being just a container and doesn't
represent the object in a proper manner. A better approach could be 
something like

```python
class Polynomial:
    def __init__(self, coefs):
        self.coefs = coefs

    def __call__(self, x):
        return sum(c * x**p for p, c in self.coefs.items())

    def degree(self):
        ...

coefs = {0: 2, 1: -1, 2: 1}
p = Polynomial(coefs)
```

If we have an existing abstraction, say `x`, and we also have
several functions that aims to add functionality to it, a more expressive and
pythonic way of accomplish that is by using inheritance.

```python
class MyObject(x):
     ...

```

With this code we make clear our intention on expanding the functionality of
the existing object `x`.


## Inheritance, abstract classes and mixins

If now we have classes that share similar states and methods, we can factor
that out in a base classes and then inherit from it. A familiar example is
the sklearn API. At high level, we can think it as 

```python
class BaseEstimator:
    ...

    def fit(X, *args, **kwargs):
        ...

    def predict(X, *args, **kwargs):
        ...

```
We can use those methods on new classes, or use them as a base for method
on the children

## The dunder magic

In order to make the most out of classes we should talk a little bit about
double under score methods, often called dunder's. These are methods that
implement special protocols or behaviors. Maybe the most familiar is the
`__init__` method which is generally called when we create a new instance of
the class, so can be considered as kind of a constructor.  With `__init__`, we
can see that usually, although it's possible, we don't call dunder's
explicitly. 

A good example of these special methods are pythorch loaders.

```python
class Dataset:
    def __init__(self, data):
        # Suppose data is a torch.tensor
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
```
It's easy to guess that the `__len__` method implements `len(self)`. The point to 
note here is, in general, we implement special methods by delegating to the same
method on another object. 

The `__getitem__` method implements `self[key]`, that is, hash the instance by
an integer or an slice of integers. Again, we see this pattern of relying on a call
to the same method on `data`, which already implements it. 

In the python data model documentation we can find all the details and a full list 
of the special, double score, methods. 


## when to avoid classes?

Although classes are certainly useful, they're not always the way to go. Instantiate
a class has its costs and at a times they end up making the code less readable.

A good example of class overuse are two or three method classes where one
method is `__init__`.  Many times the idea behind those is to store parameters
or functions.  For example

```python
class SomeFunc:
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        return x**self.p - 1

    def grad(self, x):
        return p * x**(self.p-1)
```
Which can be replaced with a closure

```python
def somefunc(p):

    def func(x):
        return x**p - 1

    def grad(x)
        return p * x**(p-1)

    func.grad = grad
    return func

f = somefunc(4)
f(1) # 0
f.grad(1) # 4
```
Of course, sometimes is fine to define two-method classes. 

