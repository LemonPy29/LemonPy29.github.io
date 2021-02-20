---
layout: page
title: Something about classes
nav: false
---
<link rel="stylesheet" href="/assets/css/main.css"/>

Most of the library code we daily use come in form of classes. For example, any
time we create a data frame, a numpy array or a pytorch `nn.Module` we're
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

    @property
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
the existing object `x`. In the following section we willtalk a little bit more
about inheritance.


## Inheritance, abstract classes and mixins

If now we have classes that share similar states and methods, we can factor
that out in a base class and then inherit from it. We can think for example in
the popular sklearn API. With that in mind, suppose there is a base class for
all the models with fit and predict methods

```python
class BaseEstimator:
    ...

    def fit(self, X, *args, **kwargs):
        ...

    def predict(self, X, *args, **kwargs):
        ...

```

We can use those methods on new classes, or use them as a base for a method
on the children. For example,

```python
class LinearEstimator(BaseEstimator):
    ...

le = LinearEstimator(**parameters)
le.fit(X) # call the parent fit method
```
Or we can wrap the parent method

```python
class OtherEstimator(BaseEstimator):
    ...
    
    def predict(self, X, *args, **kwargs):
        result = super().predict(X, *args, **kwargs)
        # do something else
        return result
```      

With `super` we call methods of the parent class and any instance of the
children can call parents methods, even if they are not defined explicitly on
the children as we saw before.

Although sklearn objects share methods, it's very likely the `fit` method on a
linear estimator is different from the one on a random forest.  Nevertheless,
we would like to have consistency accross classes, ensuring all of them
implement certain methods. For that purpose, we have some help from the `abc`
built-in module.

```python
from abc import ABCMeta, abstracmethod

class BaseEstimator(metaclass=ABCMeta):
    
    @abstracmethod
    def fit(self, **args, **kwargs):
        ...
```
Classes with the ABC metaclass aren't meant to be instantiated by themselves.
Instead they serve as interfaces to inherit from them. They enforce to 
every children to follow that interface

```python
class LinearEstimator(BaseEstimator):
    # no fit method
    ...

le = LinearEstimator(**params)
# TypeError: Can't instantiate child with abstract method fit 
```

Finally, we have Mixins. They don't have any special syntax, but still aren't 
meant to be instantiated by their own. They serve to add functionality to an
existing class by inheritance. Why it is not regular inheritance? At some 
point it is, but one the main features of Mixins is they are some kind of
agnostic respect to the extended class and in general they perform a generic
operation such as login, timing or type checking. 

Suppose for example, we have a framework with models but also with data
structures, and maybe some kind of plot objects. We would like to implement a
save method to store metadata, across of all of our classes, but obviously we
don't want to type that method for every one of them. Also, there is no base
class in common. What about creating a class for this? 

```python
class SaveMixin:
    def save(self, dir):
        name = self.__class__.__name__
        useful_metadata = super().metadata(format='json')
        some_generic_save(useful_metadata, name, dir)

class SomeEstimator(SaveMixin, BaseEstimator):
      ...
```

We can see here a common implementation of a Mixin. They usually are parents
along other classes, and can take advantage of that by invoking other parents 
methods using `super` too. 

Although we could define elsewhere a save function that takes as an argument
our object, this design can make life easier for a potential end user, because
that function would probably mean another line of importing.


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

## When to avoid classes?

Although classes are certainly useful, they're not always the way to go. Instantiate
a class has its costs and at a times they end up making the code less readable.

A good example of class overuse are classes that have two methods, a one of
them is `__init__` (or single-method classes for short).  Many times the idea
behind those is to store parameters or functions. For example

```python
import pandas as pd

class Reader:
    def __init__(self, delimiter):
        self.delimiter = delimiter

    def read_csv(self, dir, **kwargs):
        return pd.read_csv(dir, sep=self.delimiter, **kwargs)
```

A much better approach is the use of closures

```python
def reader(delimiter):
    def read_csv(dir, **kwargs):
        return pd.read_csv(dir, sep=delimiter, **kwargs)
    return read_csv
 
reader_semicolon = reader(';')
reader_semicolon('path/to/my/csv')
```

This example is kind of silly. It is actually much better just to pass the
parameter that we're trying to store in this case. We've picked it to
illustrate how people often go crazy wrapping everything in classes. If for
some reason we really want to set or store one function parameter, a nice
solution is to use `partial` from the `functools` built-in module.

```python
from functools import partial

reader = partial(pd.read_csv, sep=';')
```

Here we have another example of replacing functions with closures

```python
class FuncAndGrad:
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        return x**self.p - 1

    def grad(self, x):
        return p * x**(self.p-1)
```
Instead we can write

```python
def somefunc(p):

    def func(x):
        return x**p - 1

    def grad(x)
        return p * x**(p-1)

    func.grad = grad
    return func

f = somefunc(4)
f(1) = 0
f.grad(1) = 4
```

Here we have attached a function to another function, but there is no problem
on attaching variables. Of course, this doesn't mean every of this short
classes is bad or unnecesary, but we should think twice before creating one.

Another good alternative for replacing the classes, that maybe are there just
for storing data, are the objects from the `collections` module. For example,
`namedtuple` and `defaultdict` are great choices for this purpose. 
