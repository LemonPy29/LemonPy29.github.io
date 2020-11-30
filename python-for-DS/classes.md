---
#TODO yml
---

Most of the library code we daily use come in form of classes. For example, any
time we create a data frame, a numpy array or a pytorch module we're working
with an instance, a particular example, of a class. Roughly speaking, we can
see classes as templates which can carry states and methods. Similar to
functions, they can be useful tools to avoid duplicate code. 

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

Now you can access `mu, sigma` as instance attributes and there is no longer
the need to type those arguments for every function signature. 

In general, whenever we have functions sharing signatures, body or return, it
may be a good idea to create a class around them. For instance, if we consider
`a` as something as an array and then we look at the next code

```python
def mean(a, *args, **kwargs):
    ...

def clip(a, *args, **kwargs):
    ...

def argmax(a, *args, **kwargs):
    ...
```

Instead of passing the argument `a` to every function, it would be better if
all those functions are attached to the object `a`. We see this pattern in 
popular libraries such as numpy or pandas. 

If we want to add extra functionality to an existing object, we can inherit from
it.

```python
class MyObject(a):
     ...
```

Before going further we should stop to say that inheriting from a non-native
object, that is, objects constructed in the `C` family for example, can
sometimes be tricky an lead to some non-expected behaviors. If we check the
numpy documentation, we'll see inherit from `np.ndarray` it's not so trivial
and you have to follow some rules.

### The dunder magic

In order to make the most out of classes we should talk a little bit about
double under score methods, often called dunder's. These are methods that
implement special protocols or behaviors. Maybe the most familiar it's the
`__init__` method which is generally called when we create a new instance of
the class (we can actually avoid this call if you want with another dunder).
With `__init__` we can see that usually, although it's possible, we don't call
dunder's explicitly. 


A full dunder's list can be found in the python data model, and for sure
every one of them can be useful in certain situations, but in general we tend to use 
a couple of them which we'll review now.

The `__call__` dunder let us literally to call the class as a function. 

Sometimes we may be delighted by the `__call__` charm and start to write
classes every time we need a function that carries extra states. This is often
a bad practice and we should replace those with closures. 
