# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: MIT                                               #
#    FILE: 2_autograd.py                                     #
#    DATE: 26 NOV 2020                                       #
# ********************************************************** #

#from __future__ import print_function
import torch

# ----------------------------------------------------------
# INTRO TO AUTOMATIC DIFFERENTIATION

# define a tensor which tracks changes
x = torch.ones(2, 2, requires_grad=True)
print(x)

# perform a tensor operation
# any tensor created from an autograd tensor will also track
y = x + 2
print(y)

# query the gradient function bestowed upon y
print(y.grad_fn)

# more tensor operations
z = y * y * 3
out = z.mean()
print(z, out)

# change a tensors ``require grad`` status in place
a = torch.randn(2, 2)
a = ((a * 3)/ (a - 1))
print(a.requires_grad)

a.requires_grad_(True)
print(a.requires_grad)

b = (a * a).sum()
print(b.grad_fn)


# ----------------------------------------------------------
# BACKPROP

# scalar case (out = z.mean(), line 27)
out.backward()
print(x.grad)

# vector-Jacobian case
# a) do a few tensor operations
x = torch.randn(3, requires_grad=True)

print(x)

y = x * 2

while y.data.norm() < 1000:
    print(y)
    y = y * 2

print(y)

# extract the vector-Jacobian product
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

# NOTE! THIS IS FUCKIN WEIRD AND I DONT QUITE FOLLOW
#       REVISIT LATER.

# 1) why is the derivative stored with the variable that the
#    derivative is taken with respect to?

# 2) does v in this case represent (y - yhat) error?
#    is y.backward(v) dv/dy? dv/dx?

print()
