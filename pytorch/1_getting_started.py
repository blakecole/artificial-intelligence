# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: MIT                                               #
#    FILE: 1_getting_started.py                              #
#    DATE: 26 NOV 2020                                       #
# ********************************************************** #

from __future__ import print_function
import torch
import numpy as np

# -----------------------------------------------------------
# INITIALIZE TENSOR:

# 1) allocate an empty 5x3 tensor
x = torch.empty(5, 3)
print(x)

# 2) define a randomly initialized tensor [vals = (0:1)]
x = torch.rand(5, 3)
print(x)

# 3) define tensor with specified data type
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 4) create tensor from data
x = torch.tensor([5.5, 3])
print(x)

# 5) create tensor based on existing tensor
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.rand_like(x, dtype=torch.float)
print(x)

# 6) get size of tensor
print(x.size())


# -----------------------------------------------------------
# TENSOR OPERATIONS

# 7a) addition (syntax 1)
y = torch.rand(5, 3)
print(x + y)

# 7b) addition (syntax 2)
print(torch.add(x, y))

# 7c) addition (output tensor as argument)
result = torch.empty(5, 3)    # allocate output matrix
torch.add(x, y, out=result)   # write output matrix
print(result)

# 7d) addition (in place)
# note: Any operation that mutates a tensor in-place is
#       post-fixed with an _. For example: x.copy_(y),
#       x.t_(), will change x.
y.add_(x)
print(y)

# 8) tensor index notation (NumPy-like!)
print(x[:, 1])

# 9) resizing
x = torch.randn(4, 4)
y = x.view(16)       # single row vector
z = x.view(-1, 8)    # size -1 is inferred from other dims
print(x.size(), y.size(), z.size())

# 10) extract one-element tensor as Python number
x = torch.randn(1)
print(x)
print(x.item())


# -----------------------------------------------------------
# NUMPY BRIDGE

# 11) torch tensor --> numpy array
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

# note: these representations are 1-way coupled; a change to
#       the torch tensor will change the numpy array.

a.add_(1)
print(a)
print(b)

#12) numpy array --> torch tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# -----------------------------------------------------------
# CUDA TENSORS (GPU)

# let us run this cell only if CUDA is available.

# we will use ``torch.device`` objects to move tensors
# into and out of GPU.

if torch.cuda.is_available():
    # create CUDA device object
    device = torch.device("cuda")

    # directly create a tensor on GPU
    y = torch.ones_like(x, device=device)

    # or just use strings ``.to("cuda")``
    x = x.to(device)
    z = x + y
    print(z)

    # ``.to`` can also change the dtype together
    print(z.to("cpu", torch.double))
