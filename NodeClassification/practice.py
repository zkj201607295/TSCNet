import torch
import numpy as np
import random

row = torch.tensor([0, 1, 2, 2])
col = torch.tensor([0, 2, 1, 3])
val = torch.tensor([1, 2, 3, 4])
size = (3, 4)

self.adj_t = SparseTensor(row=row, col=col,
                                      value=val,
                                      sparse_sizes=(3, 4))

print(a.max(1)[1])