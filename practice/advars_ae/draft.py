import torch

# Example tensors
z = torch.tensor([[0, 0],
                  [0, 0]])
c = torch.tensor([[1, 1],
                  [1, 1]])

# Concatenating tensors along the second dimension (horizontal concatenation)
result = torch.cat((z, c), 1)

print(result)

