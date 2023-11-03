import torch
batches = torch.randn(5, 3, 14, 128)
samples = torch.randn(5, 3, 14, 128)

tsm = torch.matmul(batches, samples.transpose(-2, -1))

for i in range(batches.shape[1]):
    print()