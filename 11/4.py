import torch

zeros = torch.zeros(3, 3)
ones = torch.ones(3, 3)
random_matrix = torch.rand(3, 3)

print("Zeros matrix:")
print(zeros)
print("\nOnes matrix:")
print(ones)
print("\nRandom matrix:")
print(random_matrix)

print("\nAbsolute value of random matrix:")
print(torch.abs(random_matrix))

print("\nAddition of ones and random matrix:")
print(ones + random_matrix)

print("\nDivision of random matrix by ones:")
print(random_matrix / ones)

print("\nDot product of random matrix with ones:")
print(random_matrix * ones)

print("\nCross product of random matrix with ones:")
print(random_matrix @ ones)

print("\nExponentiation of random matrix:")
print(torch.exp(random_matrix))

print("\nSize of matrices:")
print(f"Shape: {random_matrix.size()}")
print(f"Dimensions: {random_matrix.dim()}")
print(f"Total elements: {random_matrix.numel()}")
