import torch

print(f"PyTorch Version: {torch.__version__}")

matrix1 = torch.tensor(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=torch.float32
)

matrix2 = torch.tensor(
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.float32
)

result = matrix1 @ matrix2

print("\nMatrix 1 (5 x 3):")
print(matrix1)
print("\nMatrix 2 (3 x 4):")
print(matrix2)
print("\nResult of matrix multiplication (5 x 4):")
print(result)
