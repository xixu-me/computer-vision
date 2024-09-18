import numpy as np
import pywt

data = np.array([22, 26, 30, 18, 16, 20, 25, 36])
coeffs0 = pywt.dwt(data, "haar")
coeffs1 = pywt.wavedec(data, "haar", level=2)

data1 = np.array([[12, 4, 26, 14], [6, 27, 32, 22], [18, 12, 30, 24], [32, 28, 10, 8]])
coeffs2 = pywt.dwt2(data1, "haar")
coeffs3 = pywt.wavedec2(data1, "haar", level=2)

print(coeffs0)
print(coeffs1)
print(coeffs2)
print(coeffs3)
