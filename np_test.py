import numpy as np

a = np.array([[1,2,3],[1,2,3]])
print(a)
b = []
b.append(a)
b.append(a)
print("----------------------")
a = a.squeeze()
print(a)
