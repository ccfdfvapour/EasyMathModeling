import numpy as np

# 原矩阵
A = np.array([[0.527, -0.574, 0.628, 2],
              [0.369, 0.819, 0.439, 5],
              [-0.766, 0, 0.643, 3],
              [0, 0, 0, 1]])
# 逆矩阵
A1 = np.linalg.inv(A)
# 保留三位小数
A2 = np.around(
    A1,  # numpy数组或列表
    decimals=3)  # 保留几位小数

print(A2)
