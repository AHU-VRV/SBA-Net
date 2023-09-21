import numpy as np

# EGNet # matlab中gradient方法 # 比直接调用opencv包中canny算子效果好
# 计算规则: [Fx,Fy]=gradient(F)，其中Fx为其水平方向上的梯度，Fy为其垂直方向上的梯度
# Fx的第一列元素为原矩阵第二列与第一列元素之差，Fx的第二列元素为原矩阵第三列与第一列元素之差除以2
# 以此类推：Fx(i,j)=(F(i,j+1)-F(i,j-1))/2，最后一列则为最后两列之差。同理，可以得到Fy
def gradient(img):
    # 输出为形状和x一致的矩阵，其元素全部为0
    x = np.zeros_like(img, dtype=np.float64)
    # x[:,0]取矩阵x所有行的第0列的元素，x[:,1]取矩阵x所有行的第1列的元素
    x[:, 0] = img[:, 1] - img[:, 0]
    # x[:, 1:-1]取矩阵x所有行的第1列-倒数第2列的元素
    # x[:, 2:]取矩阵x所有行的第2列-最后1列的元素
    # x[:, :-2]取矩阵x所有行的第0列-倒数第3列的元素
    x[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2
    x[:, -1] = img[:, -1] - img[:, -2]

    y = np.zeros_like(img, dtype=np.float64)
    y[0, :] = img[1, :] - img[0, :]
    y[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
    y[-1, :] = img[-1, :] - img[-2, :]

    return x, y