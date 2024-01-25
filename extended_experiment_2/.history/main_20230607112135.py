import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plt_cl

def beale(x1, x2):
    return (1.5-x1+x1*x2)**2+(2.25-x1+x1*x2**2)**2+(2.625-x1+x1*x2**3)**2;

def dbeale_dx(x1, x2):
    dfdx1 = 2*(1.5-x1+x1*x2)*(x2-1)+2*(2.25-x1+x1*x2**2)*(x2**2-1)+2*(2.625-x1+x1*x2**3)*(x2**3-1)
    dfdx2 = 2*(1.5-x1+x1*x2)*x1+2*(2.25-x1+x1*x2**2)*(2*x1*x2)+2*(2.625-x1+x1*x2**3)*(3*x1*x2**2)
    return dfdx1, dfdx2

def gradient_desent(x1_0, x2_0, learning_rate=0.1, tol_iter=10):
    x = np.array([x1_0, x2_0])
    dydx = np.array(dbeale_dx(x[0], x[1]))
    for i in range(tol_iter):
        x = x - learning_rate*dydx
        dydx = np.array(dbeale_dx(x[0], x[1]))
    return x

# 定义画图函数
def gd_plot(x_traj):
    plt.rcParams['figure.figsize'] = [6, 6] # 窗口大小
    plt.contour(X1, X2, Y, levels=np.logspace(0, 6, 30),
    norm=plt_cl.LogNorm(), cmap=plt.cm.jet) # 画等高线图
    plt.title('2D Contour Plot of Beale function') # 添加标题
    plt.xlabel('$x_1$') # x轴标签
    plt.ylabel('$x_2$') # y轴标签
    plt.axis('equal') # 设置坐标轴为正方形
    plt.plot(3, 0.5, 'k*', markersize=10) # 画出最低点
    if x_traj is not None:
        x_traj = np.array(x_traj) # 将x_traj转为数组
        plt.plot(x_traj[0], x_traj[1], 'r*', markersize=10)
    # 以x_traj的第一列为x轴坐标，第二列为y轴坐标进行画图
    plt.show() # 显示图像

if __name__ == "__main__":
    step_x1, step_x2 = 0.2, 0.2
    X1, X2 = np.meshgrid(np.arange(-5, 5 + step_x1, step_x1), np.arange(-5, 5 + step_x2, step_x2)) # 将图形从-5 到 5.2，步长为0.2 划分成网格点
    Y = beale(X1, X2) # 将x1,x2坐标带入beale公式
    x1_opt, x2_opt = 3, 0.5
    print("目标结果 (x_1, x_2) = (3, 0.5)")
    gd_plot(None) # 调用函数

    # 使用梯度下降
    x1_0, x2_0 = 0.5, 0
    x_gd = gradient_desent(x1_0, x2_0, learning_rate=0.01, tol_iter=1000)
    print(x_gd)
    gd_plot(x_gd)
