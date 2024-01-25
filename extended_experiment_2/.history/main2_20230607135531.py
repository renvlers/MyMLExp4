import tensorflow as tf
from tensorflow import keras
from sklearn import datasets

# 载入Iris数据集
iris = datasets.load_iris()
data = iris.data
target = iris.target

from sklearn.model_selection import train_test_split

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

