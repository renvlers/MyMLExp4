import tensorflow as tf
from tensorflow import keras
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 载入Iris数据集
iris = datasets.load_iris()
data = iris.data
target = iris.target

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

def create_model(optimizer):
    model = keras.models.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=X_train.shape[1:]),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model

if __name__ == "__main__":
    # 梯度下降
    optimizer = keras.optimizers.SGD(learning_rate=0.01)
    model = create_model(optimizer)
    history_gd = model.fit(X_train, y_train, epochs=50, validation_split=0.2)
    test_loss_gd, test_accuracy_gd = model.evaluate(X_test, y_test)
print(f"Test accuracy Gradient Descent: {test_accuracy_gd}")


    # 动量算法
    optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model = create_model(optimizer)
    history_momentum = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

    # AdaGrad
    optimizer = keras.optimizers.Adagrad(learning_rate=0.01)
    model = create_model(optimizer)
    history_adagrad = model.fit(X_train, y_train, epochs=50, validation_split=0.2)
