import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 导入ResNet50模型
base_model = ResNet50(weights='imagenet', include_top=False)

# 添加全局空间平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 添加全连接层
x = Dense(1024, activation='relu')(x)

# 添加一个分类层，假设我们有10个类
predictions = Dense(5, activation='softmax')(x)

# 构建我们需要训练的完整模型
model = Model(inputs=base_model.input, outputs=predictions)

# 首先，我们只训练顶部的几层（即全连接层）
for layer in base_model.layers:
    layer.trainable = False

# 编译模型（一定要在锁定层后进行编译）
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 现在，让我们从一些图片开始训练顶层。
# 假设我们的数据集存放在"data/train"和"data/validation"目录下
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',  # 这是目标目录
        target_size=(224, 224),  # 所有图像将被调整为150x150
        batch_size=32,
        class_mode='categorical')  # 因为我们使用binary_crossentropy损失，所以需要二进制标签

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

# 开始训练模型
model.fit(
        train_generator,
        steps_per_epoch=2000 // 32,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // 32)
