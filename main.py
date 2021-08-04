# tensorflow와 tf.keras를 임포트합니다
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 헬퍼(helper) 라이브러리를 임포트합니다
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

image_generator = ImageDataGenerator(
  rotation_range=10,
  zoom_range=0.10,
  shear_range=0.5,
  width_shift_range=0.10,
  height_shift_range=0.10,
  horizontal_flip=True,
  vertical_flip=False)

augment_size = 30000

randidx = np.random.randint(train_images.shape[0], size=augment_size)
x_augmented = train_images[randidx].copy()
y_augmented = train_labels[randidx].copy()
x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size), batch_size=augment_size, shuffle=False).next()[
  0]

train_images = np.concatenate((train_images, x_augmented))
train_labels = np.concatenate((train_labels, y_augmented))

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(input_shape=(28, 28, 1), kernel_size=(3, 3), filters=32, padding='same', activation='relu'),
  tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=64, padding='same', activation='relu'),
  tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
  tf.keras.layers.Dropout(rate=0.5),
  tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=128, padding='same', activation='relu'),
  tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=256, padding='valid', activation='relu'),
  tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
  tf.keras.layers.Dropout(rate=0.5),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(units=512, activation='relu'),
  tf.keras.layers.Dropout(rate=0.5),
  tf.keras.layers.Dense(units=256, activation='relu'),
  tf.keras.layers.Dropout(rate=0.5),
  tf.keras.layers.Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=50)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\n테스트 정확도:', test_acc)

predictions = model.predict(test_images)

predictions[0]


def plot_image(i, predictions_array, true_label, img): #번호,예측배열,정답,이미지
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i] #예상된 배열값, 정답, 이미지
  plt.grid(False) #그리드 없음
  plt.xticks([]) #x축 눈금없음
  plt.yticks([]) #y축 눈금없음

  plt.imshow(img, cmap=plt.cm.binary) #이미지 회색으로 출력하기

  predicted_label = np.argmax(predictions_array) #예상되는 배열중 제일 큰값
  if predicted_label == true_label: #예상된 배열중 제일 큰값 == 정답
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label): #번호, 예상 배열, 정답
  predictions_array, true_label = predictions_array[i], true_label[i]  #예상, 배열
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777") #각 배열은 회색으로
  plt.ylim([0, 1]) #y축은 0부터 1까지 범위 제한
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red') #제일 높은 예측 값은 빨강
  thisplot[true_label].set_color('blue') #정답은 파랑

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))  #인치를 열은 두개로 나누고 행은 그대로 5
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1) #행, 열, 홀수
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)# 행, 열, 짝수
  plot_value_array(i, predictions, test_labels)
plt.show()

