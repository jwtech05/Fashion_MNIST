# tensorflow와 tf.keras를 임포트합니다
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 헬퍼(helper) 라이브러리를 임포트합니다
import numpy as np
import matplotlib.pyplot as plt

# image_generator = ImageDataGenerator(
#             rotation_range=10, # 회전
#             zoom_range=0.10, # 확대
#             shear_range=0.5, # 기울임
#             width_shift_range=0.10, # 가로방향 평행 이동
#             height_shift_range=0.10, # 세로방향 평행 이동
#             horizontal_flip=True, # 좌우반전
#             vertical_flip=False # 상하반전
# )

augment_size = 100

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# for i in range(train_images):
#   train_images = image_generator.flow(np.tile(train_images[i].reshape(28*28),10).reshape(-1,28,28,1),
#                            np.zeros(augment_size), batch_size=augment_size,
#                            shuffle=False).next()[i]
train_images = train_images.reshape(-1, 28, 28, 1)
train_images = train_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (2,2), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Dropout(0.15),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_json = model.to_json()
with open("model.json", "w") as json_file :
    json_file.write(model_json)



model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\n테스트 정확도:', test_acc)

predictions = model.predict(test_images)

predictions[0]

model.save_weights("model.h5")
print("Saved model to disk")

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

plt.subplot()

fig, (ax1,ax2) =plt.subplots(1, 2, figsize= (12,5))

y_vloss = history.history['val_loss']

y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
ax1.plot(x_len, y_vloss, marker ='.', c="red", label='Testset_loss')
ax1.plot(x_len,y_loss,marker='.',c='blue', label = 'Trainset_loss')

ax1.legend(loc='upper right')
ax1.grid()
ax1.set(xlable='epoch', ylabel='loss')

y_vaccuracy = history.history['val_accuracy']

y_accuracy = history.history['accuracy']

x_len = np.arange(len(y_accuracy))
ax2.plot(x_len, y_vaccuracy, marker = '.', c="red", label='Testset_accuracy')
ax2.plot(x_len, y_accuracy, marker = '.', c='blue', label = 'Trainset_accuracy')

ax2.legend(loc='lower right')
ax2.grid()

ax2.set(xlabel='epoch', ylabel='accuracy')

ax2.grid(True)
plt.show()