# import cv2
# import os
# import numpy as np
# from PIL import Image
# import sqlite3
# # получаем путь к этому скрипту
# path = os.path.dirname(os.path.abspath(__file__))
# # создаём новый распознаватель лиц
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# # указываем, что мы будем искать лица по примитивам Хаара
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# # путь к датасету с фотографиями пользователей
# dataPath = path+r'/dataSet'
#
#
#
# # получаем картинки и подписи из датасета
# def get_images_and_labels(datapath):
#     # получаем путь к картинкам
#     image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]
#     # списки картинок и подписей на старте пустые
#     images = []
#     labels = []
#     # перебираем все картинки в датасете
#     for image_path in image_paths:
#         # проверяем, является ли путь файлом
#         if os.path.isfile(image_path):
#             # читаем картинку и сразу переводим в ч/б
#             image_pil = Image.open(image_path).convert('L')
#             # переводим картинку в numpy-массив
#             image = np.array(image_pil, 'uint8')
#             # получаем id пользователя из имени файл��
#             nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
#             # определяем лицо на картинке
#             faces = faceCascade.detectMultiScale(image)
#             # если лицо найдено
#             for (x, y, w, h) in faces:
#                 # добавляем его к списку картинок
#                 images.append(image[y: y + h, x: x + w])
#                 # добавляем id пользователя в спис��к подписей
#                 labels.append(nbr)
#                 # выводим текущую картинку на экран
#                 cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
#                 # делаем паузу
#                 cv2.waitKey(100)
#     # возвращаем список картинок и подписей
#     return images, labels
#
# # получаем список картинок и подписей
# images, labels = get_images_and_labels(dataPath)
# print(labels)
# # обучаем модель распознавания на наших картинках и учим сопоставлять её лица и подписи к ним
# recognizer.train(images, np.array(labels))
# # сохраняем модель
# recognizer.save(path+r'/trainer/trainer.yml')
# # удаляем из памяти все созданные окнаы
# cv2.destroyAllWindows()
#
#
#
#
#
import os
import numpy as np
import cv2
import sqlite3


path = os.path.dirname(os.path.abspath(__file__))

recognizer = cv2.face.LBPHFaceRecognizer_create()

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

databasePath = path + '/faces.db'

# получаем картинки и подписи из базы данных
def get_images_and_labels(databasePath):
    # подключаемся к базе данных
    conn = sqlite3.connect(databasePath)
    cursor = conn.cursor()
    # получаем все записи из таблицы faces
    cursor.execute("SELECT * FROM faces")
    rows = cursor.fetchall()
    # списки картинок и подписей на старте пустые
    images = []
    labels = []
    # for record in rows:
    #     # convert image data from bytes to numpy array
    #     img = np.frombuffer(record[2], np.uint8)
    #     img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    #     images.append(img)
    #     labels.append(record[1])
    # перебира��м все записи в таблице

    for row in rows:
        # получаем имя и фото пользователя
        name = row[1]
        image = row[2]

        # переводим фото в numpy-массив
        nparr = np.frombuffer(image, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        # определяем лицо на фото
        faces = faceCascade.detectMultiScale(img_np)

        # если лицо найдено
        for (x, y, w, h) in faces:
            # добавляем его к списку картинок
            images.append(img_np[y: y + h, x: x + w])
            # добавляем имя пользователя �� спис��к подписей
            labels.append(name)

    # закрываем соединение с базой данных
    cursor.close()
    conn.close()
    # возвращаем список картинок и подписей
    return images, labels

# # получаем список картинок и подписей
# images, labels = get_images_and_labels(databasePath)
# labels = "".join(labels)
# labels = int(labels)
# print(labels)
# # обучаем модель распознавания на наших картинках и учим сопоставлять её лица и подписи к ним
# recognizer.train(images, np.array(labels))
# # сохраняем модель
# recognizer.save(path+r'/trainer/trainer1.yml')
# # удаляем из памяти все созданные окнаы
# cv2.destroyAllWindows()
images, labels = get_images_and_labels(databasePath)

# convert all labels to integers
labels = [int(label) for label in labels]
print(labels)
# train the recognizer on the images and labels
recognizer.train(images, np.array(labels))
# save the model
recognizer.save(path + '/trainer/trainer1.yml')
# destroy all windows
cv2.destroyAllWindows()