# import cv2
# import os
# import sqlite3
#
# # получаем путь к этому скрипту
# path = os.path.dirname(os.path.abspath(__file__))
# # указываем, что мы будем искать лица по примитивам Хаара
# detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# # счётчик изображений
# i=0
# # расстояния от распознанного лица до рамки
# offset=50
# # запрашиваем номер пользователя
# name=input('Введите номер пользователя: ')
# # получаем доступ к камере
# video=cv2.VideoCapture(0)
#
# #создаем базу данных и таблицу для хранения изображений
# conn = sqlite3.connect('faces.db')
# cursor = conn.cursor()
# cursor.execute('''CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, image BLOB)''')
#
#
#
# while True:
#     # берём видеопоток
#     ret, im =video.read()
#     # переводим всё в ч/б для простоты
#     gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#     # настраиваем параметры распознавания и получаем лицо с камеры
#     faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
#     # обрабатываем лица
#     if len(faces) > 0:
#         for(x,y,w,h) in faces:
#             # увеличиваем счётчик кадров
#             i=i+1
#             # записываем файл на диск
#             cv2.imwrite("dataSet/face-"+name +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
#             # формируем размеры окна для вывода лица
#             cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
#             # показываем очередной кадр, который мы запомнили
#             cv2.imshow('im',im[y-offset:y+h+offset,x-offset:x+w+offset])
#             # делаем паузу
#             cv2.waitKey(100)
#     # если у нас хватает кадров
#     if i>30:
#         # освобождаем камеру
#         video.release()
#         # удалаяем все созданные окна
#         cv2.destroyAllWindows()
#         # останавливаем цикл
#         break




import cv2
import os
import sqlite3
import numpy as np

# полу��аем путь к этому скрипту
path = os.path.dirname(os.path.abspath(__file__))
# указываем, что мы будем искать лица по примитивам Хаара
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# счётчик изображений
i=0
# расстояния от распознанного лица до рамки
offset=50
# запр��шиваем номер пользовател��
name=input('Введите номер пользовател��: ')
# полу��аем доступ к камере
video=cv2.VideoCapture(0)

# создаем базу данных и таблицу для хранения изображений
conn = sqlite3.connect('faces.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, image BLOB)''')

while True:
    # берём видеопоток
    ret, im =video.read()
    # переводим всё в ч/б для простоты
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # настраиваем параметры распознавания и получаем лицо с камеры
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
    # обрабатываем лица
    if len(faces) > 0:
        for(x,y,w,h) in faces:
            # увеличиваем счётчик кадров
            i=i+1
            # записываем файл в базу данных
            img = gray[y - offset:y + h + offset, x - offset:x + w + offset]
            if img.size != 0:
                _, buffer = cv2.imencode('.jpg', img)
                cursor.execute('''INSERT INTO faces (name, image) VALUES (?, ?)''', (name, buffer.tobytes()))
                conn.commit()
            # формируем размеры окна для вывода лица
            cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
            # показываем очередной кадр, который мы запомнили
            if img.shape[0] > 0 and img.shape[1] > 0:
                cv2.imshow('im', im[y - offset:y + h + offset, x - offset:x + w + offset])
                cv2.waitKey(100)
    # если у нас хватает кадров
    if i>30:
        # освобождаем камеру
        video.release()
        # удалаяем все созданные окна
        cv2.destroyAllWindows()
        # закрываем базу данных
        conn.close()
        # останавливаем цикл
        break

