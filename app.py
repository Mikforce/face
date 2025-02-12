# from flask import Flask, render_template, Response, request, redirect, url_for
# import cv2
# import os
# import sqlite3
# import numpy as np
#
# app = Flask(__name__)
#
# # Получаем путь к этому скрипту
# path = os.path.dirname(os.path.abspath(__file__))
# # Создаём новый распознаватель лиц
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# # Добавляем в него модель, которую мы обучили на прошлых этапах
# recognizer.read("my_model.yml")
# # Указываем, что мы будем искать лица по примитивам Хаара
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#
# # Создаем или подключаемся к базе данных
# conn = sqlite3.connect('faces.db')
# cursor = conn.cursor()
# cursor.execute('''CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, image BLOB)''')
#
# # Глобальная переменная для хранения имени текущего пользователя
# current_user = ""
#
#
# # Получаем доступ к камере
# cam = cv2.VideoCapture(0)
# # Настраиваем шрифт для вывода подписей
# font = cv2.FONT_HERSHEY_SIMPLEX
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# def gen():
#     while True:
#         # Получаем видеопоток
#         ret, im = cam.read()
#         # Переводим его в ч/б
#         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         # Определяем лица на видео
#         faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
#         # Перебираем все найденные лица
#         for (x, y, w, h) in faces:
#             # Получаем id пользователя
#             nbr_predicted, coord = recognizer.predict(gray[y:y + h, x:x + w])
#             # Рисуем прямоугольник вокруг лица
#             cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
#             # Если мы знаем id пользователя
#             if (nbr_predicted == 1 or nbr_predicted == 5):
#                 # Подставляем вместо него имя человека
#                 nbr_predicted = 'Ivan'
#             if (nbr_predicted == 2):
#                 # Подставляем вместо него имя человека
#                 nbr_predicted = 'Anastasiya'
#             if (nbr_predicted == 3):
#                 # Подставляем вместо него имя человека
#                 nbr_predicted = 'Viktorya'
#             if (nbr_predicted == 4):
#                 # Подставляем вместо него имя человека
#                 nbr_predicted = 'Aleksey'
#             # Добавляем текст к рамке
#             cv2.putText(im, str(nbr_predicted), (x, y + h), font, 1.1, (0, 255, 0))
#
#         # Преобразовываем изображение OpenCV в формат JPEG
#         ret, jpeg = cv2.imencode('.jpg', im)
#         frame = jpeg.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#
# @app.route('/video_feed')
# def video_feed():
#     return Response(gen(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
#
#
# @app.route('/add_user', methods=['GET', 'POST'])
# def add_user():
#     global current_user
#     if request.method == 'POST':
#         current_user = request.form['name']
#         return redirect(url_for('capture_user'))
#     return render_template('add_user.html')
#
# @app.route('/capture_user')
# def capture_user():
#     path = os.path.dirname(os.path.abspath(__file__))
#     # указываем, что мы будем искать лица по примитивам Хаара
#     detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#     # счётчик изображений
#     i = 0
#     # расстояния от распознанного лица до рамки
#     offset = 50
#     # запр��шиваем номер пользовател��
#     name = input('Введите номер пользовател��: ')
#     # полу��аем доступ к камере
#     video = cv2.VideoCapture(0)
#
#     # создаем базу данных и таблицу для хранения изображений
#     conn = sqlite3.connect('faces.db')
#     cursor = conn.cursor()
#     cursor.execute('''CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, image BLOB)''')
#
#     while True:
#         # берём видеопоток
#         ret, im = video.read()
#         # переводим всё в ч/б для простоты
#         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         # настраиваем параметры распознавания и получаем лицо с камеры
#         faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
#         # обрабатываем лица
#         if len(faces) > 0:
#             for (x, y, w, h) in faces:
#                 # увеличиваем счётчик кадров
#                 i = i + 1
#                 # записываем файл в базу данных
#                 img = gray[y - offset:y + h + offset, x - offset:x + w + offset]
#                 if img.size != 0:
#                     _, buffer = cv2.imencode('.jpg', img)
#                     cursor.execute('''INSERT INTO faces (name, image) VALUES (?, ?)''', (name, buffer.tobytes()))
#                     conn.commit()
#                 # формируем размеры окна для вывода лица
#                 cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
#                 # показываем очередной кадр, который мы запомнили
#                 if img.shape[0] > 0 and img.shape[1] > 0:
#                     cv2.imshow('im', im[y - offset:y + h + offset, x - offset:x + w + offset])
#                     cv2.waitKey(100)
#         # если у нас хватает кадров
#         if i > 30:
#             # освобождаем камеру
#             video.release()
#             # удалаяем все созданные окна
#             cv2.destroyAllWindows()
#             # закрываем базу данных
#             conn.close()
#             # останавливаем цикл
#             break
#     return render_template('capture_user.html', user_name=current_user)
#
# @app.route('/save_image', methods=['POST'])
# def save_image():
#     global current_user
#     if current_user:
#         image = request.files['image']
#         if image:
#             # Преобразуем изображение в массив байтов
#             image_bytes = image.read()
#             cursor.execute('''INSERT INTO faces (name, image) VALUES (?, ?)''', (current_user, image_bytes))
#             conn.commit()
#             return "Изображение успешно сохранено для пользователя: " + current_user
#     return "Ошибка: Изображение не сохранено"
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
import sqlite3
import numpy as np


app = Flask(__name__)

# Получаем путь к этому скрипту
path = os.path.dirname(os.path.abspath(__file__))
# Создаём новый распознаватель лиц
recognizer = cv2.face_LBPHFaceRecognizer.create()
# Добавляем в него модель, которую мы обучили на прошлых этапах
recognizer.read("my_model.yml")
# Указываем, что мы будем искать лица по примитивам Хаара
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Создаем или подключаемся к базе данных
conn = sqlite3.connect('faces.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, image BLOB)''')

# Глобальная переменная для хранения имени текущего пользователя
current_user = ""

# Получаем доступ к камере
cam = cv2.VideoCapture(0)
# Настраиваем шрифт для вывода подписей
font = cv2.FONT_HERSHEY_SIMPLEX


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    while True:
        # Получаем видеопоток
        ret, im = cam.read()
        if not ret:
            break
        # Переводим его в ч/б
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # Определяем лица на видео
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        # Перебираем все найденные лица
        for (x, y, w, h) in faces:
            # Получаем id пользователя
            nbr_predicted, coord = recognizer.predict(gray[y:y + h, x:x + w])
            # Рисуем прямоугольник вокруг лица
            cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
            # Если мы знаем id пользователя
            # if (nbr_predicted == 1):
            #     # Подставляем вместо него имя человека
            #     nbr_predicted = 'Ivan'
            if (nbr_predicted == 2):
                # Подставляем вместо него имя человека
                nbr_predicted = 'Anastasiya'
            if (nbr_predicted == 3):
                # Подставляем вместо него имя человека
                nbr_predicted = 'Viktorya'
            if (nbr_predicted == 4):
                # Подставляем вместо него имя человека
                nbr_predicted = 'Aleksey'
            # Добавляем текст к рамке
            cv2.putText(im, str(nbr_predicted), (x, y + h), font, 1.1, (0, 255, 0))


        # Преобразовываем изображение OpenCV в формат JPEG
        ret, jpeg = cv2.imencode('.jpg', im)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    global current_user
    if request.method == 'POST':
        current_user = request.form['name']
        return redirect(url_for('capture_user'))
    return render_template('add_user.html')



@app.route('/capture_user', methods=['GET', 'POST'])
def capture_user():
    if request.method == 'POST':
        user_name = request.form.get('user_name')
        if user_name:
            user_directory = os.path.join(path, 'dataSet', user_name)
            os.makedirs(user_directory, exist_ok=True)

            detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            i = 0
            offset = 50

            conn = sqlite3.connect('faces.db')
            cursor = conn.cursor()
            cursor.execute(
                '''CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, image BLOB)''')

            video = cv2.VideoCapture(0)

            while True:
                ret, im = video.read()
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        i = i + 1
                        img = gray[y - offset:y + h + offset, x - offset:x + w + offset]
                        if img.size != 0:
                            _, buffer = cv2.imencode('.jpg', img)
                            cursor.execute('''INSERT INTO faces (name, image) VALUES (?, ?)''',
                                           (user_name, buffer.tobytes()))
                            conn.commit()

                        cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)

                        if img.shape[0] > 0 and img.shape[1] > 0:
                            cv2.imshow('im', im[y - offset:y + h + offset, x - offset:x + w + offset])
                            cv2.waitKey(100)

                if i > 30:
                    video.release()
                    cv2.destroyAllWindows()
                    conn.close()
                    obech()

                    # После завершения цикла, перенаправляем пользователя на домашнюю страницу
                    return redirect(url_for('index'))

    return render_template('capture_user.html')

def obech():
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
        for row in rows:
            # получаем имя и фото пользователя
            name = row[1]
            image = row[2]

            # проверяем, что name не пустое и состоит только из цифр
            if name and name.isdigit():
                # переводим фото в numpy-массив
                nparr = np.frombuffer(image, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                # определяем лицо на фото
                faces = faceCascade.detectMultiScale(img_np)

                # если лицо найдено
                for (x, y, w, h) in faces:
                    # добавляем его к списку картинок
                    images.append(img_np[y: y + h, x: x + w])
                    # добавляем имя пользователя к списку подписей
                    labels.append(name)

        # закрываем соединение с базой данных
        cursor.close()
        conn.close()
        # возвращаем список картинок и подписей
        return images, labels

    images, labels = get_images_and_labels(databasePath)
    # import tensorflow as tf
    # print(tf.test.is_built_with_cuda())
    # convert all labels to integers
    labels = [int(label) for label in labels]
    print(labels)
    # train the recognizer on the images and labels
    recognizer.train(images, np.array(labels))
    # save the model
    recognizer.write('my_model.yml')
    # destroy all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(debug=True)


