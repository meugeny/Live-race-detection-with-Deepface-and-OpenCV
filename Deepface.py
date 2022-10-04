import cv2
from deepface import DeepFace
font = cv2.FONT_HERSHEY_SIMPLEX

# Загрузка классификатора для детекции лиц
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Подключение к веб-камере
video_capture = cv2.VideoCapture(0)

# Покадровая детекция лица
while True:
    ret, frame = video_capture.read()

    # При помощи модели DeepFace анализируем лицо и получаем список параметров
    # из которых выберем расовую принадлежность
    try:
        res = DeepFace.analyze(frame, actions = ['race'], enforce_detection=False)
    except:
        print("no face")

    # Рисуем прямоугольник вокруг лица с помощью каскадов Хаара
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4, None, (200, 200))
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Пишем текст с вероятностью принадлежности к одной из рас
        asian = int(res['race']['asian'])
        indian = int(res['race']['indian'])
        black = int(res['race']['black'])
        white = int(res['race']['white'])
        middle_eastern = int(res['race']['middle eastern'])
        latino_hispanic = int(res['race']['latino hispanic'])

        cv2.putText(frame, str(f'Asian: {asian} %'), (x-110, y+10), font, 0.5, (255, 255, 255), 2, cv2.LINE_4)
        cv2.putText(frame, str(f'Indian: {indian} %'), (x-110, y+35), font, 0.5, (255, 255, 255), 2, cv2.LINE_4)
        cv2.putText(frame, str(f'Black: {black} %'), (x-110, y+60), font, 0.5, (255, 255, 255), 2, cv2.LINE_4)
        cv2.putText(frame, str(f'White: {white} %'), (x-110, y+85), font, 0.5, (255, 255, 255), 2, cv2.LINE_4)
        cv2.putText(frame, str(f'Arab: {middle_eastern} %'), (x-110, y+110), font, 0.5, (255, 255, 255), 2, cv2.LINE_4)
        cv2.putText(frame, str(f'Latino: {latino_hispanic} %'), (x-110, y+135), font, 0.5, (255, 255, 255), 2, cv2.LINE_4)

    # Полученный результат
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()