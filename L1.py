import time
from collections import deque
import cv2
import imutils

#1

# INFO:GayeNecet =>üst ve alt renk değerleri HSV
color_Lower = (29, 86, 6)
color_Upper = (64, 255, 255)
dq = deque(maxlen=64)   #Deque, çift taraflı bir kuyruk yapısıdır ve hem başından hem de sonundan eleman eklemeye ve çıkarmaya izin verir.

# INFO:GayeNecet =>default Kamera
vs = cv2.VideoCapture(0)

# INFO:GayeNecet =>1sec sonra aç
time.sleep(1.0)

# INFO:GayeNecet =>sekme tespiti için veriler
bounce_threshold = 20   # eğer topun yükseklik değişimi bu eşik değeri kadar veya daha fazla ise, bir sıçrama olarak kabul edilir
bounces = 0 # topun kaç kez sıçradığını takip eder
ball_prev_y = None  # önceki konumu alır, en başta konum none alınır

# INFO:GayeNecet => istenilen kare hızı girilir opsyoneldir
frame_rate = 50
delay = int(1000 / frame_rate)

while True:
    frame = vs.read()
    frame = frame[1]

    if frame is None:
        break

    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, color_Lower, color_Upper)
    mask = cv2.erode(mask, None, iterations=2)
    #x2 kere serezyonu azalt
    mask = cv2.dilate(mask, None, iterations=2) #x2 kere
    # INFO:GayeNecet =>Genişletme işlemi, beyaz bölgenin genişletilmesine ve siyah bölgenin küçülmesine neden olur.
    # INFO:GayeNecet =>bu işlem, özellikle erozyon işlemi sonrasında küçültülen nesnelerin boyutunu geri getirmek veya nesneler arasındaki boşlukları doldurmak için kullanılır.

    # INFO:GayeNecet =>mask deki kontürleri bul ve topun mevcut (x, y) merkezini başlat
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        # INFO:GayeNecet => mask deki en büyük konturu bu
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            # INFO:GayeNecet =>çerçeveye daire ve ağırlık merkezini çiz
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # INFO:GayeNecet =>topun sekme sayısını tespit etme
            if ball_prev_y is not None and center is not None:
                if ball_prev_y - center[1] > bounce_threshold:
                    bounces += 1
                    print("Bounce detected! Total Bounces:", bounces)

            ball_prev_y = center[1]

    dq.appendleft(center)

    # INFO:GayeNecet =>izlenen noktalar kümesi üzerinde döngü oluşturur
    for i in range(1, len(dq)):
        if dq[i - 1] is None or dq[i] is None:
            continue

        cv2.line(frame, dq[i - 1], dq[i], (0, 0, 255), 5)  #thickness

    cv2.imshow("Frame", frame)

    # INFO:GayeNecet =>'g' tuşu çıkış
    key = cv2.waitKey(delay) & 0xFF

    if key == ord("g"):
        break

# INFO:GayeNecet => Formların hepsini kapatır.
cv2.destroyAllWindows()
