import time
from collections import deque
import cv2
import imutils
#2

def detect_bounce(ball_prev_y, center, bounce_threshold):
    if ball_prev_y is not None and center is not None:

        if ball_prev_y - center[1] > bounce_threshold:
            return True

    return False

#INFO:GayeNecet => belirli bir top için daire ve ağırlık merkezi çizilir
def draw_ball(frame, c, radius, center):
    cv2.circle(frame, (int(c[0]), int(c[1])), int(radius), (0, 255, 255), 2)
    cv2.circle(frame, center, 5, (0, 0, 255), -1)

#INFO:GayeNecet => default kamera kullan
vs = cv2.VideoCapture(0)
time.sleep(1.0)

#INFO:GayeNecet => toplar için parametreler
balls_params = [
    {"color_lower": (29, 86, 6), "color_upper": (64, 255, 255), "bounce_threshold": 20},  # green ball
    {"color_lower": (0, 100, 100), "color_upper": (10, 255, 255), "bounce_threshold": 20},  # red ball
    # daha fazla top ve eşik değeri girilebilir
]

#  INFO:GayeNecet => her top için sekme tespiti için değişkenler
balls = [{"bounces": 0, "ball_prev_y": None, "dq": deque(maxlen=64)} for _ in balls_params]

# INFO:GayeNecet => istenilen kare hızı girilir opsyoneldir
frame_rate = 50
delay = int(1000 / frame_rate)

while True:
    #INFO:GayeNecet =>kameradan frame okunur
    ret, frame = vs.read()
    if not ret:
        break

    #INFO:GayeNecet => çerçeveyi yeniden boyutlandırır ve bulanıklaştırır
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    #INFO:GayeNecet => HSV renk uzayına dönüştürülür
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    for i, ball_params in enumerate(balls_params):
        #INFO:GayeNecet => renk alt ve üst sınırlarına göre bir mask oluştur.
        mask = cv2.inRange(hsv, ball_params["color_lower"], ball_params["color_upper"])
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        #INFO:GayeNecet => mask deki kontürleri bulur ve topun mevcut (x, y) merkezini başlatır
        cont = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cont = imutils.grab_contours(cont)
        center = None

        if len(cont) > 0:
            #INFO:GayeNecet => mask üzerindeki en büyük kontür seçilir
            c = max(cont, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10:
                #INFO:GayeNecet => çerçeve üzerinde daire ve ağırlık merkezi çizimi yapar
                draw_ball(frame, (x, y), radius, center)

                #INFO:GayeNecet =>topun sekme tespiti
                if detect_bounce(balls[i]["ball_prev_y"], center, ball_params["bounce_threshold"]):
                    #+ bounce count on each bounce
                    balls[i]["bounces"] += 1

                    #INFO:GayeNecet => yeni sıçrama tespit edilince yazdırılır
                    print(f"Ball {i + 1} Bounce detected! Total Bounces: {balls[i]['bounces']}")

                #INFO:GayeNecet => her top için önceki y konumunu günceller
                balls[i]["ball_prev_y"] = center[1]

        #INFO:GayeNecet =>deque güncel merkezi
        balls[i]["dq"].appendleft(center)

        #INFO:GayeNecet =>loop over set of tracked points
        for j in range(1, len(balls[i]["dq"])):
            if balls[i]["dq"][j - 1] is None or balls[i]["dq"][j] is None:
                continue
            # INFO:GayeNecet =>geçmiş ve mevcut merkezler arasında bir çizgi çizilir
            cv2.line(frame, balls[i]["dq"][j - 1], balls[i]["dq"][j], (0, 0, 255), 5)

    cv2.imshow("Frame", frame)

    #INFO:GayeNecet =>'g' tuşu çıkış
    key = cv2.waitKey(delay) & 0xFF
    if key == ord("g"):
        break
#INFO:GayeNecet => tüm pencereleri kapat
cv2.destroyAllWindows()
vs.release()
