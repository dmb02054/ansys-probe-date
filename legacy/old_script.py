import cv2
import numpy as np
import pyautogui

# 사용자 입력: 최소/최대 값
min_val = float(input("최소값 입력: "))
max_val = float(input("최대값 입력: "))

def map_color_to_value(bgr):
    # 예시: 빨강 채널 기준 선형 보간
    r, g, b = bgr[2], bgr[1], bgr[0]
    ratio = r / 255.0
    return min_val + (max_val - min_val) * ratio

# 1. 화면 한 번 캡처
screenshot = pyautogui.screenshot()
frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# 2. 마우스 이벤트 정의
def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 주변 5x5 영역 평균 색상
        h, w, _ = frame.shape
        region = frame[max(0,y-2):min(h,y+3), max(0,x-2):min(w,x+3)]
        avg_color = region.mean(axis=(0,1))  # BGR 평균
        value = map_color_to_value(avg_color)

        # 결과 오버레이
        temp = frame.copy()
        cv2.putText(temp, f"({x},{y}) → {value:.2f}", (x+10,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.imshow("Captured Screen", temp)
        print(f"클릭 좌표 ({x},{y}) → 평균 색상 {avg_color.astype(int)} → 값 {value:.2f}")

# 3. 캡처된 화면 띄우기
cv2.imshow("Captured Screen", frame)
cv2.setMouseCallback("Captured Screen", mouse_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
