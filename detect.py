import cv2
import numpy as np
import mediapipe as mp

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,   #!!! для глаз
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Camera not opened")

print("✅ Camera opened. ESC — exit")

# Індекси точок 
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_mask(frame, landmarks, eye_ids):
    h, w = frame.shape[:2]
    pts = np.array(
        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_ids],
        np.int32
    )
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    eye = cv2.bitwise_and(frame, frame, mask=mask)

    x, y, w_, h_ = cv2.boundingRect(pts)
    return eye[y:y+h_, x:x+w_], (x, y, w_, h_)

def gaze_from_eye(eye_img):
    if eye_img.size == 0:
        return "UNKNOWN"

    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h, w = thresh.shape
    left_part = thresh[:, :w//2]
    right_part = thresh[:, w//2:]

    left_white = cv2.countNonZero(left_part)
    right_white = cv2.countNonZero(right_part)

    if left_white + right_white == 0:
        return "UNKNOWN"

    ratio = (right_white - left_white) / (left_white + right_white)

    if ratio > 0.15:
        return "LEFT"
    elif ratio < -0.15:
        return "RIGHT"
    else:
        return "CENTER"

# Основной цикл
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    direction = "UNKNOWN"

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        left_eye_img, left_box = eye_mask(frame, lm, LEFT_EYE)
        right_eye_img, right_box = eye_mask(frame, lm, RIGHT_EYE)

        left_dir = gaze_from_eye(left_eye_img)
        right_dir = gaze_from_eye(right_eye_img)

        # финальне рішення
        if left_dir == right_dir:
            direction = left_dir
        else:
            direction = "CENTER"

        def draw_soft_box(frame, box, color=(0,255,0)):
            x, y, w, h = box
            pad_x = int(w * 0.35)   # ширина рамки
            pad_y = int(h * 0.50)   # висота рамки

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)

        draw_soft_box(frame, left_box)
        draw_soft_box(frame, right_box)

    cv2.putText(
        frame, direction, (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5, (0,255,0), 3
    )

    cv2.imshow("Eye Gaze Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()