import cv2
import numpy as np
from model import create_model, IMG_WIDTH, IMG_HEIGHT
import tensorflow as tf

img = cv2.imread("raw/EPSON009.JPG")
(h, w) = img.shape[:2]
image_size = h * w
mser = cv2.MSER_create()
mser.setMaxArea(image_size // 2)
mser.setMinArea(10)

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(grey, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

(contours, h) = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

instructions = {}
rect_img = img.copy()
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w < 5 or h < 5:
        continue
    cv2.rectangle(rect_img, (x, y), (x + w, y + h), color=(255, 0, 255), thickness=1)
    resized = cv2.resize(img[y : y + h, x : x + w], (IMG_WIDTH, IMG_HEIGHT))
    instructions[x] = resized

cv2.imwrite("test-2.jpg", rect_img)

instructions = np.array(list(map(lambda kv: kv[1], sorted(instructions.items(), key=lambda kv: kv[0]))))

model = create_model()
model.load_weights("./weights/weights")

probab_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

class_names = [
    "]",
    ",",
    "",
    "<",
    "-",
    "[",
    ".",
    "+",
    ">",
]
predictions = probab_model.predict(instructions)
predictions = np.argmax(predictions, axis=1)

program = "".join(class_names[i] for i in predictions)
print(program)
