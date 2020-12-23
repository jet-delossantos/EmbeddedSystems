'''
Accessing webcam or camera using VideoCapture in OpenCV
'''

import cv2
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image



cam = cv2.VideoCapture(0)
img_dir = "C:/Users/Jet/Desktop/image_classify/dataset/Seed/images/"
img_path = "C:/Users/Jet/Desktop/image_classify/dataset/Seed/images/0/"
train_path = "C:/Users/Jet/Desktop/image_classify/dataset/Seed/train/"
model_path = "C:/Users/Jet/Desktop/image_classify/dataset/Model/tf_model_sinigangmix.h5"
cv2.namedWindow("test")
class_list = ['Ampalaya', 'Kangkong', 'Okra', 'Raddish', 'Sitao']
img_counter = 0

while True:
    model = load_model(model_path)
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "test_seed_{}.png".format(img_counter)
        da_path = img_path+img_name
        img = image.load_img(da_path, target_size=(300,300))
        trytry = image.img_to_array(img)
        trytry = np.expand_dims(trytry, axis=0)
        trytry = trytry/255
        gg = model.predict(trytry)
        ff = np.argmax(gg)
        img.show()
        print("Predicted label: " + class_list[ff])

        img_counter += 1

cam.release()
cv2.destroyAllWindows()