import cv2
from rembg import remove
import easygui
from PIL import Image
import os
import numpy as np

def remove_background_rembg(image_path):
    input_image = Image.open(image_path)
    output_image = remove(input_image)
    output_np = np.array(output_image)

    if output_np.shape[2] == 4 and np.all(output_np[:, :, 3] == 255):
        return cv2.imread(image_path)
    else:
        return output_np

image_path = easygui.fileopenbox(title='Select image file')
nobg_image=remove_background_rembg(image_path)
output_path = os.path.join(os.path.dirname(__file__), "bgremoved.jpg")
cv2.imwrite(output_path, nobg_image)