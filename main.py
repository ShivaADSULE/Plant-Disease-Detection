# GUI IMPORTS
import kivy
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.properties import ObjectProperty
from tkinter import filedialog
from tkinter import *

# PDD MODEL IMPORTS
import time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
from PIL import Image
classifications =[
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy"
]
N = len(classifications)
MODEL = load_model("notebook/apple_final.h5")

def read_image(file_path):
    image = load_img(file_path, target_size=(64, 64))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255
    return image

def test_image(path):
    image = read_image(path)
    time.sleep(.5)
    preds = [round(i,4) for i in MODEL.predict_proba(image)[0]]
    return classifications[preds.index(max(preds))]


class Root(GridLayout):

    def show_load(self):
        root = Tk()
        root.iconify()
        root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        print (root.filename)
        self.image_path=root.filename
        self.ids["myimage"].source=root.filename
        self.ids["myimage"].reload()
        root.destroy()
    def show_predict(self):
        try:
            text = test_image(self.image_path)
            self.ids["the_info"].text=text
        except:
            self.ids["the_info"].text="ERROR : PATH NOT GIVEN "


class PDD(App):
    pass

if __name__ == '__main__':
    PDD().run()