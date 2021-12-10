from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import os
import requests
import cv2
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('modelfruit.h5',compile=False)

def preprocess(img,input_size):
    nimg = cv2.resize(img, (100, 100))
    img_arr = (np.array(nimg))/255
    return img_arr

def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)

def predict_label(img_path):
    input_size = (100,100)
    channel = (3,)
    input_shape = input_size + channel
    labels = ['Cauliflower' 'Pineapple' 'Pitahaya Red' 'Kiwi' 'Potato White'
 'Mango Red' 'Pineapple Mini' 'Maracuja' 'Papaya' 'Banana Red' 'Physalis'
 'Nectarine Flat' 'Cherry 2' 'Pear Forelle' 'Grape White 2' 'Tomato 2'
 'Grape Blue' 'Quince' 'Apple Crimson Snow' 'Pear 2' 'Potato Sweet' 'Pear'
 'Peach 2' 'Lemon' 'Watermelon' 'Cocos' 'Apple Golden 2' 'Tomato Heart'
 'Corn Husk' 'Physalis with Husk' 'Guava' 'Tomato 1' 'Dates' 'Clementine'
 'Cactus fruit' 'Melon Piel de Sapo' 'Apricot' 'Apple Red Yellow 1'
 'Carambula' 'Huckleberry' 'Strawberry Wedge' 'Banana Lady Finger'
 'Pear Monster' 'Plum 2' 'Tomato Maroon' 'Cantaloupe 2' 'Chestnut'
 'Tangelo' 'Nut Forest' 'Apple Granny Smith' 'Grape Pink' 'Granadilla'
 'Grape White' 'Apple Pink Lady' 'Grape White 3' 'Potato Red Washed'
 'Peach' 'Apple Golden 3' 'Cherry Rainier' 'Kohlrabi' 'Plum 3'
 'Pomelo Sweetie' 'Nectarine' 'Pear Williams' 'Tomato 3' 'Eggplant'
 'Cantaloupe 1' 'Grapefruit Pink' 'Apple Red 1' 'Tomato not Ripened'
 'Blueberry' 'Raspberry' 'Mango' 'Onion Red' 'Cucumber Ripe 2' 'Limes'
 'Onion White' 'Tomato 4' 'Strawberry' 'Hazelnut' 'Orange'
 'Apple Red Delicious' 'Redcurrant' 'Pepper Green' 'Beetroot'
 'Pear Kaiser' 'Peach Flat' 'Grape White 4' 'Banana' 'Apple Red 2'
 'Walnut' 'Mandarine' 'Apple Red 3' 'Apple Red Yellow 2' 'Tamarillo' 'Fig'
 'Apple Braeburn' 'Tomato Cherry Red' 'Pepino' 'Onion Red Peeled'
 'Pepper Red' 'Pear Abate' 'Nut Pecan' 'Tomato Yellow' 'Cherry 1'
 'Potato Red' 'Ginger Root' 'Cherry Wax Black' 'Plum' 'Kaki' 'Pear Red'
 'Apple Golden 1' 'Pepper Orange' 'Kumquats' 'Cherry Wax Yellow' 'Lychee'
 'Mulberry' 'Pear Stone' 'Pepper Yellow' 'Grapefruit White' 'Lemon Meyer'
 'Cherry Wax Red' 'Mangostan' 'Avocado ripe' 'Avocado' 'Passion Fruit'
 'Salak' 'Cucumber Ripe' 'Pomegranate' 'Rambutan' 'Corn']
    im = cv2.imread(img_path)
    x = preprocess(im,input_size)
    x = reshape([x])
    y = model.predict(x)
    return labels[np.argmax(y)], np.max(y)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = predict_label(img_path)
            return render_template('index.html', uploaded_image=image.filename, prediction=prediction)

    return render_template('index.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)