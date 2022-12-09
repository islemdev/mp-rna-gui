from flask import Flask
from flask import render_template, request
import os
from werkzeug.utils import secure_filename
from tensorflow import keras
import tensorflow as tf

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    print(request.files)
    pred_class = None
    if "image" in request.files:
        file = request.files['image']
        if file.name !="":
            print(file)
            filename = secure_filename(file.filename)
            dest = os.path.join("", filename)
            file.save(dest)
            print(dest)
            model = keras.models.load_model('mammo_model')
            img = keras.preprocessing.image.load_img(
                dest, target_size=(128, 128),
                color_mode="grayscale"
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis
            predictions = model.predict(img_array, verbose=0)
            index = predictions[0].argmax()
            class_names = ["BEN", "CAN", "NOR"]
            pred_class = class_names[index]
    print(pred_class)
    return render_template('uploader.html', pred_class=pred_class)