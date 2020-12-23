import numpy as np
import os
from keras.preprocessing import image
import pandas as pd
import tensorflow as tf
from flask import Flask,request,render_template
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

global graph
sess = tf.Session()
graph = tf.get_default_graph()

app = Flask(__name__)
set_session(sess)

model = load_model('braintumor.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

MODEL_PATH = './best_weights.hdf5'




def model_predict(img, model):
    img = img.resize((224, 224))

    
    x = image.img_to_array(img)
    
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
       df = pd.read_csv('patient.csv')
       f = request.files['image']
       name=request.form['name']
       age=request.form['age']

       basepath = os.path.dirname(__file__)
       file_path = os.path.join(
           basepath, 'uploads', secure_filename(f.filename))
       f.save(file_path)
       img = image.load_img(file_path, target_size=(64,64))
       x = image.img_to_array(img)
       x= np.expand_dims(x , axis=0)

       with graph.as_default():
           set_session(sess)
           prediction = model.predict_classes(x)[0][0]
       print(prediction)

       if prediction==0:
           text = "You are perfectly fine"
           inp= "No tumor"
       else:
           text = "You are infected! Please Consult Doctor" 
           inp = "Tumor detected"

       df= df.append(pd.DataFrame({'name':[name],'age':[age],'status':[inp]}),ignore_index=True)
       df.to_csv('patient.csv',index=False)
       return text          