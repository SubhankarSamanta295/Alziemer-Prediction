from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2

app = Flask(__name__)

dic = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
model = load_model('alzeimer_gaussian.h5')

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = './images/'+imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(96, 96))
    image = img_to_array(image)
    image = cv2.GaussianBlur(image,(3,3),0)
    image = image.reshape((-1, image.shape[0], image.shape[1], image.shape[2]))
    predictions = model.predict(image) 
    predictions = np.argmax(predictions,axis=1)
    label = dic[int(np.round(predictions))]

    

    return render_template('index.html', prediction = label)

if __name__ =='__main__':
	#app.debug = True
	app.run(port = 1000, debug = True)
    
