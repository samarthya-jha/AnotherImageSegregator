from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os


model = tf.keras.models.load_model('model.h5')
app = Flask(__name__)


@app.route('/')
def upload_f():
	return render_template('upload.html')

def finds():
	test_datagen = ImageDataGenerator(rescale = 1./255)
	vals = ['Melon', 'Blahaj'] 
	test_dir = 'uploaded'
	test_generator = test_datagen.flow_from_directory(
			test_dir,
			target_size =(200, 200),
			class_mode ='binary',
			batch_size = 1)

	pred = model.predict_generator(test_generator)
	print(pred)
	score = float("{:.2f}".format(pred[0][0]*100))
	print(score)
	return str("Blahaj: "+str(100.0-score)+"% Melon: "+str(score)+"%")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		f.save('uploaded\images\\'+f.filename)
		val = finds()
		os.remove('uploaded\images\\'+f.filename)
		return render_template('pred.html', ss = val)

if __name__ == '__main__':
	app.run()
