import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('Sample_model.hdf5')

st.write("""# COVID-19 DETECTOR WITH CHEST X-RAY""")
st.write("This is a simple image classification web app to predict covid19 with chest x-ray")
st.set_option('deprecation.showfileUploaderEncoding', False)
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
                                                                                                                     
def import_and_predict(image_data, model):
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
                                                                                                                     
if file is None:
   st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
if np.argmax(prediction) == 0:
    st.write("It is a Covid-19!")
elif np.argmax(prediction) == 1:
    st.write("It is a Normal!")
else:
    st.write("It is a Viral Pneumonia!")
    
st.text("Probability (0: Covid-19, 1: Normal, 2: Viral Pneumonia")
st.write(prediction)                                                                                                                     