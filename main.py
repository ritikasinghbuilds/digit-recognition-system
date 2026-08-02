#import essential libraries
import streamlit as st
import tensorflow as tf 
from PIL import Image
import numpy as np
import cv2
#load model
model=tf.keras.models.load_model("ann_model.keras")
#specify the title
st.title("Digit Recognition System using ANN")
#upload file
uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    #open Image
    image=Image.open(uploaded_file).convert('L')
    #L stands for luminance, image will contain only shades of gray
    #display image
    st.image(image,caption="Uploaded Image",width=150)
    #convert image into array
    img=np.array(image)
    #resize to 28x28
    img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)

    # Invert colors
    img = 255 - img

    # Normalize
    img = img.astype("float32") / 255.0
    #reshape for prediction 
    img=img.reshape (1,28,28)
    
    prediction=model.predict(img)
    predicted_digit=np.argmax(prediction)
    st.success(f"Expected Digit={predicted_digit}")
    
