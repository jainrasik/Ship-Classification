
  # Importing required libraries, obviously
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
from tensorflow.keras.models import load_model

# Loading pre-trained parameters for the cascade classifier
try:
    model =load_model('Modelship.h5')  #Load model
    Classes = ['cruise', 'sub', 'air_car', 'container', 'tanker'] # Emotion that will be predicted
except Exception:
    st.write("Error loading cascade classifiers")
    
def image2emotion(img):
   img_size=224
   final_image=cv2.resize(img,(img_size,img_size))
   final_image=np.expand_dims(final_image,axis=0)
   final_image=final_image/255
   predict=model.predict(final_image)
   label=Classes[np.argmax(predict)]
   label_position=(20,50)
   fmt=cv2.putText(img,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
   st.image(fmt, channels="RGB")



 

def main():
    
    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:blue ;padding:10px">
    <h2 style="color:white;text-align:center;">ship classification app</h2>
    <style>#"This app is created by Rasik Jain" {text-align: center}</style>
    </div>
    </body>
    """
  

    st.markdown(html_temp, unsafe_allow_html=True)
    st.write("This app is created by Rasik Jain")
    st.write("Model built by transfer learning from mobilenetv2")
    st.write("**Instructions while using the APP**")
    st.write('''
                
                1. click on upload image.
                
                2. select the ship image for which you want prediction. 
        
                3. Voila, You got your prediction.  
                
                ''')
    file=st.file_uploader("upload",type=["jpg", "jpeg"])
    if not file:
      st.info("please upload file")
    else:
        img=Image.open(file)
        img_array=np.array(img)
        image2emotion(img_array)
    
    
if __name__ == "__main__":
    main()

