# -*- coding: utf-8 -*-
# MLP for Pima Indians Dataset Serialize to JSON and HDF5
import numpy as np
from keras.preprocessing import image
##from keras.models import Sequential
##from keras.layers import Dense
from keras.models import model_from_json
##import os
import cv2
import matplotlib.pyplot as plt


json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")


label=["Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___Healthy",
       "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
       "Corn_(maize)___Healthy","Corn_(maize)___Northern_Leaf_Blight","Grape___Black_rot",
       "Grape___Esca_(Black_Measles)","Grape___Healthy","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
       "Potato___Early_blight","Potato___Late_blight","Potato___Healthy","Tomato___Bacterial_spot",
       "Tomato___Early_blight","Tomato___Healthy","Tomato___Late_blight","Tomato___Leaf_Mold",
       "Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
       "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus"]

test_image = image.load_img('im_for_testing_purpose/t.septleafmold.jpg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_model.predict(test_image)
print(result)
#print(result)
fresult=np.max(result)
label2=label[result.argmax()]
print(label2)

# Load the image using OpenCV
test_image = cv2.imread('im_for_testing_purpose/t.septleafmold.jpg')
# Convert image from BGR to RGB
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

# Display the input image
plt.imshow(test_image)
plt.axis('off')
plt.title('Input Image')
plt.show()

# Display the predicted label
plt.figure()
plt.barh(label, result[0])
plt.xlabel('Probability')
plt.ylabel('Classes')
plt.title('Predicted Probabilities')
plt.show()

# Display the predicted probabilities as a histogram
plt.figure()
plt.hist(label, bins=len(label), weights=result[0], edgecolor='black')
plt.xticks(rotation=90)
plt.xlabel('Classes')
plt.ylabel('Probability')
plt.title('Predicted Probabilities (Histogram)')
plt.show()

# Display the predicted probabilities as a line plot
plt.figure()
plt.plot(label, result[0], marker='o')
plt.xticks(rotation=90)
plt.xlabel('Classes')
plt.ylabel('Probability')
plt.title('Predicted Probabilities (Line Plot)')
plt.grid(True)
plt.show()



