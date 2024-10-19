import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image

model = Sequential()
model.load_weights('weights/cat_vs_dog_weights.weights.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  
    prediction = model.predict(img_array)
    prob = prediction.flatten()[0]  
    label = "Dog" if prob > 0.5 else "Cat"
    plt.imshow(img)
    plt.title(f"Prediction: {label} (Confidence: {prob:.2f})")
    plt.axis('off')
    plt.show()

def predict_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            print(f"Processing image: {img_path}")
            predict_image(img_path)

def predict_frame(frame):
    img_array = cv2.resize(frame, (64, 64)) 
    img_array = np.expand_dims(img_array, axis=0) / 255.0  
    prediction = model.predict(img_array)
    prob = prediction.flatten()[0]  
    label = "Dog" if prob > 0.65 else "Cat"
    return label, prob

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        label, prob = predict_frame(frame)
            
        text = f"{label} ({prob:.2f})"
        
        ext = f"{label} ({prob:.2f})"
        font_scale = 3
        thickness = 4
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2 
        text_y = (frame.shape[0] + text_size[1]) // 2  

        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
        
        frame = cv2.resize(frame, (640,640))
        cv2.imshow('Video Prediction', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    
    #Inferir em muitas imagens 
    #predict_images_in_folder(folder_path="path")
    
    #Inferir em uma imagem
    #predict_image(img_path="path")
    
    #Inferir em video
    process_video(video_path="videos/6853904-uhd_2160_4096_25fps.mp4")
    
    
