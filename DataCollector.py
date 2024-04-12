import random
import cv2
import os

data_dir = 'data/'
basic_bag = [0,1,2,3,4,5,6,7,8,9]
grab_bag = []


# Check if all directories exist
if not os.path.exists(f'{data_dir}'):
    os.makedirs(f'{data_dir}')
    
for i in range(11):
    if not os.path.exists(f'data/{i}'):
        os.makedirs(f'data/{i}') if i != 10 else os.makedirs(f'data/10') # This is a cringe line of code


#Grab the camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture(0)

while True:
    

    if len(grab_bag) == 0:
        grab_bag = basic_bag.copy()
        # shuffle the grab bag
        random.shuffle(grab_bag)

    number = grab_bag.pop() # grab a number from the grab bag    
            
    print("NUMBER:", number)
    user_input = input("Press enter to take a picture, q to quit, or p to change person: ")
    
    if user_input == 'q':
        running = False
        cap.release()
        exit()
    
    elif user_input == '':
        ret, frame = cap.read()
        
        if ret:
            #Generate random 6 character string a-z A-Z 0-9
            image_name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))
            
            #check if the image already exists
            while os.path.exists(f"{data_dir}/{number}/{image_name}.png"):
                image_name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))
            
            print(f"Capturing image {image_name}")
            filename = f"{data_dir}/{number}/{image_name}.png"
            cv2.imwrite(filename, frame)
            
        else:
            print("error capturing image")