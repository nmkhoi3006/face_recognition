import cv2
import face_recognition
import pickle
import os

#importing ID img
folder_path = "Images"
img_path_list = os.listdir(folder_path)
img_list = []
IDs = []
for path in img_path_list:
    img_list.append(cv2.imread(os.path.join(folder_path, path)))
    IDs.append(os.path.splitext(path)[0])

#print(IDs)

def encode_img(p_img_list):
    encode_list = []
    for img in p_img_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_locations(img)[0]
        encode_list.append(encode)

    return encode_list

encode_list_known = encode_img(img_list)
print(len(encode_list_known))
encode_list_known_withID = [encode_list_known, IDs]

file = open("Encode_file.p", "wb")
pickle.dump(encode_list_known_withID, file)
file.close()