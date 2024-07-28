import cv2
import os
import pickle
import face_recognition

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

img_bkground = cv2.imread("Resources/background.png")

folder_modes_path = "Resources/Modes"
modes_path_list = os.listdir(folder_modes_path)
img_modes_list = []
for path in modes_path_list:
    img_modes_list.append(cv2.imread(os.path.join(folder_modes_path, path)))


file = open("Encode_file.p", "rb")
encode_list_known_withID = pickle.load(file)
file.close()
encode_list_known, IDs = encode_list_known_withID


while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    curr_frame = face_recognition.face_locations(imgS)
    encode_curr_frame = face_recognition.face_encodings(imgS, curr_frame)

    img_bkground[162:162+480, 55:55+640] = img
    img_bkground[44:44+633, 808:808+414] = img_modes_list[1]

    for encode_face in encode_curr_frame:
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_diss = face_recognition.face_distance(encode_list_known, encode_face)

        print("matches: ", matches)

    cv2.imshow("face_recognition", img_bkground)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()