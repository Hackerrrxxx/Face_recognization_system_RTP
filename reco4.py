import tkinter as tk
from tkinter import messagebox
import cv2
import os
from PIL import Image
import numpy as np

# Create the main window
window = tk.Tk()
window.title("Face Recognition System")
window.config(background="Blue")

def check_id_exists(user_id):
    if not os.path.exists("user_info.txt"):
        return False
    with open("user_info.txt", "r") as file:
        for line in file:
            info = line.strip().split(",")
            if info[0] == str(user_id):
                return True
    return False

def generate_dataset_popup():
    popup = tk.Toplevel(window)
    popup.title("Generate Dataset")

    l1 = tk.Label(popup, text="Name", font=("Algerian", 20))
    l1.grid(column=0, row=0)
    name_entry = tk.Entry(popup, width=50, bd=5)
    name_entry.grid(column=1, row=0)

    l2 = tk.Label(popup, text="ID", font=("Algerian", 20))
    l2.grid(column=0, row=1)
    id_entry = tk.Entry(popup, width=50, bd=5)
    id_entry.grid(column=1, row=1)

    l3 = tk.Label(popup, text="Age", font=("Algerian", 20))
    l3.grid(column=0, row=2)
    age_entry = tk.Entry(popup, width=50, bd=5)
    age_entry.grid(column=1, row=2)

    l4 = tk.Label(popup, text="Address", font=("Algerian", 20))
    l4.grid(column=0, row=3)
    address_entry = tk.Entry(popup, width=50, bd=5)
    address_entry.grid(column=1, row=3)

    def generate_dataset():
        name = name_entry.get()
        user_id = id_entry.get()
        age = age_entry.get()
        address = address_entry.get()
        
        if name == "" or user_id == "" or age == "" or address == "":
            messagebox.showinfo('Result', 'Please provide complete details of the user')
            return
        if check_id_exists(user_id):
            messagebox.showinfo('Result', 'ID already exists. Please use a different ID.')
            return

        # Save user information to user_info.txt
        user_info = f"{user_id},{name},{age},{address}\n"
        with open("user_info.txt", "a") as file:
            file.write(user_info)

        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        if face_classifier.empty():
            messagebox.showinfo('Result', 'Error loading face classifier XML file.')
            return
        
        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                return None
            for (x, y, w, h) in faces:
                cropped_face = img[y:y + h, x:x + w]
            return cropped_face

        cap = cv2.VideoCapture(0)
        img_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            cropped_face = face_cropped(frame)
            if cropped_face is not None:
                img_id += 1
                face = cv2.resize(cropped_face, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                if not os.path.exists("data"):
                    os.makedirs("data")
                file_name_path = "data/user." + str(user_id) + "." + str(img_id) + ".jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Cropped face", face)

            if cv2.waitKey(1) == 13 or int(img_id) == 200:  # Enter key or 200 images
                break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result', 'Generating dataset completed!!')
        popup.destroy()

    b1 = tk.Button(popup, text="Generate Dataset", font=("Algerian", 20), bg="pink", fg="black", command=generate_dataset)
    b1.grid(column=1, row=4)

# Function to train the classifier
def train_classifier():
    data_dir = "data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jpg")]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
        
        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    messagebox.showinfo("Result", "Training dataset completed")

b2 = tk.Button(window, text="Training", font=("Algerian", 20), bg="orange", fg="red", command=train_classifier)
b2.grid(column=0, row=0)

# Function to detect faces
def detect_face():
    def get_user_info(user_id):
        with open("user_info.txt", "r") as file:
            for line in file:
                info = line.strip().split(",")
                if info[0] == str(user_id):
                    return info[1], info[2], info[3]
        return None, None, None

    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            id, pred = clf.predict(gray_img[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            if confidence > 65:
                name, age, address = get_user_info(id)
                cv2.putText(img, f"Name: {name}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                cv2.putText(img, f"Age: {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                cv2.putText(img, f"Address: {address}", (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

        return img

    # Loading classifier
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read()
        if not ret:
            break

        img = draw_boundary(img, faceCascade, 1.3, 6, (255, 255, 255), "Face", clf)
        cv2.imshow("Face Detection", img)

        if cv2.waitKey(1) == 13:
            break

    video_capture.release()
    cv2.destroyAllWindows()

b3 = tk.Button(window, text="Detect Faces", font=("Algerian", 20), bg="green", fg="orange", command=detect_face)
b3.grid(column=1, row=0)

b4 = tk.Button(window, text="Generate Dataset", font=("Algerian", 20), bg="pink", fg="black", command=generate_dataset_popup)
b4.grid(column=2, row=0)

# Set the window geometry and start the main loop
window.geometry("800x200")
window.mainloop()

