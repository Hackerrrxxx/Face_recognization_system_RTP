import tkinter as tk
from tkinter import messagebox
import cv2
import os
from PIL import Image, ImageTk
import numpy as np

# Function to check if the user ID already exists
def check_id_exists(user_id):
    if not os.path.exists("user_info.txt"):
        return False
    with open("user_info.txt", "r") as file:
        for line in file:
            info = line.strip().split(",")
            if info[0] == str(user_id):
                return True
    return False

# Function to create the popup for generating dataset
def generate_dataset_popup():
    popup = tk.Toplevel(window)
    popup.title("Generate Dataset")
    popup.geometry("400x300")
    popup.configure(bg="#a1c4fd")

    def on_closing():
        popup.destroy()

    popup.protocol("WM_DELETE_WINDOW", on_closing)

    l1 = tk.Label(popup, text="Name", font=("Arial", 14), bg="#a1c4fd")
    l1.grid(column=0, row=0, pady=10, padx=10)
    name_entry = tk.Entry(popup, width=30, bd=5)
    name_entry.grid(column=1, row=0, pady=10, padx=10)

    l2 = tk.Label(popup, text="ID", font=("Arial", 14), bg="#a1c4fd")
    l2.grid(column=0, row=1, pady=10, padx=10)
    id_entry = tk.Entry(popup, width=30, bd=5)
    id_entry.grid(column=1, row=1, pady=10, padx=10)

    l3 = tk.Label(popup, text="Age", font=("Arial", 14), bg="#a1c4fd")
    l3.grid(column=0, row=2, pady=10, padx=10)
    age_entry = tk.Entry(popup, width=30, bd=5)
    age_entry.grid(column=1, row=2, pady=10, padx=10)

    l4 = tk.Label(popup, text="Address", font=("Arial", 14), bg="#a1c4fd")
    l4.grid(column=0, row=3, pady=10, padx=10)
    address_entry = tk.Entry(popup, width=30, bd=5)
    address_entry.grid(column=1, row=3, pady=10, padx=10)

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

    b1 = tk.Button(popup, text="Generate Dataset", font=("Arial", 14), bg="#1c92d2", fg="white", command=generate_dataset)
    b1.grid(column=1, row=4, pady=20)

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

            if confidence > 60:
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

# Create the main window
window = tk.Tk()
window.title("Face Recognition System")
window.geometry("800x400")
window.configure(bg="#a1c4fd")

# Set up a background image
bg_image = Image.open("background1.jpg")
bg_image = bg_image.resize((718,404), Image.LANCZOS)
bg_image = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(window, image=bg_image)
bg_label.place(relwidth=1, relheight=1)

# Create buttons with enhanced styling
b1 = tk.Button(window, text="Training", font=("Arial", 20), bg="#f79d00", fg="white", command=train_classifier)
b1.place(relx=0.1, rely=0.3, relwidth=0.2, relheight=0.2)

b2 = tk.Button(window, text="Detect Faces", font=("Arial", 20), bg="#0f9b0f", fg="white", command=detect_face)
b2.place(relx=0.4, rely=0.3, relwidth=0.2, relheight=0.2)

b3 = tk.Button(window, text="Generate Dataset", font=("Arial", 20), bg="#1c92d2", fg="white", command=generate_dataset_popup)
b3.place(relx=0.7, rely=0.3, relwidth=0.2, relheight=0.2)

# Run the main loop
window.mainloop()

