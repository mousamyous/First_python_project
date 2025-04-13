# Import required libraries
import tkinter as tk                        # Tkinter for GUI creation
from tkinter import filedialog              # Module for opening file dialog windows
import cv2                                  # OpenCV for computer vision and face detection
from PIL import Image, ImageTk              # Pillow for converting images to a format Tkinter can display

# Define the function to select an image and perform face detection
def select_image():
    # Opens a file selection dialog and returns the selected file's path as a string
    path = filedialog.askopenfilename()
    
    # Check if the user selected a file (if a file is selected, the path is not an empty string)
    if len(path) > 0:
        # Read the image file from the given path using OpenCV
        image = cv2.imread(path)
        
        # Convert the image from BGR (OpenCV's default) to grayscale for the face detection process
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load a pre-trained Haar cascade classifier for frontal face detection
        # The 'haarcascade_frontalface_default.xml' file should be located in the same directory
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
        # Use the detectMultiScale method to find faces in the grayscale image
        # scaleFactor controls the image size reduction at each image scale
        # minNeighbors defines how many neighbors each candidate rectangle should have to retain it
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        # Draw rectangles around detected faces on the original (colored) image
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle with thickness of 2
        
        # Convert the image from BGR to RGB format because Pillow expects images in RGB order
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert the NumPy array image into a PIL Image object
        image_pil = Image.fromarray(image)
        
        # Convert the PIL image into an ImageTk object to display in Tkinter
        image_tk = ImageTk.PhotoImage(image_pil)
        
        # Update the panel label to show the resulting image with detected faces
        panel.config(image=image_tk)
        panel.image = image_tk  # Keep a reference to avoid Python's garbage collection

# Create the main Tkinter window
root = tk.Tk()
root.title("Face Detection GUI")          # Set the window title
root.geometry("800x600")                   # Set the window size

# Create a button that lets the user select an image file, triggering the face detection process
btn = tk.Button(root, text="اختر الصورة", command=select_image)
# Place the button at the bottom of the window with padding and expand to fill available space
btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

# Create a label to serve as a panel where images will be displayed
panel = tk.Label(root)
# Pack the label at the top, allowing it to expand to fill available space
panel.pack(side="top", fill="both", expand="yes")

# Start the Tkinter event loop to listen for events and run the application
root.mainloop()
