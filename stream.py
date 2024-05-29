import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import os
import subprocess

# Replace with the actual URLs of your ESP32 camera streams
stream_url1 = 'http://192.168.242.228:81/stream'
stream_url2 = 'http://192.168.242.107:81/stream'

# Initialize the video streams
cap1 = cv2.VideoCapture(stream_url1)
cap2 = cv2.VideoCapture(stream_url2)

def update_frame1():
    global cap1, panel1
    try:
        ret, frame = cap1.read()
        if not ret:
            print("Error: Unable to read frame 1")
            panel1.after(10, update_frame1)
            return

        # Convert the frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the frame to ImageTk format
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the panel with the new image
        panel1.imgtk = imgtk
        panel1.config(image=imgtk)
        
    except Exception as e:
        print(f"Exception occurred in frame 1 update: {e}")
    finally:
        panel1.after(10, update_frame1)

def update_frame2():
    global cap2, panel2
    try:
        ret, frame = cap2.read()
        if not ret:
            print("Error: Unable to read frame 2")
            panel2.after(10, update_frame2)
            return

        # Convert the frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the frame to ImageTk format
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the panel with the new image
        panel2.imgtk = imgtk
        panel2.config(image=imgtk)
        
    except Exception as e:
        print(f"Exception occurred in frame 2 update: {e}")
    finally:
        panel2.after(10, update_frame2)

def get_new_filename(base_name, extension):
    i = 1
    new_filename = f"{base_name}{extension}"
    while os.path.exists(new_filename):
        new_filename = f"{base_name}_{i}{extension}"
        i += 1
    return new_filename

def save_frame_cali():
    global cap1, cap2
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if ret1 and ret2:
        left_image_path = get_new_filename('captures/left/left', '.jpg')
        right_image_path = get_new_filename('captures/right/right', '.jpg')
        cv2.imwrite(left_image_path, frame1)
        cv2.imwrite(right_image_path, frame2)
        messagebox.showinfo("Info", f"Frames saved successfully as {left_image_path} and {right_image_path}!")
    else:
        messagebox.showerror("Error", "Failed to save frames")

def save_frame_meas():
    global cap1, cap2
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if ret1 and ret2:
        
        cv2.imwrite("left.jpg", frame1)
        cv2.imwrite("right.jpg", frame2)
        messagebox.showinfo("Info", f"Frames saved successfully as left.jpg and right.jpg!")
    else:
        messagebox.showerror("Error", "Failed to save frames")

def run_calibration():
    try:
        subprocess.run(["python", "scripts/calibration.py"], check=True)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run calibration: {e}")

# Create the main window
root = tk.Tk()
root.title("ESP32 Video Streams")

# Create panels to display the video streams
panel1 = tk.Label(root)
panel1.pack(side="left", padx=10, pady=10)
panel2 = tk.Label(root)
panel2.pack(side="right", padx=10, pady=10)

# Create a button to save frames from both streams
save_button = tk.Button(root, text="enregistrer pour calibrer", command=save_frame_cali)
save_button.pack()

# Create a button to save frames from both streams
save_button = tk.Button(root, text="enregistrer pour mesurer", command=save_frame_meas)
save_button.pack()

# Create a button to run calibration
calibration_button = tk.Button(root, text="calibrer et mesurer", command=run_calibration)
calibration_button.pack()

# Start updating the frames in separate threads
thread1 = threading.Thread(target=update_frame1)
thread2 = threading.Thread(target=update_frame2)
thread1.daemon = True
thread2.daemon = True
thread1.start()
thread2.start()

# Start the Tkinter main loop
root.mainloop()

# Release the video captures when the window is closed
cap1.release()
cap2.release()
cv2.destroyAllWindows()
