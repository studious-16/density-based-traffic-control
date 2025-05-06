from tkinter import *
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
import cv2
import numpy as np
from CannyEdgeDetector import *
import matplotlib.image as mpimg

main = Tk()
main.title("Density Based Smart Traffic Control System")
main.geometry("900x600")
main.configure(bg='white')

# Global vars
filename = None
refrence_pixels = 0
sample_pixels = 0

style = ttk.Style()
style.configure("TButton", font=('Segoe UI', 12), padding=6)
style.configure("TLabel", font=('Segoe UI', 12))

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def uploadTrafficImage():
    global filename
    filename = filedialog.askopenfilename(initialdir="images")
    if filename:
        path_label.config(text=filename.split('/')[-1])

def visualize(imgs):
    j = 0
    plt.figure(figsize=(10, 5))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt.subplot(1, 2, i+1)
        plt.title('Sample Image' if j == 0 else 'Reference Image')
        plt.imshow(img, cmap='gray')
        j += 1
    plt.show()

def applyCanny():
    imgs = []
    img = mpimg.imread(filename)
    img = rgb2gray(img)
    imgs.append(img)
    edge = CannyEdgeDetector(imgs, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
    imgs = edge.detect()
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
    cv2.imwrite("gray/test.png", img)
    temp = [mpimg.imread('gray/test.png'), mpimg.imread('gray/refrence.png')]
    visualize(temp)

def pixelcount():
    global refrence_pixels, sample_pixels
    sample_img = cv2.imread('gray/test.png', cv2.IMREAD_GRAYSCALE)
    sample_pixels = np.sum(sample_img == 255)

    ref_img = cv2.imread('gray/refrence.png', cv2.IMREAD_GRAYSCALE)
    refrence_pixels = np.sum(ref_img == 255)

    messagebox.showinfo("Pixel Counts",
        f"Sample White Pixels: {sample_pixels}\nReference White Pixels: {refrence_pixels}")

def timeAllocation():
    if refrence_pixels == 0:
        messagebox.showwarning("Error", "Reference pixel count is zero.")
        return
    avg = (sample_pixels / refrence_pixels) * 100
    if avg >= 90:
        time = 60
        status = "Traffic is very high"
    elif avg > 85:
        time = 50
        status = "Traffic is high"
    elif avg > 75:
        time = 40
        status = "Traffic is moderate"
    elif avg > 50:
        time = 30
        status = "Traffic is low"
    else:
        time = 20
        status = "Traffic is very low"
    messagebox.showinfo("Green Signal Allocation Time",
                        f"{status} - Green signal time: {time} secs")

def detectAmbulance():
    try:
        img = cv2.imread(filename, 0)
        template = cv2.imread('images/ambulance_template.jpeg', 0)
        if template is None:
            messagebox.showerror("Error", "Ambulance template not found!")
            return

        if template.shape[0] > img.shape[0] or template.shape[1] > img.shape[1]:
            scale = min(img.shape[0] / template.shape[0], img.shape[1] / template.shape[1])
            new_size = (int(template.shape[1] * scale), int(template.shape[0] * scale))
            template = cv2.resize(template, new_size, interpolation=cv2.INTER_AREA)

        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title("Traffic Image")
        plt.subplot(1, 2, 2)
        plt.imshow(result, cmap='hot')
        plt.title("Template Match Heatmap")
        plt.colorbar()
        plt.show()

        if max_val >= 0.6:
            messagebox.showinfo("Ambulance Detection", "🚑 Ambulance Detected!")
        else:
            messagebox.showinfo("Ambulance Detection", "No Ambulance Detected.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# --- GUI Layout ---

ttk.Label(main, text="Density Based Smart Traffic Control System", font=("Segoe UI", 16, "bold"), foreground="navy").pack(pady=15)

frame = ttk.Frame(main)
frame.pack(pady=20)

ttk.Button(frame, text="Upload Traffic Image", command=uploadTrafficImage).grid(row=0, column=0, padx=10, pady=5, sticky=W)
path_label = ttk.Label(frame, text="No file selected", foreground="gray")
path_label.grid(row=0, column=1, padx=10)

ttk.Button(frame, text="Apply Canny Edge Detection", command=applyCanny).grid(row=1, column=0, padx=10, pady=5, sticky=W)
ttk.Button(frame, text="Count White Pixels", command=pixelcount).grid(row=2, column=0, padx=10, pady=5, sticky=W)
ttk.Button(frame, text="Calculate Green Time", command=timeAllocation).grid(row=3, column=0, padx=10, pady=5, sticky=W)
ttk.Button(frame, text="Detect Ambulance", command=detectAmbulance).grid(row=4, column=0, padx=10, pady=5, sticky=W)
ttk.Button(frame, text="Exit", command=main.destroy).grid(row=5, column=0, padx=10, pady=5, sticky=W)

main.mainloop()
