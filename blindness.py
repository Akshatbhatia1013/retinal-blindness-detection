import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.filedialog import askopenfilename
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model import main

y = False  # login flag

def Signup():
    username = username_entry.get()
    password = password_entry.get()

    if not username or not password:
        messagebox.showinfo("Error", "Username and password cannot be empty.")
        return

    try:
        df = pd.read_csv("users.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["username", "password", "predict"])

    if username in df["username"].values:
        messagebox.showinfo("Error", "Username already exists.")
    else:
        df = pd.concat([df, pd.DataFrame([[username, password, ""]], columns=["username", "password", "predict"])], ignore_index=True)
        df.to_csv("users.csv", index=False)
        messagebox.showinfo("Success", f"Hi {username}, you're registered!")

def LogIn():
    global y
    username = username_entry.get()
    password = password_entry.get()

    if not username or not password:
        messagebox.showinfo("Error", "Please enter both username and password.")
        return

    try:
        df = pd.read_csv("users.csv")
    except FileNotFoundError:
        messagebox.showinfo("Error", "No users found.")
        return

    match = df[(df["username"] == username) & (df["password"] == password)]
    if not match.empty:
        y = True
        messagebox.showinfo("Success", f"Welcome {username}!")
    else:
        messagebox.showinfo("Error", "Invalid credentials.")

def OpenFile():
    if not y:
        messagebox.showinfo("Login Required", "Please log in first.")
        return

    try:
        img_path = askopenfilename(title="Select an image", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not img_path:
            return

        value, classes, gradcam_img = main(img_path)

        df = pd.read_csv("users.csv")
        df.loc[df["username"] == username_entry.get(), "predict"] = value
        df.to_csv("users.csv", index=False)

        messagebox.showinfo("Prediction", f"Predicted Label: {value}\nClass: {classes}")

        plt.imshow(gradcam_img)
        plt.title(f"Prediction: {value} ({classes})")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print("Error:", e)
        messagebox.showinfo("Error", f"Prediction failed.\n{str(e)}")

# ------------------- UI Setup -------------------

root = tk.Tk()
root.title("Retinal Blindness Detection")
root.geometry("600x400")
root.configure(bg="#f0f4f7")

style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12))
style.configure("TLabel", font=("Helvetica", 12), background="#f0f4f7")
style.configure("Header.TLabel", font=("Helvetica", 20, "bold"), background="#f0f4f7", foreground="#005f73")

header = ttk.Label(root, text="Retinal Blindness Detection", style="Header.TLabel")
header.pack(pady=20)

main_frame = ttk.Frame(root, padding=20)
main_frame.pack()

# Username
ttk.Label(main_frame, text="Username:").grid(row=0, column=0, pady=5, sticky="w")
username_entry = ttk.Entry(main_frame, width=30)
username_entry.grid(row=0, column=1, pady=5)

# Password
ttk.Label(main_frame, text="Password:").grid(row=1, column=0, pady=5, sticky="w")
password_entry = ttk.Entry(main_frame, width=30, show="*")
password_entry.grid(row=1, column=1, pady=5)

# Buttons
button_frame = ttk.Frame(root)
button_frame.pack(pady=15)

ttk.Button(button_frame, text="Signup", command=Signup).grid(row=0, column=0, padx=10)
ttk.Button(button_frame, text="Login", command=LogIn).grid(row=0, column=1, padx=10)
ttk.Button(button_frame, text="Upload Image", command=OpenFile).grid(row=0, column=2, padx=10)

root.mainloop()
