import tkinter as tk

root = tk.Tk()


#label
lbl1 = tk.Message(root, text="Hello world!!")
lbl1.pack()

#message
msg = tk.Message(root, text="fuck_you")
msg.config(bg='lightgreen', font=('times', 24, 'italic'))
msg.pack()


#frame
frame = tk.Frame(root)
frame.pack()

#text 
text = tk.Text(frame, height=1, width=25)
text.pack()

#button
#button function
def btn1_onclick():
    print(text.get("1.0", "end-1c"))

btn1 = tk.Button(frame, text="Shit", fg="blue", command=btn1_onclick)
btn1.pack()

#text 
text = tk.Text(frame, height=25, width=50)
text.pack()


#mainloop
frame.mainloop()
root.mainloop()
