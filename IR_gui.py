from IR import IR
import tkinter as tk

####################################
# Inisialisasi IR
# Buka folder "deeplearning"

ir = IR("deeplearning")

def gui_query():
    query = input_text_box.get("1.0", "end-1c")
    result = ir.query(query)
    result = [page[0] for page in result]
    output_text_box.delete("1.0", tk.END)
    [output_text_box.insert(tk.END, str.format("\nHalaman : %d" % page)) for page in result]
    
####################################
# Buat GUI sederhana
# GUI berupa 
#   Input : Text Box
#   Search : Button
#   Output : Button

## Buat sebuah Window
window = tk.Tk()

## Buat Input Text Box
input_text_box = tk.Text(window, height=1, width=25)

## Buat Search Button
serch_button = tk.Button(window, text="Search", command=gui_query) 

## Buat Output Text Box
output_text_box = tk.Text(window, height=20, width=50)

#packing
input_text_box.pack()
serch_button.pack()
output_text_box.pack()

#mainloop
window.mainloop()