import time
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
import threading
import pandas as pd
from imageio.v2 import imread, imwrite
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ttkthemes import ThemedTk
from tkinter import filedialog, messagebox
from tkinter import *
from PIL import Image, ImageTk
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import os
import imutils


try:
    from AppKit import NSApplication, NSApp, NSBundle
    from Foundation import NSProcessInfo

    app_name = "Image Comparison Tool"
    app_bundle = NSBundle.mainBundle()
    if app_bundle:
        info = app_bundle.localizedInfoDictionary() or app_bundle.infoDictionary()
        info["CFBundleName"] = app_name
        info["CFBundleDisplayName"] = app_name
    NSProcessInfo.processInfo().setValue_forKey_(app_name, "processName")
except ImportError:
    print("pyobjc not installed. Cannot change macOS app bar name.")


def load_image(path, max_size=(350, 250)):
    try:
        """Load an image from the path and resize it to fit within max_size."""
        image = Image.open(path)
        image.thumbnail(max_size, Image.LANCZOS)
        return ImageTk.PhotoImage(image)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while opening the file: {e}")


def update_image_preview(image, canvas):
    try:
        canvas.delete("all")
        canvas.image = image
        canvas.create_image(0, 0, image=image, anchor=tk.NW)
        canvas.config(width=image.width(), height=image.height())
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while opening the file: {e}")


def calculate_metrics(reference_path, target_paths, original_size, compressed_size):
    try:
        reference_image = cv2.imread(reference_path)
        reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        results = []
        for path in target_paths:
            target_image = cv2.imread(path)
            target_image_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

            histograms = compare_histograms(reference_path, path)
            compression_ratio = original_size / compressed_size
            ssim_value = ssim(reference_image_gray, target_image_gray, multichannel=True, channel_axis=-1)
            psnr_value = psnr(reference_image_gray, target_image_gray)

            results.append((ssim_value, psnr_value, compression_ratio, histograms))
        return results
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while opening the file: {e}")


def show_results(results):
    try:
        result_texts = []
        for ssim_value, psnr_value, compression_ratio, histogram_score in results:
            result_texts.append(
                f"SSIM: {ssim_value:.4f}\nPSNR: {psnr_value:.4f}, \n"
                f"Compression Ratio: {compression_ratio:.4f}\n"
                f"Histogram Score: {histogram_score:.4f}"
            )
        result_text = "\n".join(result_texts)
        result_label.config(text=result_text)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while opening the file: {e}")


def select_reference_image():
    try:
        global reference_image_path
        file_path = filedialog.askopenfilename()
        global original_size
        original_size = os.path.getsize(file_path)
        if file_path:
            reference_image_path = file_path
            image = load_image(file_path)
            update_image_preview(image, canvas_reference)
            update_image_preview(image, canvas_reference_DIFF)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while opening the file: {e}")


def select_comparison_image():
    try:
        global comparison_image_path
        file_path = filedialog.askopenfilename()
        global compressed_size
        compressed_size = os.path.getsize(file_path)
        if file_path:
            comparison_image_path = file_path
            image = load_image(file_path)
            update_image_preview(image, canvas_comparison)
            update_image_preview(image, canvas_comparison_DIFF)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while opening the file: {e}")


def on_start_button_click():
    try:
        global reference_image_path, comparison_image_path
        reference_image_path = globals().get('reference_image_path', '')
        comparison_image_path = globals().get('comparison_image_path', '')

        if not os.path.isfile(reference_image_path) or not os.path.isfile(comparison_image_path):
            messagebox.showerror("Error",
                                 "The reference image or comparison image file is not provided. Please select a valid image.")
            return

        results = calculate_metrics(reference_image_path, [comparison_image_path], original_size, compressed_size)

        result_text = "\n".join([
                                    f"SSIM: {ssim_value:.4f}\nPSNR: {psnr_value:.4f}\nCompression Ration: {compression_ratio:.4f}\nHistograms: {histograms}"
                                    for  ssim_value, psnr_value,compression_ratio, histograms in results])
        show_results(results)
        root.update()
        messagebox.showinfo("Comparison Results", result_text)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while opening the file: {e}")


###########################################################
#############   FEATURES   ################################
###########################################################
###########################################################


def compare_histograms(reference_path, target_path):
    try:
        image1 = cv2.imread(reference_path, 0)
        image2 = cv2.imread(target_path, 0)

        hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])

        hist1 = cv2.normalize(hist1, hist1)
        hist2 = cv2.normalize(hist2, hist2)

        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return score
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while opening the file: {e}")


def show_difference():
    try:
        image_png = Image.open(reference_image_path)
        image_jpeg = Image.open(comparison_image_path)

        img1 = cv2.imread(reference_image_path)
        img1 = cv2.resize(img1, (600, 360))

        img2 = cv2.imread(comparison_image_path)
        img2 = cv2.resize(img2, (600, 360))

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        diff_first = cv2.absdiff(img1, img2)
        cv2.imshow("Difference layer", diff_first)
        diff_sec = cv2.absdiff(gray1, gray2)
        thresh = cv2.threshold(diff_sec, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.imshow('Difference+Threshold', thresh)

        kernel = np.ones((3, 3), np.uint8)
        dilate = cv2.dilate(thresh, kernel, iterations=1)
        cv2.imshow("Dilate", dilate)

        contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)

        x = np.zeros((360, 10, 3), np.uint8)
        result = np.hstack((img1, x, img2))

        cv2.imshow("Differences contour", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while opening the file: {e}")


def compare_edges(reference_path, target_path):
    try:
        image1 = cv2.imread(reference_path, 0)
        image2 = cv2.imread(target_path, 0)

        edges1 = cv2.Canny(image1, 100, 200)
        edges2 = cv2.Canny(image2, 100, 200)

        difference = cv2.absdiff(edges1, edges2)
        score = np.sum(difference) / (difference.shape[0] * difference.shape[1])
        return score
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while opening the file: {e}")


def match_features(reference_path, target_path):
    try:
        image1 = cv2.imread(reference_path, 0)
        image2 = cv2.imread(target_path, 0)

        orb = cv2.ORB_create()

        kp1, des1 = orb.detectAndCompute(image1, None)
        kp2, des2 = orb.detectAndCompute(image2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.distance)

        return len(matches), matches
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while opening the file: {e}")




def calculate_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist

def plot_histograms(image1, image2):
    hist1 = calculate_histogram(image1)
    hist2 = calculate_histogram(image2)

    fig, ax = plt.subplots()
    ax.plot(hist1, color='blue', label='Reference image')
    ax.plot(hist2, color='red', label='Compressed image')
    ax.set_title('Histogram Comparison')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    ax.legend()

    return fig


def compare_histograms_visual():
    try:
        image1 = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
        if image1 is None:
            raise ValueError("First image not loaded")
        image2 = cv2.imread(comparison_image_path, cv2.IMREAD_GRAYSCALE)
        if image2 is None:
            raise ValueError("Second image not loaded")

        fig = plot_histograms(image1, image2)

        histogram_window = tk.Toplevel(root)
        histogram_window.title("Histogram Comparison")
        histogram_window.geometry("600x400")

        canvas = FigureCanvasTkAgg(fig, master=histogram_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(histogram_window)
        toolbar.pack(fill=tk.X)
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while comparing histograms: {e}")


###########################################################
#############  EXPERIMENT  ################################
###########################################################
###########################################################
formats = ['jpeg', 'png', 'webp', 'tiff', 'bmp', 'gif']




def compression_ratio(original_size, compressed_size):
    return original_size / compressed_size


def safe_psnr(original_image, compressed_image):
    mse = np.mean((original_image - compressed_image) ** 2)
    if mse == 0:
        return float('inf')
    return psnr(original_image, compressed_image, data_range=compressed_image.max() - compressed_image.min())


def compress_and_evaluate(image_path, output_dir, progress_bar, popup):
    try:
        results = []
        original_image = imread(image_path)
        original_size = os.path.getsize(image_path)
        total_formats = len(formats)

        for i, fmt in enumerate(formats):
            compressed_path = os.path.join(output_dir, f'compressed.{fmt}')

            if original_image.ndim == 3 and original_image.shape[2] == 4:
                image_rgb = Image.fromarray(original_image).convert('RGB')
                image_to_save = np.array(image_rgb)
            else:
                image_to_save = original_image

            start_time = time.time()
            imwrite(compressed_path, image_to_save, format=fmt)
            compression_time = time.time() - start_time

            compressed_image = imread(compressed_path)
            compressed_size = os.path.getsize(compressed_path)

            if compressed_image.ndim == 3 and compressed_image.shape[2] == 4:
                compressed_image_pil = Image.fromarray(compressed_image).convert('RGB')
                compressed_image = np.array(compressed_image_pil)

            if compressed_image.ndim != original_image.ndim:
                compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2GRAY)
                original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            else:
                original_image_gray = original_image

            image_psnr = safe_psnr(original_image_gray, compressed_image)
            image_ssim = ssim(original_image_gray, compressed_image, channel_axis=-1)

            comp_ratio = compression_ratio(original_size, compressed_size)

            results.append({
                'format': fmt,
                'psnr': image_psnr,
                'ssim': image_ssim,
                'compression_ratio': comp_ratio,
                'compression_time': compression_time
            })

            progress_bar['value'] = (i + 1) / total_formats * 100
            popup.update_idletasks()

        result_text = "\n".join([
            f"Format: {result['format']}\nPSNR: {result['psnr']:.4f}\nSSIM: {result['ssim']:.4f}\n"
            f"Compression Ratio: {result['compression_ratio']:.4f}\nCompression Time: {result['compression_time']:.4f}\n"
            for result in results
        ])

        results_df = pd.DataFrame(results)
        metrics_file_path = (f'{output_dir}/image_compression_metrics.csv')
        results_df.to_csv(metrics_file_path, index=False)

        popup.destroy()
        messagebox.showinfo("Compression Results", result_text)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during compression and evaluation: {e}")
        popup.destroy()

def select_output_directory():
    directory = filedialog.askdirectory()
    return directory


def on_compress_and_evaluate_click():
    try:
        output_dir = select_output_directory()
        if not output_dir:
            messagebox.showerror("Error", "No directory selected.")
            return

        popup = tk.Toplevel(root)
        popup.title("Progress")
        progress_bar = ttk.Progressbar(popup, orient='horizontal', length=300, mode='determinate')
        progress_bar.pack(pady=20, padx=20)

        progress_bar['value'] = 0
        popup.update_idletasks()

        thread = threading.Thread(target=compress_and_evaluate,
                                  args=(reference_image_path, output_dir, progress_bar, popup))
        thread.start()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during compression and evaluation: {e}")
        popup.destroy()



###########################################################
#############  GUI TKINTER ################################
###########################################################
###########################################################


root = ThemedTk(theme='aqua')

root.title("Image Comparison Tool 1.1 devBuild")

icon_path = "app_icon.png"
icon = ImageTk.PhotoImage(file=icon_path)
root.iconphoto(True, icon)
root.geometry('1100x500')
root.wm_minsize(800, 500)

s = ttk.Style()
s.configure('TNotebook.Tab', font=('URW Gothic L', '11'), padding=[10, 1])

main_notebook = ttk.Notebook(root)
main_notebook.pack();

metrics_tab = Frame(main_notebook, width=1000, height=500)
diff_tab = Frame(main_notebook, width=1000, height=500, )
experiment_tab = Frame(main_notebook, width=1000, height=500, )
metrics_tab.pack(fill='both', expand=1)
diff_tab.pack(fill='both', expand=1)
experiment_tab.pack(fill='both', expand=1)
fontObj = tkFont.Font(size=12)

main_notebook.add(metrics_tab, text="Metrics")
main_notebook.add(diff_tab, text="Difference")
main_notebook.add(experiment_tab, text="Experiment")

###########################################################
#############  MENU #######################################
###########################################################
###########################################################

menu_bar = tk.Menu(root)
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Load image", command=load_image)
file_menu.add_command(label="Calculate Metrics", command=on_start_button_click)

features = tk.Menu(menu_bar, tearoff=0)
features.add_command(label='Compare histograms', command=compare_histograms_visual)

select_submenu = tk.Menu(file_menu, tearoff=0)
select_submenu.add_command(label="Reference Image", command=select_reference_image)
select_submenu.add_command(label="Comparison Image", command=select_comparison_image)

file_menu.add_cascade(label="Select...", menu=select_submenu)

file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)


def about():
    about_window = tk.Toplevel(root)
    about_window.title("About")
    about_window.geometry("400x300")

    about_label = ttk.Label(about_window, text="Image Comparison Tool v1.1 devBuild", font=("Helvetica", 16, "bold"))
    about_label.pack(pady=10)

    about_text = ("This tool allows you to compare images before and after compression.\n\n"
                  "Features:\n"
                  "- Load and save images\n"
                  "- Compare using SSIM and PSNR metrics\n"
                  "- View differences in a detailed manner\n"
                  "\nDeveloped by Novikov David & Leyman Adil\n")
    about_message = tk.Text(about_window, wrap="word", width=50, height=10, padx=10, pady=10, bd=0)
    about_message.insert("1.0", about_text)
    about_message.config(state="disabled", font=("Helvetica", 12))
    about_message.pack(pady=10)

    close_button = ttk.Button(about_window, text="Close", command=about_window.destroy)
    close_button.pack(pady=10)


help_menu = tk.Menu(menu_bar, tearoff=0)
help_menu.add_command(label="About", command=about)

menu_bar.add_cascade(label="File", menu=file_menu)
menu_bar.add_cascade(label="Features", menu=features)
menu_bar.add_cascade(label="Help", menu=help_menu)

root.config(menu=menu_bar)

###########################################################
#############  METRICS TAB ################################
###########################################################
###########################################################

frame_reference = LabelFrame(metrics_tab, text='Reference Image', padx=15, pady=15, font=fontObj)
frame_reference.grid(row=0, column=0);

frame_comparison = LabelFrame(metrics_tab, text='Comparison Image', padx=15, pady=15, font=fontObj)
frame_comparison.grid(row=0, column=1);

frame_start = LabelFrame(metrics_tab, pady=15, width=400, height=500)
frame_start.grid(row=0, column=2, rowspan=4)

canvas_reference = tk.Canvas(frame_reference, width=350, height=250, bg="grey")
canvas_reference.pack(pady=10, padx=10)

btn_select_reference = ttk.Button(frame_reference, text="Select Reference Image", command=select_reference_image,
                                  padding=10)
btn_select_reference.pack()

canvas_comparison = tk.Canvas(frame_comparison, width=350, height=250, bg="grey")
canvas_comparison.pack(pady=10, padx=10)

btn_select_comparison = ttk.Button(frame_comparison, text="Select Comparison Image", command=select_comparison_image,
                                   padding=10)
btn_select_comparison.pack()

btn_start_comparison = ttk.Button(frame_start, text="Start Comparison", command=on_start_button_click, padding=10)
btn_start_comparison.pack()

result_label = tk.Label(metrics_tab, text="METRICS", justify=tk.LEFT)
result_label.grid(row=6, column=0, columnspan=3, sticky='w')

###########################################################
#############DIFFERENCE TAB################################
###########################################################
###########################################################


frame_reference_DIFF = LabelFrame(diff_tab, text='Reference Image', padx=15, pady=15, font=fontObj)
frame_reference_DIFF.grid(row=0, column=0);

frame_comparison_DIFF = LabelFrame(diff_tab, text='Comparison Image', padx=15, pady=15, font=fontObj)
frame_comparison_DIFF.grid(row=0, column=1);

frame_start_DIFF = LabelFrame(diff_tab, pady=15, width=400, height=500)
frame_start_DIFF.grid(row=0, column=2, rowspan=4)

canvas_reference_DIFF = tk.Canvas(frame_reference_DIFF, width=350, height=250, bg="grey")
canvas_reference_DIFF.pack(pady=10, padx=10)

btn_select_reference_DIFF = ttk.Button(frame_reference_DIFF, text="Select Reference Image", command=select_reference_image, padding=10)
btn_select_reference_DIFF.pack()

canvas_comparison_DIFF = tk.Canvas(frame_comparison_DIFF, width=350, height=250, bg="grey")
canvas_comparison_DIFF.pack(pady=10, padx=10)

btn_select_comparison_DIFF = ttk.Button(frame_comparison_DIFF, text="Select Comparison Image", command=select_comparison_image, padding=10)
btn_select_comparison_DIFF.pack()

btn_start_comparison_DIFF = ttk.Button(frame_start_DIFF, text="Show Difference", command=show_difference, padding=10)
btn_start_comparison_DIFF.pack()

###########################################################
############# Experiment tab ################################
###########################################################
###########################################################

progress_bar = ttk.Progressbar(experiment_tab, orient='horizontal', length=300, mode='determinate')

frame_reference_DIFF = LabelFrame(experiment_tab, text='Reference Image', padx=15, pady=15, font=fontObj)
frame_reference_DIFF.grid(row=0, column=0);

frame_start_DIFF = LabelFrame(experiment_tab, pady=15, width=400, height=500)
frame_start_DIFF.grid(row=0, column=2, rowspan=4)

canvas_reference_DIFF = tk.Canvas(frame_reference_DIFF, width=350, height=250, bg='grey')
canvas_reference_DIFF.pack(pady=10, padx=10)

btn_select_reference_DIFF = ttk.Button(frame_reference_DIFF, text='Select Reference Image', command=select_reference_image, padding=10)
btn_select_reference_DIFF.pack()

btn_start_comparison_DIFF = ttk.Button(frame_start_DIFF, text='Start experiment', command=on_compress_and_evaluate_click, padding=10)
btn_start_comparison_DIFF.pack()

root.mainloop()
