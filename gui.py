'''GUI application for vesuvius image processing'''

from typing import Dict
import random
import tkinter as tk
from tkinter import StringVar, ttk, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy import ndimage


class ImageProcessorGUI:
    '''Main GUI window for the image processor'''
    def __init__(self):
        self.image = None
        self.background_image = None
        self.processed_image = None

        self.canvas_width = 1450
        self.canvas_height = int(self.canvas_width/2.9)

        self.window = tk.Tk()
        self.window.title("Image Processor")

        self.canvas_frame = tk.Frame(self.window)
        self.canvas_frame.grid(row=0, column=0, padx=10, pady=10)

        # Create a canvas to display the images
        self.canvas = tk.Canvas(self.canvas_frame,
                                width=self.canvas_width,
                                height=self.canvas_height)
        self.canvas.grid(row=0, column=0)

        # Create a frame for labels and scales
        self.controls_frame = tk.Frame(self.window)
        self.controls_frame.grid(row=1, column=0, padx=10, pady=10)

        # Create a button to load the image
        self.load_button = ttk.Button(
            self.controls_frame,
            text="Load Image",
            command=self.load_image
        )
        self.load_button.grid(row=0, column=0, padx=10, pady=10)

        self.load_bg_button = ttk.Button(
            self.controls_frame,
            text="Load Background Image",
            command=self.load_background_image
        )
        self.load_bg_button.grid(row=0, column=1, padx=10, pady=10)

        self.load_bg_button = ttk.Button(
            self.controls_frame, text="Show segmentation",
            command=self.perform_segmentation
        )
        self.load_bg_button.grid(row=0, column=2, padx=10, pady=10)

        self.selected_diagonal = tk.StringVar()
        self.rb1 = ttk.Radiobutton(self.controls_frame,
                                   text='Rectangle',
                                   value='Rectangle',
                                   variable=self.selected_diagonal,
                                   command=self.process_image)
        self.rb2 = ttk.Radiobutton(self.controls_frame,
                                   text='/',
                                   value='/',
                                   variable=self.selected_diagonal,
                                   command=self.process_image)
        self.rb3 = ttk.Radiobutton(self.controls_frame, text='\\',
                                   value='\\',
                                   variable=self.selected_diagonal,
                                   command=self.process_image)
        self.rb1.grid(row=0, column=3, padx=10, pady=10)
        self.rb2.grid(row=0, column=4, padx=10, pady=10)
        self.rb3.grid(row=0, column=5, padx=10, pady=10)

        # Create a label to display the number
        self.f05 = StringVar()
        self.f05.set('Seas')
        self.number_label = ttk.Label(self.controls_frame,
                                      textvariable=self.f05)
        self.number_label.grid(row=0, column=7, columnspan=6, padx=10, pady=10)

        # Create a label and a scale for the iterations
        self.iterations_de_label = tk.Label(self.controls_frame,
                                            text="Iterations D&E:")
        self.iterations_de_label.grid(row=1, column=0, padx=10, pady=10)
        self.iterations_de_scale = tk.Scale(
            self.controls_frame,
            from_=0,
            to=30,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.on_slider_change,
        )
        self.iterations_de_scale.grid(row=1, column=1, padx=10, pady=10)

        # Create a label and a scale for the kernel size
        self.kernel_de_label_x = tk.Label(self.controls_frame,
                                          text="Kernel Size D&E_x:")
        self.kernel_de_label_x.grid(row=2, column=0, padx=10, pady=10)
        self.kernel_de_scale_x = tk.Scale(
            self.controls_frame,
            from_=0,
            to=30,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.on_slider_change,
        )
        self.kernel_de_scale_x.grid(row=2, column=1, padx=10, pady=10)

        self.kernel_de_label_y = tk.Label(self.controls_frame,
                                          text="Kernel Size D&E_y:")
        self.kernel_de_label_y.grid(row=3, column=0, padx=10, pady=10)
        self.kernel_de_scale_y = tk.Scale(
            self.controls_frame,
            from_=0,
            to=30,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.on_slider_change,
        )
        self.kernel_de_scale_y.grid(row=3, column=1, padx=10, pady=10)

        # Create a label and a scale for the dilation parameter
        self.iterations_ed_label = tk.Label(self.controls_frame,
                                            text="Iterations E&D:")
        self.iterations_ed_label.grid(row=1, column=2, padx=10, pady=10)
        self.iterations_ed_scale = tk.Scale(
            self.controls_frame,
            from_=0,
            to=30,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.on_slider_change,
        )
        self.iterations_ed_scale.grid(row=1, column=3, padx=10, pady=10)

        # Create a label and a scale for the erosion parameter
        self.kernel_ed_label_x = tk.Label(self.controls_frame,
                                          text="Kernel Size E&D_x:")
        self.kernel_ed_label_x.grid(row=2, column=2, padx=10, pady=10)
        self.kernel_ed_scale_x = tk.Scale(
            self.controls_frame,
            from_=0,
            to=30,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.on_slider_change,
        )

        self.kernel_ed_scale_x.grid(row=2, column=3, padx=10, pady=10)

        self.kernel_ed_label_y = tk.Label(self.controls_frame,
                                          text="Kernel Size E&D_y:")
        self.kernel_ed_label_y.grid(row=3, column=2, padx=10, pady=10)
        self.kernel_ed_scale_y = tk.Scale(
            self.controls_frame,
            from_=0,
            to=30,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.on_slider_change,
        )
        self.kernel_ed_scale_y.grid(row=3, column=3, padx=10, pady=10)

        # Create a label and a scale for the smoother kernel size
        self.kernel_smooth_label = tk.Label(self.controls_frame,
                                            text="Kernel Size smoother:")
        self.kernel_smooth_label.grid(row=2, column=4, padx=10, pady=10)
        self.kernel_smooth_scale = tk.Scale(
            self.controls_frame,
            from_=3,
            to=199,
            resolution=2,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.on_slider_change,
        )

        self.kernel_smooth_scale.grid(row=2, column=5, padx=10, pady=10)

        # Create a label and a scale for the smoother kernel size
        self.segmentation_label = tk.Label(self.controls_frame,
                                           text="Kernel Size smoother:")
        self.segmentation_label.grid(row=2, column=4, padx=10, pady=10)
        self.segmentation_label_scale = tk.Scale(
            self.controls_frame,
            from_=0,
            to=10000,
            resolution=1,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.on_slider_change,
        )

        self.segmentation_label_scale.grid(row=3, column=5, padx=10, pady=10)

        self.hide = tk.IntVar()
        self.checkbutton = tk.Checkbutton(self.controls_frame,
                                          text="Hide labelling",
                                          variable=self.hide,
                                          command=self.process_image)
        self.checkbutton.grid(row=1, column=4, padx=10, pady=10)

        self.hide_smallest = tk.IntVar()
        self.checkbutton = tk.Checkbutton(self.controls_frame,
                                          text="Hide smallest",
                                          variable=self.hide_smallest,
                                          command=self.process_image)
        self.checkbutton.grid(row=1, column=5, padx=10, pady=10)

        # Center the controls frame under the canvas frame
        self.window.grid_rowconfigure(1, weight=1)
        self.window.grid_columnconfigure(0, weight=1)

        # Center the controls frame horizontally
        self.controls_frame.grid_columnconfigure(0, weight=1)
        self.controls_frame.grid_columnconfigure(1, weight=1)
        self.controls_frame.grid_columnconfigure(2, weight=1)
        self.controls_frame.grid_columnconfigure(3, weight=1)

        # Place the controls frame below the canvas frame
        self.controls_frame.grid(row=2, column=0, padx=10, pady=10,
                                 sticky=tk.N)

        # Init
        self.results = 0
        self.fn = 0
        self.fp = 0
        self.tp = 0
        self.precision = 0
        self.recall = 0
        self.job = None

        self.window.mainloop()

    def score_guess(self, guess, truth, beta=0.5):
        '''F Score'''
        guess = guess.astype(bool)
        truth = truth.astype(bool)

        self.results = np.zeros_like(guess, dtype=int)
        self.results[truth] += 1
        self.results[guess] += 2
        self.fn = np.count_nonzero(self.results == 1)
        self.fp = np.count_nonzero(self.results == 2)
        self.tp = np.count_nonzero(self.results == 3)
        self.precision = self.tp / (self.tp + self.fp)
        self.recall = self.tp / (self.tp + self.fn)

        print(f"{self.tp:,} Correct Pixels")
        print(f"{self.fp:,} False Positives")
        print(f"{self.fn:,} False Negatives")
        print(f"Precision: {self.precision}")
        print(f"Recall: {self.recall}")
        print("--------------l")
        return (1 + beta ** 2) * self.precision * self.recall \
            / (self.precision * beta ** 2 + self.recall)

    def load_image(self):
        '''Load the image and display it on the canvas'''
        filename = filedialog.askopenfilename(
            filetypes=[("PNG Images", "*.png"), ("JPEG Images", "*.jpg")]
        )
        if filename:
            self.image = cv2.imread(filename, 0)
            self.processed_image = self.image.copy()
            self.display_image(self.image)

    def load_background_image(self):
        '''Load the image and display it on the canvas'''
        filename = filedialog.askopenfilename(
            filetypes=[("PNG Images", "*.png"), ("JPEG Images", "*.jpg")]
        )
        if filename:
            self.background_image = cv2.imread(filename, 0)
            self.display_image(self.image)

    def process_image(self):
        '''
        Apply the erosion and dilation operations
        with the current settings
        '''
        iterations_de = self.iterations_de_scale.get()
        kernel_size_de_x = self.kernel_de_scale_x.get()
        kernel_size_de_y = self.kernel_de_scale_y.get()

        kernel_de = self.get_kernel(kernel_size_de_x,
                                    kernel_size_de_y,
                                    self.selected_diagonal.get())

        iterations_ed = self.iterations_ed_scale.get()
        kernel_size_ed_x = self.kernel_ed_scale_x.get()
        kernel_size_ed_y = self.kernel_ed_scale_y.get()

        kernel_ed = self.get_kernel(kernel_size_ed_x,
                                    kernel_size_ed_y,
                                    self.selected_diagonal.get())

        self.processed_image = self.image

        self.perform_dilation_and_erosion(iterations_de, kernel_de)

        self.perform_erosion_and_dilation(iterations_ed, kernel_ed)

        self.perform_median_smoothing()

        self.remove_small_areas()

        self.f05.set(self.score_guess(self.processed_image,
                                      self.background_image))

        self.display_image(self.processed_image)

    def perform_segmentation(self):
        # Label contiguous regions
        labeled_image, num_features = ndimage.label(self.processed_image)

        # Create a new image to draw the colored regions on
        colored_image = np.zeros((self.processed_image.shape[0],
                                  self.processed_image.shape[1], 3),
                                 dtype=np.uint8)

        # Assign each labeled region a unique color
        for i in range(1, num_features + 1):
            colored_image[labeled_image == i] = (random.randint(0, 255),
                                                 random.randint(0, 255),
                                                 random.randint(0, 255))
            num_pixels = np.sum(labeled_image == i)
            print(f'Region {i}: {num_pixels} pixels')

        colored_image = cv2.resize(colored_image,
                                   (self.canvas_width,
                                    self.canvas_height))

        # Show the original and colored images
        cv2.imshow('Colored', colored_image)  # ('Original', image)

    def remove_small_areas(self):
        if self.hide_smallest.get():
            # Label contiguous regions
            labeled_image, num_features = ndimage.label(self.processed_image)

            # Create a new image to draw the colored regions on
            removed_small_regions_image = self.processed_image.copy()

            # Assign each labeled region a unique color
            area_dict: Dict[int, int] = {}  # key: region, value: pixel count

            for i in range(1, num_features + 1):
                num_pixels = np.sum(labeled_image == i)
                area_dict[i] = num_pixels
                print(f'Region {i}: {num_pixels} pixels')

                if num_pixels < self.segmentation_label_scale.get():
                    removed_small_regions_image[labeled_image == i] = 0

            self.processed_image = removed_small_regions_image

    def perform_median_smoothing(self):
        self.processed_image = \
            cv2.medianBlur(self.processed_image,
                           ksize=self.kernel_smooth_scale.get())

    def perform_dilation_and_erosion(self, iterations_de, kernel_de):
        if iterations_de != 0 \
            and kernel_de.shape[0] != 0 \
                and kernel_de.shape[1] != 0:
            self.processed_image = cv2.dilate(
                self.processed_image, kernel_de, iterations=iterations_de
            )
            self.processed_image = cv2.erode(
                self.processed_image, kernel_de, iterations=iterations_de
            )

    def perform_erosion_and_dilation(self, iterations_ed, kernel_ed):
        if iterations_ed != 0 \
            and kernel_ed.shape[0] != 0 \
                and kernel_ed.shape[1] != 0:
            self.processed_image = cv2.erode(
                self.processed_image, kernel_ed, iterations=iterations_ed
            )
            self.processed_image = cv2.dilate(
                self.processed_image, kernel_ed, iterations=iterations_ed
            )

    def reset_image(self):
        # Reset the processed image to the original image
        self.processed_image = self.image.copy()
        self.display_image(self.image)

    def display_image(self, image):
        if self.background_image is None:
            displayed_background_image = self.image
        else:
            displayed_background_image = self.background_image

        if self.image is None:
            displayed_image = self.background_image
            image = self.background_image
        else:
            displayed_image = image

        # ------------- used for generating new background image --------------
        # background = cv2.imread\
        # (r"data\input\vesuvius-challenge\train\1\inklabels.png", 0)

        # coordinates of batch used
        # x = 2000
        # y = 400
        # width = 2500
        # height = 1000

        # background = background[y:y+height, x:x+width]

        # background = cv2.resize\
        # (background, (self.canvas_width, self.canvas_height))

        # cv2.imwrite('background 1.png', background)
        # ------------- used for generating new background image --------------

        displayed_image = image.copy()
        displayed_image = cv2.resize(displayed_image,
                                     (self.canvas_width,
                                      self.canvas_height))
        displayed_background_image = cv2.resize(displayed_background_image,
                                                (self.canvas_width,
                                                 self.canvas_height))

        background_transparency = 0 if self.hide.get() else 0.3

        # add labelled image in the background
        displayed_image = cv2.addWeighted(displayed_background_image,
                                          background_transparency,
                                          displayed_image, 0.6, 0)

        # Convert the image to a PIL Image and display it on the canvas
        displayed_image = Image.fromarray(displayed_image)
        displayed_image = ImageTk.PhotoImage(displayed_image)
        self.canvas.delete("all")  # Clear the canvas
        # Display the image on the canvas
        self.canvas.create_image(
            0, 0, anchor="nw", image=displayed_image
        )
        # Keep a reference to the image to
        # prevent it from being garbage-collected
        self.canvas.image = displayed_image

    def on_slider_change(self, event):
        # only process image if slider did not change for delta milliseconds
        delta = 20
        try:  # if hasattr(self, 'job'):
            self.window.after_cancel(self.job)
        except AttributeError:
            pass
        self.job = self.window.after(delta, self.process_image)

    def get_kernel(self, width, height, diagonal):
        if diagonal == "/":

            res = np.zeros((height, width), dtype=np.uint8)
            x = np.linspace(0, width-1, height)
            y = np.arange(height-1, -1, -1)
            res[y.astype(int), x.astype(int)] = 1

        elif diagonal == "\\":
            res = np.zeros((height, width), dtype=np.uint8)
            x = np.linspace(0, width-1, height)
            y = np.arange(height)
            res[y.astype(int), x.astype(int)] = 1

        else:
            res = np.ones((height, width), dtype=np.uint8)

        return res


if __name__ == "__main__":
    app = ImageProcessorGUI()
    app.window.mainloop()
