#!/usr/bin/env python
import tkinter
import numpy as np
import sys
import subprocess

class Canvas(object):
    def __init__(self, height, width, true_x, true_y):
        '''
        A class to create a drawing canvas for use with the perceptron we trained earlier.

        :param height: The requested height of the window, in pixels
        :type height: int
        :param width: The requested length of the window, in pixels
        :type width: int
        :param true_x: The true length of the x dimension, in pixels
        :type width: int
        :param true_y: The true length of the y dimension, in pixels
        :type true_y: int
        '''
        # Set up the frame
        self.master = tkinter.Tk()
        self.true_y = true_x
        self.true_x = true_y
        self.true_coords = np.zeros((true_y, true_x))

        # Ensure the frame is evenly divisible by the true x and y
        self.width = int(true_x * round(float(width) / true_x))
        self.height =int(true_y * round(float(height) / true_y))

        self.canvas = tkinter.Canvas(self.master, width=self.width, height=self.height)

        # Set up keybindings
        self.canvas.bind("<Button-1>", self._mousePress)
        self.canvas.bind("<B1-Motion>", self._mouseDrag)
        self.canvas.bind("<ButtonRelease-1>", self._mouseRelease)
        self.canvas.pack()

        # Create the window and begin accepting input
        self.drawing = False
        self.master.mainloop()

    # Define and bind events
    # This is a machine with two states: drawing or not drawing
    # mousePress enables drawing
    # mouseRelease disables drawing
    # while drawing, pixels are drawn wherever the mouse is dragged
    #   in mouseDrag

    def _mousePress(self, event):
        # Turns on drawing
        #print("Clicked ", event.x, event.y)
        self.drawing = True

    def _mouseDrag(self, event):
        # The main event in this class; draws lines
        rectangle_size_x = int(self.width / self.true_x / 2)
        rectangle_size_y = int(self.height / self.true_y / 2)
        self.canvas.create_oval(event.x - rectangle_size_x, event.y - rectangle_size_y,
                                     event.x + rectangle_size_x, event.y + rectangle_size_y,
                                     fill = "black")
        self.true_coords[int(event.y / self.true_y), # Row first
                         int(event.x / self.true_x)] = 1
            
        #print("Dragged to ", event.x, event.y)

    def _mouseRelease(self, event):
        #print("Released at ", event.x, event.y)
        self.drawing = False

if __name__ == "__main__":
    done = False
    start = subprocess.Popen(["trainPerceptron.exe"], shell=True, stdin = subprocess.PIPE)
    while not done:
        canvas = Canvas(400, 400, 28, 28)

        # Generate the input vector
        input_vector = np.zeros(784)
        count = 0
        for row in canvas.true_coords:
            for col in row:
                input_vector[count] = col
                count += 1
        #print(input_vector)
                
        # Write to stdout
        for i in input_vector:
            if i == 1:
                start.stdin.write(b"1\n")
            else:
                start.stdin.write(b"0\n")
            start.stdin.flush()


