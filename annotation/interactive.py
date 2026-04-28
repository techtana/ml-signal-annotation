import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Rectangle


class InteractiveAnnotation_heatmap:
    """Annotate a sample by clicking on a heatmap.

    Displays `data` as a 2-D heatmap. The user clicks on a column to
    select its x-axis position. `.annotate()` blocks until a click is
    received, then returns the float x value.
    """

    def __init__(self, data):
        self.data = data
        self.retval = -1
        self.result = None

    @staticmethod
    def find_nearest(array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
            return array[idx - 1]
        return array[idx]

    def onclick(self, event):
        self.retval = event.xdata
        self.result = self.retval
        print(f"Selected x = {self.result}")
        plt.close()

    def onMotion(self, event):
        if not event.inaxes:
            return
        self.retval = event.xdata
        xint = int(event.xdata) - 0.5
        rect = Rectangle((xint, -0.5), 1, len(self.data),
                          fill=False, linestyle='dashed', edgecolor='green', linewidth=2.0)
        self.ax.add_patch(rect)
        self.ax.figure.canvas.draw()
        rect.remove()

    def annotate(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.data, cmap='hot', interpolation='nearest', aspect='auto')
        self.fig.canvas.mpl_connect('motion_notify_event', self.onMotion)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()
        return self.result


class InteractiveAnnotation_2dplot:
    """Annotate a sample by clicking on a multi-channel line plot.

    Displays each column of `data` as a line on a shared axis.
    The user clicks anywhere on the plot to select an x-axis index.
    `.annotate()` blocks until a click is received, then returns the
    integer x index of the click.

    Parameters
    ----------
    data : pandas.DataFrame
        Rows are time steps; columns are sensor channels.
        The DataFrame index is used as the x-axis label.
    plottitle : str, optional
        Title shown above the plot.
    """

    def __init__(self, data, plottitle=None):
        self.data = data
        self.plottitle = plottitle or "Click on graph to annotate."
        self.retval = -1
        self.result = None

    @staticmethod
    def find_nearest(array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
            return array[idx - 1]
        return array[idx]

    def onclick(self, event):
        self.retval = event.xdata
        self.result = int(self.retval)
        print(f"Selected index {self.result}")
        plt.close()

    def onMotion(self, event):
        if not event.inaxes:
            return
        self.retval = event.xdata
        xint = int(event.xdata) - 0.5
        rect = Rectangle((xint, -0.5), 1, len(self.data),
                          fill=False, linestyle='dashed', edgecolor='black', linewidth=2.0)
        self.ax.add_patch(rect)
        self.ax.figure.canvas.draw()
        rect.remove()

    def annotate(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(self.data)
        self.ax.set_title(self.plottitle)
        self.ax.set_xlabel(self.data.index.name or "Time Index")
        self.ax.set_ylabel("Normalized Value")
        self.fig.canvas.mpl_connect('motion_notify_event', self.onMotion)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()
        return self.result


if __name__ == "__main__":
    pass
