import matplotlib.pyplot as plt
import numpy as np
import math


class InteractiveAnnotation_heatmap:
    """Annotate a sample by clicking on a heatmap.

    Displays `data` as a 2-D heatmap. The user clicks on a column to
    select its x-axis position. `.annotate()` blocks until a click is
    received, then returns the float x value.

    Pass an existing ``fig`` and ``ax`` to reuse the same window across
    multiple calls instead of opening a new one each time.
    """

    def __init__(self, data):
        self.data = data
        self.retval = -1
        self.result = None
        self._vline = None

    @staticmethod
    def find_nearest(array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
            return array[idx - 1]
        return array[idx]

    def onclick(self, event):
        if event.inaxes is None:
            return
        self.retval = event.xdata
        self.result = self.retval
        print(f"Selected x = {self.result}")
        if self._standalone:
            plt.close(self.fig)

    def onMotion(self, event):
        if not event.inaxes:
            return
        if self._vline is None:
            self._vline = self.ax.axvline(
                x=event.xdata, color='lime', linestyle='--', linewidth=1.5, alpha=0.8
            )
        else:
            self._vline.set_xdata([event.xdata, event.xdata])
        self.ax.figure.canvas.draw_idle()

    def annotate(self, fig=None, ax=None):
        """Display this sample and wait for a click.

        Parameters
        ----------
        fig, ax : optional
            Existing figure and axes to reuse. When provided the window
            stays open and its contents are updated in-place. When
            omitted a new figure is created and closed after the click.

        Returns
        -------
        float
            The x-axis value at the click position, or ``None`` if the
            window was closed without clicking.
        """
        self._standalone = fig is None
        self._vline = None
        if self._standalone:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
        else:
            ax.cla()

        self.fig = fig
        self.ax = ax
        self.result = None

        ax.imshow(self.data, cmap='hot', interpolation='nearest', aspect='auto')
        fig.canvas.draw()
        fig.canvas.flush_events()

        motion_cid = fig.canvas.mpl_connect('motion_notify_event', self.onMotion)
        click_cid  = fig.canvas.mpl_connect('button_press_event', self.onclick)

        if self._standalone:
            plt.show()
        else:
            while self.result is None:
                if not plt.fignum_exists(fig.number):
                    print("Warning: window closed without a click — sample skipped.")
                    break
                plt.pause(0.05)

        fig.canvas.mpl_disconnect(motion_cid)
        fig.canvas.mpl_disconnect(click_cid)
        return self.result


class InteractiveAnnotation_2dplot:
    """Annotate a sample by clicking on a multi-channel line plot.

    Displays each column of `data` as a line on a shared axis.
    The user clicks anywhere on the plot to select an x-axis index.
    `.annotate()` blocks until a click is received, then returns the
    integer x index of the click.

    Pass an existing ``fig`` and ``ax`` to reuse the same window across
    multiple calls instead of opening a new one each time.

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
        self._vline = None

    @staticmethod
    def find_nearest(array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
            return array[idx - 1]
        return array[idx]

    def onclick(self, event):
        if event.inaxes is None:
            return
        self.retval = event.xdata
        self.result = int(self.retval)
        print(f"Selected index {self.result}")
        if self._standalone:
            plt.close(self.fig)

    def onMotion(self, event):
        if not event.inaxes:
            return
        if self._vline is None:
            self._vline = self.ax.axvline(
                x=event.xdata, color='gray', linestyle='--', linewidth=1.0, alpha=0.8
            )
        else:
            self._vline.set_xdata([event.xdata, event.xdata])
        self.ax.figure.canvas.draw_idle()

    def annotate(self, fig=None, ax=None):
        """Display this sample and wait for a click.

        Parameters
        ----------
        fig, ax : optional
            Existing figure and axes to reuse. When provided the window
            stays open and its contents are updated in-place. When
            omitted a new figure is created and closed after the click.

        Returns
        -------
        int
            The integer x-axis index at the click position, or ``None``
            if the window was closed without clicking.
        """
        self._standalone = fig is None
        self._vline = None
        if self._standalone:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
        else:
            ax.cla()

        self.fig = fig
        self.ax = ax
        self.result = None

        ax.plot(self.data)
        ax.set_title(self.plottitle)
        ax.set_xlabel(self.data.index.name or "Time Index")
        ax.set_ylabel("Normalized Value")
        fig.canvas.draw()
        fig.canvas.flush_events()

        motion_cid = fig.canvas.mpl_connect('motion_notify_event', self.onMotion)
        click_cid  = fig.canvas.mpl_connect('button_press_event', self.onclick)

        if self._standalone:
            plt.show()
        else:
            while self.result is None:
                if not plt.fignum_exists(fig.number):
                    print("Warning: window closed without a click — sample skipped.")
                    break
                plt.pause(0.05)

        fig.canvas.mpl_disconnect(motion_cid)
        fig.canvas.mpl_disconnect(click_cid)
        return self.result


if __name__ == "__main__":
    pass
