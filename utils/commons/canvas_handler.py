import os
from threading import Thread


class CanvasHandler:
    def __init__(self, window_size: (int, int), canvas_size: (int, int)):
        """

        :param window_size:
        :param canvas_size:
        In PySimpleGUI, there is a situation that while you are currently viewing the DSC panel, resizing the window
        will only change the size of the DSC canvas. DCE canvas will remain unchanged until it is switched to.
        Therefore, we need to use the canvas size of the currently selected tab. This aspect reflects in
        get_new_canvas_size method
        """
        self.window_size = window_size
        self.current_canvas_size = canvas_size
        self.current_orientation = self.get_orientation(self.current_canvas_size)

    def check(self, values, window, dsc_mrp, dce_mra):
        """
        Check if the main window is resized, if yes, perform scaling or switching canvas depending on the current size
        of the canvas
        :param values:
        :param window:
        :param dsc_mrp:
        :param dce_mra:
        :return:
        """
        if window.size != self.window_size:
            new_canvas_size = self.get_new_canvas_size(window)
            # Rescale in case of the same orientation
            new_orientation = self.get_orientation(new_canvas_size)

            if self.current_orientation == new_orientation:
                # In case that the canvas orientation does not change, scale all figures accordingly
                # Find the scaling coefficient
                scale_coef = self.get_scale_coef(new_canvas_size)
                # Update the figure size
                dsc_mrp.set_fig_size(window)
                dce_mra.set_fig_size(window)
                # Scale the canvas size
                self.scale(dsc_mrp, scale_coef, values, window)
                self.scale(dce_mra, scale_coef, values, window)
                self.update_info(new_orientation, new_canvas_size, window.size)
                return True
            # Change the canvas set (horizontal to vertical, and vice versa) in case of different orientations
            self.draw_new_canvas(dsc_mrp, values, window)
            self.draw_new_canvas(dce_mra, values, window)
            self.update_info(new_orientation, new_canvas_size, window.size)
            return True
        return False

    def draw_new_canvas(self, obj, values, window):
        """
        Switch between horizontal and vertical layout depending on the size of the canvas
        :param values:
        :param obj: dsc_mrp or dce_mra
        :param window:
        :return:
        """
        obj.set_fig_size(window)  # Re-compute the figure size on the new canvas
        cv_suffix = self.current_orientation + self.get_suffix()
        for k, v in obj.fig_canvas_aggs.items():
            window._reset_canvas(k, v, cv_suffix)
            if 'roi_reviewer' in k:
                s = obj.FIG_SIZE_ROI_REVIEWER
            elif 'smip' in k:  # Maximum image projection
                reshow_smip(obj, values, window, k, cv_suffix)
                continue
            elif 'phase' in k:  # Time - Intensity Figure
                s = obj.FIG_SIZE_PHASE_INTENSITY
            elif k == '_dce_cv_dyn_angio_':  # Dynamic Angio Preview
                obj.dyn_angio.set_fig_sz(obj.FIG_SIZE_DYN_ANGIO)
                obj.dyn_angio.show_fig_dyn_angio()
                obj.dyn_angio.dyn_angio_ax = obj.update_canvas(k, window, 5)
                obj.dyn_angio.display_preview_dyn_angio()
                continue
            else:
                s = obj.FIG_SIZE_DYN_ANGIO
            v.figure.set_size_inches(s)
            obj.update_canvas(k, window, figure=v.figure)
            # Re-connect the figure with its event handling
            if 'phase' in k:
                obj.mpl_connect_phase_intensity(obj.fig_canvas_aggs[k], values, window)
                fig = obj.fig_canvas_aggs[k].figure
                fig.subplots_adjust(0, 0, 1, 1)
                fig.tight_layout(pad=0.3)
            if 'trans_ref' in k:
                obj.trans_ref.mpl_connect(obj.fig_canvas_aggs[obj.trans_ref.cv_name], obj.process, values, window)

    def scale(self, *args):
        cv_suffix = self.current_orientation + self.get_suffix()
        scale_async(*args, cv_suffix=cv_suffix) if os.name == 'nt' else scale_sync(*args)

    def update_info(self, new_orientation, new_canvas_size, new_window_size):
        """

        :param new_orientation:
        :param new_canvas_size:
        :param new_window_size:
        :return:
        """
        self.current_orientation = new_orientation
        self.set_current_canvas_size(new_canvas_size)
        self.window_size = new_window_size

    @staticmethod
    def get_orientation(canvas_size):
        return 'vert' if canvas_size[0] <= canvas_size[1] else ''

    def get_suffix(self):
        return '_' if self.current_orientation == 'vert' else ''

    @staticmethod
    def get_new_canvas_size(window):
        window.refresh()
        dsc_cv_size = window['_dsc_canvases_'].get_size()
        dce_cv_size = window['_dce_canvases_'].get_size()
        new_canvas_size = dce_cv_size if 'dce' in window['_tab_group_'].Get() else dsc_cv_size
        return new_canvas_size

    def set_current_canvas_size(self, canvas_size):
        self.current_canvas_size = canvas_size

    def get_scale_coef(self, new_canvas_size):
        scale_coef_canvas = [n / c for n, c in zip(new_canvas_size, self.current_canvas_size)]
        scale_coef = scale_coef_canvas
        self.set_current_canvas_size(new_canvas_size)
        return scale_coef


def scale_async(obj, scale_coef: list, values, window, cv_suffix):
    fig_canvas_aggs = obj.fig_canvas_aggs
    for k in fig_canvas_aggs.keys():
        # window.refresh()  # refresh is needed for matplotlib to accurately calculate the axis size
        if 'smip' in k:  # Maximum image projection
            reshow_smip(obj, values, window, k, cv_suffix)
            continue
        widget = fig_canvas_aggs[k].get_tk_widget()
        widget.config(width=widget.winfo_width() * scale_coef[0],
                      height=widget.winfo_height() * scale_coef[1])


def scale_sync(obj, scale_coef: list, *arg):
    fig_canvas_aggs = obj.fig_canvas_aggs

    def _scale(_k):
        widget = fig_canvas_aggs[_k].get_tk_widget()
        widget.config(width=widget.winfo_width() * scale_coef[0],
                      height=widget.winfo_height() * scale_coef[1])

    for k in fig_canvas_aggs.keys():
        Thread(target=_scale, args=(k,), daemon=True).start()


def reshow_smip(obj, values, window, cv_name, cv_suffix):
    window._reset_canvas(cv_name, obj.fig_canvas_aggs[cv_name], cv_suffix)
    obj.show_smip(None, values, window)
