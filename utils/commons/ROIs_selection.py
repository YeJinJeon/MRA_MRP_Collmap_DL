import pickle
import pylab as plt
from matplotlib.widgets import RectangleSelector, EllipseSelector, PolygonSelector
from matplotlib.path import Path
import numpy as np


def get_y(x, a, b, h, k):
    """For ellipse"""
    tmp = b ** 2 - (b / a) ** 2 * (x - h) ** 2
    tmp = 0 if tmp < 0 else tmp  # In some case tmp is a negative number approaching 0
    y1 = -np.sqrt(tmp) + k
    y2 = +np.sqrt(tmp) + k
    return y1, y2


def get_x(y, a, b, h, k):
    """For ellipse"""
    tmp = a ** 2 - (a / b) ** 2 * (y - k) ** 2
    tmp = 0 if tmp < 0 else tmp  # In some case tmp is a negative number approaching 0

    x1 = -np.sqrt(tmp) + h
    x2 = +np.sqrt(tmp) + h
    return x1, x2


class RoiSelector:
    def __init__(self, selector_type, phase=None, prefix='dce'):
        self.triggered = False
        self.phase = phase
        self.selector_type = selector_type
        self.selector_dict = {'Ellipse': EllipseSelector, 'Rectangle': RectangleSelector, 'Polynomial': PolygonSelector}
        self.mask = None
        self.clicked = False
        self.phases = {
            f'_{prefix}_roi_art_': {'color': 'r'},
            f'_{prefix}_roi_vein_': {'color': 'b'},
        }
        self.set_selector()

    def set_selector(self, selector_type=None):
        self.selector_type = selector_type if selector_type is not None else self.selector_type
        edge_color = 'k' if self.phase is None else self.phases[self.phase]['color']

        selector = self.selector_dict[self.selector_type]
        if self.selector_type == 'Polynomial':
            self.rs = selector(plt.gca(), self.onselect_poly, useblit=True,
                               markerprops=dict(marker='x', markersize=None, markeredgecolor=edge_color),
                               vertex_select_radius=15,
                               )
        else:
            self.rs = selector(plt.gca(), self.onselect, drawtype='box', useblit=True, button=[1], minspanx=2,
                               minspany=2,
                               spancoords='pixels', interactive=True,
                               rectprops=dict(facecolor=edge_color, edgecolor=edge_color, alpha=1, fill=False),
                               state_modifier_keys=dict(move=' ', clear='escape', square='shift', center='ctrl'),
                               marker_props=dict(marker='x', markersize=None, markeredgecolor=edge_color),
                               )

    def update(self, phase=None):
        """Update selector"""
        if (phase != self.phase) and (phase is not None):
            self.phase = phase
            color = self.phases[phase]['color']
            if self.selector_type != 'Polynomial':
                self.rs.to_draw.set_color(color)
                self.rs._corner_handles._markers.set_markeredgecolor(color)
                self.rs._edge_handles._markers.set_markeredgecolor(color)
                self.rs._center_handle._markers.set_markeredgecolor(color)
            else:
                self.rs.line.set_color(color)
                self.rs._polygon_handles._markers.set_color(color)

    def onselect(self, eclick, erelease):
        """eclick and erelease are matplotlib events at press and release"""
        self.ecx, self.ecy = int(eclick.xdata), int(eclick.ydata)
        self.erx, self.ery = int(erelease.xdata), int(erelease.ydata)
        self.clicked = True

    def onselect_poly(self, eclicks):
        """eclick and erelease are matplotlib events at press and release"""
        self.poly_verts = eclicks
        self.clicked = True

    def toggle_selector(self, event):
        if event.dblclick:
            event.key = 'q'
            self.pressed = True
            # Re-create a new selector
            self.set_selector()
            plt.draw()
        if event.key in ['Q', 'q'] and self.rs.active:
            self.triggered = True if self.clicked else False
        if event.key in ['A', 'a'] and not self.rs.active:
            self.rs.set_active(True)
        self.event = event

    def gen_bin_roi(self, img_shape):
        if self.selector_type == 'Polynomial':
            "from https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask"
            nx, ny = img_shape
            x, y = np.meshgrid(np.arange(nx), np.arange(ny))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            path = Path(self.poly_verts)
            grid = path.contains_points(points)
            mask = grid.reshape((ny, nx))
        else:
            mask = np.zeros(img_shape)
            if self.selector_type == 'Rectangle':
                mask[self.ecy: self.ery, self.ecx:self.erx] = 1
            else:
                rx = (self.erx - self.ecx) / 2
                ry = (self.ery - self.ecy) / 2
                cx, cy = (self.erx + self.ecx) / 2, (self.ery + self.ecy) / 2
                if rx >= ry:
                    for x in np.arange(self.ecx, self.erx, 1e-2):
                        y1, y2 = get_y(x, rx, ry, cx, cy)
                        if np.isnan(y1) or np.isnan(y1):
                            print(x, rx, ry, cx, cy)
                        mask[int(round(y1)):int(round(y2)), int(round(x))] = 1
                else:
                    for y in np.arange(self.ecy, self.ery, 1e-2):
                        x1, x2 = get_x(y, rx, ry, cx, cy)
                        if np.isnan(x1) or np.isnan(x2):
                            print(y, rx, ry, cx, cy)
                        mask[int(round(y)), int(round(x1)):int(round(x2))] = 1
            self.mask = mask
        return mask


if __name__ == "__main__":
    pass
