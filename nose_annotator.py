import os
import json
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# class NoseAnnotatorWithScroll:
#     def __init__(self, nd2_paths):
#         self.nd2_paths = nd2_paths
#         self.dataset_idx = 0
#         self.annotations = {}  
#         self.label = None
#         self.spline_dict = None
#         self.out_pth = None
#         self.num_frames = 0

#         self._build_ui()
#         self._load_dataset()

#     def _build_ui(self):
#         self.fig, self.ax = plt.subplots(figsize=(7, 7))
#         self.fig.canvas.header_visible = False
#         self.fig.canvas.toolbar_position = 'left'

#         self.slider = widgets.IntSlider(
#             value=0, min=0, max=0, step=1,
#             description='Frame:', layout=widgets.Layout(width='500px'),
#             continuous_update=False,
#         )
#         self.slider.observe(self._on_slider_change, names='value')

#         self.info_label = widgets.HTML(value='')
#         self.annot_label = widgets.HTML(value='')

#         btn_save = widgets.Button(description='Save CSV', button_style='success',
#                                   icon='save', layout=widgets.Layout(width='120px'))
#         btn_prev = widgets.Button(description='Prev Dataset', icon='arrow-left',
#                                   layout=widgets.Layout(width='130px'))
#         btn_next = widgets.Button(description='Next Dataset', icon='arrow-right',
#                                   layout=widgets.Layout(width='130px'))
#         btn_del = widgets.Button(description='Delete Annot.', button_style='danger',
#                                  icon='trash', layout=widgets.Layout(width='130px'))

#         btn_save.on_click(lambda _: self.save_csv())
#         btn_prev.on_click(lambda _: self._change_dataset(-1))
#         btn_next.on_click(lambda _: self._change_dataset(1))
#         btn_del.on_click(lambda _: self._delete_annotation())

#         nav_row = widgets.HBox([btn_prev, btn_next, btn_save, btn_del])
#         self.ui = widgets.VBox([self.info_label, self.slider, nav_row, self.annot_label])
#         display(self.ui)

#         # Connect the click event
#         self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        
#         ### NEW ###: Connect the scroll event to the canvas
#         self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

#     # ... (skipping _load_dataset, save_csv, and _draw_frame to keep it concise)
#     # ... (they remain exactly as you had them)

#     def _load_dataset(self):
#         nd2 = self.nd2_paths[self.dataset_idx]
#         self.out_pth = os.path.splitext(nd2)[0]
#         label_path = os.path.join(self.out_pth, 'label.tif')
#         spline_path = os.path.join(self.out_pth, 'spline.json')

#         if not os.path.exists(label_path) or not os.path.exists(spline_path):
#             self._set_info(f'<b style="color:orange">Skipped</b> — missing files.')
#             self.label = None
#             self.spline_dict = None
#             self.annotations = {}
#             self.num_frames = 0
#             self.slider.max = 0
#             self.slider.value = 0
#             self.ax.clear()
#             self.fig.canvas.draw_idle()
#             return

#         self.label = tifffile.imread(label_path)
#         with open(spline_path, 'r') as f:
#             self.spline_dict = {int(k): v for k, v in json.load(f).items()}

#         self.num_frames = self.label.shape[0]
#         self.slider.max = self.num_frames - 1
#         self.slider.value = 0

#         csv_path = os.path.join(self.out_pth, 'orient_nose.csv')
#         self.annotations = {}
#         if os.path.exists(csv_path):
#             df = pd.read_csv(csv_path)
#             for _, row in df.iterrows():
#                 self.annotations[int(row['frame'])] = (int(row['nose_y']), int(row['nose_x']))

#         self._set_info(f'<b>Dataset {self.dataset_idx + 1}/{len(self.nd2_paths)}</b>')
#         self._draw_frame(0)

#     def save_csv(self):
#         if self.out_pth is None or not self.annotations:
#             return
#         rows = sorted(self.annotations.items())
#         df = pd.DataFrame(rows, columns=['frame', 'nose_yx'])
#         df['nose_y'] = df['nose_yx'].apply(lambda t: t[0])
#         df['nose_x'] = df['nose_yx'].apply(lambda t: t[1])
#         csv_path = os.path.join(self.out_pth, 'orient_nose.csv')
#         df[['frame', 'nose_y', 'nose_x']].to_csv(csv_path, index=False)

#     def _draw_frame(self, frame_idx):
#         ax = self.ax
#         ax.clear()
#         if self.label is None: return
#         ax.imshow(self.label[frame_idx], cmap='gray')
#         spline = self.spline_dict.get(frame_idx, [])
#         if len(spline) > 0:
#             pts = np.array(spline)
#             ax.plot(pts[:, 1], pts[:, 0], 'c-', linewidth=2)
#         if frame_idx in self.annotations:
#             ny, nx = self.annotations[frame_idx]
#             ax.plot(nx, ny, marker='*', color='yellow', markersize=18)
#         ax.set_title(f'Frame {frame_idx}')
#         self.fig.canvas.draw_idle()
#         self._update_annot_label()

#     # -------------------------------------------------------- callbacks
#     def _on_slider_change(self, change):
#         self._draw_frame(change['new'])

#     ### NEW ###: The actual scroll handling logic
#     def _on_scroll(self, event):
#         """Scroll through frames with the mouse wheel."""
#         if event.inaxes is not self.ax:
#             return
        
#         # event.step is +1 for scrolling up, -1 for scrolling down
#         increment = int(event.step)
#         new_frame = self.slider.value + increment
        
#         # Clamp value within the valid range
#         if 0 <= new_frame <= self.slider.max:
#             self.slider.value = new_frame

#     def _on_click(self, event):
#         if event.inaxes is not self.ax or self.label is None:
#             return
#         if self.fig.canvas.toolbar.mode != '':
#             return
#         ny, nx = int(round(event.ydata)), int(round(event.xdata))
#         self.annotations[self.slider.value] = (ny, nx)
#         self._draw_frame(self.slider.value)

#     def _delete_annotation(self):
#         frame = self.slider.value
#         if frame in self.annotations:
#             del self.annotations[frame]
#             self._draw_frame(frame)

#     def _change_dataset(self, direction):
#         if self.annotations: self.save_csv()
#         new_idx = self.dataset_idx + direction
#         if 0 <= new_idx < len(self.nd2_paths):
#             self.dataset_idx = new_idx
#             self._load_dataset()

#     def _set_info(self, html):
#         self.info_label.value = html

#     def _update_annot_label(self, msg=None):
#         self.annot_label.value = msg if msg else f"Frames: {list(self.annotations.keys())}"


# # Before adding in scroll wheel functionality, see code below


class NoseAnnotator:
    """Interactive widget for annotating nose positions on spline overlays.
    
    Usage (in a notebook cell with %matplotlib widget):
        annotator = NoseAnnotator(nd2_paths)
    """

    def __init__(self, nd2_paths):
        self.nd2_paths = nd2_paths
        self.dataset_idx = 0
        self.annotations = {}  # {frame: (nose_y, nose_x)}
        self.label = None
        self.spline_dict = None
        self.out_pth = None
        self.num_frames = 0

        self._build_ui()
        self._load_dataset()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.canvas.header_visible = False
        self.fig.canvas.toolbar_position = 'left'

        self.slider = widgets.IntSlider(
            value=0, min=0, max=0, step=1,
            description='Frame:', layout=widgets.Layout(width='500px'),
            continuous_update=False,
        )
        self.slider.observe(self._on_slider_change, names='value')

        self.info_label = widgets.HTML(value='')
        self.annot_label = widgets.HTML(value='')

        btn_save = widgets.Button(description='Save CSV', button_style='success',
                                  icon='save', layout=widgets.Layout(width='120px'))
        btn_prev = widgets.Button(description='Prev Dataset', icon='arrow-left',
                                  layout=widgets.Layout(width='130px'))
        btn_next = widgets.Button(description='Next Dataset', icon='arrow-right',
                                  layout=widgets.Layout(width='130px'))
        btn_del = widgets.Button(description='Delete Annot.', button_style='danger',
                                 icon='trash', layout=widgets.Layout(width='130px'))

        btn_save.on_click(lambda _: self.save_csv())
        btn_prev.on_click(lambda _: self._change_dataset(-1))
        btn_next.on_click(lambda _: self._change_dataset(1))
        btn_del.on_click(lambda _: self._delete_annotation())

        nav_row = widgets.HBox([btn_prev, btn_next, btn_save, btn_del])
        self.ui = widgets.VBox([self.info_label, self.slider, nav_row, self.annot_label])
        display(self.ui)

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

    # --------------------------------------------------------- dataset I/O
    def _load_dataset(self):
        """Load label.tif, spline.json, and any existing orient_nose.csv."""
        nd2 = self.nd2_paths[self.dataset_idx]
        self.out_pth = os.path.splitext(nd2)[0]

        label_path = os.path.join(self.out_pth, 'label.tif')
        spline_path = os.path.join(self.out_pth, 'spline.json')


        if not os.path.exists(label_path) or not os.path.exists(spline_path):
            self._set_info(f'<b style="color:orange">Skipped</b> — '
                           f'missing label.tif or spline.json for {os.path.basename(nd2)}')
            self.label = None
            self.spline_dict = None
            self.annotations = {}
            self.num_frames = 0
            self.slider.max = 0
            self.slider.value = 0
            self.ax.clear()
            self.ax.set_title('No data')
            self.fig.canvas.draw_idle()
            self._update_annot_label()
            return

        self.label = tifffile.imread(label_path)            # (T, H, W)
        with open(spline_path, 'r') as f:
            self.spline_dict = {int(k): v for k, v in json.load(f).items()}

        self.num_frames = self.label.shape[0]
        self.slider.max = self.num_frames - 1
        self.slider.value = 0

        # load existing annotations if present
        csv_path = os.path.join(self.out_pth, 'orient_nose.csv')
        self.annotations = {}
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.annotations[int(row['frame'])] = (int(row['nose_y']), int(row['nose_x']))

        self._set_info(f'<b>Dataset {self.dataset_idx + 1}/{len(self.nd2_paths)}</b> — '
                       f'{os.path.basename(self.nd2_paths[self.dataset_idx])}')
        self._draw_frame(0)

    def save_csv(self):
        """Write annotations to orient_nose.csv in the dataset directory."""
        if self.out_pth is None or not self.annotations:
            self._update_annot_label('Nothing to save.')
            return
        rows = sorted(self.annotations.items())
        df = pd.DataFrame(rows, columns=['frame', 'nose_yx'])
        df['nose_y'] = df['nose_yx'].apply(lambda t: t[0])
        df['nose_x'] = df['nose_yx'].apply(lambda t: t[1])
        df = df[['frame', 'nose_y', 'nose_x']]
        csv_path = os.path.join(self.out_pth, 'orient_nose.csv')
        df.to_csv(csv_path, index=False)
        self._update_annot_label(f'Saved {len(df)} annotation(s) to orient_nose.csv')

    # -------------------------------------------------------- drawing
    def _draw_frame(self, frame_idx):
        ax = self.ax
        ax.clear()

        if self.label is None:
            ax.set_title('No data')
            self.fig.canvas.draw_idle()
            return

        ax.imshow(self.label[frame_idx], cmap='gray')

        # spline overlay
        spline = self.spline_dict.get(frame_idx, [])
        if len(spline) > 0:
            pts = np.array(spline)
            ax.plot(pts[:, 1], pts[:, 0], 'c-', linewidth=2)
            ax.plot(pts[0, 1], pts[0, 0], 'go', markersize=8)   # first point
            ax.plot(pts[-1, 1], pts[-1, 0], 'ro', markersize=8)  # last point

        # nose annotation
        if frame_idx in self.annotations:
            ny, nx = self.annotations[frame_idx]
            ax.plot(nx, ny, marker='*', color='yellow', markersize=18,
                    markeredgecolor='k', markeredgewidth=0.5)

        title = f'Frame {frame_idx}/{self.num_frames - 1}'
        if frame_idx in self.annotations:
            ny, nx = self.annotations[frame_idx]
            title += f'  |  nose=({ny}, {nx})'
        ax.set_title(title)
        ax.axis('equal')
        self.fig.canvas.draw_idle()
        self._update_annot_label()

    # -------------------------------------------------------- callbacks
    def _on_slider_change(self, change):
        self._draw_frame(change['new'])

    def _on_click(self, event):
        if event.inaxes is not self.ax or self.label is None:
            return
        # ignore toolbar interactions (zoom/pan)
        if self.fig.canvas.toolbar.mode != '':
            return
        ny, nx = int(round(event.ydata)), int(round(event.xdata))
        frame = self.slider.value
        self.annotations[frame] = (ny, nx)
        self._draw_frame(frame)

    def _on_scroll(self, event):
        if event.inaxes is not self.ax:
            return
        new_frame = self.slider.value - int(event.step)
        new_frame = max(0, min(new_frame, self.slider.max))
        if new_frame != self.slider.value:
            self.slider.value = new_frame

    def _delete_annotation(self):
        frame = self.slider.value
        if frame in self.annotations:
            del self.annotations[frame]
            self._draw_frame(frame)
            self._update_annot_label(f'Deleted annotation for frame {frame}')

    def _change_dataset(self, direction):
        # auto-save current annotations
        if self.annotations:
            self.save_csv()
        new_idx = self.dataset_idx + direction
        if new_idx < 0 or new_idx >= len(self.nd2_paths):
            self._update_annot_label('No more datasets in that direction.')
            return
        self.dataset_idx = new_idx
        self._load_dataset()

    # -------------------------------------------------------- labels
    def _set_info(self, html):
        self.info_label.value = html

    def _update_annot_label(self, msg=None):
        parts = []
        if self.annotations:
            frames = sorted(self.annotations.keys())
            parts.append(f'Annotated frames: {frames}')
        else:
            parts.append('No annotations yet.')
        if msg:
            parts.append(msg)
        self.annot_label.value = '<br>'.join(parts)
