import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tqdm
import os
import sys
from skimage import morphology, measure
import tifffile
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.ndimage import distance_transform_edt

def _as_yx_array(data):
    return np.asarray(data, dtype=np.float32)

def _clip_yx(yx, shape):
    y, x = int(yx[0]), int(yx[1])
    y = np.clip(y, 0, shape[0] - 1)
    x = np.clip(x, 0, shape[1] - 1)
    return np.array([y, x], dtype=np.int32)

def bootstrap_nose_from_mask_and_centerline(spline0_yx, mask0, k=40):
    """
    Decide which end of the centerline is the nose on frame 0.
    Uses a thickness proxy: distance-to-boundary (via distance transform)
    averaged over first k points from each end.

    In a body mask, the tail is often thinner; the nose/head can be thicker near pharynx.
    This heuristic is not perfect, but it's decent and only used for frame 0 bootstrap.
    """
    pts = _as_yx_array(spline0_yx)
    if pts.shape[0] < 2:
        return pts[0].tolist()

    k = min(k, pts.shape[0] // 2)
    # distance to nearest background pixel (radius proxy)
    dt = distance_transform_edt(mask0.astype(bool))

    def mean_radius(prefix_pts):
        ys = np.clip(prefix_pts[:, 0].astype(int), 0, mask0.shape[0] - 1)
        xs = np.clip(prefix_pts[:, 1].astype(int), 0, mask0.shape[1] - 1)
        return float(np.mean(dt[ys, xs]))

    r0 = mean_radius(pts[:k])
    r1 = mean_radius(pts[-k:])

    # pick the end with larger average radius as "head/nose-side"
    # (swap if in your masks tail tends to be thicker)
    nose = pts[0] if r0 <= r1 else pts[-1]
    return _clip_yx(nose, mask0.shape).tolist()

def orient_all_tracked(spline_dict, mask_stack, crop_n=350,
                       bootstrap_k=40,
                       jump_mult=6.0,
                       jump_px_min=12.0):
    """
    Robust orientation using last nose position + sanity checks.

    - Bootstraps nose on frame 0 from mask+centerline thickness.
    - For each subsequent frame:
        choose endpoint closest to last nose
        sanity check: if jump is implausibly big, try the other endpoint;
                      if still bad, keep previous orientation (don't flip).
    """
    out_dict = {}

    # ---- bootstrap last_nose from frame 0
    s0 = _as_yx_array(spline_dict[0])
    last_nose = bootstrap_nose_from_mask_and_centerline(s0, mask_stack[0], k=bootstrap_k)
    last_nose = np.asarray(last_nose, dtype=np.float32)

    # keep a running estimate of typical motion (median of accepted jumps)
    accepted_jumps = []

    for i in range(len(spline_dict)):
        pts = _as_yx_array(spline_dict[i])
        if pts.shape[0] < 2:
            out_dict[i] = pts.tolist()
            continue

        p0 = pts[0]
        p1 = pts[-1]

        d0 = float(np.linalg.norm(p0 - last_nose))
        d1 = float(np.linalg.norm(p1 - last_nose))

        # choose closest to last nose
        choose_flip = d1 < d0
        chosen_d = min(d0, d1)
        other_d  = max(d0, d1)

        # dynamic jump threshold: based on history, with a floor
        if accepted_jumps:
            med = float(np.median(accepted_jumps))
            jump_thresh = max(jump_px_min, jump_mult * med)
        else:
            # early frames: be permissive but not infinite
            jump_thresh = 40.0  # tweak if needed

        # if chosen jump seems too large, try the other endpoint
        if chosen_d > jump_thresh and other_d < chosen_d:
            choose_flip = not choose_flip
            chosen_d = other_d

        # if STILL too large, we assume spline is bad this frame; don't flip,
        # and keep last_nose (i.e., orient by continuity, but don't update)
        if chosen_d > jump_thresh:
            # do not update last_nose, but still output something oriented
            # prefer orientation that keeps endpoint closer to last_nose
            choose_flip = d1 < d0
            pts_oriented = pts[::-1] if choose_flip else pts
            out_dict[i] = pts_oriented[:crop_n].tolist()
            continue

        # accept orientation
        pts_oriented = pts[::-1] if choose_flip else pts
        out_dict[i] = pts_oriented[:crop_n].tolist()

        # update last_nose and motion stats
        last_nose = pts_oriented[0]
        accepted_jumps.append(chosen_d)

    return out_dict


def visualize_frame(seg, nodes, spline_dilation=4):
    out = np.zeros(seg.shape, np.bool)
    for node in nodes:
        out[node] = True
    # out = morphology.isotropic_dilation(out, spline_dilation)
    return np.logical_and(seg, np.logical_not(out))

def main():

    fullfile_spline  = sys.argv[1]
    spline_path = os.path.dirname(fullfile_spline)

    # reads spline
    with open(os.path.join(spline_path,'spline.json'), 'r') as f:
        spline_dict = json.load(f)
    spline_dict = {int(k): v for k, v in spline_dict.items()}

    # load mask stack (dilated.tif is body mask)
    dilated = tifffile.imread(os.path.join(spline_path, 'dilated.tif')).astype(bool)

    # If user provided nose_y nose_x, you *can* still support it, but you no longer need it.
    # We'll ignore sys.argv[2:4] and do fully automatic tracked orientation:
    out_dict = orient_all_tracked(spline_dict, dilated)


    #saves outputs
    with open(os.path.join(spline_path, 'oriented.json'), 'w') as f:
        json.dump(out_dict, f, indent=4)

    #saves visual along with orientation
    dilated = tifffile.imread(os.path.join(spline_path, 'dilated.tif'))



    #plots oriented spline and saves visual as tif
    # visualization = np.zeros(dilated.shape, np.bool)
    plt.figure()
    plt.imshow(np.ones((512, 512)), cmap='gray', vmin=0, vmax=1)
    cmap = plt.get_cmap('viridis')
    for i in tqdm.tqdm(range(len(spline_dict))):
        y, x = np.array(out_dict[i]).T
        color = cmap(i / (len(spline_dict) - 1))
        plt.scatter(x[0], y[0], color=color) # plot the nose
        plt.plot(x, y, color=color)
        # dilated_frame = dilated[i,:,:]
        # visualization[i,:,:] = visualize_frame(dilated_frame, out_dict[i])
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(spline_path, 'oriented.png'))
    # tifffile.imwrite(os.path.join(spline_path, 'oriented.tif'), visualization)

    # 1. Initialize a list to hold the rendered image arrays
    visualization_frames = []

    # 2. Get the colormap (adjust 'viridis' if a different one is needed)
    cmap = cm.get_cmap('viridis')

    # 3. Create a single figure instance outside the loop
    # We will clear and reuse this figure for each frame
    fig = plt.figure(figsize=(dilated.shape[2] / 100, dilated.shape[1] / 100), dpi=100) # Adjust figsize/dpi as needed
    canvas = FigureCanvasAgg(fig) # Use the non-interactive Agg canvas

    print("Rendering and collecting frames...")

    for i in tqdm.tqdm(range(len(spline_dict))):
        # Clear the figure for the new plot
        fig.clf() 
        ax = fig.add_subplot(111)

        # Plot the base image
        ax.imshow(dilated[i,:,:], cmap='gray', vmin=0, vmax=1)

        # Get line and point data
        y, x = np.array(out_dict[i]).T
        color = cmap(i / (len(spline_dict) - 1)) # Get the color for this frame

        # Plot the point (nose) and the line
        ax.scatter(x[0], y[0], color=color, s=50) # s is the marker size
        ax.plot(x, y, color=color, linewidth=2)
        
        # Configure the plot to remove axes, which is common for visualization stacks
        ax.set_axis_off() 
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0) # Remove white margins

        # Render the figure to the canvas
        canvas.draw()
        
        # Capture the image data (RGBA format - 4 layers)
        s, (width, height) = canvas.print_to_buffer()
        
        # Convert the buffer data to a NumPy array (H, W, 4) and take only the RGB layers
        frame_rgb = np.frombuffer(s, np.uint8).reshape((height, width, 4))[:, :, :3]
        
        # Add the rendered frame to the list
        visualization_frames.append(frame_rgb)

    plt.close(fig) # Close the figure object to free up memory

    # 4. Stack and save the images
    if visualization_frames:
        # Stack all frames into a single 4D array (T, Y, X, C)
        stack_data = np.stack(visualization_frames)
        
        # Define the output path
        output_filename = os.path.join(spline_path, 'oriented_stack.tif')
        
        print(f"Saving TIFF stack to: {output_filename}")
        # Save the stack using tifffile
        # tifffile.imwrite(output_filename, stack_data, compression=1) # compress=1 is ZLIB compression
        # tifffile.imwrite(
        #     output_filename,
        #     stack_data,
        #     compression="lzma",
        # )
        tifffile.imwrite(
            output_filename,
            stack_data,
            photometric="rgb",
            compression="deflate",
            compressionargs={"level": 9},  # max deflate
        )


        print("TIFF stack saved successfully.")
    else:
        print("No frames were generated.")
