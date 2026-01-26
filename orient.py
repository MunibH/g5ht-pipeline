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


def orient_all(last_pt, spline_dict, constrain_frame=None, constrain_nose=None):
    out_dict = {}
    for i in range(len(spline_dict)):
        data = spline_dict[i]
        data_arr = np.array(data)
        # no spline sometimes if tracking failed and/or segmentation failed
        if len(data_arr) == 0:
            out_dict[i] = data
            continue
        # apply constraint if specified
        if constrain_frame is not None and i == constrain_frame:
            last_pt = np.array(constrain_nose)
        # determine orientation based on distance to last_pt
        dist_unflipped = np.linalg.norm(data_arr[0] - last_pt)
        dist_flip = np.linalg.norm(data_arr[-1] - last_pt)
        if dist_flip < dist_unflipped:
            data = data[::-1]
        # Truncate first to ensure consistency
        data = data[:350]
        # Update last_pt with the first point of the truncated, oriented spline
        last_pt = np.array(data[0])
        out_dict[i] = data
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

    # last_pt is represented as (y,x)
    last_pt = [np.clip(int(i), 0, 512) for i in sys.argv[2:4]]
    
    # optional constraints
    if len(sys.argv) == 7:
        constrain_frame = int(sys.argv[4])
        constrain_nose = [np.clip(int(i), 0, 512) for i in sys.argv[5:7]]
    else:
        constrain_frame = None
        constrain_nose = None

    #reads spline
    with open(os.path.join(spline_path,'spline.json'), 'r') as f:
        spline_dict = json.load(f)
    spline_dict = {int(k): v for k, v in spline_dict.items()}

    #orients all
    out_dict = orient_all(last_pt, spline_dict, constrain_frame, constrain_nose)

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
        if len(out_dict[i]) == 0:
            continue
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
        if len(out_dict[i]) == 0:
            # If no spline data, just capture the current frame
            canvas.draw()
            s, (width, height) = canvas.print_to_buffer()
            frame_rgb = np.frombuffer(s, np.uint8).reshape((height, width, 4))[:, :, :3]
            visualization_frames.append(frame_rgb)
            continue
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
