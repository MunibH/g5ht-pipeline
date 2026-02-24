import numpy as np              # For array operations, meshgrid creation, and numerical computations
import tifffile                # For reading and writing TIFF image stacks
import plotly.graph_objects as go     # For interactive 3D volume visualization
import matplotlib.pyplot as plt       # For 2D slice visualization
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For adding colorbars to matplotlib plots
from skimage import transform         # For image transformations like rotation, resizing, and scaling
import os
from scipy.ndimage import zoom        # For resizing images using interpolation
import imageio
import tqdm


def save_volume_movie(fig, filename='volume_rotation.mp4', fps=20, rotations=1):
    frames = []
    num_steps = 60  # Total frames for one full 360-degree rotation
    
    # Create a temporary directory for frames
    if not os.path.exists('temp_frames'):
        os.makedirs('temp_frames')

    print("Generating frames...")
    for i in tqdm.tqdm(range(num_steps * rotations), desc="Creating frames"):
        # Calculate new camera position (Rotating around Z-axis)
        # We use sine and cosine to move the 'eye' in a circle
        theta = 2 * np.pi * i / num_steps
        dist = 2.0  # Distance from the center
        
        fig.update_layout(scene_camera=dict(
            eye=dict(
                x=dist * np.cos(theta), 
                y=dist * np.sin(theta), 
                z=1.5
            )
        ))

        # Save frame as a static image
        frame_path = f'temp_frames/frame_{i}.png'
        fig.write_image(frame_path)
        frames.append(imageio.imread(frame_path))
        
    print(f"Stitching {len(frames)} frames into movie...")
    imageio.mimsave(filename, frames, fps=fps)
    
    # Cleanup
    for f in os.listdir('temp_frames'):
        os.remove(os.path.join('temp_frames', f))
    os.rmdir('temp_frames')
    print(f"Done! Video saved as {filename}")

# Usage:
# fig = show_cube_volumetric(volume_data)
# save_volume_movie(fig)

def show_volume_with_rois(volume, roi_volume, labels, plot_roi, colorscale='Greys_r'):
    nx, ny, nz = volume.shape
    x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
    
    # Flatten once for efficiency
    xf, yf, zf = x.flatten(), y.flatten(), z.flatten()
    
    from scipy.ndimage import gaussian_filter
    # volume_smoothed = gaussian_filter(volume, sigma=0.8)

    # 1. Base Data Volume (The "Cloud")
    fig = go.Figure(data=go.Volume(
        x=xf, y=yf, z=zf,
        value=volume.flatten(),
        isomin=100, isomax=500,
        opacity=0.1,
        surface_count=50,
        colorscale=colorscale,
        showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))

    # 2. Add ROIs
    if plot_roi:
        roi_ids = np.unique(roi_volume)
        roi_ids = roi_ids[roi_ids > 0]
        # roi_ids = [1,2,3,4,5,6,7] # discard (4 =DNC), render nerve ring (1) last so it's on top
        # procorpus (4)
        # metacorpus (7)
        # isthmus (3)
        # nerve ring (2)
        # VNC (1)
        # DNC (5)
        # terminal bulb(6)
        roi_ids = [4,7,3,1,6,2]
        
        roi_colors = ["#43AA8B", '#90BE6D', "#F3BD3E", '#F94144', "#000000", 
                    "#FFD981", "#FF9129", '#01FF70', '#F012BE', '#39CCCC']
        
        roi_colors = [roi_colors[i-1] for i in (roi_ids)]

        for i, roi_id in enumerate(roi_ids):
            # if i==4 or i ==1:
            #     continue # skip dorsal nerve cord (4), can't actually see it, and render nerve ring last (1)
            # 1. Binary-ish mask: Current ROI is the value, everything else is 0
            masked_roi = np.where(roi_volume == roi_id, roi_id, 0)
            # masked_roi = gaussian_filter(masked_roi, sigma=0.1) # Smooth it slightly to help with rendering
            
            # 2. Add the trace with a small THRESHOLD WINDOW
            fig.add_trace(go.Isosurface(
                x=xf, y=yf, z=zf,
                value=masked_roi.flatten(),
                # Crucial: Look for the boundary between 0 and roi_id
                isomin=roi_id * 0.5, 
                isomax=roi_id * 1.1, # Slightly above to capture the whole volume
                opacity=0.8,
                surface_count=3,     # Give it a bit more "meat" to render
                colorscale=[[0, roi_colors[i % len(roi_colors)]], 
                            [1, roi_colors[i % len(roi_colors)]]],
                showscale=False,
                name=labels[i] if i < len(labels) else f'ROI {int(roi_id)}',
                caps=dict(x_show=True, y_show=True, z_show=True), # Turn caps ON to see if it's hitting edges
                lighting=dict(
                    ambient=0.7,    # Lower ambient: makes shadows darker
                    diffuse=1.0,    # Higher diffuse: makes the shape's form more visible
                    specular=0.5,   # High specular: adds a "plastic" shine that defines the curve
                    roughness=0.1,  # Lower roughness: sharper highlights
                    fresnel=1.0     # HIGH Fresnel: This is the secret! 
                ),
            ))
            
        # # render nerve ring
        # i = 1
        # roi_id = 1
        # masked_roi = np.where(roi_volume == roi_id, roi_id, 0)
        # # masked_roi = gaussian_filter(masked_roi, sigma=0.1) # Smooth it slightly to help with rendering
        # fig.add_trace(go.Isosurface(
        #     x=xf, y=yf, z=zf,
        #     value=masked_roi.flatten(),
        #     # Crucial: Look for the boundary between 0 and roi_id
        #     isomin=roi_id * 0.5, 
        #     isomax=roi_id * 1.1, # Slightly above to capture the whole volume
        #     opacity=0.8,
        #     surface_count=3,     # Give it a bit more "meat" to render
        #     colorscale=[[0, roi_colors[i % len(roi_colors)]], 
        #                 [1, roi_colors[i % len(roi_colors)]]],
        #     showscale=False,
        #     name=labels[i] if i < len(labels) else f'ROI {int(roi_id)}',
        #     caps=dict(x_show=True, y_show=True, z_show=True), # Turn caps ON to see if it's hitting edges
        #     lighting=dict(
        #         ambient=0.7,    # Lower ambient: makes shadows darker
        #         diffuse=1.0,    # Higher diffuse: makes the shape's form more visible
        #         specular=0.5,   # High specular: adds a "plastic" shine that defines the curve
        #         roughness=0.1,  # Lower roughness: sharper highlights
        #         fresnel=1.0     # HIGH Fresnel: This is the secret! 
        #     ),
        # ))

    # fig.update_layout(
    #     scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
    #     width=900, height=900,
    #     legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    # )
    
    # 3. FIX ASPECT RATIO (Worms are long/thin, Z-slices are often thick)
    # Adjust z=0.2 if the worm looks too "tall"
    fig.update_layout(
        scene=dict(
            aspectratio=dict(x=0.15, y=0.55, z=1), 
            xaxis_title='Z',
            yaxis_title='Y',
            zaxis_title='X'
        ),
        width=900, height=900,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # rotate the whole figure
    fig.update_layout(scene_camera=dict(eye=dict(x=2, y=1.5, z=1)))
    # turn off axes
    fig.update_layout(scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    ))
    # turn off grid lines
    fig.update_layout(scene=dict(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        zaxis=dict(showgrid=False)
    ))
    # zoom in
    fig.update_layout(scene_camera=dict(eye=dict(x=0.75, y=0.75, z=0.75)))
    
    # # lighting
    # # lighting=dict(ambient=0.6, diffuse=0.5, specular=1, roughness=0.3, fresnel=0.2),
    # lighting=dict(ambient=1.0, diffuse=0.5, specular=1, roughness=1, fresnel=0.2),
    # lightposition=dict(x=100, y=100, z=1000)
    
    # show rois labels in a legend
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    
    return fig

if __name__ == "__main__":

    # --- 1. Load Data ---
    data_path = r'D:\DATA\g5ht-free\20251028\date-20251028_time-1500_strain-ISg5HT_condition-starvedpatch_worm001'
    data_fn = 'fixed_0166.tif'
    data_full_path = os.path.join(data_path, data_fn)

    z2keep = (2,25)
    # z2keep = (18,29)
    # z2keep = (12,15)

    plot_roi = True

    volume = tifffile.imread(data_full_path)[:,1,:,:] 
    volume = volume.astype(np.float32) # Ensure it's float for visualization

    # --- 2. Manual Downsampling (The 'Skip' Method) ---
    # This is faster and keeps ROI integers pure
    stride = 3 # This is your 0.25 factor (1/0.25 = 4)

    # Downsample volume (Red channel)
    volume = volume[z2keep[0]:z2keep[1], ::stride, ::stride]

    # Downsample ROI (Exact same skipping pattern)
    roi_volume = tifffile.imread(os.path.join(data_path, 'roi.tif'))
    roi_volume = roi_volume[z2keep[0]:z2keep[1], ::stride, ::stride]

    # Double check shapes match exactly
    assert volume.shape == roi_volume.shape, f"Shape mismatch: {volume.shape} vs {roi_volume.shape}"

    with tifffile.TiffFile(os.path.join(data_path, 'roi.tif')) as tif:
        labels = tif.imagej_metadata['Labels']

    print(f"Volume shape: {volume.shape}")
    print(f"ROI shape: {roi_volume.shape}")
    print(f"Labels: {labels}")

    # Create and display the visualization
    print("\nCreating visualization...")
    fig = show_volume_with_rois(volume, roi_volume, labels, plot_roi)
    save_volume_movie(fig,os.path.join(data_path, 'volume_rotation_with_roi.mp4'))
    
    # fig.show() # uncomment to view interactively in a browser instead of saving a movie
    
    
    # print("Saving interactive HTML...")
    # fig.write_html(os.path.join(data_path, "volume_plot.html"))
    
    
    plot_roi = False
    fig = show_volume_with_rois(volume, roi_volume, labels, plot_roi)
    save_volume_movie(fig,os.path.join(data_path, 'volume_rotation.mp4'))
