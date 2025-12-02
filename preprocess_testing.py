from nd2reader import ND2Reader
import numpy as np
import itk
import sys
import os
import tifffile
import warnings; warnings.filterwarnings('ignore', category=UserWarning, module='itk')

from utils import get_noise_stack

from scipy.interpolate import UnivariateSpline
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation

# noise_path = '/home/albert_w/scripts/noise_042925.tif'
# noise_tif = tifffile.imread(noise_path)
# noise_stack = np.stack([noise_tif] * 41, axis=0).astype(np.float32)

def get_stack(input_nd2, index, noise_stack, stack_shape=(41, 2, 512, 512), trim=2):
    """Extracts and preprocesses a specific stack from the ND2 file, returns float32 array with trimmed z-slices."""

    if stack_shape[0]==1:
        noise_stack = np.mean(noise_stack,axis=0)
        noise_stack = noise_stack[np.newaxis,:,:,:]
    elif stack_shape[0] != 41:
        noise_stack = np.mean(noise_stack,axis=0)
        noise_stack = noise_stack[np.newaxis,:,:,:]

    stack = np.zeros(stack_shape, np.float32)
    frame_indices = np.arange(stack_shape[0] * index, stack_shape[0] * (index + 1))
    with ND2Reader(input_nd2) as f:
        for i, j in enumerate(frame_indices):
            stack[i] = f.get_frame_2D(0, j), f.get_frame_2D(1, j)
    denoised = np.clip(stack - noise_stack, 0, 4095)

    if stack_shape[0]==1:
        return denoised
    else:
        return denoised[:-trim]

def load_parameter_object(parameter_file: str) -> itk.ParameterObject:
    """Create an ITK Elastix ParameterObject from a text parameter file."""
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(parameter_file)
    # parameter_object.ReadParameterFile(parameter_file)
    return parameter_object


def register(fixed, moving, parameter_object, threads=8):
    """Performs rigid registration between two images using ITK's elastix with binary masks."""
    #fixed_mask = itk.image_view_from_array((fixed > 0).astype(np.ubyte))
    #moving_mask = itk.image_view_from_array((moving > 0).astype(np.ubyte))
    return itk.elastix_registration_method(fixed, moving, #fixed_mask=fixed_mask, moving_mask=moving_mask,
                                           parameter_object=parameter_object, number_of_threads=threads)
                                           #log_to_file=True, log_file_name='test.log', output_directory='.')

def shear_correct(stack, parameter_object):
    """Performs shear correction using RFP channel as reference, propagating from max intensity slice."""
    rfp_means = np.mean(stack[:, 1], axis=(1, 2))
    max_rfp_z = np.argmax(rfp_means)
    output = stack.copy()

    for i in range(max_rfp_z - 1, -1, -1):
        rfp_fixed, rfp_moving = output[i+1, 1], output[i, 1]
        rfp_reg, params = register(rfp_fixed, rfp_moving, parameter_object)
        gfp_reg = itk.transformix_filter(output[i, 0], params)
        output[i] = gfp_reg, rfp_reg
    
    for i in range(max_rfp_z + 1, len(output)):
        rfp_fixed, rfp_moving = output[i-1, 1], output[i, 1]
        rfp_reg, params = register(rfp_fixed, rfp_moving, parameter_object)
        gfp_reg = itk.transformix_filter(output[i, 0], params)
        output[i] = gfp_reg, rfp_reg
    return output

def shear_correct_single_ref(stack, parameter_object):
    """Shear correction: each slice registered to a single reference RFP slice."""
    rfp = stack[:, 1]
    gfp = stack[:, 0]
    rfp_means = np.mean(rfp, axis=(1, 2))
    ref_z = np.argmax(rfp_means)

    ref_rfp = rfp[ref_z]
    corrected = stack.copy()

    for i in range(len(stack)):
        if i == ref_z:
            continue
        moving_rfp = rfp[i]
        rfp_reg, params = register(ref_rfp, moving_rfp, parameter_object)
        gfp_reg = itk.transformix_filter(gfp[i], params)
        corrected[i] = np.stack([gfp_reg, rfp_reg], axis=0)

    return corrected


def shear_correct_progressive(stack, parameter_object, threads=8):
    """
    Shear correction by progressively registering growing groups of z-slices.

    Algorithm (for the RFP channel, ch=1):
      1. Find z_ref = argmax_z mean(RFP[z]).
      2. Forward pass (0 -> z_ref):
         - Start group = {0}.
         - For z in 1..z_ref:
             * Register group-average RFP to slice z (fixed = slice z, moving = group avg).
             * Apply transform to all slices in 'group' (both channels).
             * Add z to 'group', recompute group-average.
      3. Backward pass (Z-1 -> z_ref):
         - Start group = {Z-1}.
         - For z in (Z-2 .. z_ref):
             * Register group-average RFP to slice z (fixed = slice z, moving = group avg).
             * Apply transform to all slices in 'group' (both channels).
             * Add z to 'group', recompute group-average.
      4. z_ref itself is never transformed; everything else is aligned into its frame.

    Parameters
    ----------
    stack : np.ndarray
        Array of shape (Z, 2, H, W). Channel 0 = GFP, 1 = RFP.
    parameter_object : itk.ParameterObject
        Elastix parameter object.
    threads : int
        Number of threads for elastix.

    Returns
    -------
    corrected : np.ndarray
        Shear-corrected stack (same shape, float32).
    """

    # Work in float32 for registration
    stack = stack.astype(np.float32, copy=False)
    corrected = stack.copy()

    Z, C, H, W = corrected.shape
    assert C >= 2, "Expected stack with at least 2 channels (GFP, RFP)."

    GFP_CH = 0
    RFP_CH = 1

    # --- 1. Find reference slice (max mean RFP) ---
    rfp_means = np.mean(corrected[:, RFP_CH], axis=(1, 2))
    ref_z = int(np.argmax(rfp_means))

    # Helper to compute group-average RFP image
    def group_mean_rfp(indices):
        return np.mean([corrected[idx, RFP_CH] for idx in indices], axis=0)

    # --- 2. Forward pass: from first slice up to ref_z ---
    if ref_z > 0:
        group_indices = [0]
        group_rfp = corrected[0, RFP_CH]

        for z in range(1, ref_z + 1):
            fixed_rfp = corrected[z, RFP_CH]  # treat current slice as fixed

            # Register group-average (moving) to fixed
            rfp_group_reg, params = register(
                fixed_rfp, group_rfp, parameter_object, threads=threads
            )

            # Apply this transform to all slices currently in the group
            for idx in group_indices:
                corrected[idx, RFP_CH] = itk.transformix_filter(
                    corrected[idx, RFP_CH], params
                )
                corrected[idx, GFP_CH] = itk.transformix_filter(
                    corrected[idx, GFP_CH], params
                )

            # Now extend the group with current slice and recompute group-average
            if z not in group_indices:
                group_indices.append(z)
            group_rfp = group_mean_rfp(group_indices)

    # --- 3. Backward pass: from last slice down to ref_z ---
    if ref_z < Z - 1:
        group_indices = [Z - 1]
        group_rfp = corrected[Z - 1, RFP_CH]

        # Walk downwards until and including ref_z
        for z in range(Z - 2, ref_z - 1, -1):
            fixed_rfp = corrected[z, RFP_CH]  # treat current slice as fixed

            # Register group-average (moving) to fixed
            rfp_group_reg, params = register(
                fixed_rfp, group_rfp, parameter_object, threads=threads
            )

            # Apply transform to all slices in the group
            for idx in group_indices:
                corrected[idx, RFP_CH] = itk.transformix_filter(
                    corrected[idx, RFP_CH], params
                )
                corrected[idx, GFP_CH] = itk.transformix_filter(
                    corrected[idx, GFP_CH], params
                )

            # Now extend group with current slice and recompute group-average
            if z not in group_indices:
                group_indices.append(z)
            group_rfp = group_mean_rfp(group_indices)

    return corrected


def extract_translation_from_params(param_object):
    """
    Extract (dx, dy) from an Elastix ParameterObject returned by registration.
    """

    pmap = param_object.GetParameterMap(0)
    vals = [float(v) for v in pmap["TransformParameters"]]

    if len(vals) == 2:
        # TranslationTransform
        return vals[0], vals[1]

    elif len(vals) == 3:
        # EulerTransform: [angle tx ty]
        return vals[1], vals[2]

    else:
        raise ValueError(f"Unexpected TransformParameters: {vals}")




def apply_translation_to_image(img, dx, dy):
    """
    Apply (dx, dy) translation using elastix/transformix.
    Fully compatible with ITK-Elastix builds that require explicit
    Direction, Index, ImageDimension, Size, etc.
    """

    img = img.astype(np.float32, copy=False)
    H, W = img.shape

    # Create parameter object
    po = itk.ParameterObject.New()

    # Start from default 2D translation map
    tmap = po.GetDefaultParameterMap("translation")

    # --- REQUIRED KEYS FOR YOUR BUILD ---
    tmap["Transform"] = ["TranslationTransform"]
    tmap["TransformParameters"] = [str(dx), str(dy)]
    tmap["NumberOfParameters"] = ["2"]

    tmap["ImageDimension"] = ["2"]

    # Elastix uses X,Y ordering
    tmap["Size"] = [str(W), str(H)]
    tmap["Index"] = ["0", "0"]

    tmap["Spacing"] = ["1.0", "1.0"]
    tmap["Origin"] = ["0.0", "0.0"]

    # Direction must be a flat matrix in row-major order:
    #   [1 0
    #    0 1]
    tmap["Direction"] = ["1", "0", "0", "1"]

    tmap["UseDirectionCosines"] = ["true"]

    # Clean and load transform map
    po.ClearParameterMaps() if hasattr(po, "ClearParameterMaps") else None
    po.AddParameterMap(tmap)

    # Convert image -> ITK
    itk_img = itk.image_view_from_array(img)

    # Apply transform
    out_itk = itk.transformix_filter(itk_img, po)

    # Convert back -> numpy
    out = itk.array_view_from_image(out_itk)
    return out





def fit_motion_model(z_positions, shifts, smoothing="linear"):
    """
    Fit dx(z), dy(z) with a smooth model.
    smoothing: "linear" or "spline"
    """
    z = np.array(z_positions)
    dx = np.array([s[0] for s in shifts])
    dy = np.array([s[1] for s in shifts])

    if smoothing == "linear":
        # Simple shear model: dx = a0 + a1*z
        A = np.vstack([z, np.ones_like(z)]).T
        a1_dx, a0_dx = np.linalg.lstsq(A, dx, rcond=None)[0]
        a1_dy, a0_dy = np.linalg.lstsq(A, dy, rcond=None)[0]

        def dx_fit(zz): return a1_dx * zz + a0_dx
        def dy_fit(zz): return a1_dy * zz + a0_dy

    elif smoothing == "spline":
        # Cubic spline with small smoothing factor
        dx_spline = UnivariateSpline(z, dx, s=len(z) * 1.0)
        dy_spline = UnivariateSpline(z, dy, s=len(z) * 1.0)

        def dx_fit(zz): return dx_spline(zz)
        def dy_fit(zz): return dy_spline(zz)

    else:
        raise ValueError("smoothing must be 'linear' or 'spline'")

    return dx_fit, dy_fit


def shear_correct_global_model(stack, parameter_object, smoothing="linear", threads=8):
    """
    standard short time scale, time series shear correction:
    1. Choose reference slice (max RFP intensity).
    2. Register all slices to reference slice.
    3. Extract dx(z), dy(z) from each registration.
    4. Fit smooth model dx(z), dy(z).
    5. Apply smoothed transforms to all slices (GFP and RFP).

    stack shape: (Z, 2, H, W)
    """

    stack = stack.astype(np.float32, copy=False)
    corrected = stack.copy()

    Z, C, H, W = stack.shape
    GFP, RFP = 0, 1

    # --- 1. Find reference slice ---
    rfp_means = np.mean(stack[:, RFP], axis=(1, 2))
    ref_z = int(np.argmax(rfp_means))
    ref_img = stack[ref_z, RFP]

    # --- 2. Register all slices to reference (RFP channel) ---
    raw_shifts = []

    for z in range(Z):
        if z == ref_z:
            raw_shifts.append((0.0, 0.0))
            continue

        moving = stack[z, RFP]
        _, params = register(ref_img, moving, parameter_object, threads=threads)

        dx, dy = extract_translation_from_params(params)
        raw_shifts.append((dx, dy))

    # --- 3. Fit smooth global dx(z), dy(z) ---
    z_positions = np.arange(Z)
    dx_fit, dy_fit = fit_motion_model(z_positions, raw_shifts, smoothing=smoothing)

    # --- 4. Apply smoothed transforms to each slice and channel ---
    for z in range(Z):
        dx = float(dx_fit(z))
        dy = float(dy_fit(z))

        corrected[z, RFP] = apply_translation_to_image(corrected[z, RFP], dx, dy)
        corrected[z, GFP] = apply_translation_to_image(corrected[z, GFP], dx, dy)

    return corrected

def shear_correct_pairwise_model(stack, parameter_object, smoothing="linear", threads=8):
    """
    Shear correction using pairwise (adjacent) slice registration + global motion model.

    Steps:
      1. Compute pairwise translations between slice z and z+1 for all z.
      2. Integrate those to get cumulative displacement D(z) for each slice.
      3. Choose a reference z_ref (max RFP intensity).
      4. Re-center displacements so that D(z_ref) = (0,0).
      5. Fit smooth dx(z), dy(z) (linear or spline).
      6. Apply smoothed translations to both channels.

    Parameters
    ----------
    stack : np.ndarray
        Array of shape (Z, 2, H, W). Channel 0 = GFP, 1 = RFP.
    parameter_object : itk.ParameterObject
        Elastix parameter object for translation/rigid registration.
    smoothing : {"linear", "spline"}
        Type of motion model to fit to dx(z), dy(z).
    threads : int
        Number of threads for elastix registration.

    Returns
    -------
    corrected : np.ndarray (Z, 2, H, W), float32
        Shear-corrected stack.
    """

    stack = stack.astype(np.float32, copy=False)
    corrected = stack.copy()

    Z, C, H, W = corrected.shape
    GFP_CH, RFP_CH = 0, 1

    # --- 1. Pairwise registration: for each (z, z+1) ---
    pair_shifts = []  # list of (dx, dy) for transform mapping slice z+1 -> z (fixed=z, moving=z+1)

    for z in range(Z - 1):
        fixed = corrected[z, RFP_CH]     # slice z
        moving = corrected[z + 1, RFP_CH]  # slice z+1

        _, params = register(fixed, moving, parameter_object, threads=threads)
        dx, dy = extract_translation_from_params(params)
        pair_shifts.append((dx, dy))

    pair_shifts = np.asarray(pair_shifts, dtype=np.float64)  # shape (Z-1, 2)

    # --- 2. Cumulative displacement for each slice (relative to slice 0) ---
    # cumulative_disp[z] = displacement to map slice z into coords of slice 0
    cumulative_disp = np.zeros((Z, 2), dtype=np.float64)
    for z in range(1, Z):
        cumulative_disp[z] = cumulative_disp[z - 1] + pair_shifts[z - 1]

    # --- 3. Choose reference z_ref (e.g. max RFP mean) ---
    rfp_means = np.mean(corrected[:, RFP_CH], axis=(1, 2))
    ref_z = int(np.argmax(rfp_means))

    # --- 4. Re-center so that displacement at ref_z is (0,0) ---
    ref_disp = cumulative_disp[ref_z].copy()
    disp_rel = cumulative_disp - ref_disp  # D(z) relative to ref_z

    # Prepare for model fitting: raw_shifts[z] = (dx(z), dy(z))
    z_positions = np.arange(Z)
    raw_shifts = [tuple(disp_rel[z]) for z in range(Z)]

    # --- 5. Fit smooth motion model dx(z), dy(z) ---
    dx_fit, dy_fit = fit_motion_model(z_positions, raw_shifts, smoothing=smoothing)

    # --- 6. Apply smoothed translation to each slice & channel ---
    for z in range(Z):
        dx = float(dx_fit(z))
        dy = float(dy_fit(z))

        corrected[z, RFP_CH] = apply_translation_to_image(corrected[z, RFP_CH], dx, dy)
        corrected[z, GFP_CH] = apply_translation_to_image(corrected[z, GFP_CH], dx, dy)

    return corrected

def register_pair_phasecorr(fixed, moving, upsample_factor=10):
    """
    Subpixel translation-only registration using phase correlation (Guizar-Sicairos).
    fixed, moving: 2D arrays (Y, X)
    Returns:
        shift: (dy, dx)
        error: registration error (float)
        diffphase: global phase offset (float)
    """
    shift, error, diffphase = phase_cross_correlation(
        fixed,
        moving,
        upsample_factor=upsample_factor,
    )
    return np.array(shift, dtype=np.float32), float("nan"), float("nan")


def apply_subpixel_shift(image, shift):
    """
    Apply a subpixel translation 'shift' to 'image' using Fourier shift theorem.
    image: 2D array (Y, X)
    shift: (dy, dx)
    Returns:
        shifted_image: 2D array (Y, X), float32
    """
    # FFT -> apply Fourier shift -> IFFT
    F = np.fft.fftn(image)
    F_shifted = fourier_shift(F, shift)
    shifted = np.fft.ifftn(F_shifted).real
    return shifted.astype(np.float32, copy=False)

def register_zstack_phasecorr(stack, upsample_factor=10):
    """
    Register a single-channel z-stack by chaining FFT-based translation
    registration between neighboring slices (like reg_stack_translate!).

    Parameters
    ----------
    stack : np.ndarray
        Input z-stack of shape (Z, Y, X), e.g. RFP channel only.
    upsample_factor : int
        Subpixel upsampling factor for phase correlation (>=1).
        Typical values: 5–20.

    Returns
    -------
    registered : np.ndarray
        Registered z-stack, same shape (Z, Y, X), float32.
    shifts : np.ndarray
        Array of shape (Z, 2) containing (dy, dx) shifts for each slice.
        shifts[0] = [0, 0] by definition.
    """
    stack = stack.astype(np.float32, copy=False)
    Z, Y, X = stack.shape

    registered = stack.copy()
    shifts = np.zeros((Z, 2), dtype=np.float32)

    # Slice 0 stays as is
    shifts[0] = 0.0

    for z in range(1, Z):
        fixed = registered[z - 1]   # previous (already-registered) slice
        moving = registered[z]      # current slice

        shift, error, diffphase = register_pair_phasecorr(
            fixed, moving, upsample_factor=upsample_factor
        )
        shifts[z] = shift

        registered[z] = apply_subpixel_shift(moving, shift)

    return registered, shifts

def register_zstack_phasecorr_multichannel(stack, reg_channel=1, upsample_factor=10):
    """
    Register a multi-channel z-stack using FFT-based phase correlation on one
    channel (e.g. RFP) and applying the same shift to all channels.

    Parameters
    ----------
    stack : np.ndarray
        Array of shape (Z, C, Y, X).
        Example: C=2 (GFP, RFP).
    reg_channel : int
        Index of the channel to use for registration (default: 1 = RFP).
    upsample_factor : int
        Subpixel upsampling factor (>=1). 10 is often a good starting point.

    Returns
    -------
    registered : np.ndarray
        Registered stack, same shape (Z, C, Y, X), float32.
    shifts : np.ndarray
        Array of shape (Z, 2) with (dy, dx) for each slice.
    """
    stack = stack.astype(np.float32, copy=False)
    Z, C, Y, X = stack.shape

    registered = stack.copy()
    shifts = np.zeros((Z, 2), dtype=np.float32)
    shifts[0] = 0.0

    for z in range(1, Z):
        fixed = registered[z - 1, reg_channel]   # previous slice, reg channel
        moving = registered[z, reg_channel]      # current slice, reg channel

        shift, error, diffphase = register_pair_phasecorr(
            fixed, moving, upsample_factor=upsample_factor
        )
        shifts[z] = shift

        # Apply same shift to all channels at this z
        for c in range(C):
            registered[z, c] = apply_subpixel_shift(registered[z, c], shift)

    return registered, shifts


def main():
    """Main pipeline: load/create shear-corrected stack, perform channel alignment, save parameters."""

    input_nd2 = sys.argv[1]
    index = int(sys.argv[2])
    noise_pth = sys.argv[3]
    stack_length = int(sys.argv[4])
    num_frames, height, width, num_channels = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
    stack_shape = (stack_length,num_channels,height,width)

    out_dir = os.path.splitext(input_nd2)[0]
    os.makedirs(os.path.join(out_dir, 'shear_corrected'), exist_ok=True) 
    os.makedirs(os.path.join(out_dir, 'denoised'), exist_ok=True)
    
    noise_stack = get_noise_stack(noise_pth)

    # itk parameters
    shear_param_file = r'C:\Users\munib\POSTDOC\CODE\g5ht-pipeline\parameters_shear_correct.txt'
    # shear_param_file = r'C:\Users\munib\POSTDOC\CODE\g5ht-pipeline\parameters_shear_correct_test.txt'
    shear_correct_parameter_object = load_parameter_object(shear_param_file)
    
    # # itk parameters
    # shear_correct_parameter_object = itk.ParameterObject.New()
    # shear_correct_parameter_map = shear_correct_parameter_object.GetDefaultParameterMap('rigid', 4)
    # shear_correct_parameter_object.AddParameterMap(shear_correct_parameter_map)
    # shear_correct_parameter_object.WriteParameterFile(shear_correct_parameter_map, r'C:\Users\munib\POSTDOC\CODE\g5ht-pipeline\parameters_shear_correct.txt')
    
    
    stack = get_stack(input_nd2, index, noise_stack, stack_shape=stack_shape)
    stack_pth = os.path.join(out_dir, "denoised", f"{index:04d}.tif")
    # tifffile.imwrite(stack_pth, np.clip(stack, 0, 4095).astype(np.uint16), imagej=True)
    
    # shear_corrected = shear_correct(stack, shear_correct_parameter_object)
    # shear_corrected = shear_correct_single_ref(stack, shear_correct_parameter_object)
    # shear_corrected = shear_correct_progressive(stack, shear_correct_parameter_object)
    # shear_corrected = shear_correct_global_model(stack,
    #                                              shear_correct_parameter_object,
    #                                              smoothing="spline"  # linear or spline
    # )
    
    # shear_corrected = shear_correct_pairwise_model(stack,
    #                                              shear_correct_parameter_object,
    #                                              smoothing="spline"  # linear or spline
    # )
    shear_corrected, shifts = register_zstack_phasecorr_multichannel(stack,
                                                                     reg_channel=1,       # RFP
                                                                     upsample_factor=20)
    
    
    shear_corrected = np.clip(shear_corrected, 0, 4095).astype(np.uint16)
    
    tif_path = os.path.join(out_dir, "shear_corrected", f"{index:04d}.tif")
    tifffile.imwrite(tif_path, shear_corrected, imagej=True)
    # print(f'Stack {index:04d} shear corrected')

if __name__ == '__main__':
    main()
