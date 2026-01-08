import tifffile
import numpy as np
import scipy.ndimage as ndi
import itk
import sys
import os

from itk import image_view_from_array

#get channels out of stacks
def register(fixed_stack, fixed_mask_stack, moving_stack, moving_mask_stack):
	fixed_rfp = fixed_stack[:, 1].copy()
	moving_gfp, moving_rfp = moving_stack[:, 0].copy(), moving_stack[:, 1].copy()

	#initialize registration parameters
	parameter_object = itk.ParameterObject.New()
	default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid', 4)
	parameter_object.AddParameterMap(default_rigid_parameter_map)
	default_affine_parameter_map = parameter_object.GetDefaultParameterMap('affine', 4)
	parameter_object.AddParameterMap(default_affine_parameter_map)
	default_bspline_128_parameter_map = parameter_object.GetDefaultParameterMap('bspline', 4, 128)
	parameter_object.AddParameterMap(default_bspline_128_parameter_map)
	default_bspline_64_parameter_map = parameter_object.GetDefaultParameterMap('bspline', 4, 64)
	parameter_object.AddParameterMap(default_bspline_64_parameter_map)
	default_bspline_32_parameter_map = parameter_object.GetDefaultParameterMap('bspline', 4, 32)
	parameter_object.AddParameterMap(default_bspline_32_parameter_map)

	fixed_mask_stack = itk.image_view_from_array(fixed_mask_stack.astype(np.ubyte))
	moving_mask_stack = itk.image_view_from_array(moving_mask_stack.astype(np.ubyte))

	#register rfp first and then apply transform to gfp
	registered_rfp, transform_parameters = itk.elastix_registration_method(fixed_rfp, moving_rfp, parameter_object,
																		   fixed_mask=fixed_mask_stack, moving_mask=moving_mask_stack)
	registered_gfp = itk.transformix_filter(moving_gfp, transform_parameters)

	#initialize and fill output
	output_stack = np.zeros((fixed_stack.shape[0], 2, 200, 500), np.uint16)
	output_stack[:, 0] = np.clip(registered_gfp, 0, 4095)
	output_stack[:, 1] = np.clip(registered_rfp, 0, 4095)
	return output_stack, transform_parameters


#check if output already exists
def main():
	input_dir, index, zoom = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
	registered_pth = os.path.join(input_dir,'registered_gfp+1')
	# registered_pth = os.path.join(input_dir,'registered')
	warped_pth = os.path.join(input_dir,'warped')

	os.makedirs(registered_pth, exist_ok=True)
	output_path = os.path.join(registered_pth,f'{index:04d}.tif')
	# if os.path.isfile(output_path):
	# 	raise FileExistsError(f'Stack {index} already registered!')

	#load stacks
	moving_path = os.path.join(warped_pth,f'{index:04d}.tif')
	moving_stack = tifffile.imread(moving_path).astype(np.float32)
	moving_stack = ndi.zoom(moving_stack, zoom=(zoom, 1, 1, 1))
	fixed_stack = tifffile.imread(os.path.join(input_dir, 'fixed.tif')).astype(np.float32)
	fixed_stack = ndi.zoom(fixed_stack, zoom=(zoom, 1, 1, 1))

	fixed_mask = tifffile.imread(os.path.join(input_dir, 'fixed_mask.tif'))
	fixed_mask_stack = np.stack([fixed_mask] * fixed_stack.shape[0])

	moving_mask_path = os.path.join(input_dir, 'masks', f'{index:04d}.tif')
	moving_mask = tifffile.imread(moving_mask_path)
	moving_mask_stack = np.stack([moving_mask] * fixed_stack.shape[0])
	
	#register and save
	# print('Begin registration')
	
    # trim first gfp z slice
	# trim last rfp z slice
	fixed_gfp = fixed_stack[:,0,:,:]
	fixed_rfp = fixed_stack[:,1,:,:]
	fixed_stack = np.stack((fixed_gfp[1:,:,:], fixed_rfp[:-1,:,:]), axis=1)
	fixed_mask_stack = fixed_mask_stack[1:,:,:]
	mov_gfp = moving_stack[:,0,:,:]
	mov_rfp = moving_stack[:,1,:,:]
	moving_stack = np.stack((mov_gfp[1:,:,:], mov_rfp[:-1,:,:]), axis=1)
	moving_mask_stack = moving_mask_stack[1:,:,:]

	output_stack, transform_params = register(fixed_stack, fixed_mask_stack, moving_stack, moving_mask_stack)

	tifffile.imwrite(output_path, output_stack, imagej=True)
	# print(f'Registered stack {index}!')
