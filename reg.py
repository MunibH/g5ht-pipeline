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
	output_stack = np.zeros((117, 2, 200, 500), np.uint16)
	output_stack[:, 0] = np.clip(registered_gfp, 0, 4095)
	output_stack[:, 1] = np.clip(registered_rfp, 0, 4095)
	return output_stack

def register_munib(fixed_stack, fixed_mask_stack, moving_stack, moving_mask_stack):
	 
	fixed_rfp = fixed_stack[:, 1].copy()
	moving_gfp, moving_rfp = moving_stack[:, 0].copy(), moving_stack[:, 1].copy()
	fixed_mask_stack = itk.image_view_from_array(fixed_mask_stack.astype(np.ubyte))
	moving_mask_stack = itk.image_view_from_array(moving_mask_stack.astype(np.ubyte))
	 
	#initialize registration parameters
	parameter_object = itk.ParameterObject.New()
	 # Rigid
	rigid_map = parameter_object.GetDefaultParameterMap('rigid', 4)
	rigid_map['MaximumNumberOfIterations'] = ['512']
	rigid_map['NumberOfResolutions'] = ['4']
	rigid_map['NumberOfSpatialSamples'] = ['4096']
	parameter_object.AddParameterMap(rigid_map)
	
	# Affine
	affine_map = parameter_object.GetDefaultParameterMap('affine', 4)
	affine_map['MaximumNumberOfIterations'] = ['512']
	affine_map['NumberOfResolutions'] = ['3']
	affine_map['NumberOfSpatialSamples'] = ['4096']
	parameter_object.AddParameterMap(affine_map)
	
	# # Multi-resolution B-splines
	# for grid_spacing in [64, 32, 16]:
	# 	bspline_map = parameter_object.GetDefaultParameterMap('bspline', 3, grid_spacing)
	# 	bspline_map['MaximumNumberOfIterations'] = ['512']
	# 	bspline_map['NumberOfResolutions'] = ['3']
	# 	bspline_map['NumberOfSpatialSamples'] = ['4096']
		
	# 	# Add regularization
	# 	bspline_map['Metric'] = ['AdvancedMattesMutualInformation', 'TransformBendingEnergyPenalty']
	# 	bspline_map['Metric0Weight'] = ['1.0']
	# 	bspline_map['Metric1Weight'] = ['1.0']
		
	# 	parameter_object.AddParameterMap(bspline_map)
	 
	#register rfp first and then apply transform to gfp
	registered_rfp, transform_parameters = itk.elastix_registration_method(fixed_rfp, moving_rfp, parameter_object,
																		   fixed_mask=fixed_mask_stack, moving_mask=moving_mask_stack)
	registered_gfp = itk.transformix_filter(moving_gfp, transform_parameters)
	 
	#initialize and fill output
	output_stack = np.zeros((117, 2, 200, 500), np.uint16)
	output_stack[:, 0] = np.clip(registered_gfp, 0, 4095)
	output_stack[:, 1] = np.clip(registered_rfp, 0, 4095)

	return output_stack, transform_parameters


def register_simple(fixed_stack, fixed_mask_stack, moving_stack, moving_mask_stack):
	"""
	Simplest registration with explicit ITK image creation.
	Use this first to debug.
	"""
	# Extract channels and convert to float32
	fixed_rfp = fixed_stack[:, 1].copy().astype(np.float32)
	moving_gfp = moving_stack[:, 0].copy().astype(np.float32)
	moving_rfp = moving_stack[:, 1].copy().astype(np.float32)
	
	print(f"Data shapes: fixed_rfp={fixed_rfp.shape}, moving_rfp={moving_rfp.shape}")
	print(f"Data types: fixed_rfp={fixed_rfp.dtype}, moving_rfp={moving_rfp.dtype}")
	print(f"Data ranges: fixed=[{fixed_rfp.min():.1f}, {fixed_rfp.max():.1f}], moving=[{moving_rfp.min():.1f}, {moving_rfp.max():.1f}]")
	
	# Create ITK images directly from arrays with correct type
	fixed_image = itk.image_from_array(fixed_rfp)
	moving_image = itk.image_from_array(moving_rfp)
	
	# Create masks
	fixed_mask = itk.image_from_array(fixed_mask_stack.astype(np.uint8))
	moving_mask = itk.image_from_array(moving_mask_stack.astype(np.uint8))
	
	# Setup parameters
	parameter_object = itk.ParameterObject.New()
	
	default_rigid = parameter_object.GetDefaultParameterMap('rigid', 3)  # Changed to 3D
	parameter_object.AddParameterMap(default_rigid)
	
	default_affine = parameter_object.GetDefaultParameterMap('affine', 3)
	parameter_object.AddParameterMap(default_affine)
	
	default_bspline = parameter_object.GetDefaultParameterMap('bspline', 3, 32)
	parameter_object.AddParameterMap(default_bspline)
	
	print("Starting registration...")
	
	try:
		registered_rfp, transform_params = itk.elastix_registration_method(
			fixed_image, 
			moving_image, 
			parameter_object,
			fixed_mask=fixed_mask, 
			moving_mask=moving_mask,
			log_to_console=True
		)
	except RuntimeError as e:
		print(f"\nRegistration failed with error: {e}")
		raise e
	
	print("Applying transform to GFP...")
	moving_gfp_image = itk.GetImageFromArray(moving_gfp)
	moving_gfp_image = moving_gfp_image.astype(np.uint16)
	registered_gfp = itk.transformix_filter(moving_gfp_image, transform_params)
	
	# Convert back to numpy
	registered_rfp_np = itk.GetArrayFromImage(registered_rfp)
	registered_gfp_np = itk.GetArrayFromImage(registered_gfp)
	
	# Prepare output
	z_size = registered_rfp_np.shape[0]
	output_stack = np.zeros((z_size, 2, 200, 500), dtype=np.uint16)
	output_stack[:, 0] = np.clip(registered_gfp_np, 0, 4095).astype(np.uint16)
	output_stack[:, 1] = np.clip(registered_rfp_np, 0, 4095).astype(np.uint16)
	
	print('Done')
	
	return output_stack, transform_params


def register_fast(fixed_stack, fixed_mask_stack, moving_stack, moving_mask_stack):
	"""
	Faster registration: rigid + affine + single B-spline.
	"""
	fixed_rfp = fixed_stack[:, 1].copy().astype(np.float32)
	moving_gfp = moving_stack[:, 0].copy().astype(np.float32)
	moving_rfp = moving_stack[:, 1].copy().astype(np.float32)
	
	ImageType = itk.Image[itk.F, 3]
	MaskType = itk.Image[itk.UC, 3]
	
	fixed_image = itk.GetImageFromArray(fixed_rfp).astype(ImageType)
	moving_image = itk.GetImageFromArray(moving_rfp).astype(ImageType)
	fixed_mask = itk.GetImageFromArray(fixed_mask_stack.astype(np.uint8)).astype(MaskType)
	moving_mask = itk.GetImageFromArray(moving_mask_stack.astype(np.uint8)).astype(MaskType)
	
	parameter_object = itk.ParameterObject.New()
	
	rigid_map = parameter_object.GetDefaultParameterMap('rigid', 3)
	rigid_map['MaximumNumberOfIterations'] = ['256']
	parameter_object.AddParameterMap(rigid_map)
	
	affine_map = parameter_object.GetDefaultParameterMap('affine', 3)
	affine_map['MaximumNumberOfIterations'] = ['256']
	parameter_object.AddParameterMap(affine_map)
	
	bspline_map = parameter_object.GetDefaultParameterMap('bspline', 3, 32)
	bspline_map['MaximumNumberOfIterations'] = ['256']
	parameter_object.AddParameterMap(bspline_map)
	
	registered_rfp, transform_params = itk.elastix_registration_method(
		fixed_image, moving_image, parameter_object,
		fixed_mask=fixed_mask, moving_mask=moving_mask,
		log_to_console=False
	)
	
	moving_gfp_image = itk.GetImageFromArray(moving_gfp).astype(ImageType)
	registered_gfp = itk.transformix_filter(moving_gfp_image, transform_params)
	
	registered_rfp_np = itk.GetArrayFromImage(registered_rfp)
	registered_gfp_np = itk.GetArrayFromImage(registered_gfp)
	
	z_size = registered_rfp_np.shape[0]
	output_stack = np.zeros((z_size, 2, 200, 500), dtype=np.uint16)
	output_stack[:, 0] = np.clip(registered_gfp_np, 0, 4095).astype(np.uint16)
	output_stack[:, 1] = np.clip(registered_rfp_np, 0, 4095).astype(np.uint16)
	
	return output_stack, transform_params


def register_test(fixed_stack, fixed_mask_stack, moving_stack, moving_mask_stack):
	"""
	Full multi-resolution registration pipeline.
	"""
	fixed_rfp = fixed_stack[:, 1].copy().astype(np.float32)
	moving_gfp = moving_stack[:, 0].copy().astype(np.float32)
	moving_rfp = moving_stack[:, 1].copy().astype(np.float32)
	
	ImageType = itk.Image[itk.F, 3]
	MaskType = itk.Image[itk.UC, 3]
	
	fixed_image = itk.GetImageFromArray(fixed_rfp).astype(ImageType)
	moving_image = itk.GetImageFromArray(moving_rfp).astype(ImageType)
	fixed_mask = itk.GetImageFromArray(fixed_mask_stack.astype(np.uint8)).astype(MaskType)
	moving_mask = itk.GetImageFromArray(moving_mask_stack.astype(np.uint8)).astype(MaskType)
	
	parameter_object = itk.ParameterObject.New()
	
	# Rigid
	rigid_map = parameter_object.GetDefaultParameterMap('rigid', 3)
	rigid_map['MaximumNumberOfIterations'] = ['512']
	rigid_map['NumberOfResolutions'] = ['4']
	rigid_map['NumberOfSpatialSamples'] = ['4096']
	parameter_object.AddParameterMap(rigid_map)
	
	# Affine
	affine_map = parameter_object.GetDefaultParameterMap('affine', 3)
	affine_map['MaximumNumberOfIterations'] = ['512']
	affine_map['NumberOfResolutions'] = ['3']
	affine_map['NumberOfSpatialSamples'] = ['4096']
	parameter_object.AddParameterMap(affine_map)
	
	# Multi-resolution B-splines
	for grid_spacing in [64, 32, 16]:
		bspline_map = parameter_object.GetDefaultParameterMap('bspline', 3, grid_spacing)
		bspline_map['MaximumNumberOfIterations'] = ['512']
		bspline_map['NumberOfResolutions'] = ['3']
		bspline_map['NumberOfSpatialSamples'] = ['4096']
		
		# Add regularization
		bspline_map['Metric'] = ['AdvancedMattesMutualInformation', 'TransformBendingEnergyPenalty']
		bspline_map['Metric0Weight'] = ['1.0']
		bspline_map['Metric1Weight'] = ['0.1']
		
		parameter_object.AddParameterMap(bspline_map)
	
	registered_rfp, transform_params = itk.elastix_registration_method(
		fixed_image, moving_image, parameter_object,
		fixed_mask=fixed_mask, moving_mask=moving_mask,
		log_to_console=True
	)
	
	moving_gfp_image = itk.GetImageFromArray(moving_gfp).astype(ImageType)
	registered_gfp = itk.transformix_filter(moving_gfp_image, transform_params)
	
	registered_rfp_np = itk.GetArrayFromImage(registered_rfp)
	registered_gfp_np = itk.GetArrayFromImage(registered_gfp)
	
	z_size = registered_rfp_np.shape[0]
	output_stack = np.zeros((z_size, 2, 200, 500), dtype=np.uint16)
	output_stack[:, 0] = np.clip(registered_gfp_np, 0, 4095).astype(np.uint16)
	output_stack[:, 1] = np.clip(registered_rfp_np, 0, 4095).astype(np.uint16)
	
	return output_stack, transform_params


def register_without_masks(fixed_stack, moving_stack):
	"""
	Registration without masks - use this if masks are causing issues.
	"""
	fixed_rfp = fixed_stack[:, 1].copy().astype(np.float32)
	moving_gfp = moving_stack[:, 0].copy().astype(np.float32)
	moving_rfp = moving_stack[:, 1].copy().astype(np.float32)
	
	ImageType = itk.Image[itk.F, 3]
	
	fixed_image = itk.GetImageFromArray(fixed_rfp).astype(ImageType)
	moving_image = itk.GetImageFromArray(moving_rfp).astype(ImageType)
	
	parameter_object = itk.ParameterObject.New()
	
	default_rigid = parameter_object.GetDefaultParameterMap('rigid', 3)
	parameter_object.AddParameterMap(default_rigid)
	
	default_affine = parameter_object.GetDefaultParameterMap('affine', 3)
	parameter_object.AddParameterMap(default_affine)
	
	default_bspline = parameter_object.GetDefaultParameterMap('bspline', 3, 32)
	parameter_object.AddParameterMap(default_bspline)
	
	print("Registration without masks...")
	registered_rfp, transform_params = itk.elastix_registration_method(
		fixed_image, moving_image, parameter_object,
		log_to_console=True
	)
	
	moving_gfp_image = itk.GetImageFromArray(moving_gfp).astype(ImageType)
	registered_gfp = itk.transformix_filter(moving_gfp_image, transform_params)
	
	registered_rfp_np = itk.GetArrayFromImage(registered_rfp)
	registered_gfp_np = itk.GetArrayFromImage(registered_gfp)
	
	z_size = registered_rfp_np.shape[0]
	output_stack = np.zeros((z_size, 2, 200, 500), dtype=np.uint16)
	output_stack[:, 0] = np.clip(registered_gfp_np, 0, 4095).astype(np.uint16)
	output_stack[:, 1] = np.clip(registered_rfp_np, 0, 4095).astype(np.uint16)
	
	return output_stack, transform_params


#check if output already exists
def main():
	input_dir, index = sys.argv[1], int(sys.argv[2])
	registered_pth = os.path.join(input_dir,'registered_test')
	warped_pth = os.path.join(input_dir,'warped')

	os.makedirs(registered_pth, exist_ok=True)
	output_path = os.path.join(registered_pth,f'{index:04d}.tif')
	# if os.path.isfile(output_path):
	# 	raise FileExistsError(f'Stack {index} already registered!')

	#load stacks
	moving_path = os.path.join(warped_pth,f'{index:04d}.tif')
	moving_stack = tifffile.imread(moving_path).astype(np.float32)
	moving_stack = ndi.zoom(moving_stack, zoom=(3, 1, 1, 1))
	fixed_stack = tifffile.imread(os.path.join(input_dir, 'fixed.tif')).astype(np.float32)
	fixed_stack = ndi.zoom(fixed_stack, zoom=(3, 1, 1, 1))

	fixed_mask = tifffile.imread(os.path.join(input_dir, 'fixed_mask.tif'))
	fixed_mask_stack = np.stack([fixed_mask] * 117)

	moving_mask_path = os.path.join(input_dir, 'masks', f'{index:04d}.tif')
	moving_mask = tifffile.imread(moving_mask_path)
	moving_mask_stack = np.stack([moving_mask] * 117)
	
	#register and save
	print('Begin registration')
	# output_stack, transform_params = register(fixed_stack, fixed_mask_stack, moving_stack, moving_mask_stack)
	# output_stack, transform_params = register_test(fixed_stack, fixed_mask_stack, moving_stack, moving_mask_stack)
	output_stack, transform_params = register_munib(fixed_stack, fixed_mask_stack, moving_stack, moving_mask_stack)
	# output_stack, transform_params = register_simple(fixed_stack, fixed_mask_stack, moving_stack, moving_mask_stack)
	# output_stack, transform_params = register_fast(fixed_stack, fixed_mask_stack, moving_stack, moving_mask_stack)
	# output_stack, transform_params = register_without_masks(fixed_stack, moving_stack)

	tifffile.imwrite(output_path, output_stack, imagej=True)
	# print(f'Registered stack {index}!')
