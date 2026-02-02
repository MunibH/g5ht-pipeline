ENV["CUDA_VISIBLE_DEVICES"] = "0"
ENV["JULIA_IO_BUFFER"] = "0"

using ImageDataIO, BehaviorDataNIR, UNet2D, H5Zblosc
using HDF5, PyPlot, FileIO

NAME = "2023-05-26-08"
PATH_H5 = "/data1/albert/prj_starvation/data_raw/2023-05-26/$(NAME).h5"
PATH_JLD2 = "/home/albert/data_personal/$(NAME)_data_dict.jld2"
MAX_T_NIR = size(h5open(PATH_H5)["img_nir"])[3]

param = Dict()
#for 1st compute_worm_spline!()
param["num_center_pts"] = 1000
param["img_label_size"] = (480, 360)
param["nose_confidence_threshold"] = 0.99
param["nose_crop_threshold"] = 20
#for compute_worm_thickness()
param["min_len_percent"] = 90
param["max_len_percent"] = 98
#for 2nd compute_worm_spline!()
param["worm_thickness_pad"] = 3
param["boundary_thickness"] = 5
param["close_pts_threshold"] = 30
param["trim_head_tail"] = 15
param["max_med_axis_delta"] = Inf

data_dict = Dict()
data_dict["med_axis_dict"] = Dict()
data_dict["med_axis_dict"][0] = nothing
data_dict["pts_order_dict"] = Dict()
data_dict["pts_order_dict"][0] = nothing
data_dict["is_omega"] = Dict()
data_dict["x_array"] = zeros(MAX_T_NIR, param["num_center_pts"] + 1)
data_dict["y_array"] = zeros(MAX_T_NIR, param["num_center_pts"] + 1)
data_dict["nir_worm_angle"] = zeros(MAX_T_NIR)
data_dict["eccentricity"] = zeros(MAX_T_NIR)

error_dict = Dict()

println("Loading model...")
path_weight = "/data1/shared/dl_weights/behavior_nir/worm_segmentation_best_weights_0310.pt"
worm_seg_model = create_model(1, 1, 16, path_weight)

println("Computing spline...")
error_dict["worm_spline_errors_1"] = compute_worm_spline!(param, PATH_H5, worm_seg_model, nothing,
                                                          data_dict["med_axis_dict"], data_dict["pts_order_dict"],
                                                          data_dict["is_omega"], data_dict["x_array"], data_dict["y_array"],
                                                          data_dict["nir_worm_angle"],  data_dict["eccentricity"],
                                                          timepts=1:MAX_T_NIR)

println("Detecting self-intersection...")
data_dict["worm_thickness"], count = compute_worm_thickness(param, PATH_H5, worm_seg_model, data_dict["med_axis_dict"],
                                                            data_dict["is_omega"])

println("Recomputing spline...")
error_dict["worm_spline_errors_2"] = compute_worm_spline!(param, PATH_H5, worm_seg_model, data_dict["worm_thickness"],
                                                          data_dict["med_axis_dict"], data_dict["pts_order_dict"],
                                                          data_dict["is_omega"], data_dict["x_array"], data_dict["y_array"],
                                                          data_dict["nir_worm_angle"],  data_dict["eccentricity"],
                                                          timepts=1:MAX_T_NIR)

println("Done!")
save(PATH_JLD2, "data_dict", data_dict)
