using HDF5, Statistics, BehaviorDataNIR

#used in make_confocal_to_nir()
function bin(timestamps, cutoff=0)
    timestamps = timestamps[1:end - cutoff]
    timestamps .> mean(timestamps)
end

#used in make_confocal_to_nir()
function get_stacks(camera_bin, stack=1)
    list_on = findall(diff(camera_bin) .== 1) .+ 1
    list_off = findall(diff(camera_bin) .== -1) .+ 1
    
    len = minimum([length(list_on), length(list_off)])
    max_stack = div(len, stack) * stack

    list_stack_start = list_on[1:stack:max_stack]
    list_stack_stop = list_off[stack:stack:max_stack]
    hcat(list_stack_start, list_stack_stop)
end

#used in make_confocal_to_nir()
function filter_nir_stacks(nir_stacks, h5f)
    img_metadata = read(h5f, "img_metadata")
    img_id = img_metadata["img_id"]
    q_iter_save = img_metadata["q_iter_save"]
    img_id_diff = diff(img_id)
    idx_nir_save = Bool[] 
    for (Δn, q_save) = zip(img_id_diff, q_iter_save)
        if Δn == 1
            push!(idx_nir_save, q_save)
        else
            push!(idx_nir_save, q_save)
            for i = 1:Δn-1
                push!(idx_nir_save, false)
            end
        end
    end
    if size(nir_stacks)[1] < length(idx_nir_save)
        out = idx_nir_save[1:size(nir_stacks)[1]]
    else
        out = zeros(Bool, size(nir_stacks)[1])
        out[1:length(idx_nir_save)] = idx_nir_save
    end
    nir_stacks[out, :]
end

#used in add_missing_nonspline_data()
function get_confocal_to_nir(path_h5, stack=11)
    #gets timing for each confocal stack
    h5f = h5open(path_h5, "r")
    confocal_timestamps = Float64.(read(h5f, "daqmx_ai")[:, 1])
    confocal_bin = bin(confocal_timestamps)
    confocal_stacks = get_stacks(confocal_bin, stack)

    #gets timing for each nir frame and reformats
    nir_timestamps = Float64.(read(h5f, "daqmx_di")[:, 2])
    nir_bin = bin(nir_timestamps)
    nir_stacks = get_stacks(nir_bin)
    filtered_nir_stacks = filter_nir_stacks(nir_stacks, h5f)
    one_hot_nir = zeros(Int, maximum(filtered_nir_stacks))
    for i in 1:size(filtered_nir_stacks)[1]
        one_hot_nir[filtered_nir_stacks[i, 1]:filtered_nir_stacks[i, 2]] .= i
    end

    #gets the index of all nir frames that overlap a confocal frame
    out = []
    for i = 1:size(confocal_stacks)[1]
        set = Set(one_hot_nir[confocal_stacks[i, 1]:confocal_stacks[i, 2]])
        nir_frames = sort(collect(filter(x -> x != 0, set)))
        push!(out, nir_frames)
    end
    out
end

#used in add_missing_nonspline_data()
function get_max_t_nir(path_h5)
    out = 0
    for i = 1:size(h5open(path_h5)["img_nir"])[3]
        try
            img = h5open(path_h5)["img_nir"][:, :, i]
        catch
            break
        end
        out = i
    end
    return out
end

#used in add_missing_nonspline_data()
function get_max_t_confocal(confocal_to_nir, max_t_nir)
    latest = 0
    for confocal_index in 1:length(confocal_to_nir)
        for nir_index in confocal_to_nir[confocal_index]
            if nir_index > max_t_nir
                return latest
            end
        end
        latest = confocal_index
    end
    latest
end

function add_nonspline_data!(data_dict, path_h5)
    data_dict["confocal_to_nir"] = get_confocal_to_nir(path_h5)
    data_dict["max_t_nir"] = get_max_t_nir(path_h5)
    data_dict["max_t"] = get_max_t_confocal(data_dict["confocal_to_nir"], data_dict["max_t_nir"])
    data_dict["t_range"] = 1:data_dict["max_t"]
    data_dict["confocal_to_nir"] = data_dict["confocal_to_nir"][data_dict["t_range"]]
    
    pos_feature, pos_feature_unet = read_pos_feature(path_h5)
    pos_stage = read_stage(path_h5)
    data_dict["x_stage"] = impute_list(pos_stage[1, :])
    data_dict["y_stage"] = impute_list(pos_stage[2, :])
    mn_vec, mp_vec, orthog_mp_vec = nmp_vec(pos_feature);
    data_dict["pm_angle"] = vec_to_angle(mp_vec);
    data_dict["x_med"], data_dict["y_med"] = offset_xy(data_dict["x_stage"], data_dict["y_stage"], pos_feature_unet[2, :, :])
    nothing
end

function add_behavioral_data!(data_dict)
    param = Dict()
    param["num_center_pts"] = 1000
    param["segment_len"] = 7
    param["max_pt"] = 31
    param["body_angle_t_lag"] = 40
    param["body_angle_pos_lag"] = 2
    param["head_pts"] = [1, 5, 8]
    param["filt_len_angvel"] = 150
    param["FLIR_FPS"] = 20.0
    param["v_stage_m_filt"] = 10
    param["v_stage_λ_filt"] = 250.0
    param["rev_len_thresh"] = 2
    param["rev_v_thresh"] = -0.005
    param["nose_pts"] = [1, 2, 3]

    vec_to_confocal = vec -> nir_vec_to_confocal(vec, data_dict["confocal_to_nir"], data_dict["max_t"])
    interpolate_splines!(data_dict)
    data_dict["segment_end_matrix"] = get_segment_end_matrix(param, data_dict["x_array"], data_dict["y_array"])
    data_dict["x_stage_confocal"] = vec_to_confocal(data_dict["x_stage"])
    data_dict["y_stage_confocal"] = vec_to_confocal(data_dict["y_stage"])
    data_dict["zeroed_x"], data_dict["zeroed_y"] = zero_stage(data_dict["x_med"], data_dict["y_med"])
    data_dict["zeroed_x_confocal"] = vec_to_confocal(data_dict["zeroed_x"])
    data_dict["zeroed_y_confocal"] = vec_to_confocal(data_dict["zeroed_y"])

    get_body_angles!(data_dict, param)
    get_angular_velocity!(data_dict, param)
    get_velocity!(data_dict, param)
    get_curvature_variables!(data_dict, param)
    get_nose_curling!(data_dict, param)
    nothing
end

nothing
    