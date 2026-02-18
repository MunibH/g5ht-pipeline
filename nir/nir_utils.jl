using HDF5, Statistics, BehaviorDataNIR, TotalVariation

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
#munib 2024-06-12 edits:
#one_hot_nir sizing — now uses max(maximum(filtered_nir_stacks), maximum(confocal_stacks)) so the vector covers the full sample range of both cameras.
#Clamped confocal indexing — idx_end = min(confocal_stacks[i, 2], length(one_hot_nir)) prevents out-of-bounds access for confocal stacks that extend past the last NIR sample
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

    # Fix: if the NIR DI signal starts HIGH, get_stacks mis-pairs rising/falling
    # edges so that start > stop for every frame. Detect and fix by recomputing
    # with properly aligned edges.
    if size(filtered_nir_stacks, 1) > 0 && filtered_nir_stacks[1, 1] > filtered_nir_stacks[1, 2]
        @warn "NIR DI signal starts HIGH — re-aligning rising/falling edges"
        list_on = findall(diff(nir_bin) .== 1) .+ 1
        list_off = findall(diff(nir_bin) .== -1) .+ 1
        # Drop the first falling edge (it happened before the first rising edge)
        if length(list_off) > 0 && length(list_on) > 0 && list_off[1] < list_on[1]
            list_off = list_off[2:end]
        end
        len = minimum([length(list_on), length(list_off)])
        nir_stacks_fixed = hcat(list_on[1:len], list_off[1:len])
        filtered_nir_stacks = filter_nir_stacks(nir_stacks_fixed, h5f)
    end

    max_sample = max(maximum(filtered_nir_stacks), maximum(confocal_stacks))
    one_hot_nir = zeros(Int, max_sample)
    for i in 1:size(filtered_nir_stacks)[1]
        one_hot_nir[filtered_nir_stacks[i, 1]:filtered_nir_stacks[i, 2]] .= i
    end

    #gets the index of all nir frames that overlap a confocal frame
    out = []
    for i = 1:size(confocal_stacks)[1]
        idx_start = confocal_stacks[i, 1]
        idx_end = min(confocal_stacks[i, 2], length(one_hot_nir))
        set = Set(one_hot_nir[idx_start:idx_end])
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

function add_nonspline_data!(data_dict, path_h5; stack=11)
    data_dict["confocal_to_nir"] = get_confocal_to_nir(path_h5, stack)
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

# ============================================================
# Safe replacements for library functions that crash on all-NaN data.
#
# The library's impute_list (BehaviorDataNIR.impute_list) calls
#   Impute.interp(replace(x, NaN=>missing))
# which throws AssertionError when ALL values are missing.
#
# Several library functions call impute_list INTERNALLY, so wrapping
# them externally doesn't help — the crash happens inside the library
# before the wrapper ever runs.
#
# Solution: reimplement these functions locally, replacing the internal
# impute_list call with safe_impute_list.
#
# Affected library functions:
#   get_tot_worm_curvature  (spline_data.jl:72)  — called by get_curvature_variables!
#   get_nose_curling!       (behaviors.jl:145)    — called by add_behavioral_data!
#   get_body_angles!        (behaviors.jl:60)     — called by add_behavioral_data!
#   get_angular_velocity!   (behaviors.jl:78)     — called by add_behavioral_data!
#   get_velocity!           (behaviors.jl:100)    — called by add_behavioral_data!
# ============================================================

"""
    safe_impute_list(x)

Drop-in replacement for BehaviorDataNIR.impute_list that handles all-NaN input
gracefully (returns zeros) and catches any other imputation failures.
"""
function safe_impute_list(x::Vector{<:AbstractFloat})
    # Case 1: all NaN → can't interpolate, return zeros
    if all(isnan, x)
        @warn "safe_impute_list: all values are NaN, returning zeros"
        return zeros(eltype(x), length(x))
    end

    # Case 2: no NaN at all → nothing to do
    if !any(isnan, x)
        return copy(x)
    end

    # Case 3: some NaN → try the library's impute_list first
    try
        return impute_list(x)
    catch e
        @warn "safe_impute_list: impute_list failed ($(typeof(e))), using manual fill"
    end

    # Fallback: manual forward-fill then backward-fill, then zero-fill
    result = copy(x)
    last_valid = NaN
    for i in eachindex(result)
        if !isnan(result[i])
            last_valid = result[i]
        elseif !isnan(last_valid)
            result[i] = last_valid
        end
    end
    last_valid = NaN
    for i in reverse(eachindex(result))
        if !isnan(result[i])
            last_valid = result[i]
        elseif !isnan(last_valid)
            result[i] = last_valid
        end
    end
    result[isnan.(result)] .= 0.0
    return result
end

"""
    safe_get_tot_worm_curvature(body_angle, min_len; directional=false)

Local reimplementation of BehaviorDataNIR.get_tot_worm_curvature.
Uses safe_impute_list instead of impute_list so it doesn't crash
when the curvature vector is entirely NaN.
"""
function safe_get_tot_worm_curvature(body_angle, min_len; directional::Bool=false)
    worm_curvature = zeros(size(body_angle, 2))
    for t = 1:size(body_angle, 2)
        all_angles = [body_angle[i, t] for i = 1:size(body_angle, 1) if !isnan(body_angle[i, t])]
        if length(all_angles) < min_len
            worm_curvature[t] = NaN
        else
            all_angles = local_recenter_angle(all_angles)
            if directional
                idx = min(min_len, length(all_angles))
                worm_curvature[t] = (all_angles[1] - all_angles[idx]) / length(all_angles)
            else
                worm_curvature[t] = std(all_angles)
            end
        end
    end
    return safe_impute_list(worm_curvature)
end

"""
    get_curvature_variables_safe!(data_dict, param; prefix="")

Replaces BehaviorDataNIR.get_curvature_variables! with safe versions of
get_tot_worm_curvature and impute_list throughout.
"""
function get_curvature_variables_safe!(data_dict::Dict, param::Dict; prefix::String="")
    vec_to_confocal = vec -> nir_vec_to_confocal(vec, data_dict["$(prefix)confocal_to_nir"], data_dict["$(prefix)max_t"])

    body_angle = data_dict["$(prefix)body_angle"]
    n_pts = size(body_angle, 1)
    # Use min_len = 2 (absolute minimum for std/directional) instead of n_pts.
    # The library uses n_pts which requires ALL body angles to be valid — too strict.
    min_len = max(div(n_pts, 2), 2)

    data_dict["$(prefix)worm_curvature"] = safe_get_tot_worm_curvature(body_angle, min_len)
    data_dict["$(prefix)ventral_worm_curvature"] = safe_get_tot_worm_curvature(body_angle, min_len, directional=true)

    data_dict["nir_head_angle"] = -safe_impute_list(
        get_worm_body_angle(data_dict["x_array"], data_dict["y_array"], data_dict["segment_end_matrix"], param["head_pts"])
    )
    data_dict["nir_nose_angle"] = -safe_impute_list(
        get_worm_body_angle(data_dict["x_array"], data_dict["y_array"], data_dict["segment_end_matrix"], param["nose_pts"])
    )

    data_dict["$(prefix)head_angle"] = vec_to_confocal(data_dict["nir_head_angle"])
    data_dict["$(prefix)nose_angle"] = vec_to_confocal(data_dict["nir_nose_angle"])
end

"""
    get_nose_curling_safe!(data_dict, param; prefix="")

Replaces BehaviorDataNIR.get_nose_curling! with a safe version that
uses safe_impute_list instead of impute_list.
"""
function get_nose_curling_safe!(data_dict::Dict, param::Dict; prefix::String="")
    vec_to_confocal = vec -> nir_vec_to_confocal(vec, data_dict["$(prefix)confocal_to_nir"], data_dict["$(prefix)max_t"])
    data_dict["nir_nose_curling"] = Vector{Float64}()
    for t = 1:data_dict["max_t_nir"]
        if length(data_dict["segment_end_matrix"][t]) < param["max_pt"]
            push!(data_dict["nir_nose_curling"], NaN)
        else
            push!(data_dict["nir_nose_curling"], nose_curling(
                data_dict["x_array"][t, :], data_dict["y_array"][t, :],
                data_dict["segment_end_matrix"][t][1:param["max_pt"]], max_i=1
            ))
        end
    end
    data_dict["nir_nose_curling"] = safe_impute_list(data_dict["nir_nose_curling"])
    data_dict["$(prefix)nose_curling"] = vec_to_confocal(data_dict["nir_nose_curling"])
end

"""
    get_body_angles_safe!(data_dict, param; prefix="")

Replaces BehaviorDataNIR.get_body_angles! with a safe version that
uses safe_impute_list instead of impute_list for each body position.
"""
function get_body_angles_safe!(data_dict::Dict, param::Dict; prefix::String="")
    vec_to_confocal = vec -> nir_vec_to_confocal(vec, data_dict["$(prefix)confocal_to_nir"], data_dict["$(prefix)max_t"])
    s = size(data_dict["x_array"], 1)
    m = maximum([length(x) for x in data_dict["segment_end_matrix"]]) - 1
    conf_len = length(data_dict["$(prefix)confocal_to_nir"])
    data_dict["nir_body_angle"] = zeros(param["max_pt"]-1, s)
    data_dict["nir_body_angle_all"] = zeros(m, s)
    data_dict["nir_body_angle_absolute"] = zeros(m, s)
    data_dict["$(prefix)body_angle"] = zeros(param["max_pt"]-1, conf_len)
    data_dict["$(prefix)body_angle_all"] = zeros(m, conf_len)
    data_dict["$(prefix)body_angle_absolute"] = zeros(m, conf_len)
    for pos in 1:m
        for t in 1:s
            if length(data_dict["segment_end_matrix"][t]) > pos
                Δx = data_dict["x_array"][t, data_dict["segment_end_matrix"][t][pos+1]] - data_dict["x_array"][t, data_dict["segment_end_matrix"][t][pos]]
                Δy = data_dict["y_array"][t, data_dict["segment_end_matrix"][t][pos+1]] - data_dict["y_array"][t, data_dict["segment_end_matrix"][t][pos]]
                data_dict["nir_body_angle_absolute"][pos, t] = recenter_angle(vec_to_angle([Δx, Δy])[1])
                data_dict["nir_body_angle_all"][pos, t] = recenter_angle(vec_to_angle([Δx, Δy])[1] - data_dict["nir_worm_angle"][t])
            else
                data_dict["nir_body_angle_absolute"][pos, t] = NaN
                data_dict["nir_body_angle_all"][pos, t] = NaN
            end
        end
        data_dict["nir_body_angle_absolute"][pos, :] .= local_recenter_angle(data_dict["nir_body_angle_absolute"][pos, :], delta=param["body_angle_t_lag"])
    end

    for t in 1:s
        data_dict["nir_body_angle_absolute"][:, t] .= local_recenter_angle(data_dict["nir_body_angle_absolute"][:, t], delta=param["body_angle_pos_lag"])
        data_dict["nir_body_angle_all"][:, t] .= local_recenter_angle(data_dict["nir_body_angle_all"][:, t], delta=param["body_angle_pos_lag"])
    end

    for pos in 1:m
        data_dict["$(prefix)body_angle_all"][pos, :] .= vec_to_confocal(data_dict["nir_body_angle_all"][pos, :])
        data_dict["$(prefix)body_angle_absolute"][pos, :] .= vec_to_confocal(data_dict["nir_body_angle_absolute"][pos, :])

        if pos < param["max_pt"]
            data_dict["nir_body_angle"][pos, :] .= data_dict["nir_body_angle_all"][pos, :]
            data_dict["nir_body_angle"][pos, :] .= safe_impute_list(data_dict["nir_body_angle"][pos, :])
            data_dict["$(prefix)body_angle"][pos, :] .= vec_to_confocal(data_dict["nir_body_angle"][pos, :])
            data_dict["$(prefix)body_angle_absolute"][pos, :] .= vec_to_confocal(data_dict["nir_body_angle_absolute"][pos, :])
        end
    end
end

"""
    get_angular_velocity_safe!(data_dict, param; prefix="")

Replaces BehaviorDataNIR.get_angular_velocity! with a safe version.
"""
function get_angular_velocity_safe!(data_dict::Dict, param::Dict; prefix::String="")
    vec_to_confocal = vec -> nir_vec_to_confocal(vec, data_dict["$(prefix)confocal_to_nir"], data_dict["$(prefix)max_t"])
    data_dict["$(prefix)worm_angle"] = vec_to_confocal(data_dict["nir_worm_angle"])

    nir_turning_angle = safe_impute_list(data_dict["nir_body_angle_absolute"][param["head_pts"][1], :])
    data_dict["nir_angular_velocity"] = savitzky_golay_filter(nir_turning_angle, param["filt_len_angvel"], is_derivative=true, has_inflection=false) .* param["FLIR_FPS"]
    data_dict["$(prefix)angular_velocity"] = vec_to_confocal(data_dict["nir_angular_velocity"])
end

"""
    get_velocity_safe!(data_dict, param; prefix="")

Replaces BehaviorDataNIR.get_velocity! with a safe version.
"""
function get_velocity_safe!(data_dict::Dict, param::Dict; prefix::String="")
    vec_to_confocal = vec -> nir_vec_to_confocal(vec, data_dict["$(prefix)confocal_to_nir"], data_dict["$(prefix)max_t"])
    data_dict["filt_xmed"] = gstv(Float64.(data_dict["x_med"]), param["v_stage_m_filt"], param["v_stage_λ_filt"])
    data_dict["filt_ymed"] = gstv(Float64.(data_dict["y_med"]), param["v_stage_m_filt"], param["v_stage_λ_filt"])

    Δx = [0.0]
    Δy = [0.0]
    append!(Δx, diff(data_dict["filt_xmed"]))
    append!(Δy, diff(data_dict["filt_ymed"]))
    Δt = 1.0 / param["FLIR_FPS"]
    data_dict["nir_mov_vec_stage"] = make_vec(Δx, Δy)
    data_dict["$(prefix)mov_vec_stage"] = vec_to_confocal(data_dict["nir_mov_vec_stage"])
    data_dict["nir_mov_angle_stage"] = safe_impute_list(vec_to_angle(data_dict["nir_mov_vec_stage"]))
    data_dict["$(prefix)mov_angle_stage"] = vec_to_confocal(data_dict["nir_mov_angle_stage"])
    data_dict["nir_speed_stage"] = speed(Δx, Δy, Δt)
    data_dict["$(prefix)speed_stage"] = vec_to_confocal(data_dict["nir_speed_stage"])
    data_dict["nir_velocity_stage"] = data_dict["nir_speed_stage"] .* cos.(data_dict["nir_mov_angle_stage"] .- data_dict["pm_angle"])
    data_dict["$(prefix)velocity_stage"] = vec_to_confocal(data_dict["nir_velocity_stage"])
    data_dict["$(prefix)reversal_events"], data_dict["$(prefix)all_rev"] = get_reversal_events(param, data_dict["$(prefix)velocity_stage"], data_dict["$(prefix)t_range"], data_dict["$(prefix)max_t"])
    data_dict["$(prefix)rev_times"] = compute_reversal_times(data_dict["$(prefix)all_rev"], data_dict["$(prefix)max_t"])
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

    get_body_angles_safe!(data_dict, param)
    get_angular_velocity_safe!(data_dict, param)
    get_velocity_safe!(data_dict, param)
    get_curvature_variables_safe!(data_dict, param)
    get_nose_curling_safe!(data_dict, param)
    nothing
end

nothing
    