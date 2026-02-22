function loop_control(ch_control)
    try
        img_array_uint32 = zeros(UInt32, IMG_SIZE_X, IMG_SIZE_Y)

        net_out = zeros(Float32, (3,3))
        x_nose, y_nose = 0, 0
        x_mid, y_mid = 0, 0
        x_pharynx, y_pharynx = 0, 0
        p_nose, p_pharynx, p_mid = 0., 0., 0.
        x_offset, y_offset = 0, 0

        x_displace, y_displace = 0, 0
        
        q_move_stage = false
        q_move_stage_count = 0
        
        
        for (q_iter_save, q_recording) in ch_control
            session.n_loop += 1

            # get image
            imid, imtimestamp = getimage!(cam, session.img_array,
                normalize=false, release=true)
            # println(imtimestamp)
            q_recording && push!(session.list_cam_info, (q_iter_save, q_recording, imid, imtimestamp))

            # detect features
            net_out .= dlc.get_pose(session.img_array[IMG_CROP_RG_X, IMG_CROP_RG_Y])

            # offsetting since input image to net is cropped
            y_nose, y_mid, y_pharynx = round.(Int, net_out[:,1]) .+ IMG_CROP_RG_Y[1] .+ 1
            x_nose, x_mid, x_pharynx = round.(Int, net_out[:,2]) .+ IMG_CROP_RG_X[1] .+ 1
            p_nose, p_mid, p_pharynx = net_out[:,3]

            @sync begin
                # stage control
                q_move_stage = false
                @async if session.q_tracking
                    try
                        detected_pts = [x_nose y_nose;
                            x_mid y_mid; x_pharynx y_pharynx]

                        # determine if enough info to move stage
                        if all([p_nose, p_mid, p_pharynx] .> param.θ_net) && check_pts_order_pca(detected_pts)
                            # determine offset
                            x_offset, y_offset = round.(Int, get_offset_loc(detected_pts, param.Δ_offset))
                            q_move_stage = true
                        end
                        if q_move_stage
                            # determine displacement
                            Δx, Δy = (x_offset, y_offset) .- (IMG_CENTER_X, IMG_CENTER_Y)
                            Δ_displacement = sqrt(Δx ^ 2 + Δy ^ 2)

                            x_displace = x_displace + Δx
                            y_displace = y_displace + Δy

                            if param.Δ_move_min <= Δ_displacement <= param.Δ_move_max
                                v_x_target = pid_x(Δx)
                                v_y_target = pid_y(Δy)

                                set_velocity(sp_stage, round(Int32, v_x_target), round(Int32, v_y_target))
                            else
                                q_move_stage_count += 1
                                if q_move_stage_count > 3
                                    q_move_stage_count = 0

                                    pid_x.reset()
                                    pid_y.reset()
                                    set_velocity(sp_stage, Int32(0), Int32(0))
                                end
                            end
                        else
                            q_move_stage_count += 1
                            if q_move_stage_count > 3
                                q_move_stage_count = 0
                                pid_x.reset()
                                pid_y.reset()
                                set_velocity(sp_stage, Int32(0), Int32(0))
                            end
                        end # if q_move_stage

#                         x_mm = covert_stage_unit_to_mm(session.x_stage)
#                         y_mm = covert_stage_unit_to_mm(session.y_stage)

                        #TODO: Assert that both have been zeroed and take the middle of them
                        if q_recording
                            Δt_recording = (time_ns() - session.t_recording_start) / 1e9
                        else
                            Δt_recording = 0
                        end
                    
                    catch e
                        println("Tracking async Error: ", e)
                    end
                end # async

                # display update
                @async if !q_iter_save
                    try
                        # 1s avg speed text
                        str_speed_avg = "N/A"
                        if length(session.speed_cb) > 0
                            cbnan = map(x->all(.!(isnan.(x.val))), session.speed_cb)
                            cb_first = session.speed_cb[findfirst(cbnan)]
                            cb_last = session.speed_cb[findlast(cbnan)]

                            Δt_cb = (cb_last.t - cb_first.t) / 1e9 # seconds
                            Δstage =  norm(cb_last.val .- cb_first.val, 2)
                            speed_worm_stage = Δstage / Δt_cb 
                            speed_worm_mm = covert_stage_unit_to_mm(speed_worm_stage) # mm/s
                            str_speed_avg = rpad(string(round(speed_worm_mm, digits=2)), 4, "0")
                            @emit updateTextSpeedAvg(str_speed_avg)

                        end
                    catch e
                        println("Save async Error: ", e)
                    end
                    try
                        if q_recording
                            Δt_recording = (time_ns() - session.t_recording_start) / 1e9
                            (Δt_recording_m, Δt_recording_s) = fldmod(Δt_recording, 60)
                            str_recording_duration = lpad(round(Int, Δt_recording_m), 2, "0") *
                                ":" * lpad(round(Int, Δt_recording_s), 2, "0")
                            @emit updateTextRecordingDuration(str_recording_duration)
                        end
                    catch e
                        println("Recording async Error: ", e)
                    end

                    # camera img
                    img_array_uint32 .= UInt32.(session.img_array)
                    img_array_uint32 .= (0xff000000 .+ img_array_uint32 .+
                        (img_array_uint32 .<< 8) .+ (img_array_uint32 .<< 16))

                    # mark detected location
                    if session.q_show_net
                        p_nose > param.θ_net && mark_feature!(img_array_uint32, x_nose, y_nose,
                            session.q_tracking ? param.c_nose_on : param.c_nose_off)
                        p_mid > param.θ_net && mark_feature!(img_array_uint32, x_mid, y_mid,
                            session.q_tracking ? param.c_mid_on : param.c_mid_off)
                        p_pharynx > param.θ_net && mark_feature!(img_array_uint32, x_pharynx, y_pharynx,
                            session.q_tracking ? param.c_pharynx_on : param.c_pharnyx_off)

                        q_move_stage && mark_feature!(img_array_uint32, x_offset, y_offset, param.c_offset)
                    end

                    # ruler
                    session.q_show_ruler && mark_ruler!(img_array_uint32, RULER_RG_X, RULER_RG_Y)

                    # crosshair
                    session.q_show_crosshair && mark_crosshair!(img_array_uint32, IMG_CENTER_X, IMG_CENTER_Y,
                        IMG_CROP_CENTER_X, IMG_CROP_CENTER_Y, param.Δ_move_max)


                    try
                        # updatedisplay buffer
                        canvas_buffer .= img_array_uint32[:]
                        @emit updateCanvas()
                    catch e
                        println("Canvas Update Error: ", e)
                    end

                end # @async display update

                @async if q_iter_save && q_recording
                    try
                        push!(session.list_img, deepcopy(session.img_array))
                        push!(session.list_pos_net, 
                            [x_nose y_nose p_nose; x_mid y_mid p_mid; x_pharynx y_pharynx p_pharynx])

                    catch e
                        println("Iteration Save async Error: ", e)
                    end
                end
            end



    #         yield()
        end # for
#         end # open do
    catch e
        println("Control Loop Error: ", e)
    end
end

function loop_stage(ch_stage)
    try
        x_stage, y_stage = Float64(0), Float64(0)

        for (q_iter_save, q_recording) in ch_stage
    #         push!(list_t_stage, time_ns())
            try
                sleep(0.001)
                query_position(sp_stage)
                sleep(0.025) 
                x_stage, y_stage = Float64.(read_position(sp_stage) ./ 2)
                push!(session.speed_cb, ValWithTime((x_stage, y_stage)))
                session.x_stage = x_stage
                session.y_stage = y_stage
                if !isnan(session.laser_zero_x)
                    @emit updateCoords(string(covert_stage_unit_to_mm(x_stage - session.laser_zero_x)), string(covert_stage_unit_to_mm(y_stage - session.laser_zero_y)))
                elseif !isnan(session.cooling_zero_calc_x)
                    @emit updateCoords(string(covert_stage_unit_to_mm(x_stage - session.cooling_zero_calc_x)), string(covert_stage_unit_to_mm(y_stage - session.cooling_zero_calc_y)))
                else
                    @emit updateCoords(string(round(covert_stage_unit_to_mm(x_stage),digits=2)), string(round(covert_stage_unit_to_mm(y_stage),digits=2)))
                    @emit updateFoodPatchCoords(string(round(covert_stage_unit_to_mm(session.TL_x),digits=2)), string(round(covert_stage_unit_to_mm(session.TL_y),digits=2)), 
                    string(round(covert_stage_unit_to_mm(session.TR_x),digits=2)), string(round(covert_stage_unit_to_mm(session.TR_y),digits=2)), 
                    string(round(covert_stage_unit_to_mm(session.BR_x),digits=2)), string(round(covert_stage_unit_to_mm(session.BR_y),digits=2)), 
                    string(round(covert_stage_unit_to_mm(session.BL_x),digits=2)), string(round(covert_stage_unit_to_mm(session.BL_y),digits=2)))
                end
                
                # if all food coordinate positions set, calculate polygon, track stage distance to nearest edge
                if !isnan(session.TL_x + session.TR_x + session.BR_x + session.BL_x)

                    # 1) Generate the lines once, when the polygon is first fully defined
                    # not using currently for functionality, but useful for debugging and visualization
                    if !session.polygonCalculated
                        spacing = float(param.line_space_px)  # 10.0 by default

                        # TL -> TR (top edge)
                        session.Line_TL_TR = line_points(
                            session.TL_x, session.TL_y,
                            session.TR_x, session.TR_y,
                            spacing,
                        )

                        # TR -> BR (right edge)
                        session.Line_TR_BR = line_points(
                            session.TR_x, session.TR_y,
                            session.BR_x, session.BR_y,
                            spacing,
                        )

                        # BR -> BL (bottom edge)
                        session.Line_BR_BL = line_points(
                            session.BR_x, session.BR_y,
                            session.BL_x, session.BL_y,
                            spacing,
                        )

                        # BL -> TL (left edge)
                        session.Line_BL_TL = line_points(
                            session.BL_x, session.BL_y,
                            session.TL_x, session.TL_y,
                            spacing,
                        )

#                         # Redraw the separate polygon plot
#                         redraw_polygon_canvas(x_stage, y_stage)
#                         @emit updatePolygonCanvas()

                        session.polygonCalculated = true
                    end

                    # 2) Distance from current stage position to each polygon edge
                    px = (x_stage)
                    py = (y_stage)

                    d_TL_TR, _ = dist_point_to_segment(
                        px, py,
                        session.TL_x, session.TL_y,
                        session.TR_x, session.TR_y,
                    )
                    d_TR_BR, _ = dist_point_to_segment(
                        px, py,
                        session.TR_x, session.TR_y,
                        session.BR_x, session.BR_y,
                    )
                    d_BR_BL, _ = dist_point_to_segment(
                        px, py,
                        session.BR_x, session.BR_y,
                        session.BL_x, session.BL_y,
                    )
                    d_BL_TL, _ = dist_point_to_segment(
                        px, py,
                        session.BL_x, session.BL_y,
                        session.TL_x, session.TL_y,
                    )

                    nearest_edge_dist_stage = minimum((
                        d_TL_TR, d_TR_BR, d_BR_BL, d_BL_TL
                    ))

                    nearest_edge_dist_mm = covert_stage_unit_to_mm(nearest_edge_dist_stage)
                    push!(session.dist2patch_mm, Float64(covert_stage_unit_to_mm(nearest_edge_dist_stage)))
                    @emit updateDistanceToPatch(string(round(nearest_edge_dist_mm, digits=3)))
                end

                
            catch e # Brian: This seems to happen frequently . Mostly: AssertionError("read_data[2] == 0x01") or BoundsError(UInt8[0x23, 0x01, 0xd4, 0x00, 0x05, 0x00, 0x04, 0x00, 0x3e, 0x00, 0x00, 0x00, 0x0d], (14,))
#                 println("Loop Stage Error: ", e)
                x_stage, y_stage = NaN, NaN 
            end
#             session.x_stage, session.y_stage = x_stage, y_stage
            q_recording && push!(session.list_pos_stage, Float64[x_stage, y_stage])
        end
    catch e
        println("Stage async Error: ", e)
    end
#     yield()
end

function loop_recording(ch_recording)
    try
        for (q_iter_save, q_recording) in ch_recording
            q_recording && nidaq_read_data()

#             q_recording && push!(session.list_laser_info, reshape([Float64(session.laser_temp), Float64(session.laser_pow), Float64(session.laser_enabled)], 1, 1, 3))
#             q_recording && push!(session.list_cooling_info, reshape([Float64(session.stage_temp), Float64(session.cooling_flow_rate)], 1, 1, 2))
        end 
    catch e
        println("Recording Error: ", e)
    end
end


function loop_main()    
    ch_stage = Channel{Tuple{Bool,Bool}}(16)
    ch_control = Channel{Tuple{Bool,Bool}}(16)
    ch_recording = Channel{Tuple{Bool,Bool}}(16)
    session.q_loop = true

    @sync begin
        try
            # start(task_ai_temp)
            start(task_laser)
            setLaserPower(0.0)
            
            @async loop_stage(ch_stage)
            @async loop_control(ch_control)
            @async loop_recording(ch_recording) # Threads.@spawn

            local loop_count = 1
            local q_recording = false

            start!(cam)
            Timer(0, interval=1/LOOP_INTERVAL_CONTROL) do timer
                if !q_recording && session.q_recording # start rec
                    try
                        start(task_di)
                        println("DI works")
                        start(task_ai)
                        stop!(cam)
                        sleep(0.001)
                        start!(cam)
                    catch e
                        println("Recording Start Error: ", e)
                    end
                elseif q_recording && !(session.q_recording) # stop rec
                    stop!(cam)
                    sleep(0.001)
                    start!(cam)
                    nidaq_read_data()
                    stop(task_ai)
                    stop(task_di)
                end
                q_recording = session.q_recording

                if session.q_loop == false && loop_count == 1
                    close(ch_control)
                    close(ch_stage)
                    close(timer)
                    # stop(task_ai_temp)
                    stop(task_laser)
                    stop!(cam)
                elseif isodd(loop_count)
                    put!(ch_control, (true, q_recording))
                    put!(ch_stage, (true, q_recording))
                    loop_count += 1
                elseif loop_count == 20
                    put!(ch_control, (false, q_recording))
                    put!(ch_recording, (false, q_recording))
                    loop_count = 1
                else
                    put!(ch_control, (false, q_recording)) #false #BRIAN
    #                 put!(ch_control, (true, q_recording))
    #                 put!(ch_stage, (true, q_recording))
                    loop_count += 1
                end
            end # timer
        catch e
            println("Main Loop Error: ", e)
        end
    end
end

"""
Evenly spaced points along a line segment from (x1,y1) to (x2,y2).

Returns a 2×N matrix, each column is a point [x; y].
Spacing is approximate: we take floor(L/spacing) segments.
"""
function line_points(x1::Float64, y1::Float64,
                     x2::Float64, y2::Float64,
                     spacing::Float64)
    dx = x2 - x1
    dy = y2 - y1
    L = sqrt(dx^2 + dy^2)

    if L == 0 || !isfinite(L)
        pts = Array{Float64,2}(undef, 2, 1)
        pts[:,1] .= (x1, y1)
        return pts
    end

    n_segments = max(1, floor(Int, L / spacing))
    ts = range(0.0, 1.0; length = n_segments + 1)

    pts = Array{Float64,2}(undef, 2, length(ts))
    @inbounds for (i, t) in enumerate(ts)
        pts[1,i] = x1 + dx * t
        pts[2,i] = y1 + dy * t
    end
    return pts
end

"""
Shortest distance from point P=(px,py) to segment A=(x1,y1)–B=(x2,y2).
Returns the distance (in stage units) and the projection point (projx, projy).
"""
function dist_point_to_segment(px::Float64, py::Float64,
                               x1::Float64, y1::Float64,
                               x2::Float64, y2::Float64)
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1

    seg_len2 = vx^2 + vy^2
    if seg_len2 == 0
        # A and B are the same point
        dx = px - x1
        dy = py - y1
        return sqrt(dx^2 + dy^2), (x1, y1)
    end

    t = (wx*vx + wy*vy) / seg_len2 # projection of p onto infinite line
    t_clamped = clamp(t, 0.0, 1.0) # just on the segment now

    projx = x1 + t_clamped * vx
    projy = y1 + t_clamped * vy

    dx = px - projx
    dy = py - projy
    return sqrt(dx^2 + dy^2), (projx, projy) # returns: distance from (px,py) to closest point on segment, closest point
end

# Bresenham-style line drawer on a 2D canvas
function draw_line!(canvas::Array{UInt32,2},
                    x1::Int, y1::Int, x2::Int, y2::Int,
                    color::UInt32)
    x, y = x1, y1
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = x1 < x2 ? 1 : -1
    sy = y1 < y2 ? 1 : -1
    err = dx - dy

    while true
        if 1 ≤ x ≤ size(canvas,1) && 1 ≤ y ≤ size(canvas,2)
            canvas[x,y] = color
        end
        x == x2 && y == y2 && break
        e2 = 2 * err
        if e2 > -dy
            err -= dy
            x += sx
        end
        if e2 < dx
            err += dx
            y += sy
        end
    end
end

function draw_point!(canvas::Array{UInt32,2}, x::Int, y::Int,
                     color::UInt32; r::Int=3)
    for i in max(1, x-r):min(size(canvas,1), x+r)
        for j in max(1, y-r):min(size(canvas,2), y+r)
            canvas[i,j] = color
        end
    end
end

function redraw_polygon_canvas(px::Float64, py::Float64)
    # Clear to black
    fill!(poly_canvas_buffer, 0xff000000)
    canvas = reshape(poly_canvas_buffer, POLY_CANVAS_W, POLY_CANVAS_H)

    # Need a valid polygon
    if any(isnan, (session.TL_x, session.TR_x, session.BR_x, session.BL_x))
        return
    end

    # --- 1) Compute bounds in stage coords ---
    xs = [session.TL_x, session.TR_x, session.BR_x, session.BL_x, px]
    ys = [session.TL_y, session.TR_y, session.BR_y, session.BL_y, py]

    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)

    dx = xmax - xmin
    dy = ymax - ymin
    dx = dx == 0 ? 1.0 : dx
    dy = dy == 0 ? 1.0 : dy

    padding = 0.1
    xmin -= padding * dx
    xmax += padding * dx
    ymin -= padding * dy
    ymax += padding * dy

    sx = (POLY_CANVAS_W - 1) / (xmax - xmin)
    sy = (POLY_CANVAS_H - 1) / (ymax - ymin)

    # stage -> canvas (x right, y up)
    to_canvas(x, y) = begin
        cx = Int(clamp(round((x - xmin) * sx) + 1, 1, POLY_CANVAS_W))
        cy = Int(clamp(POLY_CANVAS_H - round((y - ymin) * sy), 1, POLY_CANVAS_H))
        (cx, cy)
    end

    # --- 2) Find nearest point on polygon to stage position ---
    px_stage, py_stage = px, py

    d_TL_TR, p_TL_TR = dist_point_to_segment(
        px_stage, py_stage,
        session.TL_x, session.TL_y,
        session.TR_x, session.TR_y,
    )
    d_TR_BR, p_TR_BR = dist_point_to_segment(
        px_stage, py_stage,
        session.TR_x, session.TR_y,
        session.BR_x, session.BR_y,
    )
    d_BR_BL, p_BR_BL = dist_point_to_segment(
        px_stage, py_stage,
        session.BR_x, session.BR_y,
        session.BL_x, session.BL_y,
    )
    d_BL_TL, p_BL_TL = dist_point_to_segment(
        px_stage, py_stage,
        session.BL_x, session.BL_y,
        session.TL_x, session.TL_y,
    )

    dists = [d_TL_TR, d_TR_BR, d_BR_BL, d_BL_TL]
    projs = [p_TL_TR, p_TR_BR, p_BR_BL, p_BL_TL]
    i_min = argmin(dists)
    nearest_dist = dists[i_min]
    nearest_proj = projs[i_min]
    projx_stage, projy_stage = nearest_proj

    # --- 3) Draw polygon edges ---
    TLc = to_canvas(session.TL_x, session.TL_y)
    TRc = to_canvas(session.TR_x, session.TR_y)
    BRc = to_canvas(session.BR_x, session.BR_y)
    BLc = to_canvas(session.BL_x, session.BL_y)

    color_edge = 0xffffffff  # white
    draw_line!(canvas, TLc[1], TLc[2], TRc[1], TRc[2], color_edge)
    draw_line!(canvas, TRc[1], TRc[2], BRc[1], BRc[2], color_edge)
    draw_line!(canvas, BRc[1], BRc[2], BLc[1], BLc[2], color_edge)
    draw_line!(canvas, BLc[1], BLc[2], TLc[1], TLc[2], color_edge)

    # --- 4) Draw stage position ---
    sx_c, sy_c = to_canvas(px_stage, py_stage)
    draw_point!(canvas, sx_c, sy_c, 0xff00ff00; r=4)  # green

    # --- 5) Draw vector to nearest point on edge ---
    projx_c, projy_c = to_canvas(projx_stage, projy_stage)
    draw_line!(canvas, sx_c, sy_c, projx_c, projy_c, 0xff00ffff)  # cyan line
    draw_point!(canvas, projx_c, projy_c, 0xffff0000; r=3)        # red projection point
end



function stop_loop_main()
    session.q_loop = false
end
