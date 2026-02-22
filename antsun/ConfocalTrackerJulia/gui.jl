# gui functions
function update_img(buffer::Array{UInt32, 1},
        width32::Int32, height32::Int32)
    width::Int = width32
    height::Int = height32
    
    buffer .= canvas_buffer
    nothing
end

# function update_polygon_canvas(buffer::Array{UInt32,1},
#                                width32::Int32, height32::Int32)
#     width::Int = width32
#     height::Int = height32
#     @assert width == POLY_CANVAS_W
#     @assert height == POLY_CANVAS_H

#     buffer .= poly_canvas_buffer
#     nothing
# end

function mark_feature!(canvas::Array{UInt32,2}, x::Int, y::Int,
        color_mark::UInt32, r_pix::Int=5)
    x_ = clamp(x, 1+r_pix, IMG_SIZE_Y-r_pix)
    y_ = clamp(y, 1+r_pix, IMG_SIZE_Y-r_pix)
        
    canvas[x_-r_pix:x_+r_pix, y_-r_pix:y_+r_pix] .= color_mark
    
    nothing
end


function mark_ruler!(canvas::Array{UInt32,2}, RULER_RG_X::Array{Int64,1},
        RULER_RG_Y::Array{Int64,1})    
    for x_ = RULER_RG_X
        canvas[x_,:] .= 0xffffff00
    end
    for y_ = RULER_RG_Y
        canvas[:,y_] .= 0xffffff00
    end
    
    nothing
end

function mark_crosshair!(canvas::Array{UInt32,2}, IMG_CENTER_X::Int, IMG_CENTER_Y::Int,
    IMG_CROP_CENTER_X::Int, IMG_CROP_CENTER_Y::Int, Δ_move_max::Int)
    
    # center lines
    canvas[IMG_CENTER_X,:] .= 0xffff0000
    canvas[:,IMG_CENTER_Y] .= 0xffff0000

    # Δ_move_max
    for i = IMG_CENTER_X-Δ_move_max:IMG_CENTER_X+Δ_move_max
        for j = IMG_CENTER_Y-Δ_move_max:IMG_CENTER_Y+Δ_move_max
            if Δ_move_max - 1 < sqrt((i - IMG_CENTER_X) ^ 2 +
                    (j - IMG_CENTER_Y) ^ 2) < Δ_move_max
                canvas[i, j] = 0xffff0000
            end
        end
    end

    # cropped image field of view for deepnet
    canvas[IMG_CENTER_X-IMG_CROP_CENTER_X:IMG_CENTER_X+IMG_CROP_CENTER_X,
        IMG_CENTER_Y-IMG_CROP_CENTER_Y] .= 0xffff0000
    canvas[IMG_CENTER_X-IMG_CROP_CENTER_X:IMG_CENTER_X+IMG_CROP_CENTER_X,
        IMG_CENTER_Y+IMG_CROP_CENTER_Y] .= 0xffff0000
    canvas[IMG_CENTER_X-IMG_CROP_CENTER_X,
        IMG_CENTER_Y-IMG_CROP_CENTER_Y:IMG_CENTER_Y+IMG_CROP_CENTER_Y] .= 0xffff0000
    canvas[IMG_CENTER_X+IMG_CROP_CENTER_X,
        IMG_CENTER_Y-IMG_CROP_CENTER_Y:IMG_CENTER_Y+IMG_CROP_CENTER_Y] .= 0xffff0000

    nothing
end

# controls
function toggle_tracking()    
    if session.q_tracking
        session.q_tracking = false
        set_velocity(sp_stage, Int32(0), Int32(0))
        
        return "Start"
    else
        session.q_tracking = true
        
        return "Stop"
    end
end

function toggle_ruler()    
    if session.q_show_ruler
        session.q_show_ruler = false
        
        return "Show"
    else
        session.q_show_ruler = true
        
        return "Hide"
    end
end

function toggle_crosshair()
    if session.q_show_crosshair
        session.q_show_crosshair = false
        
        return "Show"
    else
        session.q_show_crosshair = true
        
        return "Hide"
    end
end

function toggle_recording()
    try
        if session.q_recording
            session.q_recording = false
            @emit updateTextRecordingDuration("N/A")
            return "Start"
        else
            reset_recording!(session)
            session.q_recording = true
            session.t_recording_start_str = string(Dates.now())
            session.t_recording_start = time_ns()
            return "Stop"
        end
    catch e
        println("Record Toggle Error: ", e)
    end
end

function toggle_deepnetoutput()
    if session.q_show_net
        session.q_show_net = false
        
        return "Show"
    else
        session.q_show_net = true
        
        return "Hide"
    end
end

function send_halt_stage()
    halt_stage(sp_stage)
    
    nothing
end

function window_close()    
    session.q_loop = false
    session.q_tracking = false
    session.q_recording = false
    
    set_velocity(sp_stage, Int32(0), Int32(0))
    close(sp_stage)

    nothing
end

function start_gui()
    @qmlfunction toggle_crosshair
    @qmlfunction toggle_tracking
    @qmlfunction toggle_ruler
    @qmlfunction toggle_recording
    @qmlfunction toggle_deepnetoutput
    @qmlfunction zero_cooling_right
    @qmlfunction zero_cooling_left
    @qmlfunction zero_laser
    @qmlfunction set_TL_coord
    @qmlfunction set_TR_coord
    @qmlfunction set_BR_coord
    @qmlfunction set_BL_coord
    @qmlfunction send_halt_stage
    @qmlfunction window_close

    loadqml(path_qmlfile,
        cf_update=CxxWrap.@safe_cfunction(update_img, Cvoid, 
            (Array{UInt32,1}, Int32, Int32)))
#     loadqml(path_qmlfile,
#         cf_update = CxxWrap.@safe_cfunction(update_img, Cvoid,
#             (Array{UInt32,1}, Int32, Int32)),
#         cf_update_polygon = CxxWrap.@safe_cfunction(update_polygon_canvas, Cvoid,
#             (Array{UInt32,1}, Int32, Int32))
#     )
    exec_async()
    
    nothing
end


# Stage Zeroing
function zero_cooling_right()
    if isnan(session.cooling_zero_right_x)
        session.cooling_zero_right_x = session.x_stage
        session.cooling_zero_right_y = session.y_stage
        
        return "Zeroed"
    else
        println("Currently, you must restart the kernel to re-zero the cooling program")
        
        return "Zeroed"
    end
end

function zero_cooling_left()
    if isnan(session.cooling_zero_left_x)
        session.cooling_zero_left_x = session.x_stage
        session.cooling_zero_left_y = session.y_stage
        
        return "Zeroed"
    else
        println("Currently, you must restart the kernel to re-zero the cooling program")
        
        return "Zeroed"
    end
end

function zero_laser()
    if isnan(session.laser_zero_x)
        session.laser_zero_x = session.x_stage
        session.laser_zero_y = session.y_stage
            
        return "Zeroed"
    else
        println("Currently, you must restart the kernel to re-zero the cooling program")
        # TODO: Array which logs previous zeros? Def don't want anything fully destructive

        return "Zeroed"
    end
end

function set_TL_coord()
    if isnan(session.TL_x)
        session.TL_x = copy(session.x_stage)
        session.TL_y = copy(session.y_stage)
            
        return "TL is set"
    else
        println("Currently, you must restart the kernel to reset the coordinate")
        # TODO: Array which logs previous zeros? Def don't want anything fully destructive

        return "TL already set"
    end
end

function set_TR_coord()
    if isnan(session.TR_x)
        session.TR_x = copy(session.x_stage)
        session.TR_y = copy(session.y_stage)
            
        return "TR is set"
    else
        println("Currently, you must restart the kernel to reset the coordinate")
        # TODO: Array which logs previous zeros? Def don't want anything fully destructive

        return "TR already set"
    end
end

function set_BR_coord()
    if isnan(session.BR_x)
        session.BR_x = copy(session.x_stage)
        session.BR_y = copy(session.y_stage)
            
        return "BR is set"
    else
        println("Currently, you must restart the kernel to reset the coordinate")
        # TODO: Array which logs previous zeros? Def don't want anything fully destructive

        return "BR already set"
    end
end

function set_BL_coord()
    if isnan(session.BL_x)
        session.BL_x = copy(session.x_stage)
        session.BL_y = copy(session.y_stage)
            
        return "BL is set"
    else
        println("Currently, you must restart the kernel to reset the coordinate")
        # TODO: Array which logs previous zeros? Def don't want anything fully destructive

        return "BL already set"
    end
end





# TODO: Buttons to control cooling:
    # Reset pid
    # Turn off?
    