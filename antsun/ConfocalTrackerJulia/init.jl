const py_tf = PyNULL()
const py_pid = PyNULL()
const py_dlc = PyNULL()

ENV["QSG_RENDER_LOOP"] = "basic"

function __init__()
    # data
    global canvas_buffer = zeros(UInt32, 968 * 732)
#     global poly_canvas_buffer = fill(UInt32(0xff000000), POLY_CANVAS_W * POLY_CANVAS_H)  # black BG
    global param = Param()
    global session = SessionData()

    # gui
    start_gui()
    
    # py
    copy!(py_tf, pyimport("tensorflow"))
    copy!(py_dlc, pyimport("dlclive"))
    copy!(py_pid, pyimport("simple_pid"))
 
    # stage
    global sp_stage = LibSerialPort.open(DEFAULT_PORT, SERIAL_BAUD_RATE)
    check_baud_rate(sp_stage)
    
    # camera
    init_cam()
    
    # tracking
    init_deepnet()
    init_pid()
    
    # NIDAQ
    init_nidaq()
    
    nothing
end