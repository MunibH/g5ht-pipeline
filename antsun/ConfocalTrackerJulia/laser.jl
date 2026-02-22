using NIDAQ, Optim, Statistics

# Globally Definied `task_laser` in nidaq.jl

# Change in temp (C) per cm
dTdD = 1.5

# The ambient temperature of the stage
amb = 15.0

# Calibration Terms
agarHeatOffset = 17.7695
agarHeatCalibrationFactor = 23.24999999999999

function coordsToTemp(x, y)
    ((x  * dTdD) / 10) + 18
end


function tempToPow(temp)
#     max((pow - 15) / 10, 0.0)
    return max(0.0, (temp - agarHeatOffset) / agarHeatCalibrationFactor)
end


# function coordsToPow(x, y)
#     tempToPow(coordsToTemp(x, y))
# end

# function coordsToPow()
#     max(x / 10, 0)
# end


function setLaserPower(out)
    # Takes laser power on a 0-1 scale and converts it to voltage 0-10
    if (out > 1 || out < 0)
        println(String(out) * " is not a valid laser power")
        write(task_laser, [0]) 
    end
    methods(write)
    write(task_laser, [out * 10])
    
end