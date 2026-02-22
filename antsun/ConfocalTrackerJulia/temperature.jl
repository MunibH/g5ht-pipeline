using NIDAQ, Optim, Statistics

# SAMPLE_RATE_AI = 1000 # Hz
# DEV_NAME = "Dev1"


# macro daqmx_check(exp)
#     error_code = eval(exp)
#     if error_code != 0
#         error("DAQmx error $error_code")
#     end
    
#     nothing
# end

function kOhmsToC(ohms)
#     A = 1.024546757e-3
#     B = 1.986759795e-4
#     C = 3.445466326e-7
    
    # Using thermocouple as baseline - 9/23/25
    A = 0.3788898044e-3
    B = 3.081223060e-4
    C = -1.007874924e-7

    temp = A + (B * log(ohms)) + (C * (log(ohms)^3))
    round((1/temp) - 273.15, digits=2) 
end

# empty_str = ""
# ptr_empty_str = pointer(empty_str)


# # Set up analog input
# task_ai_temp = analog_input("$DEV_NAME/ai4", terminal_config=NIDAQ.Differential)
# @daqmx_check NIDAQ.DAQmxCfgSampClkTiming(task_ai_temp.th, ptr_empty_str, SAMPLE_RATE_AI, NIDAQ.DAQmx_Val_Rising,
#     NIDAQ.DAQmx_Val_ContSamps, SAMPLE_RATE_AI)


function getCurrentTemperature()
    # start(task_ai_temp)
#     read(task_ai, -1) # Clear the buffer
    samps = read(task_ai_temp, Int(NIDAQ_SAMPLE_RATE_AI/10)) # Reads 500 samples and takes the average. If you're sampling at 5000hz then this takes .1 seconds
    avg = mean(samps)
    Rt = (avg * 20000.0) # Convert from Volts to Ohms (*20 - because of the 200C's conversion factor) to KOhm (*1000)
    temp = kOhmsToC(Rt)
    # println(samps)
    # println(temp)
    # stop(task_ai_temp)
    
    temp
end