# Sync NIR-Confocal Python Module - Usage Guide

This module provides tools for synchronizing Near-Infrared (NIR) camera data with confocal microscopy data, commonly used in neuroscience experiments to correlate behavioral observations with neural imaging.

## Installation Requirements

```bash
pip install numpy h5py
```

## Quick Start

```python
import numpy as np
from sync_nir_confocal_converted import sync_timing_from_file, get_timing_info

# Basic synchronization from HDF5 file
path_h5 = "path/to/your/data.h5"
confocal_to_nir, nir_to_confocal, timing_stack, timing_nir = sync_timing_from_file(path_h5, n_rec=1)
```

## Core Functions

### 1. **Timing Detection**

#### `detect_nir_timing(di_nir, img_id, q_iter_save, n_img_nir)`
Detects start/stop times of NIR camera recording from digital input signals.

```python
# di_nir: digital input signal from NIR camera
# img_id: image ID array
# q_iter_save: boolean array indicating which frames were saved
# n_img_nir: total number of NIR images

timing = detect_nir_timing(di_nir, img_id, q_iter_save, n_img_nir)
# Returns: (N, 2) array with [start, stop] times for each frame
```

#### `detect_confocal_timing(ai_laser)`
Detects confocal microscopy stack timing from laser analog signal.

```python
# ai_laser: analog laser signal
list_stack_start, list_stack_stop = detect_confocal_timing(ai_laser)
# Returns: arrays of stack start and stop indices
```

### 2. **Synchronization**

#### `sync_timing_from_file(path_h5, n_rec=1)`
Main synchronization function - reads HDF5 file and returns timing mappings.

```python
confocal_to_nir, nir_to_confocal, timing_stack, timing_nir = sync_timing_from_file(
    path_h5="experiment_data.h5",
    n_rec=1  # number of recording sessions in file
)

# confocal_to_nir: list of arrays, each containing NIR frame indices for a confocal frame
# nir_to_confocal: array mapping each NIR frame to its confocal frame (0 = no mapping)
# timing_stack: (N, 2) array of confocal stack [start, stop] times
# timing_nir: (M, 2) array of NIR frame [start, stop] times
```

### 3. **Data Alignment**

#### `nir_vec_to_confocal(vec, confocal_to_nir, confocal_len)`
Bins NIR behavioral data to match confocal timepoints by averaging.

```python
# Example: align behavioral velocity data
nir_velocity = np.array([...])  # velocity at NIR frame rate
confocal_velocity = nir_vec_to_confocal(
    vec=nir_velocity,
    confocal_to_nir=confocal_to_nir,
    confocal_len=len(timing_stack)
)

# Works with 2D data too (e.g., multiple behavioral features)
nir_features = np.array([[...], [...]])  # shape: (n_features, n_timepoints)
confocal_features = nir_vec_to_confocal(nir_features, confocal_to_nir, confocal_len)
```

#### `sync_stim(stim, timing_stack, timing_nir)`
Aligns stimulus signal to both confocal and NIR timing.

```python
stim_signal = np.array([...])  # raw stimulus signal
stim_confocal, stim_nir = sync_stim(stim_signal, timing_stack, timing_nir)
```

### 4. **Complete Timing Setup**

#### `get_timing_info(data_dict, param, path_h5, h5_confocal_time_lag)`
Comprehensive function that sets up all timing variables in a data dictionary.

```python
data_dict = {}
param = {
    'n_rec': 1,           # number of recordings
    'max_t': 1000,        # maximum confocal timepoints
    't_range': range(1000),
    'FLIR_FPS': 30        # NIR camera frame rate
}

get_timing_info(
    data_dict=data_dict,
    param=param,
    path_h5="data.h5",
    h5_confocal_time_lag=0  # time offset if needed
)

# data_dict now contains:
# - 'confocal_to_nir', 'nir_to_confocal': timing mappings
# - 'timestamps', 'nir_timestamps': timestamp arrays
# - 'avg_timestep': average time between confocal frames
# - 'stim_begin_confocal', 'stim_begin_nir': stimulus onset times
# - 'pre_nir_to_confocal', 'pre_confocal_to_nir': pre-recording timing
```

### 5. **Utility Functions**

#### `get_timestamps(path_h5)`
Extract timestamps from HDF5 file.

```python
timestamps = get_timestamps("data.h5")  # in seconds
```

#### `unlag_vec(vec, lag)`
Correct for temporal lag in data.

```python
lagged_data = np.array([...])
unlagged_data = unlag_vec(lagged_data, lag=5)
```

#### `nir_to_confocal_t(t, nir_to_confocal)`
Convert single NIR timepoint to confocal timepoint.

```python
nir_frame = 150
confocal_frame = nir_to_confocal_t(nir_frame, nir_to_confocal)
```

#### `fill_timeskip(traces, timestamps, min_timeskip_length=5, timeskip_step=1, fill_val=0)`
Fill gaps in data for visualization.

```python
# traces: (n_neurons, n_timepoints) array
# timestamps: time array
new_timestamps, new_traces = fill_timeskip(
    traces=neural_data,
    timestamps=timestamps,
    min_timeskip_length=5,  # gaps > 5 sec will be filled
    timeskip_step=1,        # insert point every 1 sec
    fill_val=0              # fill with zeros
)
```

#### `signal_stack_repeatability(signal, timing_stack, sampling_rate=5000)`
Analyze signal consistency across repeated stacks.

```python
mean_signal, std_signal, time_array, n_stacks = signal_stack_repeatability(
    signal=piezo_signal,
    timing_stack=timing_stack,
    sampling_rate=5000
)

import matplotlib.pyplot as plt
plt.plot(time_array, mean_signal)
plt.fill_between(time_array, mean_signal - std_signal, mean_signal + std_signal, alpha=0.3)
```

## Complete Workflow Example

```python
import numpy as np
from sync_nir_confocal_converted import (
    sync_timing_from_file, get_timing_info, nir_vec_to_confocal
)

# 1. Load and synchronize timing
path_h5 = "experiment_2025_02_01.h5"
confocal_to_nir, nir_to_confocal, timing_stack, timing_nir = sync_timing_from_file(
    path_h5, n_rec=1
)

print(f"Confocal frames: {len(confocal_to_nir)}")
print(f"NIR frames: {len(nir_to_confocal)}")

# 2. Setup complete timing information
data_dict = {}
param = {
    'n_rec': 1,
    'max_t': len(confocal_to_nir),
    't_range': range(len(confocal_to_nir)),
    'FLIR_FPS': 30
}

get_timing_info(data_dict, param, path_h5, h5_confocal_time_lag=0)

# 3. Align behavioral data
# Assume you have NIR-rate behavioral data
nir_velocity = np.random.randn(len(nir_to_confocal))  # example data
confocal_velocity = nir_vec_to_confocal(
    nir_velocity, 
    data_dict['confocal_to_nir'], 
    param['max_t']
)

# 4. Access synchronized timestamps
confocal_timestamps = data_dict['timestamps']
nir_timestamps = data_dict['nir_timestamps']

# 5. Find stimulus onsets
stim_frames_confocal = data_dict['stim_begin_confocal']
stim_frames_nir = data_dict['stim_begin_nir']

print(f"Stimulus delivered at confocal frames: {stim_frames_confocal}")
```

## Data Requirements

The HDF5 file should contain:
- `img_nir`: NIR image stack (3D array)
- `daqmx_ai`: Analog inputs (N x 3 array)
  - Column 0: Laser signal
  - Column 1: Piezo signal  
  - Column 2: Stimulus signal
- `daqmx_di`: Digital inputs (N x 2 array)
  - Column 0: Confocal camera trigger
  - Column 1: NIR camera trigger
- `img_metadata`: Group containing:
  - `img_timestamp`: Timestamp for each frame
  - `img_id`: Image ID array
  - `q_iter_save`: Boolean array of saved frames

## Common Use Cases

### Case 1: Correlate Neural Activity with Behavior
```python
# Load calcium imaging data (confocal rate)
neural_activity = load_neural_data()  # shape: (n_neurons, n_confocal_frames)

# Load behavioral data (NIR rate)
behavior = load_behavior_data()  # shape: (n_nir_frames,)

# Align behavior to neural data
behavior_aligned = nir_vec_to_confocal(behavior, confocal_to_nir, len(neural_activity[0]))

# Now both are aligned and can be correlated
correlation = np.corrcoef(neural_activity[0], behavior_aligned)[0, 1]
```

### Case 2: Identify Frames Around Stimulus
```python
# Get stimulus timing
get_timing_info(data_dict, param, path_h5, 0)
stim_frames = data_dict['stim_begin_confocal']

# Extract neural activity around each stimulus
window = 50  # frames before/after
responses = []
for stim_frame in stim_frames:
    response = neural_activity[:, stim_frame-window:stim_frame+window]
    responses.append(response)
```

### Case 3: Validate Timing Consistency
```python
from sync_nir_confocal_converted import signal_stack_repeatability

# Check if z-stack timing is consistent
piezo_signal = load_piezo_data()
mean_piezo, std_piezo, time, n_stacks = signal_stack_repeatability(
    piezo_signal, timing_stack, sampling_rate=5000
)

# Plot to verify consistency
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(time, mean_piezo, 'k-', linewidth=2, label='Mean')
plt.fill_between(time, mean_piezo-std_piezo, mean_piezo+std_piezo, 
                 alpha=0.3, label='±1 SD')
plt.xlabel('Time (s)')
plt.ylabel('Piezo Signal')
plt.title(f'Stack Repeatability (n={n_stacks} stacks)')
plt.legend()
plt.show()
```

## Troubleshooting

### Error: "more than 2 recording on detected"
- Your data contains multiple recording sessions
- Increase `n_rec` parameter in `sync_timing_from_file()`

### Error: "detected trigger count is different from image id data"
- Mismatch between hardware triggers and saved frames
- Check data acquisition settings
- Verify HDF5 file integrity

### Empty or zero values in timing arrays
- Check that HDF5 file contains all required datasets
- Verify analog/digital signals are properly recorded
- Ensure proper thresholding of analog signals

## Key Differences from Julia Version

1. **Indexing**: Python uses 0-based indexing (Julia uses 1-based)
2. **Array operations**: NumPy functions replace Julia's broadcasting
3. **File I/O**: h5py replaces Julia's HDF5.jl
4. **In-place operations**: Python functions don't use `!` suffix convention
5. **Return values**: Functions return tuples instead of multiple return values

## Performance Notes

- For large datasets, consider processing in chunks
- Use NumPy's vectorized operations for best performance
- HDF5 file reading can be slow; consider caching results

## Citation

If using this code, please cite your original work that this was converted from.
