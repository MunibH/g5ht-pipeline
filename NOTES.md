- there's nothing in the .nd2 file metadata that indicates number of z-slices per frame
    - should be able to get that automatically from somewhere? 
        - doesn't seem to be saved anywhere
        - 41 slices per frame
        - 0.36 um for each timepoint
- even though we focus in the middle of the worm, the recording starts at the bottom most z-coordinate and moves up 41 slices, then resets, and repeats per frame stack
- last 2 z-slices are trimmed? 
    - piezo starts moving back up at 39 frames, so it's not clear where those z-slices are taken in space


# TODO
## for preprocess.py
- noise stack is loaded in `run_pipeline.ipynb` rather than in `preprocess.py`
- parallelize rather than submit a job per frame
- crosstalk???
- define a baseline? 

## for preprocess_array.sh and submit_preprocess.sh
- need to figure out QOSMaxCpuPerUserLimit and how job arrays and cpus-per-task fit in
- need to make the log more verbose
- chunk_concurrency is annoying. might just get rid of it, dont know if it does anything


## for mip.py
- this should not be hardcoded `parameter_object.ReadParameterFile('/home/albert_w/scripts/template.txt')`
- `fps=5/0.53` ????