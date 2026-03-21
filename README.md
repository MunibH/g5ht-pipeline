## batch process
1. __batch_pipeline_stage1.py__
    - add full path to .nd2 in `# STAGE1` section of `datasets.txt`
    - performs steps 1-7 (through spline estimation)
2. __annotate nose positions in pipeline.ipynb__
    - add full path to .nd2 in `# ORIENT` section of `datasets.txt`
    - click on nose position for the first frame (required) and subsequent frames that need help
    - outputs a `orient_nose.csv` file
3. __batch_pipeline_stage2.py__
    - add full path to .nd2 in `# STAGE2` section of `datasets.txt`
    - performs orient, warp, register
