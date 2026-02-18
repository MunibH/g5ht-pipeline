
probably need julia version 1.8.2

### Message from Albert:

Oh yup before he left David duplicated a whole bunch of repos and made them private so the public one would be unaltered.

If you're just doing the behavioral data I would just recommend the public one I don’t anyone has even changed anything for the behavioral analysis

The setup script is also here but I definitely don't recommend running it one a computer where you plan to run anything else https://github.com/flavell-lab/flv-c-setup

Alex might also be able to give you advice on best practices to set it up. I think he sets up most people's in lab c4 etc for them

### USAGE
run behav_pipeline.jl.ipynb
then run python stuff

### Julia packages

can probably use this to install everything (https://github.com/flavell-lab/FlavellPkg.jl) along with flv-c-setup from albert's message above

## steps to install

`setx JULIA_DEPOT_PATH "$env:USERPROFILE\.julia-flv"`

`$env:JULIA_DEPOT_PATH = "$env:USERPROFILE\.julia-flv"`

`julia`

```
    import Pkg
    ENV["PYTHON"] = ""
    Pkg.add("PyCall")
    Pkg.build("PyCall")

    using PyCall
    println(PyCall.python)
```
Now it should print something like:

`C:\Users\munib\.julia-flv\conda\3\x86_64\python.exe`

```
conda create -n flavell-py310 python=3.10 -y
conda activate flavell-py310
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-cpu==2.15.*     OR   pip install tensorflow-cpu
pip install matplotlib nd2reader hdbscan tensorboard tensorboardX h5py simpleitk pyyaml

```

Stopped here on lab pc

# 4) Clone + install  lab’s private Python repos
### 4a) Make sure git clone git@github.com:... works

You need Git + SSH keys set up for GitHub (and access to flavell-lab). Quick test:

ssh -T git@github.com

If that doesn’t succeed, you’ll need to set up an SSH key + add it to GitHub before proceeding.

check if you have an SSH key:
ls ~/.ssh

getting an error here, so maybe need this:
https://github.com/flavell-lab/FlavellLabWiki/wiki/Setting-up-SSH-key-on-Windows

### 4b) Clone + pip install each repo

This mirrors flv-init.sh script, but using Windows paths.

```
$DateStr = Get-Date -Format "yyyy-MM-dd"
$TempSrc = "$env:USERPROFILE\setup-src-$DateStr"
$SrcDir  = "$env:USERPROFILE\src"
$Repos = @("unet2d","euler_gpu","DeepReg","AutoCellLabeler")

New-Item -ItemType Directory -Force -Path $TempSrc | Out-Null
Set-Location $TempSrc

foreach ($repo in $Repos) {
    if (Test-Path $repo) { Remove-Item -Recurse -Force $repo }
    git clone "git@github.com:flavell-lab/$repo.git" $repo
    Set-Location "$TempSrc\$repo"
    if ($repo -eq "unet2d") { git checkout v0.1 }
    pip install .
    Set-Location $TempSrc
}
```