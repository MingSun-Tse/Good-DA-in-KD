# --------------------------------
# Change the settings to your need
CODE=$1 # Good-DA-in-KD
PY=$2 # 3.9.6
REQ=$3 # requirements.txt

# Usage example:
# sh scripts/set_up_env.sh Good-DA-in-KD 3.9.6 requirements.txt
# --------------------------------

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

# ACT="source $HOME/anaconda3/bin/activate" 
ACT="conda activate --no-stack "

# If in ipython, exit the ipython env first
echo "==> Anaconda path: $(which conda)"
conda deactivate

# If no anaconda env created, create it
ENV="${CODE}_Py${PY}"
echo "==> Anaconda env name: $ENV"
tmp=$(conda env list | grep "$ENV ")
if [ -z "$tmp" ]; then
    echo "==> Not found Anaconda env named '$ENV', creating it..."
    echo Y | conda create --name $ENV python=$PY ipython
    $ACT $ENV
    echo "==> Now start to install the dependencies in $REQ..."
    pip3 install -r $REQ
    echo "==> Done!"
fi
echo "==> Found Anaconda env named '$ENV', activate it: $ACT $ENV"
$ACT $ENV

# Print current conda env info
echo "==> Current conda env info:"
conda info