

# Set up models
if [ ! -d save/models/resnet110_vanilla ]; then
    mkdir -p save/models
    cd save
    wget https://github.com/MingSun-Tse/Good-DA-in-KD/releases/download/v0.1/teachers_cifar100.zip # These teachers are from the CRD paper [Tian et al., ICLR, 2020]
    unzip teachers_cifar100.zip
    cd ..
fi

# These Tiny ImageNet models are trained by ourselves
if [ ! -d save/models_tinyimagenet_v2/resnet56_vanilla ]; then
    mkdir -p save/models_tinyimagenet_v2
    cd save
    wget https://github.com/MingSun-Tse/Good-DA-in-KD/releases/download/v0.1/teachers_tinyimagenet.zip
    unzip teachers_tinyimagenet.zip
    cd ..
fi