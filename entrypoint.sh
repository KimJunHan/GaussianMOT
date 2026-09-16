#!/bin/bash
clear
export PYTHONPATH=$PYTHONPATH:/workspace
# Use all 3 GPUs (container indices 0,1,2). Pair with trainer.devices=3.
export CUDA_VISIBLE_DEVICES=0,1,2

echo -e "\n------------------------------------------------------------------------------------\n"
figlet -c "GaussianCaR"
echo -e "\n------------------------------------ System info -----------------------------------\n"

CONDA_ENV="gaussiancar"
CONDA_BIN="/opt/conda/bin/conda"
ENV_PYTHON="/opt/conda/envs/$CONDA_ENV/bin/python"
ENV_PIP="/opt/conda/envs/$CONDA_ENV/bin/pip"

# ─── Conda 환경 생성 (Named Volume이라 최초 1회만 실행) ──────────────────────────
echo "🔄 Checking conda environment..."

# 유저별 ToS 동의 및 채널 설정
$CONDA_BIN tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
$CONDA_BIN tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
$CONDA_BIN config --add channels conda-forge 2>/dev/null || true
$CONDA_BIN config --set channel_priority strict 2>/dev/null || true

if ! $CONDA_BIN env list | grep -q "^$CONDA_ENV\s"; then
    echo "⚙️  Creating conda env '$CONDA_ENV' (Python 3.12.3)..."
    $CONDA_BIN create -n $CONDA_ENV python=3.12.3 --override-channels -c conda-forge -y \
        || { echo "❌ conda create failed"; exit 1; }

    echo "📦 [1/7] Installing PyTorch 2.8.0 + CUDA 12.9..."
    $ENV_PIP install torch==2.8.0 torchvision \
        --index-url https://download.pytorch.org/whl/cu129

    echo "📦 [2/7] Installing torch-scatter..."
    $ENV_PIP install torch-scatter==2.1.2+pt28cu129 \
        -f https://data.pyg.org/whl/torch-2.8.0+cu129.html

    echo "📦 [3/7] Installing spconv..."
    $ENV_PIP install spconv-cu126

    echo "📦 [4/7] Installing core dependencies..."
    $ENV_PIP install \
        addict \
        einops \
        hydra-core \
        lightning==2.5.0 \
        wandb \
        "numpy<2.0.0" \
        pillow \
        matplotlib \
        moviepy \
        nuscenes-devkit \
        timm \
        transformers \
        peft \
        fvcore \
        imgaug \
        jaxtyping \
        rootutils \
        efficientnet-pytorch \
        segmentation-models-pytorch \
        gradio \
        setuptools \
        wheel \
        ipykernel

    echo "📦 [5/7] Installing flash-attn (CUDA 빌드, 시간 소요)..."
    $ENV_PIP install flash-attn --no-build-isolation

    echo "📦 [6/7] Building diff-gaussian-rasterization..."
    cd /workspace/gaussiancar/ops/diff-gaussian-rasterization \
        && $ENV_PIP install -e . \
        && cd /workspace

    echo "📦 [7/7] Installing gaussiancar package..."
    $ENV_PIP install -e /workspace --no-deps

    echo -e "✅ Conda env '$CONDA_ENV' ready!\n"
else
    echo "✅ Conda env '$CONDA_ENV' found (named volume)"
fi

# ─── Conda 활성화 ────────────────────────────────────────────────────────────────
source /opt/conda/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# ─── nuScenes 데이터셋 확인 ──────────────────────────────────────────────────────
PATH_TO_NUSCENES="/data/nuscenes/nuscenes"
echo "🔍 Checking datasets availability..."
if [ ! -d "$PATH_TO_NUSCENES/samples" ] || [ ! -d "$PATH_TO_NUSCENES/sweeps" ]; then
    echo -e "❌ \033[91m\033[1mnuScenes not found at $PATH_TO_NUSCENES\033[0m"
else
    echo -e "✅ \033[92m\033[1mnuScenes found at $PATH_TO_NUSCENES\033[0m"
fi

# ─── GPU / CUDA 확인 ─────────────────────────────────────────────────────────────
echo -e "\n🔍 Checking GPU and CUDA availability..."
if ! $ENV_PYTHON -c "import torch" 2>/dev/null; then
    echo -e "❌ \033[91m\033[1mFailed to import torch\033[0m"
else
    CUDA_AVAILABLE=$($ENV_PYTHON -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_AVAILABLE" == "True" ]; then
        echo -e "✅ \033[92m\033[1mPyTorch + CUDA OK\033[0m"
        $ENV_PYTHON -c "import torch; print(f'   CUDA:    {torch.version.cuda}')"
        $ENV_PYTHON -c "import torch; print(f'   GPU:     {torch.cuda.get_device_name(0)}')"
        $ENV_PYTHON -c "import torch; print(f'   #GPUs:   {torch.cuda.device_count()}')"
    else
        echo -e "❌ \033[91m\033[1mCUDA not available\033[0m"
    fi
fi

echo -e "\n------------------------------------------------------------------------------------\n"
