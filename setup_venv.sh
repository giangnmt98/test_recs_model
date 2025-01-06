conda create -n py39_recmodel python=3.9.18 -y
conda activate py39_recmodel
conda install conda-forge/label/cf202003::openjdk -y
conda install cuda -c nvidia/label/cuda-12.2.0 -y
python3 -m pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==23.8.* cuml-cu12==23.8.*
python3 -m pip install -r requirements.txt
