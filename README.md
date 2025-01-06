# Recommender Model

## Setup
### Requirements
* python 3.9
### Setup the environment and python packages
#### Conda venv
```shell
source setup_venv.sh
```
#### Python venv
```shell
make venv PYTHON=python3.9
```
### Data
Get data from telegram channel: `[DEV/DS/IC] myTV`.

## Project Commands
### Quick start
#### Step 1:
```shell
mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db
```
#### Step 2:
Run training and evaluation.
```shell
CUDA_VISIBLE_DEVICES=0 python main.py
```
### API Usage
Updating...

###
### Run unit test
```shell
make test
```
### Run styling
```shell
make style
```
