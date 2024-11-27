SHELL = /bin/bash
PYTHON := python3
VENV_NAME = rec_model_env
RECMODEL_FOLDER = recmodel
TEST_FOLDER = tests

# Environment
venv:
	${PYTHON} -m venv ${VENV_NAME} && \
	source ${VENV_NAME}/bin/activate && \
	${PYTHON} -m pip install pip setuptools wheel && \
	${PYTHON} -m pip install \
		--extra-index-url=https://pypi.nvidia.com \
		cudf-cu12==23.8.* cuml-cu12==23.8.* && \
	${PYTHON} -m pip install -e .[dev] && \
	pre-commit install

# Style
style:
	black ./${RECMODEL_FOLDER}/
	flake8 ./${RECMODEL_FOLDER}/
	${PYTHON} -m isort -rc ./${RECMODEL_FOLDER}/

test:
	${PYTHON} -m flake8 ./${RECMODEL_FOLDER}/
	${PYTHON} -m mypy ./${RECMODEL_FOLDER}/
	CUDA_VISIBLE_DEVICES="" TEST_MODE="yes" ${PYTHON} -m pytest -s --durations=0 --disable-warnings ${TEST_FOLDER}/