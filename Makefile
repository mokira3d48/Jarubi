venv:
	python3.10 -m venv env

install:
	python3 --version
	pip install --upgrade pip
	pip install torch torchvision --index-url "https://download.pytorch.org/whl/cpu" && \
	pip install -r requirements.txt

run:
	jupyter lab
