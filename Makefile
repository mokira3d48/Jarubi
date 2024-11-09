venv:
	python3 -m venv env

install:
	pip install -r requirements.txt

run:
	jupyter lab

