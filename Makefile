setup:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest -m regression -v -s

run:
	python main.py
