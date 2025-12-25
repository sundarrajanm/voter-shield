setup:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

check-quality:
	./scripts/quality.sh

run:
	python main.py --delete-old
