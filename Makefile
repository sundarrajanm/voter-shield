setup:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

check-quality:
	./scripts/quality.sh

run:
	python main.py --delete-old

run-dev-docker:
	time docker run --rm \
		--cpus=1 \
		--memory=4g \
		-v .:/app \
		votershield-calib \
		--delete-old
