.PHONY: install train evaluate inference clean help

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make train      - Train model from scratch"
	@echo "  make evaluate   - Evaluate trained model"
	@echo "  make inference  - Run inference on test set"
	@echo "  make clean      - Remove generated files"
	@echo "  make format     - Format code with black"
	@echo "  make lint       - Run code quality checks"

install:
	pip install -r requirements.txt

train:
	python train.py

evaluate:
	python evaluate.py

inference:
	python inference.py --test-set

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf inference_results/
	rm -f *.log

format:
	black *.py utility/*.py

lint:
	flake8 *.py utility/*.py --max-line-length=120
