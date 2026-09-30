.PHONY: install test run clean

install:
	pip install -r requirements.txt
	pip install piper-tts

test:
	pytest tests/ -v

run:
	python -m src.main

clean:
	rm -rf __pycache__ .pytest_cache logs/*.log
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
