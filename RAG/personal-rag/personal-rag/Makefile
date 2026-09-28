.PHONY: install models run dev test reset clean docker

install:
	python -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt

models:
	python scripts/download_models.py

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

dev:
	uvicorn app.main:app --reload --port 8000

test:
	pytest -q

reset:
	python scripts/reset_db.py

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache

docker:
	docker compose up --build
