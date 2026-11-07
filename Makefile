.PHONY: check lint types test

# The same commands as the workflow: a green `make check` and a green CI are the same claim.
check: lint types test

lint:
	ruff check .
	ruff format --check .

types:
	mypy src/dlkit

test:
	pytest --cov=dlkit --cov-report=term-missing --cov-fail-under=80
