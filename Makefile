.PHONY: verify train eval live benchmark test docker

verify:
	python -m pytest tests/ -q

test:
	python -m pytest tests/ -q -v

train:
	python scripts/train.py --config configs/tiny.yaml

eval:
	python scripts/evaluate.py --config configs/tiny.yaml

live:
	python scripts/live_verify.py --config configs/finance_live.yaml

benchmark:
	python scripts/benchmark_grid.py --config configs/tiny.yaml

docker:
	docker compose up --build w2v-cpu
