# equity-alpha

Cross-Sectional Equity Alpha & Portfolio Construction

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env   # then set SHARADAR_API_KEY
pytest
```

Run notebooks and scripts from the repository root, or add it to `sys.path`, so that
`from src.data import get_universe_prices` resolves. See [`data/README.md`](data/README.md)
for the data layer.
