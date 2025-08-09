# Usage

## Installation

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=./src
```

## Demo

Feed the bundled Quest 3 sample through the CLI:

```bash
python -m omniintent demo --ingest samples/quest3_tiny.csv --seq-len 60
```

### Expected output

When the model weights are missing, the CLI prints the input tensor shapes:

```
⚠️  `omniintent.model.MultiModalTransformer` unavailable — printing input tensor shapes only.
{"input_shapes":{"gaze":[1,10,3],"hand_pose":[1,10,1]}}
```

If weights are available, a forward pass runs and a compact summary of input and output shapes is printed.

The demo relies on CPU-only tensor ops and falls back gracefully when the model package or weights are absent.
