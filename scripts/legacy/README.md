# scripts/legacy/

Kept for reproducibility of the pre-audit results, **not for use**.

`evaluate_docker_LEGACY.py` and `evaluate_accuracy_LEGACY.py` are the scripts that
produced the Docker 94% and venvy 83% figures. Both evaluate on the last 100 lines
of the same JSONL file the model was trained on:

```python
examples = [json.loads(line) for line in f][-100:]   # same file as training
```

The training notebook took a random 90/10 split of that file, so approximately 90 of
those 100 evaluation rows were in the training set. These scripts measure training-set
recall.

They are retained so the contaminated numbers can be reproduced and shown side by side
with the corrected ones in `docs/EVAL_METHODOLOGY.md`. Replacement lives in `eval/`.

`smoke_venvy_LEGACY.py` and `quick_fix_test_function_LEGACY.py` are print-based demo
scripts, never tests despite the old `test/` location.
