# Releaf

### Evaluate with ground truth (reproducibility)
```bash
releaf-eval --config configs/eval.yaml --mode repro

releaf-eval --config configs/eval.yaml --mode deploy

python -m releaf_tuning.lhs_search_procedure --config configs/default.yaml --samples 100
