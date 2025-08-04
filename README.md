## Running the Framework

The configuration is controlled by parameters in `config.py`. Modify this file to change:
- Work directory (todo: pick "local" or "drive")
- Dataset directory (todo: specify the location)
- Other parameters

### Using the runner notebook

The simplest way to run the framework is using the provided runner notebook:

```bash
jupyter notebook runner.ipynb
```

This notebook provides a step-by-step interface for configuring and running the training process.

### Running from command line

Alternatively, you can run the training script directly:

```bash
python main.py
