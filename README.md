# AI Suite
An automation environment for training, evaluating, and deploying PyTorch models. Designed to be flexible across all model architectures, assuming data-handling methods are provided.

# Usage
**Note:**
Directories are carefully defined and should be left alone, but may be extended for future use.

### Model
This tool comes with outlines for a simple model of a fully connected layer followed by a ReLU activation layer. This should be modified to match your desired model architecture.

### Data
In the `data` folder, `dataloader.py` and `dataset_handler.py` scripts should be modified to manage the real data you want to manipulate. They are currently setup for a simple supervised learning task.

### Data `/datasets`
The `/data/datasets` folder should contain the data to be used, but should be transformed into a `.parquet` file. This can easily be done using `pandas`. See their docs for how to convert `.csv`, `.xlsx`, etc. to `.parquet`. These have fast read/write speeds and handle the same amongst pandas `DataFrame` objects.

### Miscellaneous
The only file requiring additional manipulation to support your project is `env.py`. Make sure the model is handled properly according to the unique architecture of your project, and ensure that the tensor operations are valid given the unique shapes you'll be working with.

### To Launch the Script
The launch script has various flags used to detail how you want the training/evaluation cycles to be handled. Additionally, control over `Loss` readouts during training is available with the `--verbose` flag. `--epochs` denotes the number of samples the model is trained on from your dataloader in a single loop, and the loops denote the number of times testing/training occurs in a single `--cycle`. `--compile` allows you to optionally make use of the `torch.compile()` module, recently introduced in `torch --version 2.X`. It can speed up the model-training process even more and efficiency is especially gained when the model architecture is larger and more complicated.
```{bash}
. ./suite.sh [--version] [-c, --cycles INT] [--testloops INT] [--trainloops INT] [-e, --epochs INT] [-v, --verbose] [--compile]
```

## Configure `ai.sh` Script to Select Python Interpreter
When opening the file, you'll notice this section starting on line 11. Change the name from `deeplearn` to the name of you python interpreter alias.
```{bash}
...
# Configure this variable to the alias name of your python interpreter
ALIAS=deeplearn
...
```

## Install Dependencies for PyTorch Models and Automation
For specifics on installing PyTorch, check their official webpage for more information. To install the modules supported by this project, run the following in a bash terminal:
```{bash}
#!/bin/bash
pip install -r requirements.txt
```