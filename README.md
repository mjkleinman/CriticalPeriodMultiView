# Critical periods for multisensory training.

Similar to humans and animals, we show that in deep networks the ability to integrate and extract common information
from multiple sensors heavily relies on being exposed to properly correlated sensory data during an initial period of
development known as a *critical learning period*. In particular, we show that supervised multi-sensor networks are
particularly sensitive to corruptions in the data distribution as well as mismatch in the learning dynamics at the
beginning of training. In these situations, while they still learn to extract unique information from each modality,
their ability to extract and process synergistic information is permanently impaired as well as their performance on
downstream tasks, mirroring the observations in animals. To study the emergence of sensor integration, we introduce a
new measure of sensor integration, called Fractional Sensor Variance which allows us to track the internal
representations during training and as a result of the deficit.

## Requirements

- Python 3.8+
- PyTorch

You can install all required Python packages with `pip install -r requirements.txt`
In the main directory, run `bash init_repo.sh` to create the necessary folders to save runs and experiments.

## Training a network with deficit

Run the following to train the Split-ResNet-18 model for 180 epochs, with a .97 per epoch exponential decay of the
learning rate and a downsampling (blurring) deficit from the beginning to epoch 80. Add the flag `save-final` to save
the final model.

```sh
./main.py --lr=0.075 --slow --augment --weight-decay=0.0005 --arch=sresnet --is-rand-zero-input --view-size=16 --deficit=downsample --schedule=181 --deficit-start=-1 --deficit-end=80 --save-final
```

Similarly, to train with a independent half of an image deficit, using an architecture that has multiple
heads (`sresnetmulti`), run:

```sh
./main_independent.py --lr=0.075 --slow --augment --weight-decay=0.0005 --arch=sresnetmulti --view-size=16 --diff-aug --schedule=181 --deficit-start=-1 --deficit-end=80 --save-final
```

To see all available options and their description, run `./main.py -h`.

Each command will create a pickle file in the `logs` directory, which can then be loaded by the various plotting
utilities included to show the results.

## Running an experiment

Most experiments in the paper require training multiple networks. This is done through `run_experiment.py`: Open the
file and comment/uncomment the part relative to the desired experiment, then execute the file. This will automatically
train several networks in parallel, one per available gpu, and collect the results in a json file in the `experiments`
directory. The file can then be loaded by the various plotting utilities to display the results of the experiment.

## Evaluating the Fractional Sensor Variance of a Trained Model:

Add the flag `--dominance` to `./main.py`, and specify the architecture, view size, and model checkpoint. For example,
run:

```bash
./main.py --dominance --arch {arch} --view-size {view_size} --resume {resume_model}
```

You can also generate plots for an entire experiment by passing the experiment to `scripts/plot_dominance_experiment.py`
.

## Generate the plots in the paper

We have included the log files for all the experiments shown in the paper. To generate the plot, run `./generate_plots`.
This will output several pdf files in the `plots` directory.

The log files contain additional information that could be of interest. To see which of the included log files
correspond to a particular experiment, look for the corresponding code in generate_plots. The file can then be loaded
with pickle or json (see also the code of the plotting utilities to see how to load an experiment and all related files)
.

## Licenses

The code of this repository is released under the [Apache 2.0 license](LICENSE).
