Audio and Music Processing Challenge Repo (based on template)
=============================================
Environment set up
-------------
```
$ pip install -r requirements.txt
```

`detector.py`
-------------

Takes two arguments: A directory of `.wav` files to process, and the file name of a `.json` file to write the predictions to. Optionally takes the `--plot` argument to visualize computed features and predictions for each file as they are processed.

Requires `numpy`, `scipy`, `librosa`, optionally `tqdm` for a progress bar, and optionally `matplotlib` for the visualization.

`detect_everything()` computes a spectrogram and mel spectrogram, and then calls other functions to derive an onset detection function, detect onsets, estimate tempo, and detect beats. 

All functions have access to the command line parameters, so you can add parameters that you would like to alter from the command line or allow selecting different algorithms.

`evaluate.py`
-------------

Takes two arguments: The location of the ground truth and the location of the predictions. The ground truth can be a directory of `.onsets.gt`, `.tempo.gt` and `.beats.gt` files or a `.json` file. The predictions can be a directory of `.onsets.pr`, `.tempo.pr` and `.beats.pr` files or a `.json` file.

Requires `numpy` and `mir_eval` to run.

Use it to evaluate your predictions on training sets. It should gracefully handle cases where not all three tasks are included in the ground truth or predictions. Beware that scores are always averaged over the number of ground truth files, no matter whether there is a corresponding prediction.

Suggested use
-------------

The idea would be for you to predict and evaluate on the training set, changing the implementation and tuning parameters as you go (think about setting aside a part of the training set for validation, especially if you are using machine learning approaches). When happy, run the prediction script over the test set and submit the resulting `.json` file to the challenge server.

For reference, running `detector.py` over the full training set (extracted to `train/`) and evaluating the results should look like this:
```
$ ./detector.py train/ output.json
$ ./evaluate.py train/ output.json
Onsets F-score: x.xxx
Tempo p-score: x.xxx
Beats F-score: x.xxx
```
