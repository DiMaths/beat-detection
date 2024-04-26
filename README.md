Audio and Music Processing Challenge Template
=============================================

This provides two Python scripts that may serve as a template for your submission to the onset detection, beat detection and tempo estimation challenges. Usage is completely optional.

Environment set up
-------------
´´´pip install -r requirements.txt´´´

`detector.py`
-------------

Takes two arguments: A directory of `.wav` files to process, and the file name of a `.json` file to write the predictions to. Optionally takes the `--plot` argument to visualize computed features and predictions for each file as they are processed.

Requires `numpy`, `scipy`, `librosa`, optionally `tqdm` for a progress bar, and optionally `matplotlib` for the visualization.

You can read the file from top to bottom. It includes a function `detect_everything()` that computes a spectrogram and mel spectrogram, and then calls other functions to derive an onset detection function, detect onsets, estimate tempo, and detect beats. None of the latter functions do anything useful yet. It is your job to fill in some algorithms that work well. All functions have access to the command line parameters, so you can add parameters that you would like to alter from the command line or allow selecting different algorithms. Of course, feel free to change all parameters involved in the existing features, restructure the code, delete all the functions and write your `detect_everything()` function from scratch, or ignore this template altogether and write your own script or notebook.

`evaluate.py`
-------------

Takes two arguments: The location of the ground truth and the location of the predictions. The ground truth can be a directory of `.onsets.gt`, `.tempo.gt` and `.beats.gt` files or a `.json` file. The predictions can be a directory of `.onsets.pr`, `.tempo.pr` and `.beats.pr` files or a `.json` file.

Requires `numpy` and `mir_eval` to run.

You can use it to evaluate your predictions on the main training set (or a part of the training set that you set aside for validation) or the extra training sets. It should gracefully handle cases where not all three tasks are included in the ground truth or predictions. Beware that scores are always averaged over the number of ground truth files, no matter whether there is a corresponding prediction.

Suggested use
-------------

The idea would be for you to predict and evaluate on the training set, changing the implementation and tuning parameters as you go (think about setting aside a part of the training set for validation, especially if you are using machine learning approaches). When happy, run the prediction script over the test set and submit the resulting `.json` file to the challenge server.

For reference, running the dummy implementations of `detector.py` over the full training set (extracted to `train/`) and evaluating the results should look like this:
```
$ ./detector.py train/ train.json
$ ./evaluate.py train/ train.json
Onsets F-score: 0.3688
Tempo p-score: 0.0184
Beats F-score: 0.1367
```
This clearly leaves some room for improvement!
