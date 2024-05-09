Audio and Music Processing Challenge Repo
=============================================
Environment set up
-------------
```
$ pip install -r requirements.txt
```

`detector.py`
-------------

Needs two arguments: a directory of `.wav` files to process and the file name of a `.json` file to write the predictions to. 

Options:

`--plot_lvl` takes an integer value
if > 0, then visualizes computed features and predictions for each file as they are processed.
if > 1, produces extra plots of odf and its peaks (aka onsets) selection process.

`--method` takes str value
Possible methods are 'central_avg_envelope'(default value) and 'spectral_diff'.

For details on other options run `./detector.py --help`.

`evaluate.py`
-------------

Takes two arguments: path to the ground truth and to the predictions. Each can be a directory of `.onsets.gt`, `.tempo.gt` and `.beats.gt` files or a `.json` file.

Suggested use
-------------
```
$ ./detector.py train/ output.json --method method_name
$ ./evaluate.py train/ output.json
Onsets F-score: x.xxx
Tempo p-score: x.xxx
Beats F-score: x.xxx
```
    