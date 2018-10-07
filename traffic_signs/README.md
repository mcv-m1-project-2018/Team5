# Traffic Sign Detection

Tasks 1,2 and the thresholds selection done in the jupyter notebook `traffic_signs/task1.ipynb`

Tasks 3 and 4 done in `traffic_signs/main.py`
 
## Global
 
Use the `traffic_signs/main.py` file to obtain the traffic sign masks of a set of images.

After running the script, result masks will be saved into a directory and performance metrics will be 
printed on the screen.

## 1. Setup

To run the file, you need to install the dependencies listed in the `requirements.txt` file:


```
$ pip install -r requirements.txt
```

Or you could create a virtual environment and install them on it:

```
$ mkvirtualenv -p python3 venv-traffic_signs
(venv-traffic_signs) $ pip install -r requirements.txt
```

## 2. Run the script

To run the script, you will need to set some variables in the `main.py` file:

1. `TEST_DIR`: Where the target images are located
2. `RESULT_DIR`: Where the masks obtained from the target images will be saved
3. `ths_h`,  `ths_s`,  `ths_v`: Numpy arrays with the thresholds for each HSV channel

After setting those variables, run the script:

```
$ python traffic_signs/main.py
```

## 3. Results

In the `RESULT_DIR` directory the masks of each target image will be saved. Each mask is a png file with `1`'s value in 
every possible traffic sign pixel and `0`'s on the others.

As output you will get a confusion matrix with values corresponding to:

```
TP | FP
FN | TN
```

Also, precision, accuracy, specificity and sensitivity values of the used method will be displayed.
