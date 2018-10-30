# Content based image retrieval
 
## Week 1
 
The goal is to perform query by example image retrieval using the basic 
tools of course lectures.


## 1. Setup

To run the file, you need to install the dependencies listed in the 
`requirements-image_retrieval.txt` file:


```
$ pip install -r requirements-image_retrieval.txt
```

Or you could create a virtual environment and install them on it:

```
$ mkvirtualenv -p python3 venv-image_retrieval
(venv-image_retrieval) $ pip install -r requirements-image_retrieval.txt
```

## 2. Run the script

To run the script, you will need to set some variables in the `main.py` file:

1. `COLOR_SPACES`: Color spaces used to compute the image descriptors.
Values: `rgb`, `hsv`, `ycbcr`, `lab`
2. `IMAGE_DESCRIPTORS`: Types of image descriptors used. 
Values: `global`, `block`, `pyramid` 
3. `BLOCK_X`, `BLOCK_Y`: The sizes of the blocks for the block image descriptor
for each coordinate
3. `DISTANCE_FUNCTIONS`: Distances functions used for compute similarity. 
Values: `eucl`, `l1`, `chi`, `hist_inters`, `hell`
4. `K`: Number of images to retrieve for each query image

After setting those variables, run the script:

```
$ python image_retrieval/main.py
```

## 3. Results

In the `RESULT_DIR` directory a pickle file with for each combination will 
be created. The format of the pickle filename is as follows:

```
<color_space>_<image_descriptor>_<distance_function>_<k>_<map@k>.pkl
```

where `map@k` is the Mean Average Precision at K given by the formula: 

<a href="https://www.codecogs.com/eqnedit.php?latex=AP@K&space;=&space;\frac{\sum_{i_=1}^{K}P@i}{K}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?AP@K&space;=&space;\frac{\sum_{i_=1}^{K}P@i}{K}" title="AP@K = \frac{\sum_{i_=1}^{K}P@i}{K}" /></a>
