# Requirements
* tensorflow 1.10
* python 2.7 (not yet tested with python 3)

# Running the example
```bash
$ python run_hmouse.py
```

# Test Dataset
This example uses a small set of example images in ./hmouse/training_set/ to train the system. This example is to show 
 how to run mintnet, but will require a larger training set to produce meaningful results. To train on a larger dataset
 1. Either add images to ./hmouse/training_set, or alternatively you can use a different training set by modifying "image_dir" setting in ./hmouse/config_hmouse.json.
 2. Update the "K" parameter in ./hmouse/node_config/s4.json to reflect the number of images in your new training set.