# P2GNet

P2PNet is a neural network framework that uses a point regression approach to address the problem of crowd density estimation. It is capable of not only counting the number of people but also providing the location information of individual heads.

The operation has been simplified by removing the anchor and match operations, assigning the Ground Truth (GT) to an N x N grid, and predicting the number of people in each grid directly in the output layer, with the GT grid used to calculate the Mean Squared Error (MSE) loss.


### Installation
```
pip3 install -r requirements.txt
```

### Download dataset:
Download ShangHaiTech dataset

### Preprocess:
```
# First change the data_root to your dataset root
python3 mat2txt.py
```

### Training:
```
python3 train.py
```

### Test:
```
python3 run_test.py(Not completed)
```

### Result:
|               | 128x128 | 256x256 | 
|---------------|:-------:|:-------:|
| vgg16_bn      |   61    |  59.1   | 
