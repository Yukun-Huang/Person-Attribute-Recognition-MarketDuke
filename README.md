# Person-Attribute-Recognition-MarketDuke
A simple baseline implemented in PyTorch for **pedestrian attribute recognition** task, evaluating on Market-1501-attribute and DukeMTMC-reID-attribute dataset.


## Dataset
You can get [Market-1501-attribute](https://github.com/vana77/Market-1501_Attribute) and [DukeMTMC-reID-attribute](https://github.com/vana77/DukeMTMC-attribute) annotations from [here](https://github.com/vana77). Also you need to download Market-1501 and DukeMTMC-reID dataset.

Then, create a folder named 'attribute' under your dataset path, and put corresponding annotations into the folder.

For example,<br>
```
├── dataset
│   ├── DukeMTMC-reID
│       ├── bounding_box_test
│       ├── bounding_box_train
│       ├── query
│       ├── attribute
│           ├── duke_attribute.mat  
```

## Model
Trained model are provided. You may download it from [Google Drive](https://drive.google.com/drive/folders/1JTdjuEbxSLypnfUzVuuxLj1uSKAacfd0?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1bByCxZp9bSs8YYZPbuK21A) (提取码：jpks).

You may download it and move `checkpoints` folder to your project's root directory.


## Usage
```
python3  train.py  --data-path  ~/dataset  --dataset  [market | duke]  --model  resnet50

python3  test.py   --data-path  ~/dataset  --dataset  [market | duke]  --model  resnet50  [--print-table]

python3  inference.py   test_sample/test_market.jpg  [--dataset  market]  [--model  resnet50]
```

## Result

We use **binary classification** settings (considered each attribute as an independent binary classification problem), and the classification threshold is **0.5**.

***Note that some attributes may not have a positive (or negative) sample, so F1 score of these attributes will be zero.***

### Market-1501 gallery
```
+------------+----------+-----------+--------+----------+
| attribute  | accuracy | precision | recall | f1 score |
+------------+----------+-----------+--------+----------+
|   young    |  0.998   |   0.000   | 0.000  |  0.000   |
|  teenager  |  0.855   |   0.907   | 0.930  |  0.918   |
|   adult    |  0.880   |   0.502   | 0.240  |  0.325   |
|    old     |  0.945   |   0.000   | 0.000  |  0.000   |
|  backpack  |  0.760   |   0.550   | 0.229  |  0.323   |
|    bag     |  0.739   |   0.321   | 0.066  |  0.110   |
|  handbag   |  0.902   |   0.147   | 0.008  |  0.015   |
|  clothes   |  0.867   |   0.916   | 0.935  |  0.925   |
|    down    |  0.865   |   0.917   | 0.879  |  0.897   |
|     up     |  0.935   |   0.935   | 1.000  |  0.966   |
|    hair    |  0.791   |   0.777   | 0.592  |  0.672   |
|    hat     |  0.971   |   1.000   | 0.003  |  0.005   |
|   gender   |  0.780   |   0.764   | 0.725  |  0.744   |
|  upblack   |  0.904   |   0.663   | 0.574  |  0.615   |
|  upwhite   |  0.869   |   0.732   | 0.797  |  0.764   |
|   upred    |  0.941   |   0.704   | 0.742  |  0.723   |
|  uppurple  |  0.974   |   0.557   | 0.353  |  0.432   |
|  upyellow  |  0.968   |   0.904   | 0.723  |  0.803   |
|   upgray   |  0.879   |   0.621   | 0.242  |  0.349   |
|   upblue   |  0.924   |   0.705   | 0.152  |  0.250   |
|  upgreen   |  0.952   |   0.726   | 0.545  |  0.623   |
| downblack  |  0.834   |   0.756   | 0.841  |  0.797   |
| downwhite  |  0.943   |   0.474   | 0.484  |  0.479   |
|  downpink  |  0.981   |   0.682   | 0.552  |  0.610   |
| downpurple |  0.992   |   0.000   | 0.000  |  0.000   |
| downyellow |  0.995   |   0.000   | 0.000  |  0.000   |
|  downgray  |  0.848   |   0.689   | 0.238  |  0.354   |
|  downblue  |  0.822   |   0.685   | 0.206  |  0.316   |
| downgreen  |  0.973   |   0.600   | 0.042  |  0.079   |
| downbrown  |  0.930   |   0.490   | 0.309  |  0.379   |
+------------+----------+-----------+--------+----------+
Average accuracy: 0.9006
Average f1 score: 0.4491
```

### DukeMTMC-reID gallery
```
+-----------+----------+-----------+--------+----------+
| attribute | accuracy | precision | recall | f1 score |
+-----------+----------+-----------+--------+----------+
|  backpack |  0.677   |   0.660   | 0.842  |  0.740   |
|    bag    |  0.832   |   0.292   | 0.019  |  0.035   |
|  handbag  |  0.898   |   0.043   | 0.028  |  0.034   |
|   boots   |  0.785   |   0.521   | 0.429  |  0.471   |
|   gender  |  0.694   |   0.599   | 0.609  |  0.604   |
|    hat    |  0.794   |   0.755   | 0.248  |  0.373   |
|   shoes   |  0.886   |   0.531   | 0.128  |  0.206   |
|    top    |  0.876   |   0.466   | 0.108  |  0.175   |
|  upblack  |  0.766   |   0.786   | 0.861  |  0.821   |
|  upwhite  |  0.941   |   0.543   | 0.279  |  0.368   |
|   upred   |  0.961   |   0.598   | 0.459  |  0.520   |
|  uppurple |  0.996   |   0.000   | 0.000  |  0.000   |
|   upgray  |  0.852   |   0.270   | 0.173  |  0.211   |
|   upblue  |  0.920   |   0.610   | 0.291  |  0.394   |
|  upgreen  |  0.977   |   0.631   | 0.098  |  0.170   |
|  upbrown  |  0.980   |   0.333   | 0.003  |  0.006   |
| downblack |  0.749   |   0.693   | 0.785  |  0.736   |
| downwhite |  0.925   |   0.533   | 0.145  |  0.228   |
|  downred  |  0.984   |   0.454   | 0.234  |  0.309   |
|  downgray |  0.926   |   0.338   | 0.039  |  0.070   |
|  downblue |  0.763   |   0.752   | 0.461  |  0.572   |
| downgreen |  0.997   |   0.000   | 0.000  |  0.000   |
| downbrown |  0.963   |   0.652   | 0.263  |  0.375   |
+-----------+----------+-----------+--------+----------+
Average accuracy: 0.8758
Average f1 score: 0.3226
```

### Inference
```
>> python inference.py test_sample/test_market.jpg --dataset market
age: teenager
carrying backpack: no
carrying bag: no
carrying handbag: no
type of lower-body clothing: dress
length of lower-body clothing: short
sleeve length: short sleeve
hair length: long hair
wearing hat: no
gender: female
color of upper-body clothing: white
color of lower-body clothing: white

>> python inference.py test_sample/test_duke.jpg --dataset duke
carrying backpack: no
carrying bag: no
carrying handbag: no
wearing boots: no
gender: male
wearing hat: no
color of shoes: dark
length of upper-body clothing: short upper body clothing
color of upper-body clothing: black
color of lower-body clothing: blue
```

## Update
*19-09-16: Updated **inference.py**, fixed the error caused by missing data-transform.*

*19-09-06: Updated **test.py**, added **F1 score** for evaluating.*

*19-09-03: Added **inference.py**, thanks @ViswanathaReddyGajjala.*

*19-08-23: Released trained models.*

*19-01-09: Fixed the error caused by an update of market and duke attribute dataset.*

## Reference

*[1] Lin Y, Zheng L, Zheng Z, et al. Improving person re-identification by attribute and identity learning[J]. Pattern Recognition, 2019.*
