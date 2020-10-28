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

## Dependencies
* Python 3.5
* PyTorch >= 0.4.1
* torchvision >= 0.2.1
* matplotlib, sklearn, prettytable (optional)

## Usage
```
python3  train.py  --data-path  ~/dataset  --dataset  [market | duke]  --model  resnet50  [--use-id]

python3  test.py   --data-path  ~/dataset  --dataset  [market | duke]  --model  resnet50  [--print-table]

python3  inference.py   test_sample/test_market.jpg  [--dataset  market]  [--model  resnet50]
```

## Result

We use **binary classification** settings (considered each attribute as an independent binary classification problem), and the classification threshold is **0.5**.

***Note that the precision, recall and f1 score are denoted as '-' for some ill-defined cases.***

### Market-1501 gallery
```
+------------+----------+-----------+--------+----------+
| attribute  | accuracy | precision | recall | f1 score |
+------------+----------+-----------+--------+----------+
|   young    |  0.998   |   0.533   | 0.267  |  0.356   |
|  teenager  |  0.892   |   0.927   | 0.951  |  0.939   |
|   adult    |  0.895   |   0.582   | 0.450  |  0.508   |
|    old     |  0.992   |   0.037   | 0.012  |  0.019   |
|  backpack  |  0.883   |   0.828   | 0.672  |  0.742   |
|    bag     |  0.790   |   0.608   | 0.378  |  0.467   |
|  handbag   |  0.893   |   0.254   | 0.065  |  0.104   |
|  clothes   |  0.946   |   0.956   | 0.984  |  0.970   |
|    down    |  0.945   |   0.968   | 0.949  |  0.959   |
|     up     |  0.936   |   0.938   | 0.998  |  0.967   |
|    hair    |  0.877   |   0.871   | 0.773  |  0.819   |
|    hat     |  0.982   |   0.812   | 0.505  |  0.623   |
|   gender   |  0.919   |   0.947   | 0.864  |  0.903   |
|  upblack   |  0.954   |   0.859   | 0.790  |  0.823   |
|  upwhite   |  0.926   |   0.846   | 0.882  |  0.863   |
|   upred    |  0.974   |   0.904   | 0.840  |  0.871   |
|  uppurple  |  0.985   |   0.703   | 0.815  |  0.755   |
|  upyellow  |  0.976   |   0.895   | 0.836  |  0.865   |
|   upgray   |  0.909   |   0.852   | 0.391  |  0.537   |
|   upblue   |  0.946   |   0.868   | 0.420  |  0.566   |
|  upgreen   |  0.966   |   0.790   | 0.713  |  0.750   |
| downblack  |  0.879   |   0.815   | 0.889  |  0.850   |
| downwhite  |  0.956   |   0.608   | 0.550  |  0.578   |
|  downpink  |  0.989   |   0.795   | 0.782  |  0.788   |
| downpurple |  1.000   |     -     |   -    |    -     |
| downyellow |  0.999   |   0.000   | 0.000  |  0.000   |
|  downgray  |  0.878   |   0.756   | 0.443  |  0.559   |
|  downblue  |  0.861   |   0.762   | 0.446  |  0.563   |
| downgreen  |  0.978   |   0.766   | 0.295  |  0.426   |
| downbrown  |  0.958   |   0.754   | 0.590  |  0.662   |
+------------+----------+-----------+--------+----------+
Average accuracy: 0.9361
Average f1 score: 0.6492
```

### DukeMTMC-ReID gallery
```
+-----------+----------+-----------+--------+----------+
| attribute | accuracy | precision | recall | f1 score |
+-----------+----------+-----------+--------+----------+
|  backpack |  0.829   |   0.794   | 0.926  |  0.855   |
|    bag    |  0.836   |   0.496   | 0.287  |  0.364   |
|  handbag  |  0.935   |   0.469   | 0.073  |  0.126   |
|   boots   |  0.905   |   0.784   | 0.791  |  0.787   |
|   gender  |  0.858   |   0.806   | 0.828  |  0.817   |
|    hat    |  0.898   |   0.883   | 0.680  |  0.768   |
|   shoes   |  0.916   |   0.756   | 0.414  |  0.535   |
|    top    |  0.893   |   0.590   | 0.381  |  0.463   |
|  upblack  |  0.821   |   0.827   | 0.903  |  0.864   |
|  upwhite  |  0.959   |   0.750   | 0.509  |  0.606   |
|   upred   |  0.973   |   0.745   | 0.649  |  0.694   |
|  uppurple |  0.995   |   0.258   | 0.123  |  0.167   |
|   upgray  |  0.900   |   0.611   | 0.333  |  0.432   |
|   upblue  |  0.943   |   0.766   | 0.519  |  0.619   |
|  upgreen  |  0.975   |   0.463   | 0.403  |  0.431   |
|  upbrown  |  0.980   |   0.481   | 0.328  |  0.390   |
| downblack |  0.787   |   0.740   | 0.807  |  0.772   |
| downwhite |  0.945   |   0.771   | 0.395  |  0.522   |
|  downred  |  0.991   |   0.739   | 0.645  |  0.689   |
|  downgray |  0.927   |   0.471   | 0.238  |  0.317   |
|  downblue |  0.807   |   0.741   | 0.669  |  0.703   |
| downgreen |  0.997   |     -     |   -    |    -     |
| downbrown |  0.979   |   0.871   | 0.594  |  0.706   |
+-----------+----------+-----------+--------+----------+
Average accuracy: 0.9152
Average f1 score: 0.5739
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
carrying bag: yes
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
*20-06-03: Added **identity loss** for joint optimization; Adjusted the learning rate for better performace.*

*20-06-03: Updated **test.py**, settled the issue of ill-defined metrics.*

*19-09-16: Updated **inference.py**, fixed the error caused by missing data-transform.*

*19-09-06: Updated **test.py**, added **F1 score** for evaluating.*

*19-09-03: Added **inference.py**, thanks @ViswanathaReddyGajjala.*

*19-08-23: Released trained models.*

*19-01-09: Fixed the error caused by an update of market and duke attribute dataset.*

## Reference

*[1] Lin Y, Zheng L, Zheng Z, et al. Improving person re-identification by attribute and identity learning[J]. Pattern Recognition, 2019.*
