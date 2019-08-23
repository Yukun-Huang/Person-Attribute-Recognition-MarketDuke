# Person-Attribute-Recognition-MarketDuke
A baseline model ( pytorch implementation ) for person attribute recognition task, training and testing on Market1501-attribute and DukeMTMC-reID-attribute dataset.


## Dataset
You can get [Market1501-attribute](https://github.com/vana77/Market-1501_Attribute) and [DukeMTMC-reID-attribute](https://github.com/vana77/DukeMTMC-attribute) annotations from [here](https://github.com/vana77). Also you need to download Market-1501 and DukeMTMC-reID dataset.  
Then, create a folder named 'attribute' under your dataset path, and put corresponding annotations into the folder.

For example,<br>
&ensp;&ensp;  ~/dataset/DukeMTMC-reID/  
&ensp;&ensp;  ~/dataset/DukeMTMC-reID/bounding_box_test/  
&ensp;&ensp;  ~/dataset/DukeMTMC-reID/bounding_box_train/  
&ensp;&ensp;  ~/dataset/DukeMTMC-reID/query/  
&ensp;&ensp;  ~/dataset/DukeMTMC-reID/attribute/  
&ensp;&ensp;  ~/dataset/DukeMTMC-reID/attribute/duke_attribute.mat  


## Model
Trained model are provided. You may download it from [Google Drive](https://drive.google.com/drive/folders/1JTdjuEbxSLypnfUzVuuxLj1uSKAacfd0?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1bByCxZp9bSs8YYZPbuK21A) (提取码：jpks). You may download it and move `checkpoints` folder to your project's root directory.


## Usage
python3  train.py  --data-path  ~/dataset  --dataset  [market | duke]  --model  resnet50  
python3  test.py  --data-path  ~/dataset  --dataset  [market | duke]  --model  resnet50  


## Result (binary classification)
Market-1501 gallery:  
average accuracy: **0.9024**  

DukeMTMC-reID gallery:  
average accuracy: **0.8800**  


## Update
19-08-23: Released trained models.

19-01-09: Fixed the error caused by an update of market and duke attribute dataset.
