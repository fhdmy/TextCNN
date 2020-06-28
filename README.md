# TextCNN
基于TextCNN的新闻分类任务。

## 数据集说明：  
实验数据集为tnews_public，包含4个json文件，其中labels.json包含15行记录代表新闻的种类，train.json、dev.json、test.json分别包含53360、5000、5000条记录，用于模型的训练和测试，它们的记录组织形式类似，每行代表一条新闻的信息。
labels.json
| 列名称        | 说明    |
| :--------   | :-----   |
| label        | 整数类型，序列化后的新闻标签ID      |
| label_desc        | 字符串类型，新闻类别      |

train.json、dev.json和test.json
| 列名称        | 说明    |
| :--------   | :-----   |
| label        | 整数类型，序列化后的新闻标签ID      |
| label_desc        | 字符串类型，新闻类别      |
| sentence        | 字符串类型，新闻内容      |
| keywords        | 字符串类型，从新闻内容中提取出的关键词      |

## 训练：  
```
  python3 train.py --config cnn.ini
```
## 测试： 
``` 
  python3 test.py --config cnn.ini
```
## 分析： 
``` 
  python3 analyse.py --config cnn.ini
```
