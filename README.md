# 论文：STAFT-Weakly-Supervised-Few-Shot-Temporal-Action-Localization的代码

### 下载对应的数据集，也可以使用自己的数据集
### 下载链接：
通过网盘分享的文件：data
链接: https://pan.baidu.com/s/13pVTSpJ-Kkzz-FVOiDqR9A 提取码: d58y

### 将下载好的数据以下面的格式存放
```
1. 先在项目中创建一个data文件夹，以存放数据；
![image](https://github.com/user-attachments/assets/6f97786d-3ec6-4087-81c6-2e32ab0e120a)

2. 将下载好的数据集以下面的格式存放
![image](https://github.com/user-attachments/assets/ea738970-a3c9-4218-988b-77ff20ffff48)
```

### 接下来就可以训练了
# Traing
### 对于Thumos14数据集 : 1-way 5-shot, fault tolerate value 0
运行main.py
参数为：
--split=cvpr18 --encoder --num_in=4 --tsm=ip --sample_num_per_class=1 --batch_num_per_class=5 --Tolerate_error=0

### For Thumos14: 5-way 5-shot, fault tolerate value 0
运行main.py
参数为：
--split=cvpr18 --encoder --num_in=4 --tsm=ip --sample_num_per_class=5 --batch_num_per_class=5 --Tolerate_error=0

### For ActivityNet1.2: 5-way 1-shot, fault tolerate value 2
运行main.py
参数为：
--split=cvpr18 --dataset=ActivityNet1.2 --num_in=4 --encoder --tsm=ip --sample_num_per_class=1 --Tolerate_error=2

### For ActivityNet1.2: 5-way 5-shot, fault tolerate value 2
运行main.py
参数为：
--split=cvpr18 --dataset=ActivityNet1.2 --num_in=4 --encoder --tsm=ip --sample_num_per_class=5 --Tolerate_error=2

# Test
与训练类似，都是运行main.py 程序，参数变一下就行，dataset代码数据集，默认是Thumos14reduced，所以上面训练Thumos14时没指定，另外对于测试，还需要下载模型参数，让在--load参数中指定模型参数的路径；
由于我之前训练的模型已经找不到，所以这里先不提供了，有兴趣的话可以自己先跑一遍训练，这个模型跑训练很快，我使用GTX1070跑完Thumos14只需要一个小时时间；
跑完的模型参数会保存在./models文件夹中
--split=cvpr18 --dataset=Thumos14reduced --num_in=4 --encoder --tsm=ip --sample_num_per_class=1 --mode=testing --load=./models/Thumos14reduced_5way_1shot_split_cvpr18_enc_True_tsm_ip_lcent_0_indim_4_test/best.pth --Tolerate_error=0
--split=cvpr18 --dataset=Thumos14reduced --num_in=4 --encoder --tsm=ip --sample_num_per_class=5 --mode=testing --load=./models/Thumos14reduced_5way_5shot_split_cvpr18_enc_True_tsm_ip_lcent_0_indim_4_test/best.pth --Tolerate_error=0

--split=cvpr18 --dataset=ActivityNet1.2 --num_in=4 --encoder --tsm=ip --sample_num_per_class=1 --mode=testing --load=./models/ActivityNet1.2_5way_1shot_split_cvpr18_enc_True_tsm_ip_lcent_0_indim_4_test/best.pth --Tolerate_error=2
--split=cvpr18 --dataset=ActivityNet1.2 --num_in=4 --encoder --tsm=ip --sample_num_per_class=5 --mode=testing --load=./models/ActivityNet1.2_5way_5shot_split_cvpr18_enc_True_tsm_ip_lcent_0_indim_4_test/best.pth --Tolerate_error=2


# Ablation experiment
这里提供一些消融实现的测试方法，也是运行mian.py，只需要修改参数即可
### For different fault tolerate value
--split=cvpr18 --dataset=ActivityNet1.2 --num_in=4 --encoder --tsm=ip --sample_num_per_class=1 --mode=testing --load=./models/ActivityNet1.2_5way_1shot_split_cvpr18_enc_True_tsm_ip_lcent_0_indim_4_test/best.pth --Tolerate_error=1

### For different split strategies
### Put the category names of the different categories into the split_dataset function of the VideoDataset category in the dataset.py file, then set the names of the divisions, and change the value of split= to that name in the test
--split=Sequence70/30 --dataset=ActivityNet1.2 --num_in=4 --encoder --tsm=ip --sample_num_per_class=1 --Tolerate_error=2
--split=Sequence60/40 --dataset=ActivityNet1.2 --num_in=4 --encoder --tsm=ip --sample_num_per_class=1 --Tolerate_error=2
--split=Sequence50/50 --dataset=ActivityNet1.2 --num_in=4 --encoder --tsm=ip --sample_num_per_class=1 --Tolerate_error=2
--split=random --dataset=ActivityNet1.2 --num_in=4 --encoder --tsm=ip --sample_num_per_class=1 --Tolerate_error=2

--split='Sequence70/30' --dataset='ActivityNet1.2' --num_in=4 --encoder --tsm='ip' --sample_num_per_class=1 --Tolerate_error=2
--split='Sequence60/40' --dataset='ActivityNet1.2' --num_in=4 --encoder --tsm='ip' --sample_num_per_class=1 --Tolerate_error=2
--split='Sequence50/50' --dataset='ActivityNet1.2' --num_in=4 --encoder --tsm='ip' --sample_num_per_class=1 --Tolerate_error=2
