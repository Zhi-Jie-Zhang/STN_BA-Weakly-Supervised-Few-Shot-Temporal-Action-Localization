# STAFT-Weakly-Supervised-Few-Shot-Temporal-Action-Localization

### Download 

Please first create a data dir and then put all the features and annotations under it.
```
mkdir data
cd data
```

### Experiment
#Traing
For Thumos14: 1-way 5-shot, fault tolerate value 0
--split=cvpr18 --encoder --num_in=4 --tsm=ip --sample_num_per_class=1 --batch_num_per_class=5 --Tolerate_error=0
For Thumos14: 5-way 5-shot, fault tolerate value 0
--split=cvpr18 --encoder --num_in=4 --tsm=ip --sample_num_per_class=5 --batch_num_per_class=5 --Tolerate_error=0

For ActivityNet1.2: 5-way 1-shot, fault tolerate value 2
--split=cvpr18 --dataset=ActivityNet1.2 --num_in=4 --encoder --tsm=ip --sample_num_per_class=1 --Tolerate_error=2
For ActivityNet1.2: 5-way 5-shot, fault tolerate value 2
--split=cvpr18 --dataset=ActivityNet1.2 --num_in=4 --encoder --tsm=ip --sample_num_per_class=5 --Tolerate_error=2

#Test

--split=cvpr18 --dataset=Thumos14reduced --num_in=4 --encoder --tsm=ip --sample_num_per_class=1 --mode=testing --load=./models/Thumos14reduced_5way_1shot_split_cvpr18_enc_True_tsm_ip_lcent_0_indim_4_test/best.pth --Tolerate_error=0
--split=cvpr18 --dataset=Thumos14reduced --num_in=4 --encoder --tsm=ip --sample_num_per_class=5 --mode=testing --load=./models/Thumos14reduced_5way_5shot_split_cvpr18_enc_True_tsm_ip_lcent_0_indim_4_test/best.pth --Tolerate_error=0

--split=cvpr18 --dataset=ActivityNet1.2 --num_in=4 --encoder --tsm=ip --sample_num_per_class=1 --mode=testing --load=./models/ActivityNet1.2_5way_1shot_split_cvpr18_enc_True_tsm_ip_lcent_0_indim_4_test/best.pth --Tolerate_error=2
--split=cvpr18 --dataset=ActivityNet1.2 --num_in=4 --encoder --tsm=ip --sample_num_per_class=5 --mode=testing --load=./models/ActivityNet1.2_5way_5shot_split_cvpr18_enc_True_tsm_ip_lcent_0_indim_4_test/best.pth --Tolerate_error=2


#Ablation experiment

For different fault tolerate value
--split=cvpr18 --dataset=ActivityNet1.2 --num_in=4 --encoder --tsm=ip --sample_num_per_class=1 --mode=testing --load=./models/ActivityNet1.2_5way_1shot_split_cvpr18_enc_True_tsm_ip_lcent_0_indim_4_test/best.pth --Tolerate_error=1

For different split strategies
Put the category names of the different categories into the split_dataset function of the VideoDataset category in the dataset.py file, then set the names of the divisions, and change the value of split= to that name in the test
--split=Sequence70/30 --dataset=ActivityNet1.2 --num_in=4 --encoder --tsm=ip --sample_num_per_class=1 --Tolerate_error=2
--split=Sequence60/40 --dataset=ActivityNet1.2 --num_in=4 --encoder --tsm=ip --sample_num_per_class=1 --Tolerate_error=2
--split=Sequence50/50 --dataset=ActivityNet1.2 --num_in=4 --encoder --tsm=ip --sample_num_per_class=1 --Tolerate_error=2
--split=random --dataset=ActivityNet1.2 --num_in=4 --encoder --tsm=ip --sample_num_per_class=1 --Tolerate_error=2

python main.py --split='Sequence70/30' --dataset='ActivityNet1.2' --num_in=4 --encoder --tsm='ip' --sample_num_per_class=1 --Tolerate_error=2
python main.py --split='Sequence60/40' --dataset='ActivityNet1.2' --num_in=4 --encoder --tsm='ip' --sample_num_per_class=1 --Tolerate_error=2
python main.py --split='Sequence50/50' --dataset='ActivityNet1.2' --num_in=4 --encoder --tsm='ip' --sample_num_per_class=1 --Tolerate_error=2
