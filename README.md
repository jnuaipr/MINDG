<h3 align="center">
<p> MINDG: A Drug-Target Interaction Prediction Method Based on an Integrated Learning Algorithm </h3>

This repository contains script which were used to build and train the MINDG model together with the scripts for evaluating the model's performance.
### Dependency

```
python                  3.6.8
torch                   1.5.0
numpy                   1.19.1
scipy                   1.5.0
torch_sparse            0.6.4
pandas                  1.1.5
scikit-learn            0.22.1
matplotlib              3.2.2
requests                2.27.1
pandas-flavor           0.6.0
subword-nmt             0.3.8
prettytable             0.7.2
texttable               1.7.0
tqdm                    4.65.0
pytdc                   0.4.1
loguru                  0.7.2
```

### Dataset

> The dataset used in the experiments are provided as follows:
>
> **BindingDB** dataset is hosted in https://www.bindingdb.org/bind/index.jsp.
>
> **Davis** Dataset can be found in http://staff.cs.utu.fi/~aatapa/data/DrugTarget/.

The BingDB and DAVIS datasets are automatically downloaded from [TDC](https://tdcommons.ai/multi_pred_tasks/dti/) by Main.py


### Train and Test
#### Train
Navigate to the project source directory
```shell
cd src
```
Config the training parameters by adjusting the input parameters of `run` function
```python
run('DAVIS', phase="train",batch_size=32,epochs=5,learning_rate=5e-4,lr_step_size=10,seed_id=10,device=torch.device('cpu'))
```
Description of `run` function parameters
```
name            dataset name, "BingdingDB_kd" or "DAVIS"
phase           work phase, "train"/"test"
batch_size      batch size of data ,default 32
epochs          number of train epoch, default 5
learning_rate   learning rate, default 5e-4
lr_step_size    Scheduling step size for learning rate, default 10
seed_id         random seed id, default 10
```
start training
```python
python Main.py
```
The directory to save Model parameters is `output/model`.
Model file name is `model name` + `dataset name` + `epoch number`.pt
```shell
(mindg) yang@yang:~/sda/github/MINDG/output/model$ ll
total 119032
drwxrwxr-x 2 yang yang     4096 Jan  9 21:16 ./
drwxrwxr-x 6 yang yang     4096 Dec 29 19:58 ../
-rw-rw-r-- 1 yang yang 16303055 Jan  3 10:33 mindg_BindingDB_Kd_epoch10.pt
-rw-rw-r-- 1 yang yang 18788185 Jan  4 08:38 mindg_BindingDB_Kd_epoch20.pt
-rw-rw-r-- 1 yang yang 18788303 Jan  9 15:34 mindg_BindingDB_Kd_epoch5.pt
-rw-rw-r-- 1 yang yang 13408711 Dec 26 06:09 mindg_DAVIS_epoch100.pt
-rw-rw-r-- 1 yang yang 13408593 Dec 30 17:13 mindg_DAVIS_epoch20.pt
-rw-rw-r-- 1 yang yang 13408711 Jan  9 11:36 mindg_DAVIS_epoch5.pt
-rw-rw-r-- 1 yang yang 13408593 Dec 28 18:52 mindg_DAVIS_epoch90.pt
-rw-rw-r-- 1 yang yang 14355801 Jan  3 22:20 mindg_KIBA_epoch20.pt
```
#### Test
when phase is "test", The routine will load the Model file stored in the output directory.
```python
run('DAVIS', phase="test",batch_size=32,epochs=5,learning_rate=5e-4,lr_step_size=10,seed_id=10,device=torch.device('cpu'))
```
start testing
```shell
python Main.py
```
If you would like to obtain pretrained model, please email [yang hailong](mailto:yanghailong@stu.jiangnan.edu.cn).

#### Acknowledgement

The code is based on [HOGCN](https://github.com/kckishan/HOGCN-LP) and [DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose).
