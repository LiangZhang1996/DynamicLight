# 1. Introduction

Official code for article <DynamicLight: Dynamically Tuning Traffic Signal Duration with DRL>.



# 2. Requirements

`python3.6`,`tensorflow=2.4`, `cityflow`, `pandas`, `numpy`

[`cityflow`](https://github.com/cityflow-project/CityFlow.git) needs a linux environment, and we run the code on Manjaro Linux.

# 3. Quick start

## 3.1 Base experiemnts

- For `DynamicLight`, run:
```shell
python run_dynamiclight.py
```
- For tranfer, configure and run `run_test.py`

## 3.2 DynamicLight variants
- For `DynamicLight-Rand`, run:
```shell
python run_dynamic_rand.py
```
- For `DynamicLight-FT`, run:
```shell
python run_dynamic_ft.py
```
For `DynamicLight-MP`, run:
```shell
python run_dynamic_mp.py
```
For `DynamicLight-MQL`, run:
```shell
python run_dynamic_mql.py
```
- For `DynamicLight-Lite`, run:
```shell
python run_dynamic_lite.py
```

## 3.3 Ablation study

### 3.3.1 w/o duration control

- w/o duration control: Before run the model, change the duration action as fixed `1` accroding to the duration action space at `./models/dynamiclight.py` line 123.

### 3.3.2 Learning capability evaluation

First, well train the model;
Then, test the model with changing the phase control policy as `FixedTime`. 

For example, run `run_dynamiclight.py` with the model name as "dynamiclight", then run `run_test.py` with the model name as "dynamiclightft".

## 3.4 Case study
- run `run_dynamiclgiht.py` with the synthetic datasets
- run `run_test.py` for model transfer envaluation

# 4. Extended model study
## 4.1 Different duration action spaces

Cconfigure the action space in `run_dynamic_mql.py` and run it.

## 4.2 Different feature fusion methods
Configure the model name as <`Dyn01`, `Dyn02`,`Dyn03`,`Dyn05`> in `run_dynamic_mql.py` and run it.
Refer to the `./models` for more model details.
## 4.3 Duration Q-value prediction: parameter sharing

- with parameter sharing, configure the model name as `Dyn01` and run `run_dynamic_mql.py`

- w/o parameter sharing, configure the model name as `DynM` and run `run_dynamic_mql.py`

## 4.4 DynamicLight training techniques
- For `Clear Memory`, change `./utils/updater.py` at line 56
- For `fix part network`, change `./models/dynamiclight.py` at line 210
- For `change lr`, change `./utils/config.py` at line 101
- For `train seperately`, change `./models/dynamiclight.py` at line 193
- configure on `fine-tuning`, change `./models/dynamiclight.py` at line 193


## 4.5 Impact of fixed duraiton actions 
Configure line `123` at `./models/dynamiclight.py`:
- fixed duration action=10s: set the duration action as `0`
- fixed duration action=15s: set the duration action as `1`
- fixed duration action=20s: set the duration action as `2`



# 5. Baseline methods

For the baseline methods, refer to [Advanced-XLight](https://github.com/LiangZhang1996/Advanced_XLight.git) and [AttentionLight](https://github.com/LiangZhang1996/AttentionLight.git).

## License

This project is licensed under the GNU General Public License version 3 (GPLv3) - see the LICENSE file for details.