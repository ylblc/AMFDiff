#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-10-26 20:20:36

import argparse

import optuna
from omegaconf import OmegaConf
from optuna.trial import TrialState

from utils.util_common import get_obj_from_str
from utils.util_opts import str2bool

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
            "--save_dir",
            type=str,
            default="./save_dir",
            help="Folder to save the checkpoints and training log",
            )
    parser.add_argument(
            "--resume",
            type=str,
            const=True,
            default="",
            nargs="?",
            help="resume from the save_dir or checkpoint",
            )
    parser.add_argument(
            "--cfg_path",
            type=str,
            default="./configs/training/ffhq256_bicubic8.yaml",
            help="Configs of yaml file",
            )
    args = parser.parse_args()

    return args
'''
使用optuna获取最优超参数
配置参数:
    trainer:
        target: trainer.TrainerDifIROptuna
可视化:
1、pip install optuna-dashboard
2、optuna-dashboard sqlite:///db.sqlite3
'''
def optuna_main(object,n_trials=10):
    # 创建study对象
    study = optuna.create_study(
        # study_name="abc" #自定义名
        storage="sqlite:///db.sqlite3",
        directions=['maximize', 'minimize'],
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(object, n_trials=n_trials)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("Number of finished trials: ", len(study.trials))
    print("Number of pruned trials: ", len(pruned_trials))
    print("Number of complete trials: ", len(complete_trials))
    trials = study.best_trials
    print("Best hyperparameters:")
    for trial in trials:
        print(f'param: {trial.params}')
        print(f'value: {trial.values}')
        print("="*100)

if __name__ == "__main__":
    args = get_parser()

    configs = OmegaConf.load(args.cfg_path)

    # merge args to config
    for key in vars(args):
        if key in ['cfg_path', 'save_dir', 'resume', ]:
            configs[key] = getattr(args, key)

    trainer = get_obj_from_str(configs.trainer.target)(configs)
    # 1、正常训练
    trainer.train()

    # 2、调参
    # optuna_main(trainer.train)
