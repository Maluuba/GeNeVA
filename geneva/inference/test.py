# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Testing loop script"""
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from geneva.data.datasets import DATASETS
from geneva.evaluation.evaluate_metrics import report_inception_objects_score
from geneva.utils.config import keys, parse_config
from geneva.models.models import INFERENCE_MODELS
from geneva.data import codraw_dataset
from geneva.data import clevr_dataset


class Tester():
    def __init__(self, cfg, use_val=False, iteration=None, test_eval=False):
        self.model = INFERENCE_MODELS[cfg.gan_type](cfg)

        if use_val:
            dataset_path = cfg.val_dataset
            model_path = os.path.join(cfg.log_path, cfg.exp_name)
        else:
            dataset_path = cfg.dataset
            model_path = cfg.load_snapshot
        if test_eval:
            dataset_path = cfg.test_dataset
            model_path = cfg.load_snapshot

        self.model.load(model_path, iteration)
        self.dataset = DATASETS[cfg.dataset](path=keys[dataset_path],
                                             cfg=cfg,
                                             img_size=cfg.img_size)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=cfg.batch_size,
                                     shuffle=False,
                                     num_workers=cfg.num_workers,
                                     drop_last=True)

        self.iterations = len(self.dataset) // cfg.batch_size

        if cfg.dataset == 'codraw':
            self.dataloader.collate_fn = codraw_dataset.collate_data
        elif cfg.dataset == 'iclevr':
            self.dataloader.collate_fn = clevr_dataset.collate_data

        if cfg.results_path is None:
            cfg.results_path = os.path.join(cfg.log_path, cfg.exp_name,
                                            'results')
            if not os.path.exists(cfg.results_path):
                os.mkdir(cfg.results_path)

        self.cfg = cfg
        self.dataset_path = dataset_path

    def test(self):
        for batch in tqdm(self.dataloader, total=self.iterations):
            self.model.predict(batch)


if __name__ == '__main__':
    cfg = parse_config()
    tester = Tester(cfg, test_eval=True)
    tester.test()
    del tester
    torch.cuda.empty_cache()
    metrics_report = dict()
    if cfg.metric_inception_objects:
        io_jss, io_ap, io_ar, io_af1, io_cs, io_gs = report_inception_objects_score(None,
                                                                                    None,
                                                                                    None,
                                                                                    cfg.results_path,
                                                                                    keys[cfg.dataset + '_inception_objects'],
                                                                                    keys[cfg.test_dataset],
                                                                                    cfg.dataset)

        metrics_report['jaccard'] = io_jss
        metrics_report['precision'] = io_ap
        metrics_report['recall'] = io_ar
        metrics_report['f1'] = io_af1
        metrics_report['cossim'] = io_cs
        metrics_report['relsim'] = io_gs
    print(metrics_report)
