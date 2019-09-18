# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Evaluation script."""
import torch

from geneva.utils.config import keys, parse_config
from geneva.evaluation.evaluate_metrics import report_inception_objects_score
from geneva.utils.visualize import VisdomPlotter
from geneva.inference.test import Tester


class Evaluator():
    @staticmethod
    def factory(cfg, visualizer, logger):
        if cfg.gan_type == 'recurrent_gan':
            return RecurrentGANEvaluator(cfg, visualizer, logger)


class RecurrentGANEvaluator():
    def __init__(self, cfg, visualizer, logger):
        self.cfg = cfg
        self.visualizer = visualizer
        self.logger = logger

    def evaluate(self, iteration):
        tester = Tester(self.cfg, use_val=True, iteration=iteration)
        tester.test()
        del tester
        torch.cuda.empty_cache()
        metrics_report = dict()
        if self.cfg.metric_inception_objects:
            io_jss, io_ap, io_ar, io_af1, io_cs, io_gs = \
                report_inception_objects_score(self.visualizer,
                                               self.logger,
                                               iteration,
                                               self.cfg.results_path,
                                               keys[self.cfg.dataset + '_inception_objects'],
                                               keys[self.cfg.val_dataset],
                                               self.cfg.dataset)

            metrics_report['jaccard'] = io_jss
            metrics_report['precision'] = io_ap
            metrics_report['recall'] = io_ar
            metrics_report['f1'] = io_af1
            metrics_report['cossim'] = io_cs
            metrics_report['relsim'] = io_gs
        return metrics_report


if __name__ == '__main__':
    cfg = parse_config()
    visualizer = VisdomPlotter(env_name=cfg.exp_name)
    logger = None
    dataset = cfg.dataset
    evaluator = Evaluator(cfg, visualizer, logger, dataset)
    evaluator.evaluate()
