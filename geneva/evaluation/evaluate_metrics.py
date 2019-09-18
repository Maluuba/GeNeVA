# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from itertools import chain

from geneva.metrics.inception_localizer import calculate_inception_objects_accuracy


def _plot_scalar_metric(visualizer, value, iteration, metric_name):
    visualizer.plot(metric_name, 'train', iteration, value)


def report_inception_objects_score(visualizer, logger, iteration, img_path, model_path, inception_dataset, dataset_name):
    args = [('--img-dir', img_path),
            ('--model-path', model_path),
            ('--dataset-hdf5', inception_dataset),
            ('--dataset', dataset_name)]
    args = [l for l in chain(*args)]
    io_jss, io_ap, io_ar, io_af1, io_cs, io_gs = calculate_inception_objects_accuracy(args=args,
                                                                                      standalone_mode=False)

    if visualizer is not None:
        _plot_scalar_metric(visualizer, io_ap, iteration, 'Average Precision')
        _plot_scalar_metric(visualizer, io_ar, iteration, 'Average Recall')
        _plot_scalar_metric(visualizer, io_af1, iteration, 'Average F1 Score')
        _plot_scalar_metric(visualizer, io_jss, iteration, 'Jaccard Index (IoU)')
        _plot_scalar_metric(visualizer, io_cs, iteration, 'Cosine Similarity')
        _plot_scalar_metric(visualizer, io_gs, iteration, 'Relational Similarity')

    return io_jss, io_ap, io_ar, io_af1, io_cs, io_gs
