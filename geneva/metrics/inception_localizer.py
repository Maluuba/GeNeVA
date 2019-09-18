# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import absolute_import
from __future__ import print_function

import os
import random

import click
import numpy as np
from sklearn.metrics import jaccard_similarity_score, precision_score,\
    recall_score, f1_score, pairwise
import torch
import torch.backends.cudnn as cudnn
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
import torchvision.transforms as transforms
from tqdm import tqdm

from geneva.models.object_localizer import Inception3ObjectLocalizer


loaded_model = None
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


class ImageFolderNonGT(torch.utils.data.Dataset):
    """
        Code from https://github.com/pytorch/vision/blob/f566fac80e3182a8b3c0219a88ae00ed1b81d7c7/torchvision/datasets/folder.py
        License:
            BSD 3-Clause License

            Copyright (c) Soumith Chintala 2016,
            All rights reserved.

            Redistribution and use in source and binary forms, with or without
            modification, are permitted provided that the following conditions are met:

            * Redistributions of source code must retain the above copyright notice, this
              list of conditions and the following disclaimer.

            * Redistributions in binary form must reproduce the above copyright notice,
              this list of conditions and the following disclaimer in the documentation
              and/or other materials provided with the distribution.

            * Neither the name of the copyright holder nor the names of its
              contributors may be used to endorse or promote products derived from
              this software without specific prior written permission.

            THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
            AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
            IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
            DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
            FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
            DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
            SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
            CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
            OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
            OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    def __init__(self, root, transform=None, labelfile=None):
        self.root = root
        self.transform = transform

        generated_dirs = sorted([d.name for d in os.scandir(self.root) if d.is_dir() and not d.name.endswith('_gt')])
        gt_dirs = sorted([d.name for d in os.scandir(self.root) if d.is_dir() and d.name.endswith('_gt')])

        assert len(gt_dirs) == len(generated_dirs)

        gt_files = []
        generated_files = []
        for i in range(len(gt_dirs)):
            gt_files.extend([os.path.join(gt_dirs[i], x) for x in os.listdir(os.path.join(self.root, gt_dirs[i]))])
            generated_files.extend([os.path.join(generated_dirs[i], x)
                                    for x in os.listdir(os.path.join(self.root, generated_dirs[i]))])

        gt_imgs = [x for x in gt_files if has_file_allowed_extension(x, IMG_EXTENSIONS)]
        generated_imgs = [x for x in generated_files if has_file_allowed_extension(x, IMG_EXTENSIONS)]
        self.gt_filenames = gt_imgs
        self.generated_filenames = generated_imgs

        assert len(self.gt_filenames) == len(self.generated_filenames)

        if len(self.gt_filenames) == 0:
            raise(RuntimeError("Found 0 files in folder: " + root + "\n"
                               "Supported extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.loader = default_loader

    def __getitem__(self, idx):
        gt_path = self.gt_filenames[idx]
        generated_path = self.generated_filenames[idx]
        gt_sample = self.loader(os.path.join(self.root, gt_path))
        generated_sample = self.loader(os.path.join(self.root, generated_path))
        if transforms is not None:
            generated_sample = self.transform(generated_sample)
            gt_sample = self.transform(gt_sample)

        return generated_sample, gt_sample

    def __len__(self):
        return len(self.gt_filenames)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def setup_inception_model(num_classes, pretrained=False):
    if num_classes == 24:
        num_coords = 2
    else:
        num_coords = 3
    model = torch.nn.DataParallel(Inception3ObjectLocalizer(num_objects=num_classes,
                                                            pretrained=pretrained,
                                                            num_coords=num_coords)).cuda()
    return model


def construct_graph(coords, dataset):
    n = len(coords)
    graph = np.zeros((2, n, n))
    for i in range(n):
        if coords.shape[1] == 2:
            ref_x, ref_y = coords[i]
        else:
            ref_x, _, ref_y = coords[i]
        for j in range(n):
            if i == j:
                query_x, query_y = 0.5, 0.5
            else:
                if coords.shape[1] == 2:
                    query_x, query_y = coords[j]
                else:
                    query_x, _, query_y = coords[j]

            if ref_x > query_x:
                graph[0, i, j] = 1
            elif ref_x < query_x:
                graph[0, i, j] = -1

            if ref_y > query_y:
                graph[1, i, j] = 1
            elif ref_y < query_y:
                graph[1, i, j] = -1

    return graph


def get_graph_similarity(detections, label, locations, gt_locations, dataset):
    """Computes the accuracy of relationships of the intersected
    detections multiplied by recall
    """
    intersection = (detections & label).astype(bool)
    if not np.any(intersection):
        return 0

    locations = locations.data.cpu().numpy()[intersection]
    gt_locations = gt_locations.data.cpu().numpy()[intersection]

    genereated_graph = construct_graph(locations, dataset)
    gt_graph = construct_graph(gt_locations, dataset)

    matches = (genereated_graph == gt_graph).astype(int).flatten()
    matches_accuracy = matches.sum() / len(matches)
    recall = recall_score(label, detections, average='samples')

    graph_similarity = recall * matches_accuracy

    return graph_similarity


def get_obj_det_acc(dataloader, dataset):
    jss = []
    cs = []
    graph_similarity = []

    gt_all = []
    pred_all = []

    for _, (sample, gt) in enumerate(tqdm(dataloader)):
        sample = sample.cuda()
        gt = gt.cuda()
        detection_logits, locations = loaded_model(sample)
        gt_detection_logits, gt_locations = loaded_model(gt)

        pred = detection_logits > 0.5
        gt_pred = gt_detection_logits > 0.5

        pred = pred.cpu().numpy().astype('int')
        gt_pred = gt_pred.cpu().numpy().astype('int')
        gt_detection_logits = gt_detection_logits.cpu().numpy()
        detection_logits = detection_logits.cpu().numpy()

        gt_all.extend(gt_pred)
        pred_all.extend(pred)

        cs.append(pairwise.cosine_similarity(gt_detection_logits, detection_logits)[0][0])
        graph_similarity.append(get_graph_similarity(pred, gt_pred, locations, gt_locations, dataset))
        jss.append(jaccard_similarity_score(gt_pred, pred))

    ps = precision_score(np.array(gt_all), np.array(pred_all), average='samples')
    rs = recall_score(np.array(gt_all), np.array(pred_all), average='samples')
    f1 = f1_score(np.array(gt_all), np.array(pred_all), average='samples')

    return np.mean(jss), ps, rs, f1, np.mean(cs), np.mean(graph_similarity)


def _init_inception(model_dir):
    global loaded_model

    checkpoint = torch.load(model_dir)
    random.seed(1234)
    torch.manual_seed(1234)
    if checkpoint['cuda_enabled']:
        cudnn.deterministic = True
    loaded_model = setup_inception_model(checkpoint['num_classes'], pretrained=checkpoint['pretrained'])
    if checkpoint['cuda_enabled']:
        loaded_model = loaded_model.cuda()
        cudnn.benchmark = True
    loaded_model.load_state_dict(checkpoint['state_dict'])
    loaded_model.eval()


@click.command()
@click.option('--img-dir', type=click.Path(exists=True, dir_okay=True, readable=True), required=True)
@click.option('--model-path', type=click.Path(exists=True, file_okay=True, readable=True), required=True)
@click.option('--dataset-hdf5', type=click.Path(exists=True, file_okay=True, readable=True), required=True)
@click.option('--dataset', type=str, required=False)
def calculate_inception_objects_accuracy(img_dir, model_path, dataset_hdf5, dataset):
    if loaded_model is None:
        _init_inception(model_path)
    test_transforms = transforms.Compose([transforms.Resize(299),
                                          transforms.ToTensor()])
    dataset = ImageFolderNonGT(img_dir, transform=test_transforms, labelfile=dataset_hdf5)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    with torch.no_grad():
        jss, avg_precision, avg_recall, avg_f1, cs, graph_similarity = get_obj_det_acc(dataloader, dataset)
        print('\nNumber of images used: {}\nJSS: {}\n AP: {}\nAR: {}\n F1: {}\nCS: {}\nGS: {}'.format(len(dataset), jss,
                                                                                                      avg_precision, avg_recall,
                                                                                                      avg_f1, cs,
                                                                                                      graph_similarity))
    return jss, avg_precision, avg_recall, avg_f1, cs, graph_similarity


if __name__ == '__main__':
    calculate_inception_objects_accuracy()
