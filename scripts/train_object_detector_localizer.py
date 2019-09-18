# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import random
import shutil
import warnings

import cv2
import click
import h5py
import numpy as np
from sklearn.metrics import jaccard_similarity_score, precision_score, recall_score, f1_score
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from tqdm import tqdm

from geneva.models.object_localizer import inception_v3
from geneva.utils.config import keys
from geneva.utils.visualize import VisdomPlotter


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

viz = VisdomPlotter(env_name='inception_localizer', server='http://localhost')


class Inception3ObjectLocalizer(nn.Module):
    def __init__(self, num_objects=58, pretrained=True, num_coords=3):
        super().__init__()
        self.inception3 = inception_v3(pretrained=pretrained)
        self.inception3.fc = nn.Linear(self.inception3.fc.in_features, 512)

        self.detector = nn.Sequential(nn.Linear(512, 256),
                                      nn.Linear(256, num_objects),
                                      nn.Sigmoid())
        self.localizer = nn.Sequential(nn.Linear(1024, 512),
                                       nn.Linear(512, num_objects * num_coords))
        self.num_objects = num_objects
        self.num_coords = num_coords

    def forward(self, image):
        inception_feats, aux_feats = self.inception3(image)
        detections = self.detector(inception_feats)
        mixed_feats = torch.cat([inception_feats, aux_feats], dim=1)
        locations = self.localizer(mixed_feats)
        locations = locations.view(-1, self.num_objects, self.num_coords)
        return detections, locations


class AverageMeter(object):
    """ Computes and stores the average and current value

        Code from https://github.com/pytorch/examples/blob/6f62fcd361868406a95d71af9349f9a3d03a2c52/imagenet/main.py
        License:
            BSD 3-Clause License

            Copyright (c) 2017,
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
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr):
    """ LR decayed by factor of 2 every 30 epochs

        Code from https://github.com/pytorch/examples/blob/6f62fcd361868406a95d71af9349f9a3d03a2c52/imagenet/main.py
        License:
            BSD 3-Clause License

            Copyright (c) 2017,
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
    lr = lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def jaccard_accuracy(output, target, pretrained=False, eval_mode=False):
    with torch.no_grad():
        y_true = target.cpu().numpy().astype('int64')
        y_pred = [x.cpu().numpy() for x in output]
        if eval_mode:
            y_pred = (np.array(y_pred) > 0.5).astype('int64')
        elif not pretrained:
            y_pred = (np.array(y_pred).mean(axis=0) > 0.5).astype('int64')
        else:
            y_pred = (np.array(y_pred[0]) > 0.5).astype('int64')
    jss = []
    for i in range(y_true.shape[0]):
        jss.append(jaccard_similarity_score(y_true[i], y_pred[i]))
    return np.mean(jss)


class WeightedBCE(torch.nn.Module):
    def __init__(self, weight, objects_weights):
        super(WeightedBCE, self).__init__()
        self.weights = weight
        self.objects_weights = objects_weights

    def forward(self, pred, target):
        eps = np.finfo(float).eps
        total_loss = []
        target = target * 0.9
        for i in range(pred.size(1)):
            loss = - (1. / pred.size(0)) * torch.sum(self.weights[1] * target[:, i] * torch.log(pred[:, i] + eps) +
                                                     self.weights[0] * (1 - target[:, i]) * torch.log(1 - pred[:, i] + eps))
            loss *= self.objects_weights[i]
            total_loss.append(loss)
        return np.sum(total_loss)


def validate(valid_loader, model, criterion,
             cuda_enabled=False, print_freq=False, pretrained=False):
    detect_losses = AverageMeter()
    local_losses = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    f1 = AverageMeter()

    model.eval()

    with torch.no_grad():
        for i, sample in enumerate(valid_loader):
            input = sample['images']
            target = sample['bow']
            coords = sample['coords']
            if cuda_enabled:
                input = input.cuda()
                target = target.cuda()
                coords = coords.cuda()
            detections, locations = model(input)
            detection_loss = criterion(detections, target)
            localization_loss = torch.sum(
                target * torch.nn.functional.mse_loss(locations, coords, reduce=False).sum(dim=2)
            )

            prec = precision_score(target.detach().cpu().numpy(), detections.detach().cpu().numpy() > 0.5, average='samples')
            rec = recall_score(target.detach().cpu().numpy(), detections.detach().cpu().numpy() > 0.5, average='samples')
            fs = f1_score(target.detach().cpu().numpy(), detections.detach().cpu().numpy() > 0.5, average='samples')

            detect_losses.update(detection_loss.item(), input.size(0))
            local_losses.update(localization_loss.item(), input.size(0))
            precision.update(prec, input.size(0))
            recall.update(rec, input.size(0))
            f1.update(fs, input.size(0))

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Detection Loss {detection_loss.val:.4f} ({detection_loss.avg:.4f})\t'
                      'Localization Loss {localization_loss.val:.4f} ({localization_loss.avg:.4f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {rec.val:.3f} ({rec.avg:.3f})\t'
                      'F1 Score {f1.val:.3f} ({f1.avg:.3f})'.format(
                       i, len(valid_loader), detection_loss=detect_losses,
                       localization_loss=local_losses, prec=precision, rec=recall, f1=f1))
        print(' * Precision {prec.avg:.3f}\tRecall {rec.avg:.3f}\tF1 Score {f1.avg:.3f}'
              .format(prec=precision, rec=recall, f1=f1))

    return precision.avg, recall.avg, f1.avg, detect_losses.avg


def train_model(train_loader, model, criterion, optimizer, epoch,
                cuda_enabled=False, print_freq=1, pretrained=False):
    detect_losses = AverageMeter()
    local_losses = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    f1 = AverageMeter()

    model.train()
    dataset_detections = []
    dataset_targets = []
    for i, sample in enumerate(train_loader):
        input = sample['images']
        target_objects = sample['bow']
        coords = sample['coords']
        if cuda_enabled:
            input = input.cuda()
            target_objects = target_objects.cuda()
            coords = coords.cuda()

        detections, locations = model(input)
        detection_loss = criterion(detections, target_objects)
        localization_loss = torch.sum(
            target_objects * torch.nn.functional.mse_loss(locations, coords, reduce=False).sum(dim=2)
        )

        loss = detection_loss + 0.0001 * localization_loss

        dataset_targets.extend(target_objects.data.cpu().numpy())
        dataset_detections.extend(detections.data.cpu().numpy() > 0.5)

        prec = precision_score(target_objects.detach().cpu().numpy(), detections.detach().cpu().numpy() > 0.5, average='micro')
        rec = recall_score(target_objects.detach().cpu().numpy(), detections.detach().cpu().numpy() > 0.5, average='micro')
        fs = f1_score(target_objects.detach().cpu().numpy(), detections.detach().cpu().numpy() > 0.5, average='micro')

        detect_losses.update(detection_loss.item(), input.size(0))
        local_losses.update(localization_loss.item(), input.size(0))
        precision.update(prec, input.size(0))
        recall.update(rec, input.size(0))
        f1.update(fs, input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        image = show_detections(input.data.cpu().numpy()[0],
                                target_objects.cpu().data.numpy()[0],
                                coords.data.cpu().numpy()[0],
                                locations.data.cpu().numpy()[0])
        viz.draw('detections', image)

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Detection Loss {detection_loss.val:.4f} ({detection_loss.avg:.4f})\t'
                  'Localization Loss {localization_loss.val:.4f} ({localization_loss.avg:.4f})\t'
                  'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                  'Recall {rec.val:.3f} ({rec.avg:.3f})\t'
                  'F1 Score {f1.val:.3f} ({f1.avg:.3f})'.format(
                   epoch, i, len(train_loader), detection_loss=detect_losses,
                   localization_loss=local_losses, prec=precision, rec=recall, f1=f1))

    dataset_targets = np.array(dataset_targets)
    dataset_detections = np.array(dataset_detections)
    dataset_precision = precision_score(dataset_targets, dataset_detections, average='micro')
    dataset_recall = recall_score(dataset_targets, dataset_detections, average='micro')
    dataset_f1 = f1_score(dataset_targets, dataset_detections, average='micro')

    print('Epoch: [{0}]]\t'
          'Train-set Precision {1:.3f}\t'
          'Train-set Recall {2:.3f}\t'
          'Train-set F1 Score {3:.3f}'.format(epoch, dataset_precision, dataset_recall, dataset_f1))


class CoDrawSingleAllDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, transform=None, noise_fraction=0., bg_fraction_noise=0., stats_only=False,
                 num_classes=-1):
        super().__init__()
        self.dataset = None
        self.dataset_path = h5_path
        if num_classes == 24:
            obj_key_name = 'iclevr_objects'
            bg_key_name = 'iclevr_background'
        elif num_classes == 58:
            obj_key_name = 'codraw_objects'
            bg_key_name = 'codraw_background'
        with open(keys[obj_key_name], 'r') as f:
            self.objects = np.array([l.strip() for l in f])
        self.transform = transform
        self.noise_fraction = noise_fraction
        self.bg_fraction_noise = bg_fraction_noise
        if self.bg_fraction_noise > 0:
            background_img = cv2.imread(keys[bg_key_name])[..., ::-1]
            self.background_img = cv2.resize(background_img, (128, 128))
        self.stats_only = stats_only
        with h5py.File(self.dataset_path, 'r') as f:
            self.len_real = len(list(f.keys()))
        self.num_classes = num_classes
        if self.num_classes == 24:
            self.num_coords = 2
        elif self.num_classes == 58:
            self.num_coords = 3

    def __len__(self):
        with h5py.File(self.dataset_path, 'r') as f:
            length = len(list(f.keys()))
        return length + int(self.noise_fraction * length)

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_path, 'r')

        if self.stats_only or idx < self.len_real:
            example = self.dataset[str(idx)]
            image = example['image'].value[..., ::-1]
            bow = example['objects'].value
            coords = example['coords'].value.astype(float)
            coords[:, 0] /= 128.
            coords[:, 1] /= 128.
            coords[:, 2] /= 2.
        else:
            # 0 -> background; 1 -> noise
            chooser = np.random.choice([0, 1], p=[self.bg_fraction_noise, 1 - self.bg_fraction_noise])
            if chooser == 0:
                example = dict(image=self.background_img,
                               bow=np.zeros((self.num_classes,), dtype=np.int64),
                               coords=np.zeros((self.num_classes, 3)))
            elif chooser == 1:
                example = dict(image=np.random.randint(0, high=255, size=(128, 128, 3), dtype=np.uint8),
                               bow=np.zeros((self.num_classes,), dtype=np.int64),
                               coords=np.zeros((self.num_classes, 3)))
            else:
                raise ValueError
            image = example['image']
            bow = example['bow']
            coords = example['coords']

        if self.transform is not None:
            image = self.transform(image)

        assert np.max(bow) <= 1.
        assert np.min(bow) >= 0.

        objects = self.objects[bow > 0]
        objects = str.join(',', list(objects))

        coords = coords[:, :self.num_coords]

        sample = {
            'images': torch.FloatTensor(image),
            'bow': torch.FloatTensor(bow),
            'objects': objects,
            'coords': torch.FloatTensor(coords),
        }

        return sample


def save_checkpoint(state, is_best, filename='inception_latest_checkpoint.pth', prefix=''):
    filename = prefix + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, prefix + 'inception_best_checkpoint.pth')


def setup_inception_model(num_classes, pretrained=False, num_coords=3):
    model = torch.nn.DataParallel(Inception3ObjectLocalizer(num_objects=num_classes, pretrained=pretrained,
                                                            num_coords=num_coords)).cuda()
    return model


def get_class_weights(train_loader, num_objects):
    n_total = np.zeros((num_objects,))
    for i, sample in enumerate(tqdm(train_loader)):
        n_total += sample['bow'].sum(dim=0)

    n_total /= n_total.sum()
    n_total = 1. / n_total
    n_total /= n_total.sum()
    return n_total


@click.command()
@click.option('--seed', type=int, default=1234, help='Integer seed for deterministic training.')
@click.option('--cuda-enabled', is_flag=True, help='Add this flag to use GPU. Ignore for CPU.')
@click.option('--num-classes', type=int, required=True, help='Number of object classes.')
@click.option('--lr', type=float, default=1e-3, help='Learning Rate for SGD.')
@click.option('--momentum', type=float, default=0.9, help='Momentum for SGD.')
@click.option('--weight-decay', type=float, default=1e-4, help='Weight Decay for SGD.')
@click.option('--batch-size', type=int, default=128, help='Batch size for training.')
@click.option('--num-workers', type=int, default=8, help='Number of workers to use for data loader.')
@click.option('--train-hdf5', type=click.Path(exists=True, file_okay=True, dir_okay=False,
                                              readable=True, resolve_path=True),
              required=True, help='Path of HDF5 file containing training dataset.')
@click.option('--num-epochs', type=int, default=100, help='Maximum number of epochs to train for.')
@click.option('--print-freq', type=int, default=100, help='Printing frequency of updates.')
@click.option('--pretrained', is_flag=True, help='Use pre-trained inception v3 and only fine-tune.')
@click.option('--valid-hdf5', type=click.Path(exists=True, file_okay=True, dir_okay=False,
                                              readable=True, resolve_path=True),
              required=True, help='Path of HDF5 file containing validation dataset.')
@click.option('--pos-weight', type=float, default=2, help='neg weight is 1')
@click.option('--noise-fraction', type=float, default=0., help='Noise batches as a fraction of training batches.')
@click.option('--bg-fraction-within-noise', type=float, default=0.,
              help='Empty background batches as a fraction of the noise batches.')
def train_inception_model(seed, cuda_enabled, num_classes, lr, momentum, weight_decay,
                          batch_size, num_workers, train_hdf5, num_epochs, print_freq,
                          pretrained, valid_hdf5, pos_weight, noise_fraction, bg_fraction_within_noise):
    if cuda_enabled is None:
        cuda_enabled = False
    if pretrained is None:
        pretrained = False
        raise NotImplementedError

    if num_classes == 24:
        save_prefix = 'clevr_'
        num_coords = 2
    elif num_classes == 58:
        save_prefix = 'codraw_'
        num_coords = 3

    # fix random seeds
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda_enabled:
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    model = setup_inception_model(num_classes, pretrained=pretrained, num_coords=num_coords)
    optimizer = torch.optim.Adam(model.parameters(), lr, amsgrad=True)

    if cuda_enabled:
        cudnn.benchmark = True

    train_transforms = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize(299),
                                           transforms.ToTensor()])
    valid_transforms = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize(299),
                                           transforms.ToTensor()])

    train_dataset = CoDrawSingleAllDataset(train_hdf5, transform=train_transforms, noise_fraction=noise_fraction,
                                           bg_fraction_noise=bg_fraction_within_noise, num_classes=num_classes)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)

    # only for computing dataset statistics
    train_dataset_stats = CoDrawSingleAllDataset(train_hdf5, transform=train_transforms, noise_fraction=0.,
                                                 bg_fraction_noise=0., stats_only=True, num_classes=num_classes)
    train_loader_stats = torch.utils.data.DataLoader(train_dataset_stats, batch_size=batch_size, shuffle=True,
                                                     num_workers=num_workers, pin_memory=True)

    valid_dataset = CoDrawSingleAllDataset(valid_hdf5, transform=valid_transforms, num_classes=num_classes)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)

    class_weights = get_class_weights(train_loader_stats, num_classes)
    criterion = WeightedBCE((1, pos_weight), objects_weights=class_weights)
    if cuda_enabled:
        model = model.cuda()
        criterion = criterion.cuda()

    best_f1 = -1
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch, lr)
        train_model(train_loader, model, criterion, optimizer, epoch,
                    cuda_enabled=cuda_enabled, print_freq=print_freq, pretrained=pretrained)
        precision_eval, recall_eval, f1_eval, loss_eval = validate(valid_loader, model, criterion,
                                                                   cuda_enabled=cuda_enabled, print_freq=print_freq,
                                                                   pretrained=pretrained)

        is_best = f1_eval > best_f1
        if is_best:
            print('New Best!!!')
            best_f1 = f1_eval
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_f1': best_f1,
            'pretrained': pretrained,
            'cuda_enabled': cuda_enabled,
            'num_classes': num_classes,
        }, is_best, prefix=save_prefix)


def show_detections(image, objects, coords, predctions):
    image = image * 255
    image = image.transpose(1, 2, 0).copy()
    x = coords[:, 0] * 299
    y = coords[:, 1] * 299
    x_pred = predctions[:, 0] * 299
    y_pred = predctions[:, 1] * 299
    for i in range(len(x)):
        if objects[i] == 1:
            cv2.circle(image, (x[i], y[i]), 8, (0, 0, 255), -1)
            cv2.circle(image, (x_pred[i], y_pred[i]), 8, (255, 0, 0), -1)
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    image = 2 * image / 255. - 1.
    return image


if __name__ == '__main__':
    train_inception_model()
