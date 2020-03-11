import core 
import core.models
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import pytorch_utils.data.dataset as ds_utils


from torch.utils.data import DataLoader
from pytorch_utils.evaluation import apply_model, argmax_and_accuracy
from .experiment_base import ExperimentBase

class Blueprint(ExperimentBase):

    args = {k: v for k, v in ExperimentBase.args.items()}
    args.update(
        {
            'model': tuple, 
            'batch_size': int,
            'lr_init': float,
            'weight_decay': float,
            'ds_train': str, 
            'ds_test': str
        }
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def ds_setup_iter(self):
        
        for _ in range(self.args['num_runs']):
            ds_train = core.data.ds_factory(self.args['ds_train'])
            ds_test = core.data.ds_factory(self.args['ds_test'])

            ds_stats = ds_utils.ds_statistics(ds_train)

            to_tensor_tf = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(
                    ds_stats['channel_mean'],
                    ds_stats['channel_std']),
            ])

            augmenting_tf = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    to_tensor_tf
            ])

            self.ds_train = ds_utils.Transformer(ds_train, augmenting_tf)
            self.ds_test = ds_utils.Transformer(ds_test, to_tensor_tf)
            self.num_classes = ds_stats['num_classes']
            
            yield None 

    def setup_model(self):
        id, kwargs = self.args['model']
        kwargs['num_classes'] = self.num_classes
        self.model = getattr(core.models, id)(**kwargs)
        self.model.to(self.device)

    def setup_opt(self):
        self.opt = torch.optim.SGD(
            self.model.parameters(),
            weight_decay=self.args['weight_decay'],
            lr=self.args['lr_init'],
            momentum=0.9,
            nesterov=True)

    def setup_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            self.args['num_epochs'],
            eta_min=0,
            last_epoch=-1)

    def setup_dl_train(self):
        self.dl_train = DataLoader(
            self.ds_train,
            batch_size=self.args['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=2)

    def setup_batch(self):
        self.batch_x = self.batch_x.to(self.device)
        self.batch_y = self.batch_y.to(self.device)

    def compute_loss(self):
        l = nn.functional.cross_entropy(
            self.model_output, self.batch_y)
        
        self.logger.log_value('batch_loss', l)
        self.batch_loss = l

    def evaluate(self):
            
        self.model.eval()
    
        X, Y = apply_model(self.model, self.ds_train, device=self.device)
        mb_comment = ''

        acc_train = argmax_and_accuracy(X, Y)
        self.logger.log_value('acc_train', acc_train)
        mb_comment += " | acc. train {:.2f} ".format(acc_train)

        X, Y = apply_model(self.model, self.ds_test, device=self.device)
        acc_test = argmax_and_accuracy(X, Y)
        self.logger.log_value('acc_test', acc_test)
        mb_comment += " | acc. test {:.2f} ".format(acc_test)

        self.mb.main_bar.comment = mb_comment