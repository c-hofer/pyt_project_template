import uuid
import json
import torch
import torch.nn as nn


import pytorch_utils.logging as logging


from collections import defaultdict
from pathlib import Path

from fastprogress import master_bar, progress_bar


class ExperimentBase(object):

    args = {
        'output_root_dir': str,
        'num_runs': int,
        'num_epochs': int,
        'tag': str
    }

    def check_args(self):
        for k, v in self.args.items():
            if isinstance(v, type):
                assert isinstance(k, v)
            elif hasattr(v, '__call__'):
                assert v(k)


    def __init__(self, **kwargs):
        self.args = kwargs 
        self.check_args()
        self.device = 'cuda'

        output_dir = Path(
            self.args['output_root_dir']) / logging.get_an_id(self.args['tag'])
        output_dir.mkdir()

        self.logger = logging.Logger(output_dir, self.args)
        
        self.ds_train = None
        self.ds_test = None
        self.model = None
        self.opt = None
        self.scheduler = None
        self.dl_train = None

        self.batch_x = None
        self.batch_y = None
        self.epoch_i = None
        self.batch_loss = None


    def one_run(self):
        self.logger.new_run()

        self.setup_model()

        self.setup_opt()
        self.setup_scheduler()
        self.setup_dl_train()

        self.mb = master_bar(range(self.args['num_epochs']))

        for epoch_i in self.mb:
            self.epoch_i = epoch_i

            self.model.train()

            for (batch_x, batch_y), _ in zip(self.dl_train, progress_bar(range(len(self.dl_train)-1), parent=self.mb)):
                self.batch_x, self.batch_y = batch_x, batch_y

                self.setup_batch()

                self.model_output = self.model(self.batch_x)

                self.compute_loss()

                if isinstance(self.opt, list):
                    for o in self.opt:
                        o.zero_grad()
                else: 
                    self.opt.zero_grad()

                self.batch_loss.backward()

                if isinstance(self.opt, list):
                    for o in self.opt:
                        o.step()
                else:
                    self.opt.step()

            if self.scheduler is not None:
                if isinstance(self.scheduler, list):
                    for s in self.scheduler:
                        s.step()
                else:
                    self.scheduler.step()

            self.evaluate()
            self.logger.write_logged_values_to_disk()

        self.logger.write_model_to_disk('model', self.model)


    def __call__(self):
        try:
            for _ in self.ds_setup_iter():
                self.one_run()

        except Exception as ex:
            self.error = ex 
            self.handle_error()

    def ds_setup_iter(self):
        raise NotImplementedError()

    def setup_model(self):
        raise NotImplementedError()

    def setup_opt(self):
        raise NotImplementedError()

    def setup_scheduler(self):
        raise NotImplementedError()

    def setup_dl_train(self):
        raise NotImplementedError()

    def setup_batch(self):
        raise NotImplementedError()

    def compute_loss(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def handle_error(self):
        raise self.error 

    @classmethod
    def args_template(cls):
        s = " : , \n    "
        s = s.join([r"'{}'".format(k) for k in cls.args.keys()])
        s = "{ \n    " + s +"\n}"
        return s

