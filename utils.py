import logging
import torch
import numpy as np
import os


def neq_load_customized(model, pretrained_dict, only_load_backbone=False):
    ''' load pre-trained model in a not-equal way,
    when new model has been partially modified '''
    model_dict = model.state_dict()
    # for k, v in model_dict.items():
    #     print(k)
    # exit()
    tmp = {}
    print('\n=======Check Weights Loading======')
    print('Weights not used from pretrained file:')
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if only_load_backbone:
                if "enc1" not in k and "enc2" not in k and "fc" not in k:
                    print("In model_dict and in backbone module: ", k)
                    tmp[k] = v
                else:
                    print("In model_dict, but not in backbone module: ", k)
            else:
                tmp[k] = v
        else:
            print("In pretrained model, but not in model_dict: ", k)
    print('---------------------------')
    print('Weights not loaded into new model:')
    for k, v in model_dict.items():
        if k not in tmp.keys():
            print("In model_dict, but not in pretrained model: ", k)
    print('===================================\n')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model


def init_logging(log_file):
    """Init for logging
    """
    logging.basicConfig(level = logging.INFO,
                        format = '%(asctime)s: %(message)s',
                        datefmt = '%m-%d %H:%M:%S',
                        filename = log_file,
                        filemode = 'w')
    # define a Handler which writes INFO message or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%m-%d %H:%M:%S')
    # tell the handler to use this format
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):

    def _move_to_cuda(tensor):
        return tensor.cuda(non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)

class LossManager(object):
    def __init__(self, print_step, last_loss):
        self.print_step = print_step
        self.last_loss = last_loss
        self.total_loss = []

    def update(self, loss, epoch, num_updates):
        self.total_loss.append(loss)
        if (num_updates % self.print_step) == 0:
            mean_loss = np.mean(self.total_loss)
            logging.info('Epoch: {:d}, num_updates: {:d}, loss: {:.3f} -> {:.3f}'.
                         format(epoch, num_updates, self.last_loss, mean_loss))
            self.last_loss = mean_loss
            self.total_loss = []

            
class ModelManager(object):
    def __init__(self, max_num_models=5):
        self.max_num_models = max_num_models
        self.best_epoch = 0
        self.best_err = np.ones([4])*1000
        self.model_file_list = []

    def update(self, model_file, err, epoch):
        self.model_file_list.append((model_file, err))
        self.update_best_err(err, epoch)
        self.sort_model_list()
        if len(self.model_file_list) > self.max_num_models:
            worst_model_file = self.model_file_list.pop(-1)[0]
            if os.path.exists(worst_model_file):
                os.remove(worst_model_file)
        logging.info('CURRENT BEST PERFORMANCE (epoch: {:d}): WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format( \
            self.best_epoch, self.best_err[0], self.best_err[1], self.best_err[2], self.best_err[3]))
        pass

    def update_best_err(self, err, epoch):
        if err[0] < self.best_err[0]:
            self.best_err = err
            self.best_epoch = epoch

    def sort_model_list(self):
        self.model_file_list.sort(key=lambda x: x[1][0])