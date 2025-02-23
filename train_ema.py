import torch
from config.options import parse_args
from src.data.video_lang_datasets import PhoenixVideo
from utils import init_logging, LossManager, ModelManager
import os
# from src.model.dilated_slr import DilatedSLRNet
from src.criterion.ctc_loss import CtcLoss
from src.model.full_conv_v5 import MainStream
from src.trainer_ema import Trainer
import logging
import numpy as np
import uuid
from metrics.wer import get_phoenix_wer
from tqdm import tqdm
from src.data.vocabulary import Vocabulary
import random
from src.ema import EMA


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    opts = parse_args()
    setup_seed(opts.seed)
    init_logging(os.path.join(opts.log_dir, '{:s}_seed{}_log.txt'.format(opts.task, opts.seed)))

    if torch.cuda.is_available():
        torch.cuda.set_device(opts.gpu)
        logging.info("Using GPU!")
        device = "cuda"
    else:
        logging.info("Using CPU!")
        device = "cpu"
        
    logging.info(opts)

    train_datasets = PhoenixVideo(opts.vocab_file, opts.corpus_dir, opts.video_path, phase="train", DEBUG=opts.DEBUG)
    valid_datasets = PhoenixVideo(opts.vocab_file, opts.corpus_dir, opts.video_path, phase="dev", DEBUG=opts.DEBUG)
    vocab_size = valid_datasets.vocab.num_words
    blank_id = valid_datasets.vocab.word2index['<BLANK>']
    vocabulary = Vocabulary(opts.vocab_file)
    model = MainStream(vocab_size, opts.bn_momentum)
    criterion = CtcLoss(opts, blank_id, device, reduction="none")
    ema = EMA(model, decay=0.999)
    # 初始化
    ema.register()

    # print(model)
    # Build trainer
    trainer = Trainer(opts, model, criterion, vocabulary, vocab_size, blank_id)

    if os.path.exists(opts.check_point):
        logging.info("Loading checkpoint file from {}".format(opts.check_point))
        epoch, num_updates, loss = trainer.load_checkpoint(opts.check_point)
    elif os.path.exists(opts.pretrain):
        logging.info("Loading checkpoint file from {}".format(opts.pretrain))
        trainer.pretrain(opts)
        epoch, num_updates, loss = 0, 0, 0.0
    else:
        logging.info("No checkpoint file in found in {}".format(opts.check_point))
        epoch, num_updates, loss = 0, 0, 0.0

    logging.info('| num. module params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    trainer.set_num_updates(num_updates)
    model_manager = ModelManager(max_num_models=25)
    while epoch < opts.max_epoch and trainer.get_num_updates() < opts.max_updates:
        epoch += 1
        trainer.adjust_learning_rate(epoch)
        loss = train(opts, train_datasets, valid_datasets, trainer, epoch, num_updates, loss, ema)

        if epoch <= opts.stage_epoch:
            eval_train(opts, train_datasets, trainer, epoch)
            # phoenix_eval_err = eval_tf(opts, valid_datasets, trainer, epoch)
            phoenix_eval_err = eval(opts, valid_datasets, trainer, epoch, ema)
        else:
            # eval_train(opts, train_datasets, trainer, epoch)
            phoenix_eval_err = eval(opts, valid_datasets, trainer, epoch, ema)

        save_ckpt = os.path.join(opts.log_dir, 'ep{:d}_{:.4f}.pkl'.format(epoch, phoenix_eval_err[0]))
        trainer.save_checkpoint(save_ckpt, epoch, num_updates, loss)
        model_manager.update(save_ckpt, phoenix_eval_err, epoch)


def train(opts, train_datasets, valid_datasets, trainer, epoch, num_updates, last_loss, ema):
    train_iter = trainer.get_batch_iterator(train_datasets, batch_size=opts.batch_size, shuffle=True)
    ctc_epoch_loss, dec_epoch_loss = [], []
    for samples in train_iter:
        # trainer.warm_learning_rate(num_updates)
        loss, num_updates = trainer.train_step(samples, ema)
        ctc_loss = loss.item()
        ctc_epoch_loss.append(ctc_loss)
        lrs = trainer.get_lr()
        if (num_updates % opts.print_step) == 0:
            logging.info('Epoch: {:d}, num_updates: {:d}, loss: {:.3f}, lr: {}'.
                         format(epoch, num_updates, ctc_loss, ", ".join(str(round(lr, 8)) for lr in lrs)))
    logging.info("--------------------- ctc training ------------------------")
    logging.info('Epoch: {:d}, ctc loss: {:.3f} -> {:.3f}'.format(epoch, last_loss, np.mean(ctc_epoch_loss)))
    last_loss = np.mean(ctc_epoch_loss)
    return last_loss


def eval(opts, valid_datasets, trainer, epoch, ema):
    ema.apply_shadow()

    eval_iter = trainer.get_batch_iterator(valid_datasets, batch_size=opts.batch_size, shuffle=False)
    decoded_dict = {}
    val_err, val_correct, val_count = np.zeros([4]), 0, 0
    for samples in tqdm(eval_iter):
        err, correct, count = trainer.valid_step(samples, decoded_dict)
        val_err += err
        val_correct += correct
        val_count += count
    logging.info('-' * 50)
    logging.info('Epoch: {:d}, DEV ACC: {:.5f}, {:d}/{:d}'.format(epoch, val_correct / val_count, val_correct, val_count))
    logging.info('Epoch: {:d}, DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(epoch,
        val_err[0] / val_count, val_err[1] / val_count, val_err[2] / val_count, val_err[3] / val_count))

    # ------ Evaluation with official script (merge synonyms) --------
    list_str_for_test = []
    for k, v in decoded_dict.items():
        start_time = 0
        for wi in v:
            tl = np.random.random() * 0.1
            list_str_for_test.append('{} 1 {:.3f} {:.3f} {}\n'.format(k, start_time, start_time + tl,
                                                                       valid_datasets.vocab.index2word[wi]))
            start_time += tl
    tmp_prefix = str(uuid.uuid1())
    txt_file = '{:s}.txt'.format(tmp_prefix)
    result_file = os.path.join('evaluation_relaxation', txt_file)
    with open(result_file, 'w') as fid:
        fid.writelines(list_str_for_test)
    phoenix_eval_err = get_phoenix_wer(txt_file, 'dev', tmp_prefix)
    logging.info('[Relaxation Evaluation] Epoch: {:d}, DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(epoch,
        phoenix_eval_err[0], phoenix_eval_err[1], phoenix_eval_err[2], phoenix_eval_err[3]))
    ema.restore()
    return phoenix_eval_err


def eval_tf(opts, valid_datasets, trainer, epoch):
    eval_iter = trainer.get_batch_iterator(valid_datasets, batch_size=opts.batch_size, shuffle=False)
    decoded_dict = {}
    val_err, val_correct, val_count = np.zeros([4]), 0, 0
    for samples in tqdm(eval_iter):
        err, correct, count = trainer.valid_step_tf(samples, decoded_dict)
        val_err += err
        val_correct += correct
        val_count += count
    logging.info('-' * 50)
    logging.info('Epoch: {:d}, DEV ACC: {:.5f}, {:d}/{:d}'.format(epoch, val_correct / val_count, val_correct, val_count))
    logging.info('Epoch: {:d}, DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(epoch,
        val_err[0] / val_count, val_err[1] / val_count, val_err[2] / val_count, val_err[3] / val_count))

    # ------ Evaluation with official script (merge synonyms) --------
    list_str_for_test = []
    for k, v in decoded_dict.items():
        start_time = 0
        for wi in v:
            tl = np.random.random() * 0.1
            list_str_for_test.append('{} 1 {:.3f} {:.3f} {}\n'.format(k, start_time, start_time + tl,
                                                                       valid_datasets.vocab.index2word[wi]))
            start_time += tl
    tmp_prefix = str(uuid.uuid1())
    txt_file = '{:s}.txt'.format(tmp_prefix)
    result_file = os.path.join('evaluation_relaxation', txt_file)
    with open(result_file, 'w') as fid:
        fid.writelines(list_str_for_test)
    phoenix_eval_err = get_phoenix_wer(txt_file, 'dev', tmp_prefix)
    logging.info('[Relaxation Evaluation] Epoch: {:d}, DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(epoch,
        phoenix_eval_err[0], phoenix_eval_err[1], phoenix_eval_err[2], phoenix_eval_err[3]))
    return phoenix_eval_err

def eval_train(opts, train_datasets, trainer, epoch):
    eval_iter = trainer.get_batch_iterator(train_datasets, batch_size=opts.batch_size, shuffle=False)
    decoded_dict = {}
    val_err, val_correct, val_count = np.zeros([4]), 0, 0
    for i, samples in tqdm(enumerate(eval_iter)):
        if i > 500:
            break
        # print("decoder_label: ", samples["decoder_label"])
        # print("label: ", samples["label"])
        err, correct, count = trainer.valid_step(samples, decoded_dict)
        val_err += err
        val_correct += correct
        val_count += count
    logging.info('-' * 50)
    logging.info(
        'Epoch: {:d}, Train ACC: {:.5f}, {:d}/{:d}'.format(epoch, val_correct / val_count, val_correct, val_count))
    logging.info('Epoch: {:d}, Train WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.
                 format(epoch, val_err[0] / val_count, val_err[1] / val_count,
                        val_err[2] / val_count, val_err[3] / val_count))

if __name__ == "__main__":
    main()
