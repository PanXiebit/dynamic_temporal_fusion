
import torch
import logging, os
from torch.utils.data import DataLoader
import utils
import torch.nn.functional as F
# import ctcdecode
from itertools import groupby
from metrics.wer import get_wer_delsubins
import numpy as np
import tensorflow as tf
import ctcdecode
from utils import neq_load_customized

class Trainer(object):
    def __init__(self, opts, model, criterion, vocabulary, vocab_size, blank_id):
        self.opts = opts
        self.model = model
        self.criterion = criterion
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.pad = vocabulary.pad()
        self.unk = vocabulary.unk()
        self.eos = vocabulary.eos()
        self.bos = vocabulary.bos()


        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.criterion = self.criterion.cuda()
            self.model = self.model.cuda()

        self._num_updates = 0

        # params = []
        # for params in self.model.parameters():
        #     if params not in self.model.decoder.parameters():
        #         params.append(params)
        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate,
                                          weight_decay=self.opts.weight_decay)


        logging.info('| num. module params: {} (num. trained: {})'.format(
            sum(p.numel() for p in params),
            sum(p.numel() for p in params if p.requires_grad),
        ))

        # self._build_optimizer(params, self.opts.optimizer, lr=self.opts.learning_rate,
        #                       momentum=self.opts.momentum, weight_decay=self.opts.weight_decay)
        self.decoder_vocab = [chr(x) for x in range(20000, 20000 + self.vocab_size)]
        self.decoder = ctcdecode.CTCBeamDecoder(self.decoder_vocab, beam_width=self.opts.beam_width,
                                                    blank_id=self.blank_id, num_processes=10)


    def train_step(self, samples, update_freq=10):
        self._set_seed()  # seed is changed with the update_steps
        self.model.train()
        self.criterion.train()

        self.optimizer.zero_grad()

        samples = self._prepare_sample(samples)
        loss = self.criterion(self.model, samples)
        loss.backward()
        self.set_num_updates(self.get_num_updates() + 1)
        self.optimizer.step()
        return loss, self.get_num_updates()

    def train_unlike_step(self, samples, update_freq=10):
        self._set_seed()  # seed is changed with the update_steps
        self.model.train()
        self.criterion.train()

        self.optimizer.zero_grad()

        samples = self._prepare_sample(samples)
        loss, ctc_loss, custom_loss = self.criterion.unlikelihood_ctc(self.model, samples)
        # loss /= update_freq
        loss.backward()
        self.set_num_updates(self.get_num_updates() + 1)
        self.optimizer.step()

        if self.get_num_updates() % self.opts.print_step == 0:
            logging.info("num_updates: {}, ctc loss: {:.4f}, custom_loss: {:.4f}".
                         format(self.get_num_updates(), ctc_loss, custom_loss))

        return loss, self.get_num_updates()

    def train_decoder_step(self, samples):
        self._set_seed()  # seed is changed with the update_steps
        self.model.train()
        self.criterion.train()
        self.optimizer.zero_grad()

        samples = self._prepare_sample(samples)

        loss, _, _ = self.criterion.forward_decoder(self.model, samples)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.set_num_updates(self.get_num_updates() + 1)
        return loss, self.get_num_updates()


    def valid_step(self, samples, decoded_dict):
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            samples = self._prepare_sample(samples)
            video = samples["data"]
            len_video = samples["len_data"]
            label = samples["label"]
            len_label = samples["len_label"]
            video_id = samples['id']

            logits, len_video = self.model(video, len_video)
            logits = F.softmax(logits, dim=-1)
            pred_seq, _, _, out_seq_len = self.decoder.decode(logits, len_video)

            err_delsubins = np.zeros([4])
            count = 0
            correct = 0
            start = 0
            for i, length in enumerate(len_label):
                end = start + length
                ref = label[start:end].tolist()
                # hyp = [x for x in pred_seq[i] if x != 0]
                hyp = [x[0] for x in groupby(pred_seq[i][0][:out_seq_len[i][0]].tolist())]
                decoded_dict[video_id[i]] = hyp
                correct += int(ref == hyp)
                err = get_wer_delsubins(ref, hyp)
                err_delsubins += np.array(err)
                count += 1
                start = end
            assert end == label.size(0)
        return err_delsubins, correct, count

    def valid_step_greedy(self, samples, decoded_dict):
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            samples = self._prepare_sample(samples)
            video = samples["data"]
            len_video = samples["len_data"]
            label = samples["label"]
            len_label = samples["len_label"]
            video_id = samples['id']

            logits, len_video = self.model(video, len_video)
            logits = F.softmax(logits, dim=-1)
            # pred_seq, _, _, out_seq_len = self.decoder.decode(logits, len_video)
            pred_seq = logits.argmax(-1)
            out_seq_len = len_video

            err_delsubins = np.zeros([4])
            count = 0
            correct = 0
            start = 0
            for i, length in enumerate(len_label):
                end = start + length
                ref = label[start:end].tolist()
                # hyp = [x for x in pred_seq[i] if x != 0]
                hyp = [x[0] for x in groupby(pred_seq[i][:out_seq_len[i].item()].tolist()) if x[0] != self.blank_id]
                if i== 0:
                    if len(hyp) == 0:
                        logging.info("Here hyp is None!!!!")
                    logging.info("video id: {}".format(video_id[i]))
                    logging.info("ref: {}".format(" ".join(str(i) for i in ref)))
                    logging.info("hyp: {}".format(" ".join(str(i) for i in hyp)))

                    logging.info("\n")
                decoded_dict[video_id[i]] = hyp
                correct += int(ref == hyp)
                err = get_wer_delsubins(ref, hyp)
                err_delsubins += np.array(err)
                count += 1
                start = end
            assert end == label.size(0)
        return err_delsubins, correct, count

    def valid_step_tf(self, samples, decoded_dict):
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            samples = self._prepare_sample(samples)
            video = samples["data"]
            len_video = samples["len_data"]
            label = samples["label"]
            len_label = samples["len_label"]
            video_id = samples['id']

            logits, len_video = self.model(video, len_video)
            logits = F.softmax(logits, dim=-1)
            # pred_seq, _, _, out_seq_len = self.decoder.decode(logits, len_video)

            logits = tf.transpose(tf.constant(logits.cpu().numpy()), [1, 0, 2])  # [len, batch, vocab_size]
            len_video = tf.constant(len_video.cpu().numpy(), dtype=tf.int32)
            decoded, _ = tf.nn.ctc_beam_search_decoder(logits, len_video, beam_width=5, top_paths=1)
            pred_seq = tf.sparse.to_dense(decoded[0]).numpy()  # print(pred_seq.shape, decoded[0].dense_shape)
            
            
            err_delsubins = np.zeros([4])
            count = 0
            correct = 0
            start = 0
            for i, length in enumerate(len_label):
                end = start + length
                ref = label[start:end].tolist()
                hyp = [x for x in pred_seq[i] if x != 0]
                # hyp = [x[0] for x in groupby(pred_seq[i][0][:out_seq_len[i][0]].tolist())]
                # if i== 0:
                #     if len(hyp) == 0:
                #         logging.info("Here hyp is None!!!!")
                #     logging.info("video id: {}".format(video_id[i]))
                #     logging.info("ref: {}".format(" ".join(str(i) for i in ref)))
                #     logging.info("hyp: {}".format(" ".join(str(i) for i in hyp)))
                #
                #     logging.info("\n")
                decoded_dict[video_id[i]] = hyp
                correct += int(ref == hyp)
                err = get_wer_delsubins(ref, hyp)
                err_delsubins += np.array(err)
                count += 1
                start = end
            assert end == label.size(0)
        return err_delsubins, correct, count

    def get_batch_iterator(self, datasets, batch_size, shuffle, num_workers=32, drop_last=True):
        return DataLoader(datasets,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          collate_fn=datasets.collate_fn_video,
                          drop_last=drop_last,
                          pin_memory=True)


    def save_checkpoint(self, filename, epoch, num_updates, loss):
        state_dict = {
            'epoch': epoch,
            'num_updates': num_updates,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            "loss": loss,
        }
        torch.save(state_dict, filename)


    def load_checkpoint(self, filename):
        state_dict = torch.load(filename)
        epoch = state_dict["epoch"]
        num_updates = state_dict["num_updates"]
        loss = state_dict["loss"]
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        return epoch, num_updates, loss

    def pretrain(self, args):
        if os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            self.model = neq_load_customized(self.model, checkpoint['model_state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})"
                  .format(args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    def cnn_freeze(self):
        child_num = 0
        for child in self.model.children():
            child_num += 1
            # print(child_num, child.parameters)
            if child_num < 7:
                for param in child.parameters():
                    param.requires_grad = False

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates

    def _set_seed(self):
        # Set seed based on opts.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.opts.seed + self.get_num_updates()
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.manual_seed(seed)

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None

        if self.cuda:
            sample = utils.move_to_cuda(sample)
            
        return sample
    
    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
        if epoch >= 40:
            lr = self.opts.learning_rate * 0.5
        elif epoch >= 60:
            lr = self.opts.learning_rate * 0.25
        else:
            lr = self.opts.learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def post_process_prediction(self, tensor):
        return [x.item() for x in tensor if x not in [self.pad, self.unk, self.bos, self.eos]]
