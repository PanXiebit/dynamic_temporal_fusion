import torch
from config.options import parse_args
from src.data.video_lang_datasets import PhoenixVideo
from utils import init_logging, LossManager, ModelManager
import os
from src.criterion.ctc_loss import CtcLoss
from src.trainer import Trainer
import logging
import numpy as np
import uuid
from metrics.wer import get_phoenix_wer
import time
from tqdm import tqdm
from src.data.vocabulary import Vocabulary
import torch.nn.functional as F
import ctcdecode
from metrics.wer import get_wer_delsubins
from itertools import groupby
from src.model.full_conv_v10 import MainStream

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    opts = parse_args()
    init_logging(os.path.join(opts.log_dir, '{:s}_win0_win4_log_test.txt'.format(opts.task)))

    if torch.cuda.is_available():
        torch.cuda.set_device(opts.gpu)
        logging.info("Using GPU!")
        device = "cuda"
    else:
        logging.info("Using CPU!")
        device = "cpu"

    logging.info(opts)

    test_datasets = PhoenixVideo(opts.vocab_file, opts.corpus_dir, opts.video_path, phase=opts.task, DEBUG=opts.DEBUG)
    vocab_size = test_datasets.vocab.num_words
    blank_id = test_datasets.vocab.word2index['<BLANK>']
    vocabulary = Vocabulary(opts.vocab_file)
#     model = DilatedSLRNet(opts, device, vocab_size, vocabulary,
#                           dilated_channels=512, num_blocks=5, dilations=[1, 2, 4], dropout=0.0)
    model = MainStream(vocab_size)
    criterion = CtcLoss(opts, blank_id, device, reduction="none")
    trainer = Trainer(opts, model, criterion, vocabulary, vocab_size, blank_id)

    # ctcdeocde
    ctc_decoder_vocab = [chr(x) for x in range(20000, 20000 + vocab_size)]
    ctc_decoder = ctcdecode.CTCBeamDecoder(ctc_decoder_vocab, beam_width=opts.beam_width,
                                           blank_id=blank_id, num_processes=10)

    if os.path.exists(opts.check_point):
        logging.info("Loading checkpoint file from {}".format(opts.check_point))
        epoch, num_updates, loss = trainer.load_checkpoint(opts.check_point)
    else:
        logging.info("No checkpoint file in found in {}".format(opts.check_point))
        epoch, num_updates, loss = 0, 0, 0.0

    test_iter = trainer.get_batch_iterator(test_datasets, batch_size=opts.batch_size, shuffle=False)
    decoded_dict = {}
    val_err, val_correct, val_count = np.zeros([4]), 0, 0

    with open("Data/output/hypo_ctc.txt", "w") as f, open("Data/output/ref_ctc.txt", "w") as f2:
        with torch.no_grad():
            model.eval()
            criterion.eval()
            for samples in tqdm(test_iter):
                samples = trainer._prepare_sample(samples)
                video = samples["data"]
                len_video = samples["len_data"]
                label = samples["label"]
                len_label = samples["len_label"]
                video_id = samples['id']

                logits, _ = model(video, len_video)
                len_video /= 4
                logits = F.softmax(logits, dim=-1)
                pred_seq, _, _, out_seq_len = ctc_decoder.decode(logits, len_video)
                start = 0
                for i, length in enumerate(len_label):
                    end = start + length
                    ref = label[start:end].tolist()
                    hyp = [x[0] for x in groupby(pred_seq[i][0][:out_seq_len[i][0]].tolist())]
                    ref_sent = " ".join([vocabulary.index2word[r] for r in ref])
                    hyp_sent = " ".join([vocabulary.index2word[r] for r in hyp])
                    f.write(hyp_sent + "\n")
                    f2.write(ref_sent + "\n")

                    decoded_dict[video_id[i]] = hyp
                    val_correct += int(ref == hyp)
                    err = get_wer_delsubins(ref, hyp)
                    val_err += np.array(err)
                    val_count += 1
                    start = end
                assert end == label.size(0)
            logging.info('-' * 50)
            logging.info(
                'Epoch: {:d}, DEV ACC: {:.5f}, {:d}/{:d}'.format(epoch, val_correct / val_count, val_correct, val_count))
            logging.info('Epoch: {:d}, DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(
                epoch, val_err[0] / val_count, val_err[1] / val_count, val_err[2] / val_count, val_err[3] / val_count))


            list_str_for_test = []
            for k, v in decoded_dict.items():
                start_time = 0
                for wi in v:
                    tl = np.random.random() * 0.1
                    list_str_for_test.append('{} 1 {:.3f} {:.3f} {}\n'.format(k, start_time, start_time + tl,
                                                                              test_datasets.vocab.index2word[wi]))
                    start_time += tl
            tmp_prefix = str(uuid.uuid1())
            txt_file = '{:s}.txt'.format(tmp_prefix)
            result_file = os.path.join('evaluation_relaxation', txt_file)
            with open(result_file, 'w') as fid:
                fid.writelines(list_str_for_test)
            phoenix_eval_err = get_phoenix_wer(txt_file, opts.task, tmp_prefix)
            logging.info(
                '[Relaxation Evaluation] Epoch: {:d}, DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(
                    epoch, phoenix_eval_err[0], phoenix_eval_err[1], phoenix_eval_err[2], phoenix_eval_err[3]))
            return phoenix_eval_err


def main_argmax():
    opts = parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(opts.gpu)
        logging.info("Using GPU!")
        device = "cuda"
    else:
        logging.info("Using CPU!")
        device = "cpu"

    logging.info(opts)

    test_datasets = PhoenixVideo(opts.vocab_file, opts.corpus_dir, opts.video_path, phase=opts.task, DEBUG=opts.DEBUG)
    vocab_size = test_datasets.vocab.num_words
    blank_id = test_datasets.vocab.word2index['<BLANK>']
    vocabulary = Vocabulary(opts.vocab_file)
#     model = DilatedSLRNet(opts, device, vocab_size, vocabulary,
#                           dilated_channels=512, num_blocks=5, dilations=[1, 2, 4], dropout=0.0)
    model = MainStream(vocab_size)
    criterion = CtcLoss(opts, blank_id, device, reduction="none")
    trainer = Trainer(opts, model, criterion, vocabulary, vocab_size, blank_id)

    # ctcdeocde
    # ctc_decoder_vocab = [chr(x) for x in range(20000, 20000 + vocab_size)]
    # ctc_decoder = ctcdecode.CTCBeamDecoder(ctc_decoder_vocab, beam_width=opts.beam_width,
    #                                        blank_id=blank_id, num_processes=10)

    if os.path.exists(opts.check_point):
        logging.info("Loading checkpoint file from {}".format(opts.check_point))
        epoch, num_updates, loss = trainer.load_checkpoint(opts.check_point)
    else:
        logging.info("No checkpoint file in found in {}".format(opts.check_point))
        epoch, num_updates, loss = 0, 0, 0.0

    test_iter = trainer.get_batch_iterator(test_datasets, batch_size=opts.batch_size, shuffle=False)
    decoded_dict = {}
    val_err, val_correct, val_count = np.zeros([4]), 0, 0

    with open("Data/output/hypo_argmax.txt", "w") as f, open("Data/output/ref_argmax.txt", "w") as f2:
        with torch.no_grad():
            model.eval()
            criterion.eval()
            for samples in tqdm(test_iter):
                samples = trainer._prepare_sample(samples)
                video = samples["data"]
                len_video = samples["len_data"]
                label = samples["label"]
                len_label = samples["len_label"]
                video_id = samples['id']

                logits, _ = model(video, len_video)
                len_video /= 4
                # pred_seq, _, _, out_seq_len = ctc_decoder.decode(logits, len_video)
                logits = F.softmax(logits, dim=-1)
                pred_seq = logits.argmax(-1)
                out_seq_len = len_video
                start = 0
                for i, length in enumerate(len_label):
                    end = start + length
                    ref = label[start:end].tolist()
                    # hyp = [x[0] for x in groupby(pred_seq[i][0][:out_seq_len[i][0]].tolist())]
                    hyp = [x[0] for x in groupby(pred_seq[i][:out_seq_len[i].item()].tolist()) if x[0] != vocabulary.blank()]
                    ref_sent = " ".join([vocabulary.index2word[r] for r in ref])
                    hyp_sent = " ".join([vocabulary.index2word[r] for r in hyp])
                    f.write(hyp_sent + "\n")
                    f2.write(ref_sent + "\n")

                    decoded_dict[video_id[i]] = hyp
                    val_correct += int(ref == hyp)
                    err = get_wer_delsubins(ref, hyp)
                    val_err += np.array(err)
                    val_count += 1
                    start = end
                assert end == label.size(0)
            logging.info('-' * 50)
            logging.info(
                'Epoch: {:d}, DEV ACC: {:.5f}, {:d}/{:d}'.format(epoch, val_correct / val_count, val_correct, val_count))
            logging.info('Epoch: {:d}, DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(
                epoch, val_err[0] / val_count, val_err[1] / val_count, val_err[2] / val_count, val_err[3] / val_count))


            list_str_for_test = []
            for k, v in decoded_dict.items():
                start_time = 0
                for wi in v:
                    tl = np.random.random() * 0.1
                    list_str_for_test.append('{} 1 {:.3f} {:.3f} {}\n'.format(k, start_time, start_time + tl,
                                                                              test_datasets.vocab.index2word[wi]))
                    start_time += tl
            tmp_prefix = str(uuid.uuid1())
            txt_file = '{:s}.txt'.format(tmp_prefix)
            result_file = os.path.join('evaluation_relaxation', txt_file)
            with open(result_file, 'w') as fid:
                fid.writelines(list_str_for_test)
            phoenix_eval_err = get_phoenix_wer(txt_file, opts.task, tmp_prefix)
            logging.info(
                '[Relaxation Evaluation] Epoch: {:d}, DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(
                    epoch, phoenix_eval_err[0], phoenix_eval_err[1], phoenix_eval_err[2], phoenix_eval_err[3]))
            return phoenix_eval_err


if __name__ == "__main__":
    main()
    main_argmax()