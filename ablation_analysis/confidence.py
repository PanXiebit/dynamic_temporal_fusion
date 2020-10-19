import torch
from config.options import parse_args
from src.data.video_lang_datasets import PhoenixVideo
from utils import init_logging, LossManager, ModelManager
import os
from src.criterion.ctc_loss import CtcLoss
from src.trainer import Trainer
import logging
from tqdm import tqdm
from src.data.vocabulary import Vocabulary
import ctcdecode
from src.model.full_conv_v9 import MainStream
import pickle
from ablation_analysis.alignment import get_alignment

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    opts = parse_args()
    init_logging(os.path.join(opts.log_dir, '{:s}_log.txt'.format(opts.task)))

    if torch.cuda.is_available():
        torch.cuda.set_device(opts.gpu)
        logging.info("Using GPU!")
        device = "cuda"
    else:
        logging.info("Using CPU!")
        device = "cpu"

    logging.info(opts)

    test_datasets = PhoenixVideo(opts.vocab_file, opts.corpus_dir, opts.video_path, phase="train", DEBUG=opts.DEBUG, sample=False)
    vocab_size = test_datasets.vocab.num_words
    blank_id = test_datasets.vocab.word2index['<BLANK>']
    pad_id = test_datasets.vocab.pad()
    vocabulary = Vocabulary(opts.vocab_file)
    # model = DilatedSLRNet(opts, device, vocab_size, vocabulary,
    #                       dilated_channels=512, num_blocks=5, dilations=[1, 2, 4], dropout=0.0)
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

    with torch.no_grad():
        model.eval()
        criterion.eval()
        prob_results = {}
        for i, samples in enumerate(test_iter):
            if i > 500:
                break
            samples = trainer._prepare_sample(samples)
            video = samples["data"]
            len_video = samples["len_data"]
            label = samples["label"]
            len_label = samples["len_label"]
            video_id = samples['id']
            dec_label = samples["decoder_label"]
            len_dec_label = samples["len_decoder_label"]

            # print("video: ", video.shape)
            logits, _ = model(video, len_video)
            len_video /= 4
            # print("logits: ", logits.shape)
            # print(len_video)

            params = logits[0, :len_video[0], :].transpose(1, 0).detach().cpu().numpy()  # [T, vocab_size]
            seq = dec_label[0, :len_dec_label[0]].cpu().numpy()
            alignment = get_alignment(params, seq, blank=blank_id, is_prob=False)  # [length]
            # print("video_id:", video_id[0])
            # print("gt label:", seq)
            # print("alignment:", alignment)

            probs = logits.softmax(-1)[0]  # [length ,vocab_size]
            align_probs = []
            for i in range(alignment.shape[0]):
                align_probs.append(probs[i, alignment[i]].detach().cpu().numpy().tolist())
            # print(align_probs)
            # exit()
            count = 0
            total_cnt = 0
            for i in range(len(align_probs)):
                total_cnt += 1
                if alignment[i] == blank_id:
                    align_probs[i] = 0
                    count += 1
            print("video_id: {}, and blank count / total count: {}/{} = {:.4f}".format(video_id[0], count, total_cnt, count/total_cnt))
            prob_results[video_id[0]] = (align_probs, alignment)
            # print(align_probs)
    return prob_results
            # exit()
            # backtrace to get the label


if __name__ == "__main__":
    prob_results = main()
    import pickle
    with open("ablation_analysis/prob_align_v9.pkl", "wb") as f:
        pickle.dump(prob_results, f)

    # import pickle
    # with open("ablation_analysis/prob_align_v9.pkl", "rb") as f:
    #     align_probs_v9 = pickle.load(f)
    # print(len(align_probs_v9))
    # with open("ablation_analysis/prob_align_v0.pkl", "rb") as f:
    #     align_probs_v0 = pickle.load(f)
    # print(len(align_probs_v0))
    # f_cnt = 0
    # t_cnt = 0
    # for id, prob in align_probs_v0.items():
    #     print(id)
    #     print(prob)
    #     print(align_probs_v9[id])
    #     if sum(prob) > sum(align_probs_v9[id]):
    #         t_cnt += 1
    #     else:
    #         f_cnt += 1
    # print(t_cnt, f_cnt)
    # import matplotlib as mpl
    #

