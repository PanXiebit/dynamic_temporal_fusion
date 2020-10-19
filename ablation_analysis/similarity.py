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
from src.model.full_conv_v9_similarity import MainStream
import pickle


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

    test_datasets = PhoenixVideo(opts.vocab_file, opts.corpus_dir, opts.video_path, phase="train", DEBUG=opts.DEBUG)
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

    video_sim = {}

    with torch.no_grad():
        model.eval()
        criterion.eval()
        for i, samples in tqdm(enumerate(test_iter)):
            if i > 50:
                break
            samples = trainer._prepare_sample(samples)
            video = samples["data"]
            len_video = samples["len_data"]
            label = samples["label"]
            len_label = samples["len_label"]
            video_id = samples['id']

            logits, _, scores1, scores2 = model(video, len_video)
            print(scores1)
            ids = scores1.topk(k=16, dim=-1)[1].sort(-1)[0]  # [bs, t, t]
            bs, t, _ = scores1.size()
            for i in range(bs):
                for j in range(t):
                    select_id = ids[i, j, :].cpu().numpy().tolist()
                    for k in range(t):
                        if k not in select_id:
                            scores1[i, j, k] = 1e-9
            print("scores1: ", scores1)
            scores1 = scores1.softmax(-1)

            mask = scores1 > 0.02
            print(scores1, mask)
            scores1 *= mask.float()
            # sim_matrix = scores1.softmax(-1)
            # print(scores1[0, 0, :20])
            # exit()
            for i in range(len(video_id)):
                video_sim[video_id[i]] = scores1[i].cpu().numpy()
    # print(video_sim)
    with open("Data/output/sim_matrix.pkl", "wb") as f:
        pickle.dump(video_sim, f)

if __name__ == "__main__":
    main()