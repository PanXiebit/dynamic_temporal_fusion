
import argparse

def parse_args():
    p = argparse.ArgumentParser(description='SLR')
    p.add_argument('-t', '--task', type=str, default='train')
    p.add_argument('-g', '--gpu', type=int, default=0)
    p.add_argument('-sd', '--seed', type=int, default=8)

    # data
    p.add_argument('-dw', '--data_worker', type=int, default=32)
    p.add_argument('-fd', '--feature_dim', type=int, default=512)
    p.add_argument('-corp_dir', '--corpus_dir', type=str, default='Data/slr-phoenix14')
    p.add_argument('-voc_fl', '--vocab_file', type=str, default='Data/slr-phoenix14/newtrainingClasses.txt')
    p.add_argument('-corp_tr', '--corpus_train', type=str, default='Data/slr-phoenix14/train.corpus.csv')
    p.add_argument('-corp_te', '--corpus_test', type=str, default='Data/slr-phoenix14/test.corpus.csv')
    p.add_argument('-corp_de', '--corpus_dev', type=str, default='Data/slr-phoenix14/dev.corpus.csv')
    p.add_argument('-vp', '--video_path', type=str, default='../c3d_res_phoenix_body_iter5_120k')

    p.add_argument('--stage_epoch', type=int, default=10)
    p.add_argument('--dropout', type=float, default=0.3,
                   help='dropout probability')

    # optimizer
    p.add_argument('-op', '--optimizer', type=str, default='adam')
    p.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    p.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    p.add_argument('-mt', '--momentum', type=float, default=0.9)
    p.add_argument('-nepoch', '--max_epoch', type=int, default=1000)
    p.add_argument('-mupdates', '--max_updates', type=int, default=1e7)
    p.add_argument('-us', '--update_step', type=int, default=1)
    p.add_argument('-upm', '--update_param', type=str, default='all')

    # train
    p.add_argument('-db', '--DEBUG', type=bool, default=False)
    p.add_argument('-lg_d', '--log_dir', type=str, default='./log/debug')
    p.add_argument('-bs', '--batch_size', type=int, default=20)
    p.add_argument('-ckpt', '--check_point', type=str, default='')
    p.add_argument('-ps', '--print_step', type=int, default=20)
    p.add_argument('-siu', '--save_interval_updates', type=int, default=100)
    # test (decoding)
    p.add_argument('-bwd', '--beam_width', type=int, default=5)
    p.add_argument('-vbs', '--valid_batch_size', type=int, default=1)
    p.add_argument('-evalset', '--eval_set', type=str, default='test', choices=['test', 'dev'])

    parameter = p.parse_args()
    return parameter


if __name__ == "__main__":
    opts = parse_args()
    print(opts)