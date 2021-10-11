import argparse
import importlib
import os
import glob


def define_arguments(parser):
    parser.add_argument('--root', type=str, default="./data", help="")
    parser.add_argument('--batch-size', type=int, default=4, help="")
    parser.add_argument('--num-workers', type=int, default=0, help="")
    parser.add_argument('--no-gpu', action="store_true", help="don't use gpu")
    parser.add_argument('--gpu', type=str, default='0', help="gpu")
    parser.add_argument('--max-grad-norm', type=float, default=1, help="")
    parser.add_argument('--learning-rate', type=float, default=1e-5, help="")
    parser.add_argument('--decay', type=float, default=1e-2, help="")
    parser.add_argument('--warmup-step', type=float, default=1200, help="")
    parser.add_argument('--seed', type=int, default=44739242, help="random seed")
    parser.add_argument('--log-dir', type=str, default="./log/", help="path to save log file")
    parser.add_argument('--model-name', type=str, default="bert-large-cased", help="pretrained lm name")
    parser.add_argument('--train-epoch', type=int, default=25, help='epochs to train')
    parser.add_argument('--test-only', action="store_true", help='is testing')
    parser.add_argument('--clean-log-dir', action="store_true", help='is testing')


def parse_arguments():
    parser = argparse.ArgumentParser()
    define_arguments(parser)
    args = parser.parse_args()
    args.log = os.path.join(args.log_dir, f"logfile.log")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if args.clean_log_dir and (not args.test_only) and os.path.exists(args.log_dir):
        existing_logs = glob.glob(os.path.join(args.log_dir, "*"))
        for _t in existing_logs:
            os.remove(_t)
    return args