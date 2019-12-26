import numpy as np
import torch
import torch.cuda
import random
import os
import warnings
import hashlib
import _pickle as pickle
import logging
from fastNLP.io import EmbedLoader
from fastNLP.core.metrics import MetricBase
from fastNLP import logger
import argparse
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def md5_for_file(fn):
    md5 = hashlib.md5()
    with open(fn, "rb") as f:
        for chunk in iter(lambda: f.read(128 * md5.block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def embedding_match_vocab(
    vocab,
    emb,
    ori_vocab,
    dtype=np.float32,
    padding="<pad>",
    unknown="<unk>",
    normalize=True,
    error="ignore",
    init_method=None,
):
    dim = emb.shape[-1]
    matrix = np.random.randn(len(vocab), dim).astype(dtype)
    hit_flags = np.zeros(len(vocab), dtype=bool)

    if init_method:
        matrix = init_method(matrix)
    for word, idx in ori_vocab.word2idx.items():
        try:
            if word == padding and vocab.padding is not None:
                word = vocab.padding
            elif word == unknown and vocab.unknown is not None:
                word = vocab.unknown
            if word in vocab:
                index = vocab.to_index(word)
                matrix[index] = emb[idx]
                hit_flags[index] = True
        except Exception as e:
            if error == "ignore":
                warnings.warn("Error occurred at the {} line.".format(idx))
            else:
                print("Error occurred at the {} line.".format(idx))
                raise e

    total_hits = np.sum(hit_flags)
    print(
        "Found {} out of {} words in the pre-training embedding.".format(
            total_hits, len(vocab)
        )
    )
    if init_method is None:
        found_vectors = matrix[hit_flags]
        if len(found_vectors) != 0:
            mean = np.mean(found_vectors, axis=0, keepdims=True)
            std = np.std(found_vectors, axis=0, keepdims=True)
            unfound_vec_num = len(vocab) - total_hits
            r_vecs = np.random.randn(unfound_vec_num, dim).astype(dtype) * std + mean
            matrix[hit_flags == False] = r_vecs

    if normalize:
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    return matrix


def embedding_load_with_cache(emb_file, cache_dir, vocab, **kwargs):
    def match_cache(file, cache_dir):
        md5 = md5_for_file(file)
        cache_files = os.listdir(cache_dir)
        for fn in cache_files:
            if md5 in fn.split("-")[-1]:
                return os.path.join(cache_dir, fn), True
        return (
            "{}-{}.pkl".format(os.path.join(cache_dir, os.path.basename(file)), md5),
            False,
        )

    def get_cache(file):
        if not os.path.exists(file):
            return None
        with open(file, "rb") as f:
            emb = pickle.load(f)
        return emb

    os.makedirs(cache_dir, exist_ok=True)
    cache_fn, match = match_cache(emb_file, cache_dir)
    if not match:
        print("cache missed, re-generating cache at {}".format(cache_fn))
        emb, ori_vocab = EmbedLoader.load_without_vocab(
            emb_file, padding=None, unknown=None, normalize=False
        )
        with open(cache_fn, "wb") as f:
            pickle.dump((emb, ori_vocab), f)

    else:
        print("cache matched at {}".format(cache_fn))

    # use cache
    print("loading embeddings ...")
    emb = get_cache(cache_fn)
    assert emb is not None
    return embedding_match_vocab(vocab, emb[0], emb[1], **kwargs)


class MetricInForward(MetricBase):
    def __init__(self, val_name="loss"):
        super().__init__()
        self._init_param_map(value=val_name)
        self.total_val = 0.0
        self.num_val = 0
        self.val_name = val_name

    def evaluate(self, value):
        self.total_val += value.item()
        self.num_val += 1

    def get_metric(self, reset=True):
        result = {
            "total_{}".format(self.val_name): round(self.total_val, 5),
            "avg_{}".format(self.val_name): round(self.total_val / self.num_val, 5),
        }
        if reset:
            self.total_val = 0.0
            self.num_val = 0
        return result


def init_logger(log_path, level="info"):
    if isinstance(level, int):
        level_int = level
    else:
        level = level.lower()
        level_int = {
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "warn": logging.WARN,
            "warning": logging.WARN,
            "error": logging.ERROR,
        }[level]

    dirname = os.path.abspath(os.path.dirname(log_path))
    os.makedirs(dirname, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level_int,
    )


def get_logger(name):
    # return logging.getLogger(name)
    return logger


def pdump(obj, fn):
    with open(fn, "wb") as f:
        pickle.dump(obj, f)


def pload(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)


def find_task(task_id, task_lst):
    if task_lst[task_id].task_id == task_id:
        return task_lst[task_id]
    for task in task_lst:
        if task_id == task.task_id:
            return task
    raise RuntimeError("Cannot find task with task_id={}.".format(task_id))


def get_data(data_dir):
    DB = pload(data_dir)
    task_lst = DB["task_lst"]
    vocabs = DB["vocabs"]
    task_lst = [init_task(task) for task in task_lst]
    return task_lst, vocabs


def init_task(task):
    def find_rename_field(ds, names, new_name):
        for n in names:
            if ds.has_field(n):
                ds.rename_field(n, new_name)
                break

    task_name = task.task_name
    for ds in [task.train_set, task.dev_set, task.test_set]:
        if not ds.has_field("task_id"):
            ds.apply_field(lambda x: task.task_id, "words", "task_id")
        else:
            assert ds[0]["task_id"] == task.task_id

        find_rename_field(ds, ("words", "words_idx"), "x")

        y_names = ("label", "target", task_name)
        find_rename_field(ds, y_names, "y")

        ds.set_input("x", "y", "task_id")
        ds.set_target("y")
        ds.set_pad_val("x", 0)
        ds.set_pad_val("y", 0)

        if task_name in ["ner", "chunk"] or "pos" in task_name:
            ds.set_input("seq_len")
            ds.set_target("seq_len")
    return task


def get_default_parser():
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--seed", dest="seed", type=int, default=42, help="Random seed to set")

    parser.add_argument("--arch", type=str, required=True, help='The architecture of model')

    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--print_every', type=int, default=23, help='the logging intervals')
    parser.add_argument('--exp_name', type=str, default='exp1', help='the experiment name')
    parser.add_argument('--data_path', type=str, required=True, help='the processed data path')
    parser.add_argument('--save_dir', type=str, default='/exp/saved_models')

    parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate the model')
    parser.add_argument('--save_ep', action='store_true', default=False, help='save model when every epoch ends')

    parser.add_argument("--epochs", default=10, dest="epochs", type=int,
                        help="Number of full passes through training set")
    parser.add_argument("--batch_size", default=10, dest="batch_size", type=int,
                        help="Minibatch size of training set")

    parser.add_argument('--init_weights', type=str, default=None, help='init weights(checkpoints) for training')

    parser.add_argument('--hidden_size', type=int, default=200, help='hidden size of encoder')
    parser.add_argument('--n_layer', type=int, default=1, help='num of layers of encoder')
    parser.add_argument('--crf', action='store_true', default=False, help='whether use crf')
    parser.add_argument("--dropout", dest='dropout', type=float, default=0.5, help='the dropout probability')

    parser.add_argument("--optim", type=str, default='sgd(lr=0.1, momentum=0.9)', help='optimizer to use')
    parser.add_argument("--scheduler", type=str, default='fix', help='scheduler to use')
    parser.add_argument("--debug", dest='debug', action='store_true', default=False, help='whether to enter debug mode')
    # fmt: on
    return parser


def load_model(model, path):
    dumps = torch.load(path, map_location="cpu")

    if model is None:
        assert isinstance(dumps, nn.Module), "model is None but load %s" % type(dumps)
        model = dumps
    else:
        if isinstance(dumps, nn.Module):
            dumps = dumps.state_dict()
        else:
            assert isinstance(dumps, dict), type(dumps)
        res = model.load_state_dict(dumps, strict=False)
        assert len(res.unexpected_keys) == 0, res.unexpected_keys
        logger.info("missing keys in init-weights %s", res.missing_keys)
    logger.info("load init-weights from %s", path)
    return model


def need_acc(task_name):
    return task_name not in ("ner", "chunk")


def init_prog(args):
    set_seed(args.seed)

    args.log_path = os.path.join(args.save_dir, "log", args.exp_name)
    args.save_path = os.path.join(args.save_dir, "cp", args.exp_name)
    args.tb_path = os.path.join(args.save_dir, "tb", args.exp_name)
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.tb_path, exist_ok=True)

    if args.evaluate:
        logger.add_file(os.path.join(args.log_path, "eval.log"))
    else:
        logger.add_file(os.path.join(args.log_path, "train.log"))


def num_parameters(model):
    sum_params = 0
    for name, param in model.named_parameters():
        logger.info("{}: {}".format(name, param.shape))
        sum_params += param.numel()
    return sum_params


def parse_dict_args(dict_str):
    p = dict_str.strip()
    l = p.find("(")
    r = p.find(")")
    if l == -1:
        return dict_str, {}
    name = p[:l].strip().lower()
    res_dict = eval("dict({})".format(p[l + 1 : r]))
    return name, res_dict


def get_optim(optim_str, params):
    name, optim_args = parse_dict_args(optim_str)
    if name == "sgd":
        return SGD(params, **optim_args)
    elif name == "adam":
        return Adam(params, **optim_args)
    else:
        raise ValueError(optim_str)


def get_lr(optimizer):
    lr = -1.0
    for pg in optimizer.param_groups:
        lr = max(pg["lr"], lr)
    return lr


def get_scheduler(args, optimizer):
    name, args = parse_dict_args(args.scheduler)
    if name == "fix":
        return None
    elif name == "inverse_sqrt":
        warmup = args.get('warmup', 4000)
        lr = get_lr(optimizer)
        lr_step = lr / warmup
        decay = lr * warmup ** 0.5

        def warm_decay(n):
            if n < warmup:
                return lr_step * n
            return decay * n ** -0.5

        return LambdaLR(optimizer, warm_decay)
    elif name == 'decay':
        return LambdaLR(optimizer, lambda ep: 1 / (1 + 0.05 * ep))

def data_summary(task_lst, vocabs=None):
    logger.info("******** DATA SUMMARY ********")
    logger.info("Contain {} tasks".format(len(task_lst)))
    for task in task_lst:
        logger.info(
            "Task {}: {},\tnum of samples: train {}, dev {}, test {}".format(
                task.task_id,
                task.task_name,
                len(task.train_set),
                len(task.dev_set),
                len(task.test_set),
            )
        )
    if vocabs is None:
        return
    logger.info("Contain {} vocabs".format(len(vocabs)))
    for name, v in vocabs.items():
        logger.info("Vocab {}: has length {},\t{}".format(name, len(v), v))
