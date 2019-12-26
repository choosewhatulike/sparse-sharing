import argparse
import os
from fastNLP import Vocabulary, logger, DataSet
from fastNLP.io import Conll2003Pipe
from fastNLP.io import OntoNotesNERPipe
from fastNLP.io import ConllLoader
from src.task import Task
from src.utils import pdump, data_summary, set_seed
from copy import deepcopy
from fastNLP.core.utils import iob2bioes, iob2

def filter_docstart(db):
    def is_docstart(ins):
        return '-DOCSTART-' in ins['words']
    for ds in db.values():
        ds.drop(is_docstart, inplace=True)

def prepare_conll03(args):
    args.chunk = args.chunk or args.pos
    args.ner = args.ner or args.pos
    assert args.pos == args.chunk == args.ner
    pipe = Conll2003Pipe(
        chunk_encoding_type="bio", ner_encoding_type="bioes", lower=False
    )
    db = pipe.process_from_file(args.pos)
    task_lst = []
    for idx, task_name in enumerate(["pos", "chunk", "ner"]):
        task_lst.append(
            Task(
                idx,
                task_name,
                deepcopy(db.get_dataset("train")),
                deepcopy(db.get_dataset("dev")),
                deepcopy(db.get_dataset("test")),
            )
        )
    return task_lst, db.vocabs


def prepare_ontonotes(args):
    raise NotImplementedError


def prepare_ptb(args):
    datas = {}
    datas["pos"] = (
        ConllLoader(headers=["words", "pos"], indexes=[0, 1]).load(args.pos).datasets
    )
    chunk_data = (
        ConllLoader(headers=["words", "chunk"], indexes=[0, 2])
        .load(args.chunk)
        .datasets
    )
    chunk_data['train'], chunk_data['dev'] = chunk_data['train'].split(0.1)
    datas['chunk'] = chunk_data
    datas["ner"] = (
        ConllLoader(headers=["words", "ner"], indexes=[0, 3]).load(args.ner).datasets
    )

    for ds in datas['chunk'].values():
        ds.apply_field(lambda x: iob2(x), 'chunk', 'chunk')
    for ds in datas['ner'].values():
        ds.apply_field(lambda x: iob2bioes(iob2(x)), 'ner', 'ner')

    vocabs = {}
    src_vocab = Vocabulary()
    for idx, task_name in enumerate(["pos", "chunk", "ner"]):
        data = datas[task_name]
        filter_docstart(data)
        vocab = Vocabulary(padding=None, unknown=None)
        vocab.from_dataset(*list(data.values()), field_name=task_name)
        src_vocab.from_dataset(*list(data.values()), field_name="words")
        vocabs[task_name] = vocab

    task_lst = []
    for idx, task_name in enumerate(["pos", "chunk", "ner"]):
        data = datas[task_name]
        src_vocab.index_dataset(
            *list(data.values()), field_name="words", new_field_name="words"
        )
        vocabs[task_name].index_dataset(
            *list(data.values()), field_name=task_name, new_field_name=task_name
        )
        for ds in data.values():
            ds.apply_field(len, 'words', 'seq_len')
        task_lst.append(Task(idx, task_name, data["train"], data["dev"], data["test"]))
    vocabs["words"] = src_vocab
    return task_lst, vocabs


def get_args():
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument('--pos', type=str, help='raw pos data path')
    parser.add_argument('--chunk', type=str, help='raw chunk data path', default=None)
    parser.add_argument('--ner', type=str, help='raw ner data path', default=None)

    parser.add_argument('--type', choices=['conll03', 'ontonotes', 'ptb'], help='multi task data type')
    parser.add_argument('--out', type=str, default='data', help='processed data output dir')
    # fmt: on
    args = parser.parse_args()
    assert args.pos is not None
    return args


if __name__ == "__main__":
    args = get_args()
    set_seed(1)
    parse_table = {
        "conll03": prepare_conll03,
        "ontonotes": prepare_ontonotes,
        "ptb": prepare_ptb,
    }
    logger.info(args)
    assert args.type in parse_table
    task_lst, vocabs = parse_table[args.type](args)
    os.makedirs(args.out, exist_ok=True)
    data_summary(task_lst, vocabs)
    path = os.path.join(args.out, args.type + ".pkl")
    logger.info("saving data to " + path)
    pdump({"task_lst": task_lst, "vocabs": vocabs}, path)
