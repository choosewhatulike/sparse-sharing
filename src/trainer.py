import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import fastNLP
from fastNLP import AccuracyMetric, SpanFPreRecMetric
# from src.metric import YangJieSpanMetric as SpanFPreRecMetric
from fastNLP.core.batch import DataSetIter
from fastNLP.core.sampler import RandomSampler, BucketSampler
from src.utils import find_task, get_scheduler

g_metric = None


def has_acc(task_name):
    return task_name not in ("chunk", "ner")


def format_metric(dev_acc):
    res = {}
    for name, val in dev_acc.items():
        if name == "avg":
            res["AVG"] = val
        else:
            if has_acc(name):
                res[name] = val["acc"]
            else:
                res[name] = val["f"]
    return res


def get_trainer_cls(args):
    if args.trainer == 'seq-label':
        return SeqLabelTrainer
    elif args.trainer == 're-seq-label':
        return ReSampleSeqLabelTrainer
    else:
        raise ValueError(args.trainer)


class SeqLabelTrainer(object):
    def __init__(self, masker, task_lst, vocabs, optimizer, args):
        """
        :param model: 模型
        :param description: 模型描述
        :param task_lst: 任务列表
        :param optimizer: 优化器
        :param log_path: TensorboardX存储文件夹
        :param save_path: 模型存储位置
        :param accumulation_steps: 累积梯度
        :param print_every: 评估间隔
        """
        self.logger = fastNLP.logger

        self.masker = masker
        self.task_lst = task_lst
        self.save_path = args.save_path
        self.description = args.exp_name
        self.optim = optimizer
        self.vocabs = vocabs
        n_steps = (
            int(len(task_lst) * len(task_lst[0].train_set) * 100 / args.batch_size) + 1
        )
        args.n_steps = n_steps
        self.epoch_scheduler = get_scheduler(args, self.optim)
        self.scheduler = None
        self.logger.info('Using scheduler {}'.format(self.scheduler))
        self.accumulation_steps = args.accumulation_steps
        self.print_every = args.print_every
        self.batch_size = args.batch_size
        self.save_ep = args.save_ep

        include_tasks = args.tasks
        if include_tasks is None:
            self.empty_tasks = set()
        else:
            self.empty_tasks = set(range(len(self.task_lst))) - set(include_tasks)

        self.steps = 0
        self.best_acc = 0
        self.best_epoch = 0

        self.metrics = []
        for t in task_lst:
            if has_acc(t.task_name):
                self.metrics.append(AccuracyMetric())
            else:
                self.metrics.append(
                    SpanFPreRecMetric(
                        self.vocabs[t.task_name],
                        encoding_type="bioes" if t.task_name == "ner" else "bio",
                    )
                )
        # self.logger.info(self.metrics)

        tb_path = "eval" if args.evaluate else "train"
        self.summary_writer = SummaryWriter(os.path.join(args.tb_path, tb_path))

    def train(self, n_epoch):
        self.model = self.masker.model

        total_time = time.time()
        self.logger.info("Start training...")
        for i_epoch in range(n_epoch):
            start_time = time.time()
            self.cur_epoch = i_epoch
            self.logger.info("Epoch {}".format(i_epoch))
            self._train_epoch()
            self.logger.info(
                "Epoch {} finished. Elapse: {:.3f}s.".format(
                    i_epoch, time.time() - start_time
                )
            )

            dev_loss, dev_acc = self._eval_epoch(dev=False)
            self._dump_model_state("%d.th" % i_epoch)
            self.summary_writer.add_scalar("dev_loss", dev_loss, i_epoch)
            if "chunk" in dev_acc:
                self.summary_writer.add_scalars(
                    "dev_acc",
                    {
                        "AVG": dev_acc["avg"],
                        "pos": dev_acc["pos"]["acc"],
                        "chunk": dev_acc["chunk"]["f"],
                        "ner": dev_acc["ner"]["f"],
                    },
                    i_epoch,
                )
            else:
                self.summary_writer.add_scalars(
                    "dev_acc", format_metric(dev_acc), i_epoch
                )
            eval_str = "Validation loss {}, avg acc {:.4f}%".format(
                dev_loss, dev_acc["avg"]
            )
            for task, value in dev_acc.items():
                if has_acc(task) and task != "avg":
                    eval_str += ", {} acc {:.4f}%".format(task, value["acc"])
                elif task != "avg":
                    eval_str += ", {} f1 {:.4f}%".format(task, value["f"])
            self.logger.info(eval_str)

            if dev_acc["avg"] > self.best_acc:
                self.best_acc = dev_acc["avg"]
                self.best_epoch = i_epoch
                self.logger.info("Updating best model...")
                self._save_model()
                self.logger.info("Model saved.")

            self.logger.info(
                "Current best acc [{:.4f}%] occured at epoch [{}].".format(
                    self.best_acc, self.best_epoch
                )
            )
        self.logger.info(
            "Training finished. Elapse {:.4f} hours.".format(
                (time.time() - total_time) / 3600
            )
        )

    def _train_epoch(self):

        total_loss = 0
        corrects, samples = 0, 0

        n_tasks = len(self.task_lst)
        task_seq = list(np.random.permutation(n_tasks))
        empty_task = copy.deepcopy(self.empty_tasks)
        self.model.train()
        self.model.zero_grad()
        for task in self.task_lst:
            task.train_data_loader = iter(task.train_data_loader)
        while len(empty_task) < n_tasks:
            for task_id in task_seq:
                if task_id in empty_task:
                    continue
                task = find_task(task_id, self.task_lst)
                batch = next(task.train_data_loader, None)
                if batch is None:
                    empty_task.add(task_id)
                    task.train_data_loader = DataSetIter(
                        task.train_set,
                        self.batch_size,
                        sampler=BucketSampler(batch_size=self.batch_size),
                    )
                    continue
                x, y = batch
                batch_task_id = x["task_id"].cuda()
                batch_x = x["x"].cuda()
                batch_y = y["y"].cuda()

                self.masker.before_forward(batch_task_id[0].item())
                if "seq_len" in x:
                    seq_len = x["seq_len"].cuda()
                    out = self.model(batch_task_id, batch_x, batch_y, seq_len)
                else:
                    seq_len = None
                    out = self.model(batch_task_id, batch_x, batch_y)
                loss, pred = out["loss"], out["pred"]
                self.steps += 1

                total_loss += loss.item()
                loss = loss / self.accumulation_steps
                loss.backward()
                self.masker.after_forward(batch_task_id[0].item())
                self.metrics[task_id].evaluate(pred, batch_y, seq_len)

                if self.steps % self.accumulation_steps == 0:
                    nn.utils.clip_grad_value_(self.model.parameters(), 5)

                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optim.step()
                    self.optim.zero_grad()

                if self.steps % self.print_every == 0:
                    self.summary_writer.add_scalar(
                        "train_loss", total_loss / self.print_every, self.steps
                    )
                    score = self.metrics[task_id].get_metric()
                    metric_name = "acc" if "acc" in score else "f1"
                    score = score["acc"] if "acc" in score else score["f"]
                    self.summary_writer.add_scalar("train_acc", score, self.steps)
                    self.logger.info(
                        " - Step {}: loss {}\t{}\t{}: {}".format(
                            self.steps,
                            total_loss / self.print_every,
                            task.task_name,
                            metric_name,
                            score,
                        )
                    )
                    total_loss = 0
                    # corrects, samples = 0, 0
        if self.epoch_scheduler is not None:
            self.epoch_scheduler.step()

    def _eval_epoch(self, dev=True):
        self.logger.info("Evaluating...")
        dev_loss = 0
        e_steps = 0
        avg_acc = 0
        dev_acc = {}
        self.model = self.masker.model
        self.model.eval()
        metrics = []
        for task in self.task_lst:
            if has_acc(task.task_name):
                metrics.append(fastNLP.AccuracyMetric())
            else:
                metrics.append(
                    SpanFPreRecMetric(
                        self.vocabs[task.task_name],
                        encoding_type="bioes" if task.task_name == "ner" else "bio",
                    )
                )

        with torch.no_grad():
            for i in range(len(self.task_lst)):
                corrects, samples = 0, 0
                task = find_task(i, self.task_lst)
                if task.task_id in self.empty_tasks:
                    continue
                if dev:
                    data_loader = task.dev_data_loader
                else:
                    data_loader = task.test_data_loader
                for batch in data_loader:
                    x, y = batch
                    batch_task_id = x["task_id"].cuda()
                    batch_x = x["x"].cuda()
                    batch_y = y["y"].cuda()
                    if "seq_len" in x:
                        seq_len = x["seq_len"].cuda()
                    else:
                        seq_len = None

                    self.masker.before_forward(batch_task_id[0].item())
                    # loss, pred = self.model(batch_task_id, batch_x, batch_y, seq_len)
                    if seq_len is not None:
                        out = self.model(batch_task_id, batch_x, batch_y, seq_len)
                    else:
                        out = self.model(batch_task_id, batch_x, batch_y)
                    loss, pred = out["loss"], out["pred"]
                    self.masker.after_forward(batch_task_id[0].item())

                    dev_loss += loss.item()
                    e_steps += 1

                    metrics[i].evaluate(pred, batch_y, seq_len)

                    samples += batch_x.size(0)

            for i in range(len(self.task_lst)):
                task = find_task(i, self.task_lst)
                eval_res = metrics[i].get_metric()
                dev_acc[task.task_name] = eval_res
                avg_acc += eval_res["acc"] if "acc" in eval_res else eval_res["f"]

        avg_acc /= len(self.task_lst) - len(self.empty_tasks)
        dev_acc["avg"] = avg_acc
        dev_loss = dev_loss / e_steps
        return dev_loss, dev_acc

    def _save_model(self):
        save_path = os.path.join(self.save_path, "best.th")
        torch.save(self.model.state_dict(), save_path)

    def _dump_model_state(self, name):
        if not self.save_ep:
            return
        save_path = os.path.join(self.save_path, "models", name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)


class ReSampleSeqLabelTrainer(SeqLabelTrainer):
    def __init__(self, masker, task_lst, vocabs, optimizer, args):
        """
        :param model: 模型
        :param description: 模型描述
        :param task_lst: 任务列表
        :param optimizer: 优化器
        :param log_path: TensorboardX存储文件夹
        :param save_path: 模型存储位置
        :param accumulation_steps: 累积梯度
        :param print_every: 评估间隔
        """
        super().__init__(masker, task_lst, vocabs, optimizer, args)

        self.n_steps_per_epoch = 0
        for t in task_lst:
            self.n_steps_per_epoch += math.ceil(
                len(t.train_set) * 1.0 / args.batch_size
            )
        self.logger.info(
            "use ReSampled Trainer, steps per epoch %d", self.n_steps_per_epoch
        )
        for task in self.task_lst:
            if task.task_id not in self.empty_tasks:
                task.train_data_loader = iter(task.train_data_loader)

    def _train_epoch(self):
        total_loss = 0
        corrects, samples = 0, 0

        n_tasks = len(self.task_lst)
        task_seq = list(np.random.permutation(n_tasks))
        empty_task = copy.deepcopy(self.empty_tasks)
        self.model.train()
        self.model.zero_grad()

        for cur_step in range(self.n_steps_per_epoch):
            for task_id in task_seq:
                if task_id in empty_task:
                    continue
                task = find_task(task_id, self.task_lst)
                batch = next(task.train_data_loader, None)
                if batch is None:
                    # empty_task.add(task_id)
                    task.train_data_loader = DataSetIter(
                        task.train_set, self.batch_size, sampler=RandomSampler()
                    )
                    task.train_data_loader = iter(task.train_data_loader)
                    continue
                x, y = batch
                batch_task_id = x["task_id"].cuda()
                batch_x = x["x"].cuda()
                batch_y = y["y"].cuda()

                self.masker.before_forward(batch_task_id[0].item())
                if "seq_len" in x:
                    seq_len = x["seq_len"].cuda()
                    out = self.model(batch_task_id, batch_x, batch_y, seq_len)
                else:
                    seq_len = None
                    out = self.model(batch_task_id, batch_x, batch_y)
                loss, pred = out["loss"], out["pred"]
                self.steps += 1

                total_loss += loss.item()
                loss = loss / self.accumulation_steps
                loss.backward()
                self.masker.after_forward(batch_task_id[0].item())
                self.metrics[task_id].evaluate(pred, batch_y, seq_len)

                if self.steps % self.accumulation_steps == 0:
                    nn.utils.clip_grad_value_(self.model.parameters(), 5)

                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optim.step()
                    self.optim.zero_grad()

                if self.steps % self.print_every == 0:
                    self.summary_writer.add_scalar(
                        "train_loss", total_loss / self.print_every, self.steps
                    )
                    score = self.metrics[task_id].get_metric()
                    metric_name = "acc" if "acc" in score else "f1"
                    score = score["acc"] if "acc" in score else score["f"]
                    self.summary_writer.add_scalar("train_acc", score, self.steps)
                    self.logger.info(
                        " - Step {}: loss {}\t{}\t{}: {}".format(
                            self.steps,
                            total_loss / self.print_every,
                            task.task_name,
                            metric_name,
                            score,
                        )
                    )
                    total_loss = 0
        if self.epoch_scheduler is not None:
            self.epoch_scheduler.step()
