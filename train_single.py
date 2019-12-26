import os
import fastNLP
from fastNLP import (
    Trainer,
    Tester,
    Callback,
    LRScheduler,
    LossInForward,
    AccuracyMetric,
    SpanFPreRecMetric,
    GradientClipCallback,
    logger,
)
from src.metric import YangJieSpanMetric
from src import utils
from src.utils import MetricInForward, get_optim
from src.models import get_model
from torch.optim.lr_scheduler import LambdaLR
import torch
from tensorboardX import SummaryWriter
from src.prune import Pruning


class LogCallback(Callback):
    def __init__(self, path, print_every=50):
        super().__init__()
        self._log = utils.get_logger(__name__)
        self.avg_loss = 0.0
        self.print_every = print_every
        self.writer = SummaryWriter(log_dir=path)

    def on_backward_begin(self, loss):
        self.avg_loss += loss.item()
        if self.print_every > 0 and self.step % self.print_every == 0:
            self.writer.add_scalar("loss", self.avg_loss, self.step)
            self.avg_loss = 0.0

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        scalars = {}
        for res in eval_result.values():
            if "acc" in res:
                scalars.update(res)
            elif "f" in res:
                scalars.update(res)
        self.writer.add_scalars("dev acc", scalars, self.epoch)


class LRStep(Callback):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def on_batch_end(self):
        self.scheduler.step()


SEQ_LABEL_TASK = {"pos", "chunk", "ner"}


def get_metric(res):
    if "acc" in res:
        return "acc", res["acc"]
    elif "f" in res:
        return "f", res["f"]
    for n, v in res.items():
        if isinstance(v, dict):
            ans = get_metric(v)
            if ans is not None:
                return ans
    return None


def train_mlt_single(args):
    global logger
    logger.info(args)
    task_lst, vocabs = utils.get_data(args.data_path)
    task_db = task_lst[args.task_id]
    train_data = task_db.train_set
    dev_data = task_db.dev_set
    test_data = task_db.test_set
    task_name = task_db.task_name

    if args.debug:
        train_data = train_data[:200]
        dev_data = dev_data[:200]
        test_data = test_data[:200]
        args.epochs = 3
        args.pruning_iter = 3

    summary_writer = SummaryWriter(
        log_dir=os.path.join(args.tb_path, "global/%s" % task_name)
    )

    logger.info("task name: {}, task id: {}".format(task_db.task_name, task_db.task_id))
    logger.info(
        "train len {}, dev len {}, test len {}".format(
            len(train_data), len(dev_data), len(test_data)
        )
    )

    # init model
    model = get_model(args, task_lst, vocabs)

    logger.info("model: \n{}".format(model))
    if args.init_weights is not None:
        utils.load_model(model, args.init_weights)

    if utils.need_acc(task_name):
        metrics = [AccuracyMetric(target="y"), MetricInForward(val_name="loss")]
        metric_key = "acc"

    else:
        metrics = [
            YangJieSpanMetric(
                tag_vocab=vocabs[task_name],
                pred="pred",
                target="y",
                seq_len="seq_len",
                encoding_type="bioes" if task_name == "ner" else "bio",
            ),
            MetricInForward(val_name="loss"),
        ]
        metric_key = "f"
    logger.info(metrics)

    need_cut_names = list(set([s.strip() for s in args.need_cut.split(",")]))
    prune_names = []
    for name, p in model.named_parameters():
        if not p.requires_grad or "bias" in name:
            continue
        for n in need_cut_names:
            if n in name:
                prune_names.append(name)
                break

    # get Pruning class
    pruner = Pruning(
        model, prune_names, final_rate=args.final_rate, pruning_iter=args.pruning_iter
    )
    if args.init_masks is not None:
        pruner.load(args.init_masks)
        pruner.apply_mask(pruner.remain_mask, pruner._model)
    # save checkpoint
    os.makedirs(args.save_path, exist_ok=True)

    logger.info('Saving init-weights to {}'.format(args.save_path))
    torch.save(
        model.cpu().state_dict(), os.path.join(args.save_path, "init_weights.th")
    )
    torch.save(args, os.path.join(args.save_path, "args.th"))
    # start training and pruning
    summary_writer.add_scalar("remain_rate", 100.0, 0)
    summary_writer.add_scalar("cutoff", 0.0, 0)

    if args.init_weights is not None:
        init_tester = Tester(
            test_data,
            model,
            metrics=metrics,
            batch_size=args.batch_size,
            num_workers=4,
            device="cuda",
            use_tqdm=False,
        )
        res = init_tester.test()
        logger.info("No init testing, Result: {}".format(res))
        del res, init_tester

    for prune_step in range(pruner.pruning_iter + 1):
        # reset optimizer every time
        optim_params = [p for p in model.parameters() if p.requires_grad]
        # utils.get_logger(__name__).debug(optim_params)
        utils.get_logger(__name__).debug(len(optim_params))
        optimizer = get_optim(args.optim, optim_params)
        # optimizer = TriOptim(optimizer, args.n_filters, args.warmup, args.decay)
        factor = pruner.cur_rate / 100.0
        factor = 1.0
        # print(factor, pruner.cur_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = factor * pg["lr"]
        utils.get_logger(__name__).info(optimizer)

        trainer = Trainer(
            train_data,
            model,
            loss=LossInForward(),
            optimizer=optimizer,
            metric_key=metric_key,
            metrics=metrics,
            print_every=200,
            batch_size=args.batch_size,
            num_workers=4,
            n_epochs=args.epochs,
            dev_data=dev_data,
            save_path=None,
            sampler=fastNLP.BucketSampler(batch_size=args.batch_size),
            callbacks=[
                pruner,
                # LRStep(lstm.WarmupLinearSchedule(optimizer, args.warmup, int(len(train_data)/args.batch_size*args.epochs)))
                GradientClipCallback(clip_type="norm", clip_value=5),
                LRScheduler(
                    lr_scheduler=LambdaLR(optimizer, lambda ep: 1 / (1 + 0.05 * ep))
                ),
                LogCallback(path=os.path.join(args.tb_path, "No", str(prune_step))),
            ],
            use_tqdm=False,
            device="cuda",
            check_code_level=-1,
        )
        res = trainer.train()
        logger.info("No #{} training, Result: {}".format(pruner.prune_times, res))
        name, val = get_metric(res)
        summary_writer.add_scalar("prunning_dev_acc", val, prune_step)
        tester = Tester(
            test_data,
            model,
            metrics=metrics,
            batch_size=args.batch_size,
            num_workers=4,
            device="cuda",
            use_tqdm=False,
        )
        res = tester.test()
        logger.info("No #{} testing, Result: {}".format(pruner.prune_times, res))
        name, val = get_metric(res)
        summary_writer.add_scalar("pruning_test_acc", val, prune_step)

        # prune and save
        torch.save(
            model.state_dict(),
            os.path.join(
                args.save_path,
                "best_{}_{}.th".format(pruner.prune_times, pruner.cur_rate),
            ),
        )
        pruner.pruning_model()
        summary_writer.add_scalar("remain_rate", pruner.cur_rate, prune_step + 1)
        summary_writer.add_scalar("cutoff", pruner.last_cutoff, prune_step + 1)

        pruner.save(
            os.path.join(
                args.save_path, "{}_{}.th".format(pruner.prune_times, pruner.cur_rate)
            )
        )


def eval_mtl_single(args):
    global logger
    # import ipdb; ipdb.set_trace()
    args = torch.load(os.path.join(args.save_path, "args"))
    print(args)
    logger.info(args)
    task_lst, vocabs = utils.get_data(args.data_path)
    task_db = task_lst[args.task_id]
    train_data = task_db.train_set
    dev_data = task_db.dev_set
    test_data = task_db.test_set
    task_name = task_db.task_name

    # text classification
    for ds in [train_data, dev_data, test_data]:
        ds.rename_field("words_idx", "x")
        ds.rename_field("label", "y")
        ds.set_input("x", "y", "task_id")
        ds.set_target("y")
    # seq label
    if task_name in SEQ_LABEL_TASK:
        for ds in [train_data, dev_data, test_data]:
            ds.set_input("seq_len")
            ds.set_target("seq_len")

    logger = utils.get_logger(__name__)
    logger.info("task name: {}, task id: {}".format(task_db.task_name, task_db.task_id))
    logger.info(
        "train len {}, dev len {}, test len {}".format(
            len(train_data), len(dev_data), len(test_data)
        )
    )

    # init model
    model = get_model(args, task_lst, vocabs)
    # logger.info('model: \n{}'.format(model))

    if task_name not in SEQ_LABEL_TASK or task_name == "pos":
        metrics = [
            AccuracyMetric(target="y"),
            # MetricInForward(val_name='loss')
        ]
    else:
        metrics = [
            SpanFPreRecMetric(
                tag_vocab=vocabs[task_name],
                pred="pred",
                target="y",
                seq_len="seq_len",
                encoding_type="bioes" if task_name == "ner" else "chunk",
            ),
            AccuracyMetric(target="y")
            # MetricInForward(val_name='loss')
        ]

    cur_best = 0.0
    init_best = None
    eval_time = 0
    paths = [path for path in os.listdir(args.save_path) if path.startswith("best")]
    paths = sorted(paths, key=lambda x: int(x.split("_")[1]))
    for path in paths:
        path = os.path.join(args.save_path, path)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        tester = Tester(
            test_data,
            model,
            metrics=metrics,
            batch_size=args.batch_size,
            num_workers=4,
            device="cuda",
            use_tqdm=False,
        )
        res = tester.test()
        val = 0.0
        for metric_name, metric_dict in res.items():
            if task_name == "pos" and "acc" in metric_dict:
                val = metric_dict["acc"]
                break
            elif "f" in metric_dict:
                val = metric_dict["f"]
                break

        if init_best is None:
            init_best = val
        logger.info(
            "No #%d: best %f, %s, path: %s, is better: %s",
            eval_time,
            val,
            tester._format_eval_results(res),
            path,
            val > init_best,
        )

        eval_time += 1


def main():
    parser = utils.get_default_parser()

    # fmt: off
    parser.add_argument("--final_rate", dest='final_rate', type=float, default=0.1, help='percent of params to remain not to pruning')
    parser.add_argument("--pruning_iter", dest='pruning_iter', type=int, default=10, help='max times to pruning')
    parser.add_argument('--init_masks', dest='init_masks', type=str, default=None, help='initial masks for late reseting pruning')
    parser.add_argument('--need_cut', default='lstm,conv', type=str, dest='need_cut', help='parameters names that not cut')
    parser.add_argument("--task_id", dest='task_id', type=int, default=0, help='the task to use')
    # fmt: on

    args, unk = parser.parse_known_args()

    print(args)
    print("unknown args ", unk)

    utils.init_prog(args)

    if args.evaluate:
        eval_mtl_single(args)
    else:
        train_mlt_single(args)
    # print(args)


if __name__ == "__main__":
    main()
