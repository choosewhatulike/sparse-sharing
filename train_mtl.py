import os
import copy
import torch
import torch.optim as optim
from fastNLP import logger
from src.models import get_model
from src import utils
from src.trainer import get_trainer_cls


def load_masks(masks_dir):
    masks_path = [
        os.path.join(masks_dir, f)
        for f in os.listdir(masks_dir)
        if not f.startswith("init")
    ]
    masks_path = list(
        sorted(
            filter(lambda f: os.path.isfile(f), masks_path),
            key=lambda s: int(os.path.basename(s).split("_")[0]),
        )
    )
    masks = []
    logger.info("loading masks")
    for path in masks_path:
        dump = torch.load(path, "cpu")
        assert "mask" in dump and "pruning_time" in dump
        logger.info(
            "loading pruning_time {}, mask in {}".format(dump["pruning_time"], path)
        )
        masks.append(dump["mask"])

    # sanity check
    assert len(masks) == len(masks_path)
    for mi in masks:
        for name, m in mi.items():
            assert isinstance(m, torch.Tensor)
            mi[name] = m.bool()
    return masks


def load_weights(weights_dir):
    return torch.load(os.path.join(weights_dir, "init_weights"), map_location="cpu")


def check_masks(masks, weights):
    for mask_i in masks:
        for name, m in mask_i.items():
            assert name in weights
            assert weights[name].shape == m.shape


class MTL_Masker:
    def __init__(self, model, masks):
        self.model = model
        self.masks = masks
        self.weights = []
        if self.masks is None:
            mask = {}
            for name, param in self.model.named_parameters():
                m = torch.zeros_like(param.data).bool()
                mask[name] = m
            self.masks = mask
        logger.info("has masks %d, %s", len(self.masks), type(self.masks))

    def to(self, device):
        # logger.info(type(self.model), type(self.masks), device)
        logger.info("model to %s", device)
        self.model.to(device)
        if self.masks is None:
            return
        if isinstance(self.masks, dict):
            masks = [self.masks]
        else:
            masks = self.masks
        for i, mask in enumerate(masks):
            logger.info("mask {} to {}".format(i, device))
            for name, m in mask.items():
                mask[name] = m.to(device)

    def before_forward(self, task_id):
        # backup weights
        self.weights.append(copy.deepcopy(self.model.state_dict()))
        # apply mask to param
        self.apply_mask(task_id)

    def after_forward(self, task_id):
        # resume weights
        weights = self.weights.pop()
        self.model.load_state_dict(weights)
        # apply mask to grad
        self.mask_grad(task_id)

    def apply_mask(self, task_id):
        if isinstance(self.masks, dict):
            mask = self.masks
        else:
            mask = self.masks[task_id]
        for name, param in self.model.named_parameters():
            if name in mask:
                param.data.masked_fill_(mask[name], 0.0)

    def mask_grad(self, task_id):
        # zero-out all the gradients corresponding to the pruned connections
        if isinstance(self.masks, dict):
            mask = self.masks
        else:
            mask = self.masks[task_id]
        for name, p in self.model.named_parameters():
            if name in mask and p.grad is not None:
                p.grad.data.masked_fill_(mask[name], 0.0)


if __name__ == "__main__":
    parser = utils.get_default_parser()

    # fmt: off
    parser.add_argument("--masks_path", type=str, default=None, help='the task specific mask paths')
    parser.add_argument("--tasks", type=str, default=None, help='the task ids for MTL, default using all tasks')
    parser.add_argument("--trainer", type=str, choices=['re-seq-label', 'seq-label'], default='seq-label', help='the trainer type')
    # fmt: on

    args = parser.parse_args()

    utils.init_prog(args)

    logger.info(args)
    torch.save(args, os.path.join(args.save_path, "args.th"))

    n_gpu = torch.cuda.device_count()
    print("# of gpu: {}".format(n_gpu))

    logger.info("========== Loading Datasets ==========")
    task_lst, vocabs = utils.get_data(args.data_path)
    if args.tasks is not None:
        args.tasks = list(map(int, map(lambda s: s.strip(), args.tasks.split(","))))
        logger.info("activate tasks %s", args.tasks)
    logger.info("# of Tasks: {}.".format(len(task_lst)))
    for task in task_lst:
        logger.info("Task {}: {}".format(task.task_id, task.task_name))
    for task in task_lst:
        if args.debug:
            task.train_set = task.train_set[:200]
            task.dev_set = task.dev_set[:200]
            task.test_set = task.test_set[:3200]
            args.epochs = 3
        task.init_data_loader(args.batch_size)
    logger.info("done.")

    model_descript = args.exp_name
    # print('====== Loading Word Embedding =======')

    logger.info("========== Preparing Model ==========")

    n_class_per_task = []
    for task in task_lst:
        n_class_per_task.append(len(vocabs[task.task_name]))
    logger.info("n_class %s", n_class_per_task)

    model = get_model(args, task_lst, vocabs)

    if args.masks_path is None:
        masks = None
    else:
        masks = load_masks(args.masks_path)

    if args.init_weights is not None:
        utils.load_model(model, args.init_weights)
    elif args.masks_path is not None:
        utils.load_model(model, os.path.join(args.masks_path, "init_weights"))
    masker = MTL_Masker(model, masks)

    logger.info("Model parameters:")
    params = list(model.named_parameters())
    sum_param = 0
    for name, param in params:
        if param.requires_grad:
            logger.info("{}: {}".format(name, param.shape))
            sum_param += param.numel()
    logger.info("# Parameters: {}.".format(sum_param))
    masker.to("cuda" if torch.cuda.is_available() else "cpu")
    Trainer = get_trainer_cls(args)

    if not args.evaluate:
        logger.info("========== Training Model ==========")
        base_params = filter(lambda p: p.requires_grad, model.parameters())
        opt = utils.get_optim(args.optim, base_params)
        logger.info(opt)
        trainer = Trainer(masker, task_lst, vocabs, opt, args)

        trainer.train(args.epochs)

        logger.info("========== Testing Model ==========")
        trainer.model = utils.load_model(model, os.path.join(args.save_path, "best.th"))
        test_loss, test_acc = trainer._eval_epoch(dev=False)
        logger.info(args.exp_name)
        for acc in test_acc.items():
            logger.info(acc)

    else:
        logger.info("========== Evaluating Model ==========")
        trainer = Trainer(masker, task_lst, vocabs, None, args)
        model_paths = os.listdir(os.path.join(args.save_path, "models"))
        model_paths = [os.path.join(args.save_path, "models", p) for p in model_paths]
        best_acc = (-1, 0)
        logger.info(args.exp_name)
        for i, path in enumerate(model_paths):
            trainer.masker.model = utils.load_model(model, path)
            test_loss, test_acc = trainer._eval_epoch(dev=False)
            logger.info("at epoch [%d]", i)
            for acc in test_acc.items():
                logger.info(acc)
                if acc[0] == "avg" and acc[1] > best_acc[1]:
                    logger.info("update best!")
                    best_acc = (i, acc[1])
        logger.info("best at epoch [%d], avg %f", best_acc[0], best_acc[1])
