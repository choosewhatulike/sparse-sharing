import numpy as np
from functools import partial
from src import utils
from copy import deepcopy
import torch
from fastNLP import logger
from fastNLP import Callback


def prune_by_percent_once(percent, mask, param):
    # Put the weights that aren't masked out in sorted order.
    sorted_weights = np.sort(np.abs(param[mask == 1]), axis=None)

    # Determine the cutoff for weights to be pruned.
    if sorted_weights.size <= 0:
        utils.get_logger(__name__).warning(
            "cutoff all of params, shape: {}".format(param.shape)
        )
        utils.get_logger(__name__).warning("last cutoff mask {}".format(np.sum(mask)))
        # print('cut all of params')
        return np.zeros(mask.shape)

    cutoff_index = np.round(percent * sorted_weights.size).astype(int)
    cutoff = sorted_weights[cutoff_index]
    utils.get_logger(__name__).debug(
        "cutoff index{}, cutoff weights {}".format(cutoff_index, cutoff)
    )
    # Prune all weights below the cutoff.
    return np.where(np.abs(param) <= cutoff, np.zeros(mask.shape), mask)


def prune_by_percentile_once(percent, mask, param):

    cutoff = np.percentile(np.abs(param[mask]), percent * 100.0, axis=None)
    utils.get_logger(__name__).debug("cutoff weights {}".format(cutoff))
    # Prune all weights below the cutoff.
    return np.where(np.abs(param) <= cutoff, np.zeros(mask.shape), mask)


class Pruning(Callback):
    def __init__(
        self,
        model,
        pruning_param_names,
        final_rate=0.1,
        pruning_iter=1,
        prune_once=None,
    ):
        super().__init__()
        assert pruning_iter >= 0
        self.final_rate = final_rate
        self.pruning_iter = pruning_iter
        prune_once = prune_once or prune_by_percent_once
        self.pruning_names = set(pruning_param_names)
        print(self.pruning_names)
        self._log = utils.get_logger(__name__)
        self._log.info(self.pruning_names)
        self.prune_times = 0
        self.one_rate = (
            1 - (self.final_rate ** (1.0 / self.pruning_iter))
            if self.pruning_iter > 0
            else 1.0
        )
        self.prune_once = partial(prune_once, self.one_rate)
        self._log.info(
            "Pruning iter {}, pruning once persent {}, final remain rate {}".format(
                self.pruning_iter, self.one_rate, self.final_rate
            )
        )

        # backup initial weights
        # self.backup_optim = copy(self.optimizer)
        self.backup_weights = deepcopy(model.state_dict())
        if hasattr(model, "module"):
            self._model = model.module
        else:
            self._model = model
        self._log.debug(
            "model params :{}".format(
                [name for name, _ in self._model.named_parameters()]
            )
        )
        remain_mask = {
            name: torch.zeros(p.size()).to(p).bool()
            for name, p in self._model.named_parameters()
            if name in self.pruning_names
        }
        self.remain_mask = remain_mask
        self.pruning_names = set(self.remain_mask.keys())

        self._log.info("Pruning params are in following ...")
        total_m = 0
        for name, p in self.remain_mask.items():
            self.logger.info("Need pruning {}, params: {}".format(name, p.numel()))
            total_m += p.numel()
        self.logger.info("Total need pruning params: {}".format(total_m))
        self.total_params = total_m
        self.cur_rate = 100.0
        self.last_cutoff = 0

    def on_batch_end(self):
        if self.final_rate < 1.0 and self.prune_times > 0:
            self.apply_mask(self.remain_mask, self._model)

    def apply_weights(self, weights, model):
        """flush weights to model"""
        model.load_state_dict(weights)

    def apply_mask(self, remain_mask, model):
        """apply mask on them"""
        for name, param in model.named_parameters():
            if name in remain_mask:
                param.data.masked_fill_(remain_mask[name], 0.0)
                self._log.debug(
                    "apply masks for {}, mean: {}".format(name, param.data.mean())
                )

    def pruning_model(self):
        self.prune_times += 1
        if self.prune_times <= self.pruning_iter:
            # self.prune(self.one_rate, self.remain_mask, self._model)
            self.prune_global(self.one_rate, self.remain_mask, self._model)
            total_m = sum(m.sum().item() for m in self.remain_mask.values())
            total_m = self.total_params - total_m
            self.cur_rate = round(100.0 * total_m / self.total_params, 2)
            self._log.info(
                "No #{} pruning, remain params {}, remain percent {}%".format(
                    self.prune_times, total_m, self.cur_rate
                )
            )
            # reset optimizer & model weights when pruning
            self.apply_weights(self.backup_weights, self._model)
            self.apply_mask(self.remain_mask, self._model)
            # self.trainer.optimizer = copy(self.backup_optim)
            # self._log.debug('reset optimizer to {}'.format(self.trainer.optimizer))
        else:
            self._log.warning(
                "No #{} pruning, exceed max pruning times".format(self.prune_times + 1)
            )

    def save(self, path):
        state = {
            "init_weights": self.backup_weights,
            "mask": self.remain_mask,
            "pruning_time": self.prune_times,
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        # self.backup_weights = state['init_weights']
        self.remain_mask = state["mask"]
        self.prune_times = state["pruning_time"]

        # sanity check
        for name, _ in self._model.named_parameters():
            assert name in self.backup_weights
        for name, m in self.remain_mask.items():
            assert name in self.backup_weights
        self._model.to("cuda")
        logger.info("load mask from %s", path)
        logger.info("current pruning time %d", self.prune_times)

    def prune(self, rate, remain_mask, model):
        """get new pruning mask"""
        for name, param in model.named_parameters():
            if name in remain_mask:
                mask = remain_mask[name] == 0
                new_m = self.prune_once(
                    mask.data.cpu().numpy(), param.data.cpu().numpy()
                )
                new_m = torch.tensor(new_m).byte().to(param.device)
                remain_mask[name] = new_m == 0

    def prune_global(self, rate, remain_mask, model):
        names = []
        masks = []
        params = []
        weights = []
        for name, param in model.named_parameters():
            if name in remain_mask:
                # mask = remain_mask[name] == 0
                names.append(name)
                mask = remain_mask[name]
                m = mask.data.cpu().numpy()
                p = param.data.cpu().numpy()
                masks.append(m)
                params.append(p)
                weights.append(p[m == 0])
        all_w = np.concatenate([np.reshape(w, -1) for w in weights], axis=0)
        sorted_w = np.sort(np.abs(all_w))
        if sorted_w.size <= 0:
            self.logger.warning("cutoff all of params")
            return
        cutoff_index = np.round(rate * sorted_w.size).astype(int)
        cutoff = sorted_w[cutoff_index]
        new_masks = []
        for m, p in zip(masks, params):
            new_m = np.where(np.abs(p) < cutoff, np.ones(m.shape), m)
            new_masks.append(new_m)
        device = next(model.parameters()).device
        for name, new_m in zip(names, new_masks):
            remain_mask[name] = torch.tensor(new_m).bool().to(device)
        self.logger.info(
            "No #{}, cutoff weights {}, weights mean {}, weights std {}".format(
                self.prune_times, cutoff, all_w.mean(), all_w.std()
            )
        )
        self.last_cutoff = cutoff
