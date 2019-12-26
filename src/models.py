import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import fastNLP.modules
from fastNLP.core.utils import seq_len_to_mask
from fastNLP.embeddings import StaticEmbedding, CNNCharEmbedding


def get_model(args, task_lst, vocabs):
    arch = args.arch
    src_vocab = vocabs["words"]
    n_class_per_task = list(map(len, [vocabs[t.task_name] for t in task_lst]))

    if arch == "cnn-lstm":
        model = CNNBiLSTM(
            vocab=src_vocab,
            hidden_size=args.hidden_size,
            num_layers=args.n_layer,
            n_class_per_task=n_class_per_task,
            dropout=args.dropout,
            crf=args.crf,
        )
    elif arch == "hier-lstm":
        model = HierarchicalShareCNNLSTM(
            vocab=src_vocab,
            hidden_size=args.hidden_size,
            n_class_per_task=n_class_per_task,
            dropout=args.dropout,
        )
    elif arch == "random-lstm":
        model = RandomLSTM(
            src_vocab, args.hidden_size, args.n_layer, n_class_per_task, args.dropout
        )
    else:
        raise ValueError(arch)
    return model


def init_embedding(input_embedding, seed=1337):
    """initiate weights in embedding layer
    """

    scope = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -scope, scope)


def init_linear(input_linear, seed=1337):
    """initiate weights in linear layer
    """

    for p in input_linear.parameters():
        nn.init.normal_(p, 0.0, 0.01)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def ce_loss(logit, target, mask):
    ignore_index = -100
    target = target.masked_fill(~mask, ignore_index)
    return F.cross_entropy(
        logit.view(-1, logit.size(-1)),
        target.view(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )


class CNNBiLSTM(nn.Module):
    def __init__(
        self, vocab, hidden_size, num_layers, n_class_per_task, dropout=0.5, crf=False
    ):
        super().__init__()
        # logger.info(n_class_per_task)
        word_embed = StaticEmbedding(
            vocab=vocab,
            model_dir_or_name="en-glove-6b-100d",
            word_dropout=0.01,
            dropout=dropout,
            lower=True,
        )
        char_embed = CNNCharEmbedding(
            vocab=vocab,
            embed_size=30,
            char_emb_size=30,
            filter_nums=[30],
            kernel_sizes=[3],
            word_dropout=0,
            dropout=dropout,
            include_word_start_end=False
        )
        self.embedding = word_embed
        self.char = char_embed
        self.lstm = fastNLP.modules.LSTM(
            input_size=self.embedding.embedding_dim + self.char.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.out = nn.ModuleList()
        for i, n_class in enumerate(n_class_per_task):
            self.out.append(nn.Linear(hidden_size * 2, n_class))

        self.dropout = nn.Dropout(dropout, inplace=True)
        if crf:
            self.crf = nn.ModuleList(
                [
                    fastNLP.modules.ConditionalRandomField(n_class)
                    for n_class in n_class_per_task
                ]
            )
        else:
            self.crf = None
            self.criterion = nn.CrossEntropyLoss()

        for name, param in self.named_parameters():
            if "out" in name:
                if param.data.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.constant_(param, 0)

    def forward(self, task_id, x, y, seq_len):
        words_emb = self.embedding(x)
        char_emb = self.char(x)
        x = torch.cat([words_emb, char_emb], dim=-1)
        x, _ = self.lstm(x, seq_len)
        self.dropout(x)
        logit = self.out[task_id[0]](x)

        seq_mask = seq_len_to_mask(seq_len, x.size(1))
        if self.crf is not None:
            logit = torch.log_softmax(logit, dim=-1)
            loss = self.crf[task_id[0]](logit, y, seq_mask).mean()
            pred = self.crf[task_id[0]].viterbi_decode(logit, seq_mask)[0]
        else:
            loss = ce_loss(logit, y, seq_mask)
            pred = torch.argmax(logit, dim=2)
        return {"loss": loss, "pred": pred}


class HierarchicalShareCNNLSTM(nn.Module):
    def __init__(self, vocab, hidden_size, n_class_per_task, dropout=0.5):
        super(HierarchicalShareCNNLSTM, self).__init__()
        word_embed = StaticEmbedding(
            vocab=vocab,
            model_dir_or_name="en-glove-6b-100d",
            word_dropout=0.01,
            dropout=dropout,
            lower=True,
        )
        char_embed = CNNCharEmbedding(
            vocab=vocab,
            embed_size=30,
            char_emb_size=30,
            filter_nums=[30],
            kernel_sizes=[3],
            word_dropout=0,
            dropout=dropout,
        )
        self.word_embed = word_embed
        self.char_embed = char_embed
        self.lstm1 = fastNLP.modules.LSTM(
            input_size=self.word_embed.embedding_dim + self.char_embed.embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.lstm2 = fastNLP.modules.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.out = nn.ModuleList(
            [nn.Linear(hidden_size * 2, i) for i in n_class_per_task]
        )
        self.dropout = nn.Dropout(dropout)
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

        for name, param in self.named_parameters():
            if "out" in name:
                if param.data.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.constant_(param, 0)

    def forward(self, task_id, x, y, seq_len):
        tid = task_id[0].item()
        word_embedding = self.word_embed(x)
        char_embedding = self.char_embed(x)
        x = torch.cat((word_embedding, char_embedding), dim=-1)
        seq_mask = seq_len_to_mask(seq_len, x.shape[1])

        out, _ = self.lstm1(x, seq_len)
        if tid != 0:
            out, _ = self.lstm2(out, seq_len)

        batch_size, sent_len, _ = x.shape
        logit = self.out[tid](self.dropout(out))
        loss = ce_loss(logit, y, seq_mask)
        pred = torch.argmax(logit, dim=2)
        return {"pred": pred, "loss": loss}


class RandomLSTM(nn.Module):
    def __init__(self, vocab, hidden_size, num_layers, n_class_per_task, dropout=0.5):
        super().__init__()
        # word_embed = nn.Embedding(len(vocab), 50)
        word_embed = StaticEmbedding(
            vocab=vocab, embedding_dim=50, word_dropout=0, dropout=dropout, lower=True
        )
        self.word_embed = word_embed
        emb_dim = self.word_embed.embedding_dim
        self.lstm = fastNLP.modules.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.out = nn.ModuleList(
            [nn.Linear(hidden_size * 2, i) for i in n_class_per_task]
        )
        self.dropout = nn.Dropout(dropout)
        self.loss = nn.CrossEntropyLoss()

        for name, param in self.named_parameters():
            if "out" in name:
                if param.data.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.constant_(param, 0)

    def forward(self, task_id, x, y, seq_len):
        tid = task_id[0].item()
        x = self.word_embed(x)
        seq_mask = seq_len_to_mask(seq_len, x.shape[1])
        out, _ = self.lstm(x, seq_len)
        logit = self.out[tid](self.dropout(out))
        loss = ce_loss(logit, y, seq_mask)
        pred = torch.argmax(logit, dim=2)
        return {"pred": pred, "loss": loss}
