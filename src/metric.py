from fastNLP.core.metrics import SpanFPreRecMetric


def get_ner_bioes(label_list, ignore_labels=None):
    list_len = len(label_list)
    begin_label = "B-"
    end_label = "E-"
    single_label = "S-"
    whole_tag = ""
    index_tag = ""
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != "":
                tag_list.append(whole_tag + "," + str(i - 1))
            whole_tag = current_label.replace(begin_label, "", 1) + "[" + str(i)
            index_tag = current_label.replace(begin_label, "", 1)

        elif single_label in current_label:
            if index_tag != "":
                tag_list.append(whole_tag + "," + str(i - 1))
            whole_tag = current_label.replace(single_label, "", 1) + "[" + str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != "":
                tag_list.append(whole_tag + "," + str(i))
            whole_tag = ""
            index_tag = ""
        else:
            continue
    if (whole_tag != "") & (index_tag != ""):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + "]"
            insert_list = fnlp_reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix


def get_ner_bio(label_list, ignore_labels=None):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = "B-"
    inside_label = "I-"
    whole_tag = ""
    index_tag = ""
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == "":
                whole_tag = current_label.replace(begin_label, "", 1) + "[" + str(i)
                index_tag = current_label.replace(begin_label, "", 1)
            else:
                tag_list.append(whole_tag + "," + str(i - 1))
                whole_tag = current_label.replace(begin_label, "", 1) + "[" + str(i)
                index_tag = current_label.replace(begin_label, "", 1)

        elif inside_label in current_label:
            if current_label.replace(inside_label, "", 1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != "") & (index_tag != ""):
                    tag_list.append(whole_tag + "," + str(i - 1))
                whole_tag = ""
                index_tag = ""
        else:
            if (whole_tag != "") & (index_tag != ""):
                tag_list.append(whole_tag + "," + str(i - 1))
            whole_tag = ""
            index_tag = ""

    if (whole_tag != "") & (index_tag != ""):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + "]"
            insert_list = fnlp_reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def fnlp_reverse_style(input_string):
    # 需要转换为(tag, (start, end))的形式
    target_position = input_string.index("[")
    input_len = len(input_string)
    output_string = (
        input_string[target_position:input_len] + input_string[0:target_position]
    )
    index = output_string.index("]")
    try:
        col_index = output_string.index(",")
    except:
        col_index = -1
    if col_index != -1:
        span = (
            output_string[index + 1 :],
            (
                int(output_string[1:col_index]),
                int(output_string[col_index + 1 : index]) + 1,
            ),
        )
    else:
        span = (
            output_string[index + 1 :],
            (int(output_string[1:index]), int(output_string[1:index]) + 1),
        )
    return span


class YangJieSpanMetric(SpanFPreRecMetric):
    def __init__(
        self,
        tag_vocab,
        pred=None,
        target=None,
        seq_len=None,
        encoding_type=None,
        ignore_labels=None,
        only_gross=True,
        f_type="micro",
        beta=1,
    ):
        r"""

        :param tag_vocab: 标签的 :class:`~fastNLP.Vocabulary` 。支持的标签为"B"(没有label)；或"B-xxx"(xxx为某种label，比如POS中的NN)，
            在解码时，会将相同xxx的认为是同一个label，比如['B-NN', 'E-NN']会被合并为一个'NN'.
        :param str pred: 用该key在evaluate()时从传入dict中取出prediction数据。 为None，则使用 `pred` 取数据
        :param str target: 用该key在evaluate()时从传入dict中取出target数据。 为None，则使用 `target` 取数据
        :param str seq_len: 用该key在evaluate()时从传入dict中取出sequence length数据。为None，则使用 `seq_len` 取数据。
        :param str encoding_type: 目前支持bio, bmes, bmeso, bioes。默认为None，通过tag_vocab自动判断.
        :param list ignore_labels: str 组成的list. 这个list中的class不会被用于计算。例如在POS tagging时传入['NN']，则不会计算'NN'个label
        :param bool only_gross: 是否只计算总的f1, precision, recall的值；如果为False，不仅返回总的f1, pre, rec, 还会返回每个label的f1, pre, rec
        :param str f_type: `micro` 或 `macro` . `micro` :通过先计算总体的TP，FN和FP的数量，再计算f, precision, recall; `macro` : 分布计算每个类别的f, precision, recall，然后做平均（各类别f的权重相同）
        :param float beta: f_beta分数， :math:`f_{beta} = \frac{(1 + {beta}^{2})*(pre*rec)}{({beta}^{2}*pre + rec)}` . 常用为 `beta=0.5, 1, 2` 若为0.5则精确率的权重高于召回率；若为1，则两者平等；若为2，则召回率权重高于精确率。
        """
        super().__init__(
            tag_vocab,
            pred=pred,
            target=target,
            seq_len=seq_len,
            encoding_type=encoding_type,
            ignore_labels=ignore_labels,
            only_gross=only_gross,
            f_type=f_type,
            beta=beta,
        )

        if self.encoding_type == "bmeso":
            self.tag_to_span_func = get_ner_bioes
        elif self.encoding_type == "bioes":
            self.tag_to_span_func = get_ner_bioes
        elif self.encoding_type == "bio":
            self.tag_to_span_func = get_ner_bio
        else:
            raise
