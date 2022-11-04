from __future__ import absolute_import
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from bleu import _bleu
import re
import torch
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from transformers import (BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
from datasets import load_dataset
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   tree_to_token_nodes,
                   index_to_code_token,
                   tree_to_variable_index,
                   detokenize_code)
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript,DFG_csharp
from tree_sitter import Language, Parser
sys.path.append('CodeBLEU')
from calc_code_bleu import calc_code_bleu, calc_code_bleu_multilang
sys.path.append('CodeBLEU')
keywords_dir = 'CodeBLEU/keywords'

MODEL_CLASSES = {'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

dfg_function = {
    'java': DFG_java
}

parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


def extract_structure(code, parser, lang):
    try:
        # ast
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        ast_token_nodes = tree_to_token_nodes(root_node)
        tokens_index = [(node.start_point, node.end_point) for node in ast_token_nodes]
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]

        # dfg
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg, ast_token_nodes


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(split, args):
    examples = []
    with open('/content/drive/MyDrive/datasets/concode/' + split + '.json') as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 target_dfg,
                 target_ast,
                 target_ast_sim
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.target_dfg = target_dfg
        self.target_ast = target_ast
        self.target_ast_sim = target_ast_sim


def get_node_types(node, l):
    l.append(node.type)
    for child in node.children:
        get_node_types(child, l)


def gather_node_types(examples, args):
    global node_types
    filename = args.output_dir + '/node_types.pkl'
    if os.path.exists(filename):
        node_types = pickle.load(open(filename, 'rb'))
        node_types = {t: i for i, t in enumerate(node_types)}
        return
    node_types = []
    for example in tqdm(examples):
        root = parsers['java'][0].parse(bytes(example.target, 'utf8')).root_node
        get_node_types(root, node_types)
    node_types = sorted(list(set(node_types)))
    pickle.dump(node_types, open(filename, 'wb'))
    node_types = {t: i for i, t in enumerate(node_types)}


def get_lr_path(leaf):
    if leaf == -1:
        return -1
    path = [leaf]
    while path[-1].parent is not None:
        path.append(path[-1].parent)
    return path


def convert_path_to_idx(path, max_depth):
    if path == -1:
        return [-1] * max_depth
    path = [node_types.get(node.type, -1) for node in path][:max_depth]
    path = path + [-1] * (max_depth - len(path))
    return path


def get_ll_sim(p1, p2):
    if (p1 == -1) or (p2 == -1):
        return -1
    common = 1
    for i in range(2, min(len(p1), len(p2)) + 1):
        if p1[-i] == p2[-i]:
            common += 1
        else:
            break
    return common * common / (len(p1) * len(p2))


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    global target_sim_bps
    features = []
    match, nomatch = 1, 1
    bar = tqdm(enumerate(examples), total=len(examples))
    for example_index, example in bar:
        # source
        source_tokens = [tokenizer.cls_token] + tokenizer.tokenize(example.source)[:args.max_source_length - 2] + [
            tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)

        # target
        if stage == "test":
            target_ids = -1
            target_dfg = -1
            target_ast = -1
            target_ast_sim = -1
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
            code_tokens, dfg, ast = extract_structure(example.target, parsers['java'], 'java')
            for i in range(1, len(ast)):
                if (ast[i].start_point[0] < ast[i - 1].start_point[0]) or \
                        ((ast[i].start_point[0] == ast[i - 1].start_point[0]) and (
                                ast[i].start_point[1] < ast[i - 1].start_point[1])):
                    raise Exception("Leaves not ordered by position in sequence.")

            tcode = list(''.join(target_tokens).replace('Ġ', ' ').replace('ĉ', '\t'))
            scode = list(''.join(code_tokens))
            tcode_to_scode = []
            j = 0
            for i in range(len(tcode)):
                if j < len(scode):
                    if tcode[i] == scode[j]:
                        tcode_to_scode.append(j)
                        j += 1
                        match += 1
                    else:
                        tcode_to_scode.append(-1)
                        if (tcode[i] != ' '):
                            #                             logger.info(tcode[i])
                            nomatch += 1
                else:
                    tcode_to_scode.append(-1)
                    if (tcode[i] != ' '):
                        #                         logger.info(tcode[i])
                        nomatch += 1

            tcode_to_target = []
            for i in range(len(target_tokens)):
                tcode_to_target += [i] * len(target_tokens[i])
            scode_to_code = []
            for i in range(len(code_tokens)):
                scode_to_code += [i] * len(code_tokens[i])

            target_to_code = [[] for i in range(len(target_tokens))]
            for i in range(len(tcode)):
                if tcode_to_scode[i] >= 0:
                    target_to_code[tcode_to_target[i]].append(scode_to_code[tcode_to_scode[i]])

            target_to_code = [set(v) for v in target_to_code]
            max_code_tokens = max([max(v) for v in target_to_code if len(v) > 0]) + 1

            code_to_target = [[] for i in range(max_code_tokens)]
            for i in range(len(target_to_code)):
                for c in target_to_code[i]:
                    code_to_target[c].append(i + 1)  # don't account for adding CLS at beginning

            dfg_small = []
            for t in dfg:
                if t[1] < max_code_tokens:
                    rights = [i for i in t[4] if i < max_code_tokens]
                    if len(rights) > 0:
                        dfg_small.append((t[0], t[1], t[2], t[3], rights))
            dfg = dfg_small.copy()
            ast = ast[:max_code_tokens]

            target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

            target_len = len(target_ids)
            target_dfg = np.zeros((target_len, target_len))
            target_ast = -np.ones((target_len, args.max_ast_depth))
            target_ast_sim = -np.ones((target_len, target_len))

            tlr_paths = [get_lr_path(leaf) for leaf in ast]
            tleaf_to_leaf = np.ones((len(ast), len(ast)))
            for i in range(len(ast)):
                for j in range(i + 1, len(ast)):
                    sim = get_ll_sim(tlr_paths[i], tlr_paths[j])
                    tleaf_to_leaf[i, j] = sim
                    tleaf_to_leaf[j, i] = sim

            tlr_paths = [convert_path_to_idx(path, args.max_ast_depth) for path in tlr_paths]
            for i, ts in enumerate(code_to_target):
                target_ast[ts, :] = np.array(tlr_paths[i]).reshape((1, -1))
                for i2, ts2 in enumerate(code_to_target):
                    sim = tleaf_to_leaf[i, i2]
                    for ts_ in ts:
                        target_ast_sim[ts_, ts2] = sim

            for _, l, _, _, rs in dfg:
                for lt in code_to_target[l]:
                    for r in rs:
                        target_dfg[lt, code_to_target[r]] = 1
            target_dfg[-1, :] = -1
            target_dfg[:, -1] = -1
            target_dfg[0, :] = -1
            target_dfg[:, 0] = -1

        bar.set_description(str(stage) + ' ' + str(nomatch / (match + nomatch)))
        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                target_dfg,
                target_ast,
                target_ast_sim
            )
        )

    if stage == 'train':
        tsims = []
        for eg in tqdm(features):
            tsims.append(eg.target_ast_sim.flatten())
        tsims = np.concatenate(tsims)
        tsims = tsims[tsims != -1]
        target_sim_bps = [np.percentile(tsims, p) for p in range(0, 100, 20)] + [100]

    return features


class TextDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--model_type", default="codet5", type=str,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default="Salesforce/codet5-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--tokenizer_name", default="Salesforce/codet5-base",
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--load_model_path", default="saved_models/pretrain/checkpoint-12000/pytorch_model.bin",
                        type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--config_name", default="Salesforce/codet5-base", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--max_source_length", default=325, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_ast_depth", default=12, type=int)
    parser.add_argument("--max_target_length", default=155, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--alpha1", default=None, type=float)
    parser.add_argument("--alpha2", default=None, type=float)
    parser.add_argument("--alpha1_clip", default=-4, type=float)
    parser.add_argument("--alpha2_clip", default=-4, type=float)
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.output_dir = 'saved_models/generation_fixed_' + str(args.max_source_length) + '_' + str(
        args.max_target_length) + '/'
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)    # Get node types.
    train_examples = read_examples('train', args)
    gather_node_types(train_examples, args)
    args.node_types = node_types
    filename = args.output_dir + 'train_features.pkl'
    if os.path.exists(filename):
        train_features = pickle.load(open(filename, 'rb'))
    else:
        train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
        pickle.dump(train_features, open(filename, 'wb'))
    model = model_class.from_pretrained(args.model_name_or_path)
    model = Seq2Seq(model=model, beam_size=args.beam_size, max_length=args.max_target_length, args=args)

    if args.load_model_path!='none':
        logger.info("reload model from {}".format(args.load_model_path))
        pt_dict = torch.load(args.load_model_path)
        my_dict = model.state_dict()
        for k,v in pt_dict.items():
            if k not in ['ast_type_emb.weight', 'ast_path_head.weight', 'ast_path_head.bias']:
                my_dict[k] = v
        pt_node_types = pickle.load(open('pretrain/pt_node_types_3L_cls_target.pkl','rb'))
        pt_node_types = {k:i for i,k in enumerate(pt_node_types)}
        my_to_pt_node_types = [-1 for i in range(len(pt_node_types))]

        for k,i in pt_node_types.items():
            if k in pt_node_types:
                my_to_pt_node_types[i] = pt_node_types[k]
                with torch.no_grad():
                    print(k, len(pt_dict['ast_path_head.weight']))
                    my_dict['ast_type_emb.weight'][i,:] = pt_dict['ast_type_emb.weight'][pt_node_types[k], :]
                    my_dict['ast_path_head.weight'][i::len(pt_node_types), :] = pt_dict['ast_path_head.weight'][i::len(pt_node_types), :]
        logger.info("*********** No. of new node types = %d", (np.array(my_to_pt_node_types)==-1).sum())
        model.load_state_dict(my_dict)
    if args.alpha1 is not None:
        model.set_alpha(args.alpha1, args.alpha2)
    if args.alpha1_clip is not None:
        model.set_alpha_clip(args.alpha1_clip, args.alpha2_clip)
    model.to(device)

    # budild model
    # model = model_class.from_pretrained(args.model_name_or_path)
    # model = Seq2Seq(model=model, beam_size=args.beam_size, max_length=args.max_target_length, args=args)
    text = """
    function to draw square using opengl
    """
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    ##simply generate a single sequence

    generated_ids=model.generate(input_ids.to(device=0), max_new_tokens=1000)
    out = tokenizer.decode(generated_ids[0],skip_special_tokens=True)
    print(out)


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_default_device():
    """Picking GPU if available or else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

if __name__ == "__main__":
    main()