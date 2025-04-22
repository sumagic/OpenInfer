#!/bin/python3

import onnx
import onnxoptimizer
import argparse
import os

import logging
import matplotlib.pyplot as plt

'''
    统计当前常见的onnx网络中，都有哪些op
    以及都有哪些op是连接在一起的，这里可以输入最长搜索链的长度
'''

def _print_args(args):
    logging.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        logging.info('%s: %s' % (arg, value))
    logging.info('------------------------------------------------')


def parse_args():
    parser = argparse.ArgumentParser(description='onnx ops analysis....')
    parser.add_argument('--model_zoo', type=str, default='models', help='onnx model file')
    parser.add_argument('--search_len', type=int, default=3, help='max search length')
    parser.add_argument('--log_level', type=str, default='INFO', help='log level')
    args = parser.parse_args()

    if args.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        raise ValueError("log level must be one of ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']")
    logging.basicConfig(level=args.log_level)
    _print_args(args)
    return args

def get_all_onnx_files(model_zoo):
    all_files = []
    for root, dirs, files in os.walk(model_zoo):
        for file in files:
            if file.endswith('.onnx'):
                all_files.append(os.path.join(root, file))
    logging.info('total onnx files: %d', len(all_files))
    return all_files

def load_model_and_simplified(model_path):
    logging.info("loading model: %s", model_path)
    model = onnx.load(model_path)
    model_simp = onnxoptimizer.optimize(model)
    return model_simp

def get_model_ops(model_simplified, all_ops={}):
    for node in model_simplified.graph.node:
        all_ops[node.op_type] = all_ops.get(node.op_type, 0) + 1

def get_model_ops_chain(model_simplified, all_ops_chain={}, search_len=3):
    logging.info("graph node len: %d", len(model_simplified.graph.node))

    before_idx = []
    next_idx = []
    total_nodes = model_simplified.graph.node
    for i, node in zip(range(len(total_nodes)), total_nodes):
        before_idx.append([])
        next_idx.append([])
        for j, node_j in zip(range(len(total_nodes)), total_nodes):
            # check node inputs
            for output in node_j.output:
                for input_name in node.input:
                    if output == input_name:
                        before_idx[i].append(j)
            # check node outputs
            for input_name in node_j.input:
                for output in node.output:
                    if output == input_name:
                        next_idx[i].append(j)
        logging.info('node id: %d, name: {}, type: {}, before: {},  next: {}'.format(i,
            node.name, node.op_type, before_idx[i], next_idx[i]))

    # for i, node in zip(range(len(total_nodes)), total_nodes):
    #     handle_keys = [node.op_type]
    #     for count in range(search_len):
    #         for node_key in handle_keys:
    #             all_ops_chain[node_key] = all_ops.get(node_key, 0) + 1
    #             for j in next_idx[i]:
    #                 fuse_key = node_key + " + " + total_nodes[j].op_type
    #                 all_ops_chain[fuse_key] = all_ops.get(fuse_key, 0) + 1




if __name__ == '__main__':
    args = parse_args()
    all_onnxes = get_all_onnx_files(args.model_zoo)
    all_onnx_files = []
    all_onnx_files.append(all_onnxes[0])
    all_onnx_files.extend(all_onnxes[1:])
    all_ops = {}
    all_ops_chain = {}
    for model_path in all_onnx_files:
        model_simplified = load_model_and_simplified(model_path)
        get_model_ops(model_simplified, all_ops)
        # get_model_ops_chain(model_simplified, all_ops_chain, args.search_len)
    logging.info('all ops: %s', all_ops)
    plt.bar(range(len(all_ops)), list(all_ops.values()))
    plt.xticks(range(len(all_ops)), list(all_ops.keys()), rotation=90)
    plt.show()