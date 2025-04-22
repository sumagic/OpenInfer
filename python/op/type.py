#!/bin/python3

import onnx

class OpType:
    OP_TYPE_MAJOR_NEIGHBOR = 0
    OP_TYPE_MAJOR_CHANNEL = 1
    

class OpTypeConv2D:
    op_type = OpType.OP_TYPE_NEIGHBOR

class OpTypeMul:
    op_type = OpType.OP_TYPE_CHANNEL_WISE

    def __init__(self, onnx_node: onnx.NodeProto):
        self.op_type = OpType.OP_TYPE_ELEMENT_WISE


if __name__ == "__main__":
    print("OpType.OP_TYPE_NEIGHBOR: ", OpType.OP_TYPE_NEIGHBOR)
    print("OpType.OP_TYPE_CHANNEL_WISE: ", OpType.OP_TYPE_CHANNEL_WISE)
    print("OpType.OP_TYPE_ELEMENT_WISE: ", OpType.OP_TYPE_ELEMENT_WISE)