import tensorflow as tf
from tensorflow import keras
import numpy as np


def getParentNodes(layer, inputs, nodes=None, node_parents=None):
    def isConnectedWithInputs(layer):
        if np.any([i.name == layer.name for i in inputs]):
            return True
        for node in layer._inbound_nodes:
            if isinstance(node.inbound_layers, list):
                input_layers = node.inbound_layers
            else:
                input_layers = [node.inbound_layers]
            for input_layer in input_layers:
                if np.any([i.name == input_layer.name for i in inputs]):
                    return True
                else:
                    return isConnectedWithInputs(input_layer)
        return False

    if nodes is None:
        nodes = {}
        node_parents = {}
    nodes[layer.name] = layer
    node_parents[layer] = []
    for node in layer._inbound_nodes:
        if isinstance(node.inbound_layers, list):
            input_layers = node.inbound_layers
        else:
            input_layers = [node.inbound_layers]
        for input_layer in input_layers:
            if isConnectedWithInputs(input_layer):
                node_parents[layer].append(input_layer)
    for parent in node_parents[layer]:
        getParentNodes(parent, inputs, nodes, node_parents)
    return nodes, node_parents


def constructModelCopy(model, layer_name):
    layer = model.get_layer(layer_name)
    inputs = model.inputs

    nodes, node_parents = getParentNodes(model.get_layer("v1"), inputs)
    outputs = {}
    inputs = []

    def getLayerOutput(layer):
        if layer not in outputs:
            if isinstance(layer, keras.layers.InputLayer):
                outputs[layer] = keras.layers.Input(layer.input.shape[1:])
                inputs.append(outputs[layer])
            else:
                l = [getLayerOutput(l) for l in node_parents[layer]]
                outputs[layer] = layer(*l)
        return outputs[layer]

    return keras.models.Model(inputs=inputs, outputs=getLayerOutput(layer))

