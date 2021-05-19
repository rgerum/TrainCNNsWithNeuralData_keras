from tensorflow import keras
import numpy as np


class MergedGenerators(keras.utils.Sequence):
    def __init__(self, *generators, use_min_length=False):
        self.generators = generators
        self.lambd = 1
        self.use_min_length = use_min_length

    def __len__(self):
        lengths = np.array([len(g) for g in self.generators])
        if not self.use_min_length:
            for l in lengths:
                if not np.all(lengths == l):
                    raise ValueError("Generators have different lengths", lengths)
        return np.min(lengths)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        for generator in self.generators:
            generator.on_epoch_end()

    def __getitem__(self, index):
        inputs = []
        outputs = []
        for generator in self.generators:
            inp, outp = generator[index]
            if isinstance(inp, (list, tuple)):
                inputs.extend(list(inp))
            else:
                inputs.append(inp)
            if isinstance(outp, (list, tuple)):
                outputs.extend(list(outp))
            else:
                outputs.append(outp)
        #inputs.append(np.ones((inputs[0].shape[0], 1)) * self.lambd)
        # print(index, "inputs", len(inputs), [i.shape for i in inputs], "outputs", len(outputs), [i.shape for i in outputs])
        return inputs, outputs
