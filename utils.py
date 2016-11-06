import numpy as np

def scalar_to_one_hot(idx):
    one_hot = np.zeros((1,8))
    one_hot[0][idx] = 1
    return one_hot

def vec_to_one_hot(vec):
    return [scalar_to_one_hot(idx) for idx in vec]

def batch_to_one_hot(batch):
    return [vec_to_one_hot(vec)[0] for vec in batch]


class MaskedEmbedding(Embedding):
    def __init__(self, mask_value=0, **kwargs):
        self.mask_value=mask_value
        super(MaskedEmbedding, self).__init__(**kwargs)
        self.mask_zero = True

    def compute_mask(self, x, mask=None):
        return K.not_equal(x, self.mask_value)
