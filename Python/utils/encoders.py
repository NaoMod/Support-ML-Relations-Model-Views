import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

# SequenceEncoder was adapted from @see https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/load_csv.py
class SequenceEncoder(object):
    # The 'SequenceEncoder' encodes raw column strings into embeddings.
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df, debug=True):
        x = self.model.encode(df.values, show_progress_bar=debug,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()

# IdentityEncoder was adapted from @see https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/load_csv.py
class IdentityEncoder(object):
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)
    

class EnumEncoder(object):
    # The 'EnumEncoder' takes a list of categories as a | splitted string and uses a one-hot-encoding approach to encode it to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        categories = pd.Series(df.values).str.get_dummies('|')

        categories_feat = torch.from_numpy(categories.values).to(self.dtype)
        return categories_feat
    
class NoneEncoder(object):
    # The 'NoneEncoder' just returns None to explict that the attributte was not
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return None


