import torch
from sentence_transformers import SentenceTransformer

# Encoders were adapted from @see https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/load_csv.py
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


class IdentityEncoder(object):
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)
