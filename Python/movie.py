from torch_geometric.data import download_url, extract_zip
from os import path as osp

import torch
from torch import Tensor

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F

import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer

import pandas as pd

def print_pr_curve(precision, recall, title, pr_auc, no_skill):
    #create precision recall curve
    _, ax = plt.subplots()
    ax.plot(recall, precision, color='purple', label='AUC = %0.4f' % pr_auc)
    ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    
    #add axis labels to plot
    ax.set_title(title)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    plt.legend(loc='best')
    #display plot
    plt.savefig("PR_" + title + '.png')

def print_roc_curve(fpr, tpr, title, roc_auc):
    #create ROC curve
    _, ax = plt.subplots()
    ax.plot(fpr, tpr, color='purple', label='AUC = %0.4f' % roc_auc)
    ax.plot([0, 1], [0, 1], linestyle='--', label='No Skill')

    #add axis labels to plot
    ax.set_title(title)
    ax.set_ylabel('True Positive Rate(TPR)')
    ax.set_xlabel('False Positive Rate(FPR)')

    plt.legend(loc='best')
    #display plot
    plt.savefig("ROC_" + title + '.png')

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

# Necessary just in the first time for download dataset
url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
data_folder = osp.join(osp.dirname(osp.realpath(__file__)), '../data/')
extract_zip(download_url(url, data_folder), data_folder)

movies_path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/' + 'ml-latest-small/movies.csv')
ratings_path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/' + 'ml-latest-small/ratings.csv')


# Load the entire movie data frame into memory:
movies_df = pd.read_csv(movies_path, index_col='movieId')

# Split genres and convert into indicator variables:
genres = movies_df['genres'].str.get_dummies('|')
# Use genres as movie input features:
encoder = SequenceEncoder("all-MiniLM-L6-v2")
movie_feat =  torch.from_numpy(genres.values).to(torch.float) # encoder(movies_df['title'])

#assert movie_feat.size() == (9742, 20)  # 20 genres in total.

# Load the entire ratings data frame into memory:
ratings_df = pd.read_csv(ratings_path)

# Create a mapping from unique user indices to range [0, num_user_nodes):
unique_user_id = ratings_df['userId'].unique()
unique_user_id = pd.DataFrame(data={
    'userId': unique_user_id,
    'mappedID': pd.RangeIndex(len(unique_user_id)),
})

# Create a mapping from unique movie indices to range [0, num_movie_nodes):
unique_movie_id = ratings_df['movieId'].unique()
unique_movie_id = pd.DataFrame(data={
    'movieId': unique_movie_id,
    'mappedID': pd.RangeIndex(len(unique_movie_id)),
})

# Perform merge to obtain the edges from users and movies:
ratings_user_id = pd.merge(ratings_df['userId'], unique_user_id,
                            left_on='userId', right_on='userId', how='left')
ratings_user_id = torch.from_numpy(ratings_user_id['mappedID'].values)
ratings_movie_id = pd.merge(ratings_df['movieId'], unique_movie_id,
                            left_on='movieId', right_on='movieId', how='left')
ratings_movie_id = torch.from_numpy(ratings_movie_id['mappedID'].values)
# With this, we are ready to construct our `edge_index` in COO format
# following PyG semantics:
edge_index_user_to_movie = torch.stack([ratings_user_id, ratings_movie_id], dim=0)

#assert edge_index_user_to_movie.size() == (2, 100836)

data = HeteroData()

# Save node indices:
data["user"].node_id = torch.arange(len(unique_user_id))
data["movie"].node_id = torch.arange(len(movies_df))

# Add the node features and edge indices:
data["movie"].x = movie_feat
data["user", "rates", "movie"].edge_index = edge_index_user_to_movie

# We also need to make sure to add the reverse edges from movies to users
# in order to let a GNN be able to pass messages in both directions.
# We can leverage the `T.ToUndirected()` transform for this from PyG:
data = T.ToUndirected()(data)

#assert data.node_types == ["user", "movie"]
#assert data.edge_types == [("user", "rates", "movie"),
                           #("movie", "rev_rates", "user")]
#assert data["user"].num_nodes == 610
#assert data["user"].num_features == 0
#assert data["movie"].num_nodes == 9742
#assert data["movie"].num_features == 20
#assert data["user", "rates", "movie"].num_edges == 100836
#assert data["movie", "rev_rates", "user"].num_edges == 100836

# For this, we first split the set of edges into
# training (80%), validation (10%), and testing edges (10%).
# Across the training edges, we use 70% of edges for message passing,
# and 30% of edges for supervision.
# We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
# Negative edges during training will be generated on-the-fly.
# We can leverage the `RandomLinkSplit()` transform for this from PyG:
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=("user", "rates", "movie"),
    rev_edge_types=("movie", "rev_rates", "user"), 
)
train_data, val_data, test_data = transform(data)

# In the first hop, we sample at most 20 neighbors.
# In the second hop, we sample at most 10 neighbors.
# In addition, during training, we want to sample negative edges on-the-fly with
# a ratio of 2:1.
# We can make use of the `loader.LinkNeighborLoader` from PyG:
from torch_geometric.loader import LinkNeighborLoader

# Define seed edges:
edge_label_index = train_data["user", "rates", "movie"].edge_label_index
edge_label = train_data["user", "rates", "movie"].edge_label

#Define mini-batch for training
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("user", "rates", "movie"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)

# sampled_data = next(iter(train_loader))

# print("Sampled mini-batch:")
# print("===================")
# print(sampled_data)


# print(data["user", "rates", "movie"].edge_index.size());print(data["user", "rates", "movie"].edge_index.size())


# print(sampled_data["user", "rates", "movie"].edge_index)

# print(sampled_data["user", "rates", "movie"].edge_label)
# exit()


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x    

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.movie_lin = torch.nn.Linear(20, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.movie_emb = torch.nn.Embedding(data["movie"].num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
        } 
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )
        return pred
        
model = Model(hidden_channels=64)

import tqdm
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1, 2):
    total_loss = total_examples = 0
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data.to(device)
        pred = model(sampled_data)
        ground_truth = sampled_data["user", "rates", "movie"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

edge_label_index = val_data["user", "rates", "movie"].edge_label_index
edge_label = val_data["user", "rates", "movie"].edge_label

val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    edge_label_index=(("user", "rates", "movie"), edge_label_index),
    edge_label=edge_label,
    batch_size=3 * 128,
    shuffle=False,
)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as area_curve
from sklearn.metrics import f1_score
preds = []
ground_truths = []
probs = []
for sampled_data in tqdm.tqdm(val_loader):
    with torch.no_grad():
        sampled_data.to(device)
        pred = model(sampled_data)
        preds.append(pred)
        prob = torch.sigmoid(pred)
        probs.append(prob)
        #pred_labels = (prob > 0.5).long()
        ground_truths.append(sampled_data["user", "rates", "movie"].edge_label)
pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
probs = torch.cat(probs, dim=0).cpu()

#ROC
auc = roc_auc_score(ground_truth, pred)
fpr, tpr, thresholds = roc_curve(ground_truth, pred)
print_roc_curve(fpr, tpr, "MovieUser", auc)
print()
print(f"Validation ROC_AUC: {auc:.4f}")

# PR
no_skill = len(ground_truth[ground_truth==1]) / len(ground_truth)
precision, recall, _ = precision_recall_curve(ground_truth, probs)
pr_auc = area_curve (recall, precision)
print_pr_curve(precision, recall, "MovieUser", pr_auc, no_skill)
tr = 0.5
pred_labels = ((probs > tr).long())
f1 = f1_score(ground_truth, pred_labels)
print(f"Validation PR_AUC {pr_auc:.4f}")
print(f"Validation F1 {f1:.4f} with TR {tr}")

# threshold = 0.5  # Adjust threshold as needed
# pred_labels = np.where(pred >= threshold, 1, 0)
# confusion = confusion_matrix(ground_truth, pred_labels)
# print("Confusion Matrix:")
# print("                  Predicted Negative   Predicted Positive")
# print("Actual Negative         TN                   FP")
# print("Actual Positive         FN                   TP")
# print(confusion)
#thresholds = np.arange(0.1, 1.1, 0.1)  # Threshold values from 0.1 to 1.0
# pred_labels = np.where(pred >= thresholds[:, np.newaxis], 1, 0)
# confusion = confusion_matrix(ground_truth, pred_labels)

# print("Confusion Matrix:")
# print("                  Predicted Negative   Predicted Positive")
# print("Actual Negative         TN                   FP")
# print("Actual Positive         FN                   TP")
# print(confusion)

# Calculate FPR and TPR for each threshold
# fpr, tpr, thresholds = roc_curve(ground_truth, pred)

print("Threshold\tFPR\t\tTPR")
i = 0
for threshold, fpr_value, tpr_value in zip(thresholds, fpr, tpr):
    if i % 100 == 0:
        print(f"{threshold:.4f}\t\t{fpr_value:.4f}\t\t{tpr_value:.4f}")
    i += 1