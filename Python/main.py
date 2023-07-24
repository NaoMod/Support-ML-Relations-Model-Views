import numpy as np
import pandas as pd
import torch

from pyecore.resources import URI

from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.nn import to_hetero

import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as area_curve

from os import path as osp
import glob
import json
from pathlib import Path

# interbal libraries
from utils.print_curves import print_pr_curve, print_roc_curve
from utils.to_graph import ToGraph
from modeling.metamodels import Metamodels

VIEWS_DIRECTORY = 'Views'
VIEW_NAME = 'Movies_Users'

# Read JSON files to get GNN properties
json_path_gnn_parameters = glob.glob(osp.join(Path(__file__).parent, '..', VIEWS_DIRECTORY, VIEW_NAME, 'src-gen', "recommended.json"))[0]

gnn_parameters_exist = len(json_path_gnn_parameters) != 0

gnn_parameters = None
if gnn_parameters_exist:
    gnn_parameters = json_path_gnn_parameters

with open(gnn_parameters) as json_data_gnn_parameters:
    parameters = json.load(json_data_gnn_parameters)

    for relation_name, relation_props in parameters.items():
        s = relation_props['S']
        t = relation_props['T']
        rev_relation_name = "rev_" + relation_name

        EPOCHS = relation_props["TRAINING_PARAMETERS"]["EPOCHS"]
        LEARNING_RATE = relation_props["TRAINING_PARAMETERS"]["LEARNING_RATE"]
        ADD_NEGATIVE_TRAINING = relation_props["TRAINING_PARAMETERS"]["ADD_NEGATIVE_TRAINING"]
        NEG_SAMPLING_RATIO = relation_props["TRAINING_PARAMETERS"]["NEG_SAMPLING_RATIO"]

        GNN_OPERATOR = relation_props["ARCHITECTURE"]["OPERATOR"]
        CONVOLUTIONS = relation_props["ARCHITECTURE"]["CONVOLUTIONS"]
        ACTIVATION = relation_props["ARCHITECTURE"]["ACTIVATION"]
        HIDDEN_CHANNELS = relation_props["ARCHITECTURE"]["HIDDEN_CHANNELS"]
        CLASSIFIER = relation_props["ARCHITECTURE"]["CLASSIFIER"]

        # Register the metamodels in the resource set
        metamodels = Metamodels()
        metamodels.register()

        modeling_resources_path = glob.glob(osp.join(Path(__file__).parent, '..', 'Modeling_Resources'))[0]
        resource_set = metamodels.get_resource_set()


        dataset_func = ToGraph(embeddings_information=relation_props["EMBEDDINGS"], features_for_embedding_left=None, features_for_embedding_right=None)
        # Register the models in the resource set
        xmi_path_left = glob.glob(osp.join(modeling_resources_path, relation_props["TRAINING_PARAMETERS"]["SOURCE_MODEL_PATH"]))[0]
        m_resource_left = resource_set.get_resource(URI(xmi_path_left))
        model_root_left = m_resource_left.contents

        xmi_path_right = glob.glob(osp.join(modeling_resources_path, relation_props["TRAINING_PARAMETERS"]["TARGET_MODEL_PATH"]))[0]
        m_resource_right = resource_set.get_resource(URI(xmi_path_right))
        model_root_right = m_resource_right.contents

        relations_path = glob.glob(osp.join(modeling_resources_path, relation_props["TRAINING_PARAMETERS"]["LINK_PATH"]))
        relations_exist = len(relations_path) != 0
        relations_for_graph = None
        if relations_exist:
            relations_for_graph = relations_path[0]  

        data, left_original_mapping, right_original_mapping = dataset_func.xmi_to_graph(model_root_left, model_root_right, relations_for_graph, s, t, relation_name)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)

        # Add a reverse (t, rev_relation_name, s) relation for message passing:
        data = T.ToUndirected()(data)
        del data[t, rev_relation_name, s].edge_label  # Remove "reverse" label.

        # Perform a link-level split into training, validation, and test edges:
        train_data, val_data, test_data = T.RandomLinkSplit(
            num_val=0.1, # 10% of links for validation
            num_test=0.1, # 10% os links for test
            disjoint_train_ratio=0.4, # use 60% for message passing and 40% for supervision
            neg_sampling_ratio=2.0, # generate 2:1 negative edges
            add_negative_train_samples=ADD_NEGATIVE_TRAINING,
            edge_types=(s, relation_name, t),
            rev_edge_types=(t, rev_relation_name, s)
        )(data)

        train_edge_label_index = train_data[s, relation_name, t].edge_label_index
        train_edge_label = train_data[s, relation_name, t].edge_label

        # LOADERS
        train_loader = LinkNeighborLoader(
            data=train_data,
            num_neighbors=[20, 10],
            neg_sampling_ratio=2.0,
            edge_label_index=((s, relation_name, t), train_edge_label_index),
            edge_label=train_edge_label,
            batch_size=128,
            shuffle=True,
        )

        def get_operator():
            if GNN_OPERATOR == "SAGEConv":
                return SAGEConv
            
        def get_activation():
            if ACTIVATION == "relu":
                return F.relu
            
        class DotProduct(torch.nn.Module):
            """
            The final classifier applies the dot-product between source and destination node embeddings to derive edge-level predictions
            """

            def forward(self, x_left: torch.Tensor, x_right: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
                """
                This method is responsible for executing the forward pass of the dot-product classifier.
                
                Parameters
                ----------
                x_left: torch.Tensor
                    input tensor for Left class
                x_right: torch.Tensor
                    input tensor for Right class
                edge_label_index: torch.Tensor
                    edge label index

                Returns
                -------
                x: torch.Tensor
                    output tensor
                """

                # Convert node embeddings to edge-level representations:
                edge_feat_left  = x_left[edge_label_index[0]]
                edge_feat_right = x_right[edge_label_index[1]]

                # Apply dot-product to get a prediction per supervision edge:
                return (edge_feat_left * edge_feat_right).sum(dim=-1)
        class GNN(torch.nn.Module):
            """
            This class is responsible for creating the ML model
            """

            def __init__(self, hidden_channels):
                super().__init__()
                self.convolutions = []
                operator = get_operator()
                self.conv1 = operator(hidden_channels, hidden_channels)
                self.conv2 = operator(hidden_channels, hidden_channels)

            def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
                """
                This method is responsible for executing the forward pass of the convolutional GNN.

                Parameters
                ----------
                x: torch.Tensor
                    input tensor
                edge_index: torch.Tensor
                    edge index

                Returns
                -------
                x: torch.Tensor
                    output tensor
                """
                # The GNN computation graph.
                # `ReLU` is the standard non-lineary function used in-between, but it can be changed by users' option in the function get_activation
                activation = get_activation()
            
                x = activation(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x

        class Model(torch.nn.Module):
            """
            This class is responsible for creating the model with two SageConv layers with a addiotinal classifier to be used in the training.
            """

            def __init__(self, hidden_channels, data):
                super().__init__()
                # When the dataset does not come with rich features, we also learn two
                # embedding matrices for Left and Right:
                if hasattr(data[s], 'x') and data[s].x != None:
                    self.left_lin = torch.nn.Linear(data[s].num_features, hidden_channels)
                if hasattr(data[t], 'x') and data[t].x != None:
                    self.right_lin = torch.nn.Linear(data[t].num_features, hidden_channels)
                self.left_emb = torch.nn.Embedding(data[s].num_nodes, hidden_channels)
                self.right_emb = torch.nn.Embedding(data[t].num_nodes, hidden_channels)

                # Instantiate homogeneous GNN:
                self.gnn = GNN(hidden_channels)
                # Convert GNN model into a heterogeneous variant:
                self.gnn = to_hetero(self.gnn, metadata=data.metadata())
                if CLASSIFIER == "dot_product":
                    self.classifier = DotProduct()

            def forward(self, data) -> torch.Tensor:
                """	
                This method is responsible for executing the forward pass of the model.

                Parameters
                ----------
                data: HeteroData
                    input data

                Returns
                -------
                pred: torch.Tensor
                    output tensor
                """
                
                # if not hasattr(self.__class__, 'left_lin') or not callable(getattr(self.__class__, 'left_lin')):
                #     x_dict = {
                #         s: self.left_emb(data[s].node_id),
                #         t: self.right_lin(data[t].x) + self.right_emb(data[t].node_id),
                #     }
                # elif not hasattr(self.__class__, 'right_lin') or not callable(getattr(self.__class__, 'right_lin')):
                #     x_dict = {
                #         s: self.left_lin(data[s].x) + self.left_emb(data[s].node_id),
                #         t: self.right_emb(data[t].node_id),
                #     }
                # else:
                #     x_dict = {
                #         s: self.left_lin(data[s].x) + self.left_emb(data[s].node_id),
                #         t: self.right_lin(data[t].x) + self.right_emb(data[t].node_id),
                #     }
                
                x_dict = {}

                if hasattr(self.__class__, 'left_lin') and callable(getattr(self.__class__, 'left_lin')):
                    x_dict[s] = self.left_lin(data[s].x) + self.left_emb(data[s].node_id)
                else:
                    x_dict[s] = self.left_emb(data[s].node_id)

                if hasattr(self.__class__, 'right_lin') and callable(getattr(self.__class__, 'right_lin')):
                    x_dict[t] = self.right_lin(data[t].x) + self.right_emb(data[t].node_id)
                else:
                    x_dict[t] = self.right_emb(data[t].node_id)
                
                # `x_dict` holds feature matrices of all node types
                # `edge_index_dict` holds all edge indices of all edge types
                x_dict = self.gnn(x_dict, data.edge_index_dict)
                pred = self.classifier(
                    x_dict[s],
                    x_dict[t],
                    data[s, relation_name, t].edge_label_index,
                )
                return pred
            
        model = Model(hidden_channels=int(HIDDEN_CHANNELS), data=data)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(LEARNING_RATE))

        for epoch in range(1, int(EPOCHS)):
            total_loss = total_examples = 0
            for sampled_data in tqdm.tqdm(train_loader):
                optimizer.zero_grad()

                sampled_data.to(device)
                pred = model(sampled_data)

                ground_truth = sampled_data[s, relation_name, t].edge_label
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

                loss.backward()
                optimizer.step()
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()
            if epoch % 50 == 0:
                print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

        # Define the validation seed edges:
        edge_label_index = val_data[s, relation_name, t].edge_label_index
        edge_label = val_data[s, relation_name, t].edge_label

        val_loader = LinkNeighborLoader(
            data=val_data,
            num_neighbors=[20, 10],
            edge_label_index=(
                (s, relation_name, t), 
                val_data[(s, relation_name, t)].edge_label_index,
            ),
            edge_label=val_data[(s, relation_name, t)].edge_label,
            batch_size=3 * 128,
            shuffle=False,
        )

        preds = []
        ground_truths = []
        for sampled_data in tqdm.tqdm(val_loader):
            with torch.no_grad():
                sampled_data.to(device)
                preds.append(model(sampled_data))
                ground_truths.append(sampled_data[s, relation_name, t].edge_label)

        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

        # Metric 1 - ROC
        auc = roc_auc_score(ground_truth, pred)
        print(f"Validation ROC_AUC: {auc:.4f}")

        fpr, tpr, roc_tr = roc_curve(ground_truth, torch.sigmoid(torch.from_numpy(pred)))
        print_roc_curve(fpr, tpr, relation_name + "_Validation", auc, True)


        # Metric 2 - Precision Recall
        no_skill = len(ground_truth[ground_truth==1]) / len(ground_truth)
        precision, recall, _ = precision_recall_curve(ground_truth, (torch.sigmoid(torch.from_numpy(pred))))
        pr_auc = area_curve (recall, precision)
        print_pr_curve(precision, recall, relation_name + "_Validation", pr_auc, no_skill, True)

        roc_auc = area_curve(fpr, tpr)
        print("Area under the ROC curve : %f" % roc_auc)

        from sklearn.metrics import confusion_matrix
        # threshold = 0.5  # Adjust threshold as needed
        # pred_labels = np.where(pred >= threshold, 1, 0)
        # confusion = confusion_matrix(ground_truth, pred_labels)
        # print("Confusion Matrix:")
        # print("                  Predicted Negative   Predicted Positive")
        # print("Actual Negative         TN                   FP")
        # print("Actual Positive         FN                   TP")
        # print(confusion)
        thresholds = np.arange(0.1, 1.1, 0.1)  # Threshold values from 0.1 to 1.0
        # pred_labels = np.where(pred >= thresholds[:, np.newaxis], 1, 0)
        # confusion = confusion_matrix(ground_truth, pred_labels)

        # print("Confusion Matrix:")
        # print("                  Predicted Negative   Predicted Positive")
        # print("Actual Negative         TN                   FP")
        # print("Actual Positive         FN                   TP")
        # print(confusion)

        # Calculate FPR and TPR for each threshold
        fpr, tpr, _ = roc_curve(ground_truth, pred)

        print("Threshold\tFPR\t\tTPR")
        for threshold, fpr_value, tpr_value in zip(thresholds, fpr, tpr):
            print(f"{threshold:.1f}\t\t{fpr_value:.4f}\t\t{tpr_value:.4f}")

        ####################################
        # The optimal cut off would be where tpr is high and fpr is low
        # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
        ####################################
        #TODO: Check again this method
        # i = np.arange(len(tpr)) # index for df
        # roc = pd.DataFrame({
        #     'fpr' : pd.Series(fpr, index=i),
        #     'tpr' : pd.Series(tpr, index = i), 
        #     '1-fpr' : pd.Series(1-fpr, index = i), 
        #     'tf' : pd.Series(tpr - fpr, index = i), 
        #     'thresholds' : pd.Series(roc_tr, index = i)})
        
        # print(roc)

        # # Convert index to strings
        # roc.index = roc.index.map(str)

        # # Specify the desired range of columns
        # thresholds_range = ['thresholds'] + [str(x) for x in np.arange(0, 1.1, 0.1)]

        # # Round the threshold values to desired decimal places
        # roc['thresholds_rounded'] = roc['thresholds'].round(decimals=1)

        # # Check if rounded threshold values exist in the desired range
        # not_found = [col for col in thresholds_range if col not in roc['thresholds_rounded'].astype(str)]

        # if not_found:
        #     for key in not_found:
        #         print(f"Key '{key}' not found in DataFrame columns.")

        # print(roc.loc[:, thresholds_range])
        
        # model = model.cpu()
        # model.eval() 

        # total_s = len(left_original_mapping) 
        # total_t = len(right_original_mapping)

        # threshold = roc.iloc[(roc.tf-0).abs().argsort()[:1]]['thresholds']
        # predictions = {}

        # for uri_idenifier_left, left_id in tqdm.tqdm(left_original_mapping.items()):
        #     predictions[uri_idenifier_left] = []

        #     left_row = torch.tensor([left_id] * total_t) 
        #     all_right_ids = torch.arange(total_t) 
        #     edge_label_index = torch.stack([left_row, all_right_ids], dim=0) 
        #     data[s, relation_name, t].edge_label_index = edge_label_index
            
        #     with torch.no_grad(): 
        #         pred = model(data)
        #     # cut off by threshold
        #     optimal_pred = (pred > threshold[0]).long()
        #     probabilities = torch.sigmoid(pred)
        #     pred_labels = (probabilities > threshold[0]).long()

        #     recommended_links = all_right_ids[pred_labels == 1].tolist()

        #     predictions[uri_idenifier_left] = recommended_links

        #     predicted_path = osp.join(Path(__file__).parent, '..', VIEWS_DIRECTORY, VIEW_NAME, "recommendations.json")

        #     inv_right_mapping = {v: k for k, v in right_original_mapping.items()}
        #     json_dict = {}
        #     #iterate over the predictions and create the JSON
        #     for uri_idenifier_left, potential_links in predictions.items():
        #         json_dict[uri_idenifier_left] = [inv_right_mapping[x] for x in potential_links]

        #     with open(predicted_path, 'w+') as f:
        #         json.dump(json_dict, f)