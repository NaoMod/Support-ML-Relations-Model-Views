import numpy as np
import pandas as pd
import torch

from pyecore.resources import ResourceSet, URI

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

from utils.print_curves import print_pr_curve, print_roc_curve
from utils.to_graph import ToGraph

VIEWS_DIRECTORY = 'Views'
VIEW_NAME = 'Movies_Users'

# Read JSON files to get GNN properties
json_path = glob.glob(osp.join(Path(__file__).parent, '..', VIEWS_DIRECTORY, VIEW_NAME, 'src-gen', "recommended.json"))[0]
json_training_path = glob.glob(osp.join(Path(__file__).parent, '..', VIEWS_DIRECTORY, VIEW_NAME, "training.json"))[0]
#TODO: replace Data by the models in training.json
data_path = glob.glob(osp.join(Path(__file__).parent, '..', 'Data', "AB"))[0]

parameters_for_view_exist = len(json_path) != 0
training_exist = len(json_training_path) != 0

parameters_for_view = None
if parameters_for_view_exist:
    parameters_for_view = json_path

with open(parameters_for_view) as json_data:
    parameters = json.load(json_data)

    for relation_name, relation_props in parameters.items():
        class_left = relation_props['CLASS_LEFT']
        class_right = relation_props['CLASS_RIGHT']
        rev_relation_name = "rev_" + relation_name

        parameters_for_gnn = None

        if training_exist:
            parameters_for_gnn = json_training_path

        with open(parameters_for_gnn) as json_training_data:
            training_parameters = json.load(json_training_data)

            EPOCHS = training_parameters[relation_name]["TRAINING_PARAMETERS"]["EPOCHS"]
            LEARNING_RATE = training_parameters[relation_name]["TRAINING_PARAMETERS"]["LEARNING_RATE"]
            ADD_NEGATIVE_TRAINING = training_parameters[relation_name]["TRAINING_PARAMETERS"]["ADD_NEGATIVE_TRAINING"]
            NEG_SAMPLING_RATIO = training_parameters[relation_name]["TRAINING_PARAMETERS"]["NEG_SAMPLING_RATIO"]

            GNN_OPERATOR = training_parameters[relation_name]["ARCHITECTURE"]["OPERATOR"]
            CONVOLUTIONS = training_parameters[relation_name]["ARCHITECTURE"]["CONVOLUTIONS"]
            ACTIVATION = training_parameters[relation_name]["ARCHITECTURE"]["ACTIVATION"]
            HIDDEN_CHANNELS = training_parameters[relation_name]["ARCHITECTURE"]["HIDDEN_CHANNELS"]
            CLASSIFIER = training_parameters[relation_name]["ARCHITECTURE"]["CLASSIFIER"]

            # Register the metamodels in the resource set    
            resource_set = ResourceSet()
            modeling_resources_path = glob.glob(osp.join(Path(__file__).parent, '..','Modeling_Resources'))[0]

            #TODO: Include as parameters in training
            ecore_path_left = glob.glob(osp.join(modeling_resources_path, 'metamodels/UserMovies.ecore'))[0]
            ecore_path_right =glob.glob(osp.join(modeling_resources_path, 'metamodels/UserMovies.ecore'))[0]

            resource_path = resource_set.get_resource(URI(ecore_path_left))
            root_pkg = resource_path.contents[0]
            
            contents = root_pkg.eContents

            resource_set.metamodel_registry[contents[0].nsURI] = contents[0]
            resource_set.metamodel_registry[contents[1].nsURI] = contents[1]
	                
                

            # resource_left = resource_set.get_resource(URI(ecore_path_left))
            # mm_root_left = resource_left.contents[0]
            # resource_right = resource_set.get_resource(URI(ecore_path_right))
            # mm_root_right = resource_right.contents[0]

            # resource_set.metamodel_registry[mm_root_left.nsURI] = mm_root_left
            # resource_set.metamodel_registry[mm_root_right.nsURI] = mm_root_right

            #if EMBEDDINGS_LEFT is not None:
            features_for_embedding_left = relation_props['CLASS_LEFT_EMBEDDINGS'].split(',')
            #else:
            #    features_for_embedding_left = None
            #if EMBEDDINGS_RIGHT is not None:
            features_for_embedding_right = relation_props['CLASS_RIGHT_EMBEDDINGS'].split(',')
            #else:
            #    features_for_embedding_right = None

            dataset_func = ToGraph(embeddings_information=training_parameters[relation_name]["EMBEDDINGS"], features_for_embedding_left=features_for_embedding_left, features_for_embedding_right=features_for_embedding_right)
            # Register the models in the resource set
            xmi_path_left = glob.glob(osp.join(modeling_resources_path, training_parameters[relation_name]["TRAINING_PARAMETERS"]["LEFT_PATH"]))[0]
            m_resource_left = resource_set.get_resource(URI(xmi_path_left))
            model_root_left = m_resource_left.contents

            xmi_path_right = glob.glob(osp.join(modeling_resources_path, training_parameters[relation_name]["TRAINING_PARAMETERS"]["RIGHT_PATH"]))[0]
            m_resource_right = resource_set.get_resource(URI(xmi_path_right))
            model_root_right = m_resource_right.contents

            relations_path = glob.glob(osp.join(modeling_resources_path, training_parameters[relation_name]["TRAINING_PARAMETERS"]["LINK_PATH"]))
            relations_exist = len(relations_path) != 0
            relations_for_graph = None
            if relations_exist:
                relations_for_graph = relations_path[0]  

            data, left_original_mapping, right_original_mapping = dataset_func.xmi_to_graph(model_root_left, model_root_right, relations_for_graph, class_left, class_right, relation_name)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data = data.to(device)

            # Add a reverse (class_right, rev_relation_name, class_left) relation for message passing:
            data = T.ToUndirected()(data)
            del data[class_right, rev_relation_name, class_left].edge_label  # Remove "reverse" label.

            # Perform a link-level split into training, validation, and test edges:
            train_data, val_data, test_data = T.RandomLinkSplit(
                num_val=0.1, # 10% of links for validation
                num_test=0.1, # 10% os links for test
                disjoint_train_ratio=0.4, # use 60% for message passing and 40% for supervision
                neg_sampling_ratio=2.0, # generate 2:1 negative edges
                add_negative_train_samples=ADD_NEGATIVE_TRAINING,
                edge_types=(class_left, relation_name, class_right),
                rev_edge_types=(class_right, rev_relation_name, class_left)
            )(data)

            train_edge_label_index = train_data[class_left, relation_name, class_right].edge_label_index
            train_edge_label = train_data[class_left, relation_name, class_right].edge_label

            # LOADERS
            train_loader = LinkNeighborLoader(
                data=train_data,
                num_neighbors=[20, 10],
                neg_sampling_ratio=2.0,
                edge_label_index=((class_left, relation_name, class_right), train_edge_label_index),
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
                    This method is responsible for executing the forward pass of the classifier.
                    
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
                    self.left_lin = torch.nn.Linear(data[class_left].num_features, hidden_channels)
                    self.right_lin = torch.nn.Linear(data[class_right].num_features, hidden_channels)
                    self.left_emb = torch.nn.Embedding(data[class_left].num_nodes, hidden_channels)
                    self.right_emb = torch.nn.Embedding(data[class_right].num_nodes, hidden_channels)

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
                    
                    if not hasattr(data[class_left], 'x') or data[class_left].x == None:
                        x_dict = {
                            class_left: self.left_emb(data[class_left].node_id),
                            class_right: self.right_lin(data[class_right].x) + self.right_emb(data[class_right].node_id),
                        }
                    elif not hasattr(data[class_right], 'x') or data[class_right].x == None:
                        x_dict = {
                            class_left: self.left_lin(data[class_left].x) + self.left_emb(data[class_left].node_id),
                            class_right: self.right_emb(data[class_right].node_id),
                        }
                    else:
                        x_dict = {
                            class_left: self.left_lin(data[class_left].x) + self.left_emb(data[class_left].node_id),
                            class_right: self.right_lin(data[class_right].x) + self.right_emb(data[class_right].node_id),
                        }
                    
                    # `x_dict` holds feature matrices of all node types
                    # `edge_index_dict` holds all edge indices of all edge types
                    x_dict = self.gnn(x_dict, data.edge_index_dict)
                    pred = self.classifier(
                        x_dict[class_left],
                        x_dict[class_right],
                        data[class_left, relation_name, class_right].edge_label_index,
                    )
                    return pred
                
            model = Model(hidden_channels=HIDDEN_CHANNELS, data=data)

            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=float(LEARNING_RATE))

            for epoch in range(1, int(EPOCHS)):
                total_loss = total_examples = 0
                for sampled_data in tqdm.tqdm(train_loader):
                    optimizer.zero_grad()

                    sampled_data.to(device)
                    pred = model(sampled_data)

                    ground_truth = sampled_data[class_left, relation_name, class_right].edge_label
                    loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss) * pred.numel()
                    total_examples += pred.numel()
                if epoch % 50 == 0:
                    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

            # Define the validation seed edges:
            edge_label_index = val_data[class_left, relation_name, class_right].edge_label_index
            edge_label = val_data[class_left, relation_name, class_right].edge_label

            val_loader = LinkNeighborLoader(
                data=val_data,
                num_neighbors=[20, 10],
                edge_label_index=(
                    (class_left, relation_name, class_right), 
                    val_data[(class_left, relation_name, class_right)].edge_label_index,
                ),
                edge_label=val_data[(class_left, relation_name, class_right)].edge_label,
                batch_size=3 * 128,
                shuffle=False,
            )

            preds = []
            ground_truths = []
            for sampled_data in tqdm.tqdm(val_loader):
                with torch.no_grad():
                    sampled_data.to(device)
                    preds.append(model(sampled_data))
                    ground_truths.append(sampled_data[class_left, relation_name, class_right].edge_label)

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

            ####################################
            # The optimal cut off would be where tpr is high and fpr is low
            # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
            ####################################
            #TODO: Check again this method
            i = np.arange(len(tpr)) # index for df
            roc = pd.DataFrame({
                'fpr' : pd.Series(fpr, index=i),
                'tpr' : pd.Series(tpr, index = i), 
                '1-fpr' : pd.Series(1-fpr, index = i), 
                'tf' : pd.Series(tpr - fpr, index = i), 
                'thresholds' : pd.Series(roc_tr, index = i)})
            
            model = model.cpu()
            model.eval() 

            total_class_left = len(left_original_mapping) 
            total_class_right = len(right_original_mapping)

            threshold = roc.iloc[(roc.tf-0).abs().argsort()[:1]]['thresholds']
            predictions = {}

            for uri_idenifier_left, left_id in tqdm.tqdm(left_original_mapping.items()):
                predictions[uri_idenifier_left] = []

                left_row = torch.tensor([left_id] * total_class_right) 
                all_right_ids = torch.arange(total_class_right) 
                edge_label_index = torch.stack([left_row, all_right_ids], dim=0) 
                data[class_left, relation_name, class_right].edge_label_index = edge_label_index
                
                with torch.no_grad(): 
                    pred = model(data)
                # cut off by threshold
                optimal_pred = (pred > threshold[0]).long()
                probabilities = torch.sigmoid(pred)
                pred_labels = (probabilities > threshold[0]).long()

                recommended_links = all_right_ids[pred_labels == 1].tolist()

                predictions[uri_idenifier_left] = recommended_links

                predicted_path = osp.join(Path(__file__).parent, '..', VIEWS_DIRECTORY, VIEW_NAME, "recommendations.json")

                inv_right_mapping = {v: k for k, v in right_original_mapping.items()}
                json_dict = {}
                #iterate over the predictions and create the JSON
                for uri_idenifier_left, potential_links in predictions.items():
                    json_dict[uri_idenifier_left] = [inv_right_mapping[x] for x in potential_links]

                with open(predicted_path, 'w+') as f:
                    json.dump(json_dict, f)