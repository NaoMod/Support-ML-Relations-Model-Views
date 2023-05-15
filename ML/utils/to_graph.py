from typing import Tuple, Optional
import torch
import pandas as pd
import json

from torch_geometric.data import HeteroData

from utils.encoders import IdentityEncoder, SequenceEncoder

class Func():

    def __init__(self, sentence_encoding_name: Optional[str] = "all-MiniLM-L6-v2", features_to_ignore_left = None, features_to_ignore_right = None, unique_id_left= None, unique_id_right = None):
        
        self.sentence_encoding_name = sentence_encoding_name
        self.features_to_ignore_left = features_to_ignore_left
        self.features_to_ignore_right = features_to_ignore_right
        self.unique_id_left = unique_id_left
        self.unique_id_right = unique_id_right


    def xmi_to_graph(self, model_root_left, model_root_right, relations_csv_path, class_left, class_right, relation_name):
        data = HeteroData()

        Left_wrapper = {}
        Right_wrapper = {}

        for element in model_root_left:
            if self.unique_id_left is None:
                if 'uriFragment' not in Left_wrapper:
                    Left_wrapper['uriFragment'] = []
                Left_wrapper['uriFragment'].append(element.eURIFragment())

            for attribute in element.eClass.eAttributes:
                attributeName = attribute.name
                if attributeName not in Left_wrapper:
                    Left_wrapper[attributeName] = []
                Left_wrapper[attributeName].append(element.eGet(attribute))       

        for element in model_root_right:
            if self.unique_id_right is None:
                if 'uriFragment' not in Right_wrapper:
                    Right_wrapper['uriFragment'] = []
                Right_wrapper['uriFragment'].append(element.eURIFragment())

            for attribute in element.eClass.eAttributes:
                attributeName = attribute.name
                if attributeName not in Right_wrapper:
                    Right_wrapper[attributeName] = []
                Right_wrapper[attributeName].append(element.eGet(attribute))
                
        df_left = pd.DataFrame(Left_wrapper)
        df_right = pd.DataFrame(Right_wrapper)
        
        df_rels = pd.read_csv(relations_csv_path)
        left_id, right_id = df_rels.columns.values
        if self.unique_id_left is None or self.unique_id_right is None:
            #adjust the problem with / and /0 in URI fragments
            df_rels.loc[(df_rels[left_id] == "/"), left_id] = "/0"
            df_rels.loc[(df_rels[right_id] == "/"), right_id] = "/0"

        left_index = self.unique_id_left or 'uriFragment'
        right_index = self.unique_id_right or 'uriFragment'

        df_left = df_left.set_index(left_index, drop=False)
        df_right = df_right.set_index(right_index, drop=False)

        data[class_left].x, left_mapping = self._load_nodes(df_left, self.features_to_ignore_left)
        data[class_left].num_nodes = len(left_mapping)
        data[class_left].node_id = torch.Tensor(list(left_mapping.values())).long()

        data[class_right].x, right_mapping = self._load_nodes(df_right, self.features_to_ignore_right)
        data[class_right].num_nodes = len(right_mapping)
        data[class_right].node_id = torch.Tensor(list(right_mapping.values())).long()

        
        edge_index, edge_label = self._load_edges(
            src_index_col=left_id,
            src_mapping=left_mapping,
            dst_index_col=right_id,
            dst_mapping=right_mapping,
            df_rels=df_rels,
            encoders=None,
        )

        # Add the edge indices
        data[class_left, relation_name, class_right].edge_index = edge_index

        if edge_label is not None:
            data[class_left, relation_name, class_right].y = edge_label

        return data, left_mapping, right_mapping

    def _load_nodes(self, df, features_to_ignore) -> Tuple[torch.Tensor, dict]:
        """
        Load the node features

        Parameters
        ---------- 
        df: DataFrame
            Dataframe that contains nodes information to be encoded.
        features_to_ignore: lst
            List of features to be inored during encoding.

        Returns
        -------
        node_features: Tensor
            node features for the graph
        mapping: dict
            mapping of the node index (Used to create the edge index)
        """

        encoders = {}
        
        for column_name, _ in df.items():
            if features_to_ignore is None or column_name not in features_to_ignore:
                # Define the encoders for each column
                if df[column_name].dtype.kind in 'biufc':
                    # biufc means "numeric" columns. bool, int, uint, float, complex
                    encoders[column_name] = IdentityEncoder(dtype=torch.float)
                else:
                    encoders[column_name] = SequenceEncoder(self.sentence_encoding_name)

        mapping = {index: i for i, index in enumerate(df.index.unique())}

        x = None
        if encoders is not None and len(encoders) > 0:
            xs = [encoder(df[col]) for col, encoder in encoders.items()]
            x = torch.cat(xs, dim=-1)

        return x, mapping

    def _load_edges(self, src_index_col, src_mapping, dst_index_col, dst_mapping,
                    df_rels, encoders=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load the edge features and edge index

        Parameters
        ----------
        src_index_col: str
            column name of the source node index
        src_mapping: dict
            mapping of the source node index
        dst_index_col: str
            column name of the destination node index
        dst_mapping: dict
            mapping of the destination node index
        df_rels: Dataframe
            Dataframe connecting src and dst
        encoders: dict
            dictionary of encoders for the edge attributes

        Returns
        -------
        edge_index: Tensor
            edge index for the graph
        edge_attr: Tensor
            edge attributes for the graph
        """

        edge_attr = None
        edge_index = None
        if df_rels is None:
            return edge_index, edge_attr
        # src = []
        # for index in df_rels[src_index_col]:
        #     src.append(src_mapping[index]) 

        src = [src_mapping[index] for index in df_rels[src_index_col]]
        dst = [dst_mapping[index] for index in df_rels[dst_index_col]]
        edge_index = torch.tensor([src, dst])

        edge_attr = None
        if encoders is not None:
            edge_attrs = [encoder(df_rels[col])
                            for col, encoder in encoders.items()]
            edge_attr = torch.cat(edge_attrs, dim=-1)

        return edge_index, edge_attr
