from typing import Tuple, Optional
import torch
import pandas as pd

from torch_geometric.data import HeteroData

from utils.encoders import IdentityEncoder, SequenceEncoder, EnumEncoder, NoneEncoder

class ToGraph():

    def __init__(self, embeddings_information, features_for_embedding_left = None, features_for_embedding_right = None):
        
        self.features_for_embedding_left = [feature for feature in embeddings_information if feature.startswith("s")]
        self.features_for_embedding_right = [feature for feature in embeddings_information if feature.startswith("t")]
        self.embeddings_information = embeddings_information


    def xmi_to_graph(self, model_root_left, model_root_right, relations_csv_path, class_left, class_right, relation_name):
        data = HeteroData()

        Left_wrapper = {}
        Right_wrapper = {}

        attributes_left_class = [s.split(".", 1)[1] for s in self.features_for_embedding_left if s.count(".") == 1]
        #TODO: To get attributes from other classes and implement navigation
        unique_id_left = None
        for element in model_root_left:
            className = element.eClass.name
            if className == class_left:
                for attribute in element.eClass.eAttributes:
                    attributeName = attribute.name
                    if attribute.unique and unique_id_left is None:
                        # important to store the unique attribute to identify the element among all models
                        unique_id_left = attribute.name
                    if  self.features_for_embedding_left is not None and (attributeName in attributes_left_class or attributeName == unique_id_left):
                        if attributeName not in Left_wrapper:
                            Left_wrapper[attributeName] = []
                        Left_wrapper[attributeName].append(element.eGet(attribute))      

        attributes_right_class = [s.split(".",1)[1] for s in self.features_for_embedding_right if s.count(".") == 1]
        sub_right_class = [s.split("->",1)[1] for s in self.features_for_embedding_right if s.count("->") == 1]
        unique_id_right = None
        for element in model_root_right:
            className = element.eClass.name
            if className == class_right:
                for attribute in element.eClass.eAttributes:
                    attributeName = attribute.name
                    if attribute.unique and unique_id_right is None:
                        # important to store the unique attribute to identify the element among all models
                        unique_id_right = attribute.name
                    if  self.features_for_embedding_right is not None and (attributeName in attributes_right_class or attributeName == unique_id_right):
                        if attributeName not in Right_wrapper:
                            Right_wrapper[attributeName] = []
                        Right_wrapper[attributeName].append(element.eGet(attribute))
                #TODO: Too much specific for the example
                if sub_right_class != []:
                    sub_class_name, attributes_sub_class = sub_right_class[0].split(".")
                    values = ""
                    for sub_class_instance in element.eGet(sub_class_name):
                        for attribute in sub_class_instance.eClass.eAttributes:
                            attributeName = attribute.name
                            if  self.features_for_embedding_right is not None and (attributeName in attributes_sub_class):
                                if values == "":
                                    values = sub_class_instance.eGet(attribute)
                                else:
                                    values += "|" + sub_class_instance.eGet(attribute)
                    if sub_right_class[0] not in attributes_right_class:
                        attributes_right_class.append(sub_right_class[0])
                    if sub_right_class[0] not in Right_wrapper:
                        Right_wrapper[sub_right_class[0]] = []
                    Right_wrapper[sub_right_class[0]].append(values)
                
        df_left = pd.DataFrame(Left_wrapper)
        df_right = pd.DataFrame(Right_wrapper)
        
        df_rels = pd.read_csv(relations_csv_path)
        left_id, right_id = df_rels.columns.values[0:2]

        df_left = df_left.set_index(unique_id_left, drop=False)
        df_right = df_right.set_index(unique_id_right, drop=False)

        data[class_left].x, left_mapping = self._load_nodes(class_left, df_left, attributes_left_class)
        data[class_left].num_nodes = len(left_mapping)
        data[class_left].node_id = torch.Tensor(list(left_mapping.values())).long()

        data[class_right].x, right_mapping = self._load_nodes(class_right, df_right, attributes_right_class)
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

    def _load_nodes(self, class_name, df, features_for_embedding) -> Tuple[torch.Tensor, dict]:
        """
        Load the node features

        Parameters
        ---------- 
        class_name: str
            Name of the class to check the embedding type.
        df: DataFrame
            Dataframe that contains nodes information to be encoded.
        features_for_embedding: lst
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
            if features_for_embedding is not None and column_name in features_for_embedding:
                #get the complete identifier of attribute to verify the list
                idx_name = class_name + '.' + column_name
                #check if the user defined a specific embedding technique
                if idx_name in self.embeddings_information:
                    if self.embeddings_information[idx_name] == 'number':
                        encoders[column_name] = IdentityEncoder(dtype=torch.float)
                    elif self.embeddings_information[idx_name] == 'enum':
                        encoders[column_name] = EnumEncoder(dtype=torch.float)
                    elif self.embeddings_information[idx_name] == 'string':
                        encoders[column_name] = SequenceEncoder()
                    elif self.embeddings_information[idx_name] == 'one-hot-encoding':
                         #TODO
                        raise NotImplemented
                    elif self.embeddings_information[idx_name] == 'id':
                        encoders[column_name] = NoneEncoder(dtype=torch.float)
                    else:
                        #consider pre-trained language model from sentence-transformers
                        encoders[column_name] = SequenceEncoder(self.embeddings_information[idx_name])
                else:
                    # Fallback to define the encoders for each column when user didn't define it
                    if df[column_name].dtype.kind in 'biufc':
                        # biufc means "numeric" columns. bool, int, uint, float, complex
                        encoders[column_name] = IdentityEncoder(dtype=torch.float)
                    else:
                        encoders[column_name] = SequenceEncoder()

        mapping = {index: i for i, index in enumerate(df.index.unique())}

        x = None
        if encoders is not None and len(encoders) > 0:
            xs = [encoder(df[col]) for col, encoder in encoders.items()]
            if xs is not None and xs != [None]:
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
        
        src = [src_mapping[index] for index in df_rels[src_index_col]]
        dst = [dst_mapping[index] for index in df_rels[dst_index_col]]
        edge_index = torch.tensor([src, dst])

        edge_attr = None
        if encoders is not None:
            edge_attrs = [encoder(df_rels[col])
                            for col, encoder in encoders.items()]
            edge_attr = torch.cat(edge_attrs, dim=-1)

        return edge_index, edge_attr
    
    def get_attributtes(model_root, check_class_name, features_for_embedding, wrapper):
        attributes_of_class = [s.split(".",1)[1] for s in features_for_embedding if s.count(".") == 1]

        # not_filtered_attributes = [s for s in features_for_embedding if s not in attributes_of_class]
        # for element in model_root:
        #     className = element.eClass.name
        #     if className == check_class_name:
        #         for attribute in element.eClass.eAttributes:
        #             attributeName = attribute.name
        #             if  features_for_embedding is not None and attributeName in attributes_of_class:
        #                 if attributeName not in wrapper:
        #                     wrapper[attributeName] = []
        #                 wrapper[attributeName].append(element.eGet(attribute))

        #         if len(not_filtered_attributes) > 0:
        #             # call the function again for the children
        #             get_attributtes(element, check_class_name, features_for_embedding, wrapper)

        unique_id = None
        for element in model_root:
            className = element.eClass.name
            if className == check_class_name:
                for attribute in element.eClass.eAttributes:
                    attributeName = attribute.name
                    if attribute.unique and unique_id is None:
                        # important to store the unique attribute to identify the element among all models
                        unique_id = attribute.name
                    if  features_for_embedding is not None and (attributeName in attributes_of_class or attributeName == unique_id):
                        if attributeName not in wrapper:
                            wrapper[attributeName] = []
                        wrapper[attributeName].append(element.eGet(attribute))
        
