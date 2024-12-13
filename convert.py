"""for loading data into format for DeepSnap"""

import torch
from torch_sparse import SparseTensor
import numpy as np
import pickle
from pathlib import Path

def load_separate_graphs(input_dir='original'):
    """Load separate conference graphs from PKL files"""
    conference_graphs = {}
    for pkl_file in Path(input_dir).glob('*_graph.pkl'):
        conf_name = pkl_file.stem.replace('_graph', '')
        try:
            # Use a safer loading approach with explicit protocol
            with open(pkl_file, 'rb') as f:
                # Add error handling for pickle loading
                try:
                    conference_graphs[conf_name] = pickle.load(f)
                    print(f"Loaded graph for {conf_name}")
                    # breakpoint()
                except pickle.UnpicklingError as e:
                    print(f"Error loading {conf_name}: {e}")
                    continue
        except Exception as e:
            print(f"Error opening {conf_name}: {e}")
            continue
    return conference_graphs

def combine_conference_graphs(input_dir='original', 
                            output_dir='processed_graphs_combined',
                            train_confs=['USENIX', 'ACM'], 
                            val_confs=['IEEE'], 
                            test_confs=['NDSS'],
                            subject_threshold=0.3,
                            use_american_gt=True
                          ):
    """Load and combine multiple conference graphs from separate PKLs"""
    Path(output_dir).mkdir(exist_ok=True)
    
    conference_graphs = load_separate_graphs(input_dir)
    
    all_features = []
    all_labels = []
    
    curr_paper_offset = 0
    curr_author_offset = 0
    
    pap_edges_list = []
    psp_edges_list = []
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for conf_name, graph in conference_graphs.items():
        curr_indices = torch.arange(curr_paper_offset, curr_paper_offset + len(graph['paper'].x))
        if conf_name in train_confs:
            train_indices.append(curr_indices)
        elif conf_name in val_confs:
            val_indices.append(curr_indices)
        elif conf_name in test_confs:
            test_indices.append(curr_indices)
        else:
            print(f"Skipping conference {conf_name} - not in any split")
            continue
            
        num_papers = len(graph['paper'].x)
        
        # breakpoint()
        all_features.append(graph['paper'].x)
        if use_american_gt:
            # For American GT, use the last column of university features
            # Get university affiliations for each person
            person_univ_edges = graph['person', 'affiliated_with', 'university'].edge_index
            university_features = graph['university'].x
            
            # Create labels based on university American status
            person_labels = torch.zeros(len(graph['person'].x))
            for person_idx, univ_idx in zip(person_univ_edges[0], person_univ_edges[1]):
                person_labels[person_idx] = university_features[univ_idx, -1]  # Last column has American status
                
            # breakpoint()
            all_labels.append(person_labels.long())
        else:
            # Original GT: collusion labels from person features column 1
            all_labels.append((graph['person'].x[:, 1] > 0).long())
        
        # Paper-author-paper connections
        paper_author_edges = graph['paper', 'written_by', 'person'].edge_index.clone()
        paper_author_edges[0] += curr_paper_offset
        paper_author_edges[1] += curr_author_offset
        
        paper_author_adj = SparseTensor(
            row=paper_author_edges[0],
            col=paper_author_edges[1],
            sparse_sizes=(curr_paper_offset + num_papers, curr_author_offset + len(graph['person'].x))
        )
        
        pap = paper_author_adj @ paper_author_adj.t()
        indices = torch.stack([
            pap.storage.row(),
            pap.storage.col()
        ])
        pap_edges_list.append(indices)
        
        # Paper-subject-paper connections according to similarity threshold
        paper_embeddings = graph['paper'].x
        similarity_matrix = torch.mm(paper_embeddings, paper_embeddings.t())
        paper_subject_edges = (similarity_matrix > subject_threshold).nonzero().t()
        paper_subject_edges[0] += curr_paper_offset
        paper_subject_edges[1] += curr_paper_offset
        psp_edges_list.append(paper_subject_edges)
        
        curr_paper_offset += num_papers
        curr_author_offset += len(graph['person'].x)

    # Combine all data
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Concatenate and format edge indices
    pap_edges = torch.cat(pap_edges_list, dim=1)
    psp_edges = torch.cat(psp_edges_list, dim=1)
    
    # Create sparse tensors for the final edge indices
    num_nodes = features.size(0)
    pap_sparse = SparseTensor(
        row=pap_edges[0],
        col=pap_edges[1],
        sparse_sizes=(num_nodes, num_nodes)
    )
    psp_sparse = SparseTensor(
        row=psp_edges[0],
        col=psp_edges[1],
        sparse_sizes=(num_nodes, num_nodes)
    )
    
    # Convert sparse tensors to edge indices in COO format
    pap_final = torch.stack([pap_sparse.storage.row(), pap_sparse.storage.col()])
    psp_final = torch.stack([psp_sparse.storage.row(), psp_sparse.storage.col()])
    
    combined_data = {
        'feature': features,
        'label': labels,
        'train_idx': torch.cat(train_indices) if train_indices else torch.tensor([], dtype=torch.long),
        'val_idx': torch.cat(val_indices) if val_indices else torch.tensor([], dtype=torch.long),
        'test_idx': torch.cat(test_indices) if test_indices else torch.tensor([], dtype=torch.long),
        'pap': pap_final,
        'psp': psp_final
    }
    
    # Save with different filename based on GT type
    filename = "combined_conferences_american.pkl" if use_american_gt else "combined_conferences.pkl"
    output_path = Path(output_dir) / filename
    save_combined_data(combined_data, output_path)
    
    print(f"\nSaved combined conference data to {output_path}")
    print(f"Number of papers: {len(combined_data['feature'])}")
    print(f"Number of PAP edges: {pap_final.size(1)}")
    print(f"Number of PSP edges: {psp_final.size(1)}")
    
    # Verify data format
    print("\nData format verification:")
    print(f"Feature shape: {combined_data['feature'].shape}")
    print(f"Label shape: {combined_data['label'].shape}")
    print(f"PAP edge index shape: {combined_data['pap'].shape}")
    print(f"PSP edge index shape: {combined_data['psp'].shape}")
    print(f"Train indices shape: {combined_data['train_idx'].shape}")
    print(f"Val indices shape: {combined_data['val_idx'].shape}")
    print(f"Test indices shape: {combined_data['test_idx'].shape}")

def save_combined_data(combined_data, output_path):
    """Safely save the combined data"""
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(combined_data, f, protocol=4)
        print(f"Successfully saved combined data to {output_path}")
    except Exception as e:
        print(f"Error saving combined data: {e}")

def load_and_process_combined_data(pkl_path, device):
    """Load and process the combined conference data for training"""
    try:
        with open(pkl_path, 'rb') as f:
            try:
                data = pickle.load(f)
            except pickle.UnpicklingError as e:
                print(f"Error unpickling data: {e}")
                return None
    except Exception as e:
        print(f"Error opening file: {e}")
        return None
    
    # Move data to device
    try:
        for split in data:
            # Move node features
            for node_type in data[split]['node_feature']:
                data[split]['node_feature'][node_type] = data[split]['node_feature'][node_type].to(device)
            
            # Move node labels
            for node_type in data[split]['node_label']:
                data[split]['node_label'][node_type] = data[split]['node_label'][node_type].to(device)
            
            # Move edge indices
            for message_type in data[split]['edge_index']:
                data[split]['edge_index'][message_type] = data[split]['edge_index'][message_type].to(device)
            
            # Move paper indices
            if isinstance(data[split]['paper_indices'], torch.Tensor):
                data[split]['paper_indices'] = data[split]['paper_indices'].to(device)
    except Exception as e:
        print(f"Error moving data to device: {e}")
        return None
    
    return data

def main():
    # Combine separate PKL files and create train/val/test splits
    combine_conference_graphs()
    
    # Example of loading for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    combined_data = load_and_process_combined_data(
        'processed_graphs_combined/combined_conferences.pkl', 
        device
    )

if __name__ == '__main__':
    main()