"""for loading data of just one graph into format for DeepSnap (for debugging)"""

import torch
from torch_sparse import SparseTensor
import numpy as np
import pickle
from pathlib import Path

def load_usenix_graph(input_dir='original'):
    """Load USENIX conference graph from PKL file"""
    pkl_file = Path(input_dir) / 'USENIX_graph.pkl'
    try:
        with open(pkl_file, 'rb') as f:
            try:
                graph = pickle.load(f)
                print("Loaded USENIX graph successfully")
                return graph
            except pickle.UnpicklingError as e:
                print(f"Error loading USENIX graph: {e}")
                return None
    except Exception as e:
        print(f"Error opening USENIX graph file: {e}")
        return None

def process_usenix_graph(input_dir='original', 
                        output_dir='processed_usenix',
                        train_ratio=0.7,
                        val_ratio=0.15,
                        test_ratio=0.15,
                        subject_threshold=0.5,
                        use_american_gt=True):
    """Process USENIX conference graph and create train/val/test splits"""
    Path(output_dir).mkdir(exist_ok=True)
    
    graph = load_usenix_graph(input_dir)
    if graph is None:
        return
    
    # Get paper features and author labels
    features = graph['paper'].x
    if use_american_gt:
        person_univ_edges = graph['person', 'affiliated_with', 'university'].edge_index
        university_features = graph['university'].x
        
        person_labels = torch.zeros(len(graph['person'].x))
        for person_idx, univ_idx in zip(person_univ_edges[0], person_univ_edges[1]):
            person_labels[person_idx] = university_features[univ_idx, -1]
        
        labels = person_labels.long()
    else:
        labels = (graph['person'].x[:, 1] > 0).long()
    
    # Create paper-author-paper connections
    paper_author_edges = graph['paper', 'written_by', 'person'].edge_index
    paper_author_adj = SparseTensor(
        row=paper_author_edges[0],
        col=paper_author_edges[1],
        sparse_sizes=(len(features), len(labels))
    )
    
    pap = paper_author_adj @ paper_author_adj.t()
    pap_edges = torch.stack([
        pap.storage.row(),
        pap.storage.col()
    ])
    
    # Create paper-subject-paper connections
    paper_embeddings = features
    similarity_matrix = torch.mm(paper_embeddings, paper_embeddings.t())
    psp_edges = (similarity_matrix > subject_threshold).nonzero().t()
    
    # Create random splits
    num_papers = len(features)
    indices = torch.randperm(num_papers)
    
    train_size = int(train_ratio * num_papers)
    val_size = int(val_ratio * num_papers)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    processed_data = {
        'feature': features,
        'label': labels,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'pap': pap_edges,
        'psp': psp_edges
    }
    
    # Save processed data
    filename = "usenix_processed_american.pkl" if use_american_gt else "usenix_processed.pkl"
    output_path = Path(output_dir) / filename
    save_processed_data(processed_data, output_path)
    
    print(f"\nSaved processed USENIX data to {output_path}")
    print(f"Number of papers: {len(features)}")
    print(f"Number of PAP edges: {pap_edges.size(1)}")
    print(f"Number of PSP edges: {psp_edges.size(1)}")
    
    # Verify data format
    print("\nData format verification:")
    print(f"Feature shape: {processed_data['feature'].shape}")
    print(f"Label shape: {processed_data['label'].shape}")
    print(f"PAP edge index shape: {processed_data['pap'].shape}")
    print(f"PSP edge index shape: {processed_data['psp'].shape}")
    print(f"Train indices shape: {processed_data['train_idx'].shape}")
    print(f"Val indices shape: {processed_data['val_idx'].shape}")
    print(f"Test indices shape: {processed_data['test_idx'].shape}")

def save_processed_data(processed_data, output_path):
    """Safely save the processed data"""
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f, protocol=4)
        print(f"Successfully saved processed data to {output_path}")
    except Exception as e:
        print(f"Error saving processed data: {e}")

def load_processed_data(pkl_path, device):
    """Load and process the USENIX data for training"""
    try:
        with open(pkl_path, 'rb') as f:
            try:
                data = pickle.load(f)
                
                # Move data to device
                data['feature'] = data['feature'].to(device)
                data['label'] = data['label'].to(device)
                data['pap'] = data['pap'].to(device)
                data['psp'] = data['psp'].to(device)
                data['train_idx'] = data['train_idx'].to(device)
                data['val_idx'] = data['val_idx'].to(device)
                data['test_idx'] = data['test_idx'].to(device)
                
                return data
            except pickle.UnpicklingError as e:
                print(f"Error unpickling data: {e}")
                return None
    except Exception as e:
        print(f"Error opening file: {e}")
        return None

def main():
    # Process USENIX graph and create train/val/test splits
    process_usenix_graph()
    
    # Example of loading for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processed_data = load_processed_data(
        'processed_usenix/usenix_processed.pkl', 
        device
    )

if __name__ == '__main__':
    main()