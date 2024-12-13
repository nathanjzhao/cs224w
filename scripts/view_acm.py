import torch
import pickle

def load_and_print_tensor(file_path):
    """Load a PyTorch tensor from a pickle file and print its contents."""
    try:
        with open(file_path, 'rb') as f:  # Open the file in read-binary mode
            data = pickle.load(f)  # Load the data from the file object
        breakpoint() 
        print(data)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Specify the path to your acm.pkl file
    file_path = 'acm.pkl'  # Change this to the full path if necessary
    file_path = 'data/processed_graphs_combined/combined_conferences.pkl'  # Change this to the full path if necessary
    # file_path = 'original/USENIX_graph.pkl'
    load_and_print_tensor(file_path) 