import pandas as pd
import os
import torch
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from thefuzz import fuzz, process
import pickle
from pathlib import Path
import colorama

colorama.init(autoreset=True)

def normalize_author_name(name):
    """Normalize author names to handle variations and duplicates"""
    name = name.strip().lower()
    
    # Handle specific name variations
    name_mappings = {
        # Jason/Minhui Xue variations
        'jason (minhui) xue': 'jason xue',
        'minhui (jason) xue': 'jason xue',
        'jason xue': 'jason xue',
        'minhui xue': 'jason xue',
        
        # Fish/Ruoyu Wang variations
        'fish wang': 'ruoyu wang',
        'ruoyu "fish" wang': 'ruoyu wang',
        'ruoyu \'fish\' wang': 'ruoyu wang',
        
        # Yao Liu variations
        'yao  liu': 'yao liu',
        
        # Xiapu Luo variations
        'xiapu luo': 'xiapu luo',
        'daniel xiapu luo': 'xiapu luo'
    }
    
    return name_mappings.get(name, name)

def normalize_university_name(name, threshold=85):
    """Fuzzy match university names to handle variations and typos"""
    if not hasattr(normalize_university_name, "known_universities"):
        normalize_university_name.known_universities = set()
    
    if not name:
        return "Unknown"
        
    if normalize_university_name.known_universities:
        matches = process.extractOne(name, normalize_university_name.known_universities)
        if matches and matches[1] >= threshold:
            return matches[0]
    
    normalize_university_name.known_universities.add(name)
    return name

def load_gt_mappings(gt_file_path):
    """Load ground truth mappings from CSV file"""
    gt_df = pd.read_csv(gt_file_path)
    
    # Create GT score mapping
    gt_scores = {
        float('nan'): 0,
        'definite': 1,
        'likely': 2/3,
        'possible': 1/3
    }
    
    # Convert GT values to numerical scores
    gt_df['GT_score'] = gt_df['GT'].map(lambda x: gt_scores.get(x, 0))
    
    # Create mapping of person_name to GT score
    return dict(zip(gt_df['person_name'], gt_df['GT_score']))

def load_american_university_mappings(gt_file_path):
    """Load ground truth mappings for American universities from CSV file"""
    gt_df = pd.read_csv(gt_file_path)
    
    # Create GT score mapping
    gt_scores = {
        float('nan'): 0,  # Non-American
        'definite': 1,  # American
        'likely': 0     # International
    }
    
    # Clean up the american column name and values
    gt_df.columns = [col.strip('*') for col in gt_df.columns]
    gt_df['american'] = gt_df['american'].str.strip('*') if 'american' in gt_df.columns else gt_df['american']
    
    # Convert GT values to numerical scores
    gt_df['american_score'] = gt_df['american'].map(lambda x: gt_scores.get(x, 0))
    
    # Create mapping of university_name to GT score
    return dict(zip(gt_df['university_name'].str.lower(), gt_df['american_score']))

def create_conference_graph(conference_name, papers_df, data_dir, gt_mappings=None, american_gt_mappings=None):
    """Create a graph for a specific conference with temporal committee-author relationships"""
    # Filter papers for this conference
    conf_papers = papers_df[papers_df['Conference'] == conference_name].reset_index(drop=True)
    
    # Load committee data
    committee_file = os.path.join(data_dir, f"{conference_name}.csv")
    if not os.path.exists(committee_file):
        print(f"No committee data found for {conference_name}")
        return None
    
    committee_df = pd.read_csv(committee_file)
    committee_df.columns = [col.strip('*') for col in committee_df.columns]
    
    # Initialize HeteroData object
    data = HeteroData()
    
    # Initialize sentence transformer for paper embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create paper node features
    paper_embeddings = model.encode(conf_papers['Titles'].tolist())
    data['paper'].x = torch.tensor(paper_embeddings, dtype=torch.float)
    
    # Process authors and create person nodes
    all_authors = set()
    for authors in conf_papers['Authors']:
        all_authors.update([normalize_author_name(author.strip().lower()) 
                          for author in authors.split(',')])
    
    # Add committee members to all_authors
    committee_members = set(normalize_author_name(name.strip().lower()) 
                          for name in committee_df['Name'])
    all_authors.update(committee_members)
    
    # Create person mapping
    person_mapping = {name: idx for idx, name in enumerate(sorted(all_authors))}
    
    # Create person features [is_author, gt_score]
    num_people = len(person_mapping)
    person_features = torch.zeros((num_people, 2))  # Updated to track only is_author and gt_score
    
    # Mark authors
    for _, row in conf_papers.iterrows():
        for author in row['Authors'].split(','):
            author_name = normalize_author_name(author.strip().lower())
            if author_name in person_mapping:
                idx = person_mapping[author_name]
                person_features[idx, 0] = 1  # is_author
    
    # Add GT scores if available
    if gt_mappings:
        for person_name, idx in person_mapping.items():
            person_features[idx, 1] = gt_mappings.get(person_name, 0)  # GT score
    
    data['person'].x = person_features
    
    # Process universities
    universities = set()
    for _, row in committee_df.iterrows():
        if pd.notna(row['Affiliation']):
            universities.add(normalize_university_name(row['Affiliation']))
    
    university_mapping = {univ: idx for idx, univ in enumerate(sorted(universities))}
    
    # Create university features [one_hot_encoding, is_american]
    num_universities = len(university_mapping)
    university_features = torch.zeros((num_universities, num_universities + 1))
    
    # Set one-hot encoding
    university_features[:, :num_universities] = torch.eye(num_universities)
    
    # Add American university scores if available
    if american_gt_mappings:
        for univ_name, idx in university_mapping.items():
            university_features[idx, -1] = american_gt_mappings.get(univ_name.lower(), 0)  # American score
    
    data['university'].x = university_features
    
    # CREATE EDGES
    
    # Paper <-> Author edges
    paper_author_edges = []
    for paper_idx, row in conf_papers.iterrows():
        for author in row['Authors'].split(','):
            author_name = normalize_author_name(author.strip().lower())
            if author_name in person_mapping:
                paper_author_edges.append((paper_idx, person_mapping[author_name]))
            else:
                print(f"Author {author_name} not found in person_mapping")
    
    if paper_author_edges:
        data['paper', 'written_by', 'person'].edge_index = torch.tensor(paper_author_edges).t()
    else:
        data['paper', 'written_by', 'person'].edge_index = torch.empty((2, 0), dtype=torch.long)
        print(f"No paper author edges found for {conference_name}")
    
    # Person <-> University edges
    person_univ_edges = []
    for _, row in committee_df.iterrows():
        person_name = normalize_author_name(row['Name'].strip().lower())
        if pd.notna(row['Affiliation']) and person_name in person_mapping:
            univ_name = normalize_university_name(row['Affiliation'])
            if univ_name in university_mapping:
                person_univ_edges.append((
                    person_mapping[person_name],
                    university_mapping[univ_name]
                ))
    
    if person_univ_edges:
        data['person', 'affiliated_with', 'university'].edge_index = torch.tensor(person_univ_edges).t()
    else:
        data['person', 'affiliated_with', 'university'].edge_index = torch.empty((2, 0), dtype=torch.long)
        print(f"No person university edges found for {conference_name}")

    # Co-author edges
    coauthor_edges = set()
    for _, row in conf_papers.iterrows():
        authors = [a.strip().lower() for a in row['Authors'].split(',')]
        for i in range(len(authors)):
            for j in range(i+1, len(authors)):
                if authors[i] in person_mapping and authors[j] in person_mapping:
                    author1_id = person_mapping[authors[i]]
                    author2_id = person_mapping[authors[j]]
                    coauthor_edges.add((author1_id, author2_id))
                    coauthor_edges.add((author2_id, author1_id))
    
    if coauthor_edges:
        data['person', 'coauthor', 'person'].edge_index = torch.tensor(list(coauthor_edges)).t()
    else:
        data['person', 'coauthor', 'person'].edge_index = torch.empty((2, 0), dtype=torch.long)
        print(f"No coauthor edges found for {conference_name}")
        
    # Create publishedWhileInCommittee edges
    published_in_committee_edges = []
    
    # Create a mapping of committee members by year
    committee_by_year = {}
    for _, row in committee_df.iterrows():
        year = row['Year']
        member_name = normalize_author_name(row['Name'].strip().lower())
        if year not in committee_by_year:
            committee_by_year[year] = set()
        committee_by_year[year].add(member_name)
    
    # Check each paper's authors against committee members in the same year
    for _, paper_row in conf_papers.iterrows():
        paper_year = paper_row['Year']
        paper_authors = [normalize_author_name(author.strip().lower()) 
                        for author in paper_row['Authors'].split(',')]
        
        # Get committee members for this year
        committee_members = committee_by_year.get(paper_year, set())
        
        # Create edges between authors and committee members
        for author in paper_authors:
            if author in person_mapping:
                author_idx = person_mapping[author]
                for committee_member in committee_members:
                    if committee_member in person_mapping and committee_member != author:
                        committee_idx = person_mapping[committee_member]
                        published_in_committee_edges.append((author_idx, committee_idx))
    
    if published_in_committee_edges:
        data['person', 'publishedWhileInCommittee', 'person'].edge_index = torch.tensor(published_in_committee_edges).t()
    else:
        data['person', 'publishedWhileInCommittee', 'person'].edge_index = torch.empty((2, 0), dtype=torch.long)
        print(f"No publishedWhileInCommittee edges found for {conference_name}")
    
    # Store person_mapping in the graph object for later use
    data.person_mapping = person_mapping
    
    return data

def save_graphs(conference_graphs, output_dir='processed_graphs'):
    """Save PyG graphs to pickle files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for conf_name, graph in conference_graphs.items():
        output_path = os.path.join(output_dir, f"{conf_name}_graph.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(graph, f)
        print(f"Saved graph for {conf_name} to {output_path}")

def create_all_conference_graphs(papers_df, data_dir):
    """Create separate graphs for each conference"""
    conference_graphs = {}
    
    for conference in papers_df['Conference'].unique():
        graph = create_conference_graph(conference, papers_df, data_dir)
        if graph is not None:
            conference_graphs[conference] = graph
    
    return conference_graphs

def show_graph_stats(data, conference_name):
    """Display detailed statistics about the conference graph"""
    print(f"\n{colorama.Fore.CYAN}=== Graph Statistics for {conference_name} ==={colorama.Style.RESET_ALL}\n")
    
    # Node statistics
    print(colorama.Fore.GREEN + "Node Statistics:")
    print(f"- Papers: {len(data['paper'].x)}")
    print(f"- People: {len(data['person'].x)}")
    print(f"- Universities: {len(data['university'].x)}")
    print(f"Total nodes: {data.num_nodes}")
    
    # Edge statistics
    print("\nEdge Statistics:")
    paper_author_edges = len(data['paper', 'written_by', 'person'].edge_index[0])
    person_univ_edges = len(data['person', 'affiliated_with', 'university'].edge_index[0])
    coauthor_edges = len(data['person', 'coauthor', 'person'].edge_index[0])
    published_in_committee = len(data['person', 'publishedWhileInCommittee', 'person'].edge_index[0])
    
    print(f"- Paper-Author connections: {paper_author_edges}")
    print(f"- Person-University affiliations: {person_univ_edges}")
    print(f"- Co-authorship connections: {coauthor_edges}")
    print(f"- Published while in committee connections: {published_in_committee}")
    print(f"Total edges: {data.num_edges}")
    
    # Person node feature analysis
    person_features = data['person'].x
    num_authors = int(torch.sum(person_features[:, 0]).item())  # is_author feature
    
    # GT score analysis
    gt_scores = person_features[:, 1]  # GT score feature
    num_gt_positive = int(torch.sum(gt_scores > 0).item())
    
    print("\nPerson Node Analysis:")
    print(f"- Total people: {len(person_features)}")
    print(f"- Number of authors: {num_authors}")
    print(f"- People with positive GT scores: {num_gt_positive}")
    
    # Connectivity analysis
    avg_papers_per_author = paper_author_edges / num_authors if num_authors > 0 else 0
    avg_coauthors = coauthor_edges / len(person_features) if len(person_features) > 0 else 0
    
    print("\nConnectivity Metrics:")
    print(f"- Average papers per author: {avg_papers_per_author:.2f}")
    print(f"- Average co-authors per person: {avg_coauthors:.2f}")
    print(f"- Average university affiliations per person: {person_univ_edges/len(person_features):.2f}")
    
    print("\n" + "="*50 + "\n")

def visualize_conference_graph(data, conference_name, output_dir='figures'):
    G = to_networkx(data.to_homogeneous())
    
    node_colors = {
        'paper': 'skyblue',
        'person': 'lightgreen',
        'university': 'salmon'
    }
    
    edge_colors = {
        ('paper', 'written_by', 'person'): 'blue',
        ('person', 'affiliated_with', 'university'): 'green',
        ('person', 'publishedWhileInCommittee', 'person'): 'red',
        ('person', 'coauthor', 'person'): 'orange'
    }
    
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G)
    
    for node_type in node_colors:
        node_indices = [i for i, x in enumerate(data.metadata()[0]) if x == node_type]
        nx.draw_networkx_nodes(G, pos, 
                             nodelist=node_indices,
                             node_color=node_colors[node_type],
                             node_size=20,
                             label=node_type)
    
    for edge_type, color in edge_colors.items():
        edge_indices = [(u.item(), v.item()) for u, v in zip(*data[edge_type].edge_index)]
        nx.draw_networkx_edges(G, pos,
                             edgelist=edge_indices,
                             edge_color=color,
                             alpha=0.2,
                             label=edge_type[1])
    
    # Create custom legend handles for edges
    edge_legends = [plt.Line2D([0], [0], color=color, label=f"{edge_type[1]}", 
                              linewidth=2, alpha=0.5) 
                   for edge_type, color in edge_colors.items()]
    
    # Get node legend handles
    node_legends = [plt.scatter([0], [0], c=color, label=node_type, s=100) 
                   for node_type, color in node_colors.items()]
    
    # Combine all legend handles
    all_legends = node_legends + edge_legends
    
    plt.title(f"Conference Graph: {conference_name}")
    plt.legend(handles=all_legends, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('off')
    plt.tight_layout()
    # plt.show()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{conference_name}_graph.png"), bbox_inches='tight')
    plt.close()

def load_and_process_papers(filepath='data/papers.csv', data_dir='data'):
    papers_df = pd.read_csv(filepath)
    conference_graphs = create_all_conference_graphs(papers_df, data_dir)
    return conference_graphs

def save_non_committee_authors(conference_graphs, output_dir='processed_graphs'):
    """Save list of authors who are not on any committee"""
    os.makedirs(output_dir, exist_ok=True)
    
    for conf_name, graph in conference_graphs.items():
        # Get person features [is_chair, is_committee, is_author]
        person_features = graph['person'].x
        
        # Find indices of authors who are not chairs or committee members
        non_committee_indices = torch.where(
            (person_features[:, 2] == 1) &  # is author
            (person_features[:, 0] == 0) &  # not chair
            (person_features[:, 1] == 0)    # not committee
        )[0]
        
        # Get the mapping from index to person name
        reverse_person_mapping = {v: k for k, v in graph.person_mapping.items()}
        
        # Get names of non-committee authors
        non_committee_authors = [reverse_person_mapping[idx.item()] for idx in non_committee_indices]
        
        # Save to file
        output_path = os.path.join(output_dir, f"{conf_name}_non_committee_authors.txt")
        with open(output_path, 'w') as f:
            f.write(f"Non-committee authors for {conf_name}:\n")
            for author in sorted(non_committee_authors):
                f.write(f"{author}\n")
        
        print(f"Saved non-committee authors for {conf_name} to {output_path}")

def save_prediction_scores(conference_graphs, output_dir='predictions'):
    """Save prediction scores for each person in the graphs"""
    os.makedirs(output_dir, exist_ok=True)
    
    for conf_name, graph in conference_graphs.items():
        predictions = []
        reverse_person_mapping = {v: k for k, v in graph.person_mapping.items()}
        
        person_features = graph['person'].x
        for idx in range(len(person_features)):
            person_name = reverse_person_mapping[idx]
            gt_score = person_features[idx, 1].item()  # GT score is in the 4th column
            
            predictions.append({
                'person_name': person_name,
                'person_id': idx,
                'GT_score': gt_score
            })
        
        # Save to CSV
        pred_df = pd.DataFrame(predictions)
        output_path = os.path.join(output_dir, f"{conf_name}_predictions.csv")
        pred_df.to_csv(output_path, index=False)
        print(f"Saved predictions for {conf_name} to {output_path}")

def main():
    # Create output directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs('processed_graphs', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)
    
    # Load GT mappings
    gt_mappings = load_gt_mappings('data/GT.csv')
    american_gt_mappings = load_american_university_mappings('data/GT_american.csv')
    
    # Load papers and create graphs with both GT data
    papers_df = pd.read_csv('data/papers.csv')
    conference_graphs = {}
    
    for conference in papers_df['Conference'].unique():
        graph = create_conference_graph(conference, papers_df, 'data', 
                                     gt_mappings=gt_mappings,
                                     american_gt_mappings=american_gt_mappings)
        if graph is not None:
            conference_graphs[conference] = graph
    
    # Save graphs and create visualizations
    save_graphs(conference_graphs)
    # for conf_name, graph in conference_graphs.items():
    #     visualize_conference_graph(graph, conf_name)

    for conf_name, graph in conference_graphs.items():
        show_graph_stats(graph, conf_name)
    
    # Save prediction scores
    save_prediction_scores(conference_graphs)

if __name__ == '__main__':
    main()