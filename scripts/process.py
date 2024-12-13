import pandas as pd
import os
from thefuzz import fuzz
from thefuzz import process
import numpy as np
from collections import defaultdict
import pycountry

def load_csv_files(data_dir):
    """Load all CSV files from the data directory and return list of (filename, dataframe) tuples."""
    csv_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(data_dir, file))
            csv_files.append((file, df))
    return csv_files

def normalize_name(name):
    """Basic name normalization."""
    return str(name).lower().strip()

def are_similar_names(name1, name2, threshold=85):
    """Check if two names are similar using fuzzy matching."""
    name1 = normalize_name(name1)
    name2 = normalize_name(name2)
    
    if name1 == name2:
        return True
    
    ratio = fuzz.ratio(name1, name2)
    partial_ratio = fuzz.partial_ratio(name1, name2)
    token_sort_ratio = fuzz.token_sort_ratio(name1, name2)
    
    return (ratio >= threshold or 
            partial_ratio >= threshold or 
            token_sort_ratio >= threshold)

def guess_country(university_name):
    """Attempt to determine country from university name."""
    university_name = university_name.lower()
    
    # Dictionary of common country indicators
    country_indicators = {
        'usa': 'United States',
        'united states': 'United States',
        'uk': 'United Kingdom',
        'united kingdom': 'United Kingdom',
        'china': 'China',
        'chinese': 'China',
        'japan': 'Japan',
        'germany': 'Germany',
        'france': 'France',
        'canada': 'Canada',
        'australia': 'Australia',
        'india': 'India',
        # Add more as needed
    }
    
    # Check for country names or indicators in university name
    for indicator, country in country_indicators.items():
        if indicator in university_name:
            return country
            
    # Search for country names from pycountry
    for country in pycountry.countries:
        if country.name.lower() in university_name:
            return country.name
            
    return 'Unknown'

def create_entity_mappings(file_df_pairs):
    """Create mappings for people and universities."""
    person_mapping = {}  # name -> id
    university_mapping = {}  # name -> id
    person_university_pairs = set()  # (name, university) pairs
    person_conferences = defaultdict(set)  # person_id -> set of conference names
    person_papers = defaultdict(lambda: defaultdict(int))  # person_id -> {conference -> count}
    university_papers = defaultdict(lambda: defaultdict(int))  # university_id -> {conference -> count}
    university_countries = {}  # university_id -> country
    
    current_person_id = 0
    current_university_id = 0
    
    for filename, df in file_df_pairs:
        conference_name = os.path.splitext(filename)[0]
        
        for _, row in df.iterrows():
            person_name = normalize_name(row['Name'])
            university_name = normalize_name(row['Affiliation'])
            
            # Handle university mapping
            university_match = None
            for existing_univ in university_mapping:
                if are_similar_names(university_name, existing_univ, threshold=90):
                    university_match = existing_univ
                    break
            
            if university_match is None:
                university_mapping[university_name] = current_university_id
                university_countries[current_university_id] = guess_country(university_name)
                current_university_id += 1
            else:
                university_mapping[university_name] = university_mapping[university_match]
            
            # Handle person mapping
            person_match = None
            for existing_name, existing_id in person_mapping.items():
                if are_similar_names(person_name, existing_name):
                    existing_pair = (existing_name, normalize_name(university_name))
                    if existing_pair in person_university_pairs:
                        person_match = existing_name
                        break
            
            if person_match is None:
                person_mapping[person_name] = current_person_id
                current_person_id += 1
            else:
                person_mapping[person_name] = person_mapping[person_match]
            
            person_id = person_mapping[person_name]
            university_id = university_mapping[university_name]
            
            # Update statistics
            person_university_pairs.add((person_name, university_name))
            person_conferences[person_id].add(conference_name)
            person_papers[person_id][conference_name] += 1
            university_papers[university_id][conference_name] += 1
    
    return (person_mapping, university_mapping, person_conferences, 
            person_papers, university_papers, university_countries)

def generate_statistics(person_mapping, university_mapping, person_conferences, 
                       person_papers, university_papers, university_countries):
    """Generate comprehensive statistics about conference attendance and publications."""
    
    # Conference attendance frequency
    attendance_stats = defaultdict(int)
    for person_id, conferences in person_conferences.items():
        attendance_stats[len(conferences)] += 1
    
    attendance_df = pd.DataFrame([
        {'conferences_attended': k, 'num_people': v} 
        for k, v in sorted(attendance_stats.items())
    ])
    
    # Top authors by conference
    top_authors = defaultdict(list)
    for person_id, conference_counts in person_papers.items():
        for conference, count in conference_counts.items():
            top_authors[conference].append({
                'person_id': person_id,
                'papers': count
            })
    
    top_authors_df = pd.DataFrame([
        {
            'conference': conf,
            'person_id': sorted(authors, key=lambda x: x['papers'], reverse=True)[0]['person_id'],
            'paper_count': sorted(authors, key=lambda x: x['papers'], reverse=True)[0]['papers']
        }
        for conf, authors in top_authors.items()
    ])
    
    # Top universities by conference
    top_universities = defaultdict(list)
    for univ_id, conference_counts in university_papers.items():
        for conference, count in conference_counts.items():
            top_universities[conference].append({
                'university_id': univ_id,
                'papers': count,
                'country': university_countries[univ_id]
            })
    
    top_universities_df = pd.DataFrame([
        {
            'conference': conf,
            'university_id': sorted(univs, key=lambda x: x['papers'], reverse=True)[0]['university_id'],
            'paper_count': sorted(univs, key=lambda x: x['papers'], reverse=True)[0]['papers'],
            'country': sorted(univs, key=lambda x: x['papers'], reverse=True)[0]['country']
        }
        for conf, univs in top_universities.items()
    ])
    
    # Countries statistics
    country_stats = defaultdict(int)
    for univ_id, country in university_countries.items():
        country_stats[country] += 1
    
    country_stats_df = pd.DataFrame([
        {'country': k, 'university_count': v}
        for k, v in sorted(country_stats.items(), key=lambda x: x[1], reverse=True)
    ])
    
    return attendance_df, top_authors_df, top_universities_df, country_stats_df

def process_csvs():
    os.makedirs('out', exist_ok=True)
    
    # Load data and generate mappings
    file_df_pairs = load_csv_files('data')
    mappings = create_entity_mappings(file_df_pairs)
    (person_mapping, university_mapping, person_conferences, 
     person_papers, university_papers, university_countries) = mappings
    
    # Generate statistics
    stats = generate_statistics(person_mapping, university_mapping, person_conferences,
                              person_papers, university_papers, university_countries)
    attendance_df, top_authors_df, top_universities_df, country_stats_df = stats
    
    # Save statistics
    attendance_df.to_csv('out/conference_attendance_stats.csv', index=False)
    top_authors_df.to_csv('out/top_authors_by_conference.csv', index=False)
    top_universities_df.to_csv('out/top_universities_by_conference.csv', index=False)
    country_stats_df.to_csv('out/country_statistics.csv', index=False)
    
    # Process original files with IDs
    for filename, df in file_df_pairs:
        new_df = pd.DataFrame()
        new_df['person_name'] = df['Name']
        new_df['university_name'] = df['Affiliation']
        new_df['person_id'] = df['Name'].apply(lambda x: person_mapping[normalize_name(x)])
        new_df['university_id'] = df['Affiliation'].apply(lambda x: university_mapping[normalize_name(x)])
        new_df['country'] = new_df['university_id'].apply(lambda x: university_countries[x])
        
        base_name = os.path.splitext(filename)[0]
        output_path = f'out/processed_{base_name}.csv'
        new_df.to_csv(output_path, index=False)
    
    # Save mapping reference files
    pd.DataFrame({
        'person_name': list(person_mapping.keys()),
        'person_id': list(person_mapping.values())
    }).to_csv('out/person_mapping.csv', index=False)
    
    pd.DataFrame({
        'university_name': list(university_mapping.keys()),
        'university_id': list(university_mapping.values()),
        'country': [university_countries[univ_id] for univ_id in university_mapping.values()]
    }).to_csv('out/university_mapping.csv', index=False)
    
    # Print summary
    print("\nConference Attendance Statistics:")
    print(attendance_df.to_string(index=False))
    print("\nTop Authors by Conference:")
    print(top_authors_df.to_string(index=False))
    print("\nTop Universities by Conference:")
    print(top_universities_df.to_string(index=False))
    print("\nCountry Statistics:")
    print(country_stats_df.to_string(index=False))

if __name__ == "__main__":
    process_csvs()