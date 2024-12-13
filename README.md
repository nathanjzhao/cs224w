`convert.py` is used to convert the original data into a format that DeepSNAP can use. We create PAP and PSP edges in order to create a heterogeneous graph while minimizing the number of relationships we have to process.

`convert_usenix.py` is used to convert the USENIX data into a format that DeepSNAP can use (we do this to test to see whether DeepSNAP can handle the USENIX data)

`view_acm.py` is used to view the original ACM data in a format that DeepSNAP can use (for debugging with breakpoints() to understand the data format)

`graph.py` is used to create an original PyG general dataset with all the data we have (including as much data as possible, though not all data is used in the future) -- we use a sentence transformer on titles in order to create node embeddings for the paper.

`process.py` is used to process the original data (e.g. create mappings so that we understand the universities, duplicate authors if there are typos in the names, etc. using fuzzy matching)

`Heterogeneous_Node_Classification_with_DeepSNAP.ipynb` is used to run the DeepSNAP model on the original ACM data (we have different systems for testing different datasets)
