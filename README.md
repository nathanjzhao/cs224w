`convert.py` is used to convert the original data into a format that DeepSNAP can use.

`convert_usenix.py` is used to convert the USENIX data into a format that DeepSNAP can use (we do this to test to see whether DeepSNAP can handle the USENIX data)

`view_acm.py` is used to view the original ACM data in a format that DeepSNAP can use (for debugging with breakpoints() to understand the data format)

`graph.py` is used to create an original PyG general dataset with all the data we have (skimmed down for the future)

`process.py` is used to process the original data (e.g. create mappings so that we understand the universities, duplicate authors if there are typos in the names, etc. using fuzzy matching)

`Heterogeneous_Node_Classification_with_DeepSNAP.ipynb` is used to run the DeepSNAP model on the original ACM data (we have different systems for testing different datasets)
