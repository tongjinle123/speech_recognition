speech recognition

dataset: aishell1 

Setup:

    1.run docker 
    2.setup beamdecodectc
    
use:

    download aishell:
        bash scripts/data/download_aishell.sh
        
    preprocess:
        bash scripts/data/*
        bash scripts/train/*
        
result:

    beam decode with lm cer:7.2 

