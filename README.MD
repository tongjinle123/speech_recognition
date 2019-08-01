**Speech Recognition**

dataset: aishell1 

Setup:
    use docker: docker pull tongjinle/tasr:v1
    
use:
    download aishell:
        bash scripts/data/download_aishell.sh
    preprocess:
        bash scripts/data/*
        bash scripts/train/*
        
result:
    beam decode with lm cer:7.2 

curves:

  <img src="images/1563870010489.jpg">