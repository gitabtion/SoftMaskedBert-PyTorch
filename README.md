# SoftMaskedBert-PyTorch
🙈 An unofficial implementation of SoftMaskedBert based on huggingface/transformers.

[中文README](readme_zh.md)

## prepare env
1. install python 3.6+
2. run the following command in terminal.
```shell
pip install -r requirements.txt
```

## prepare data
1. download sighan data from [http://nlp.ee.ncu.edu.tw/resource/csc.html](http://nlp.ee.ncu.edu.tw/resource/csc.html)
2. unzip the file and copy all the ''.sgml'' file to data/
3. copy ''SIGHAN15_CSC_TestInput.txt'' and ''SIGHAN15_CSC_TestTruth.txt'' to data/
4. download [https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml](https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml) to data/
5. check following files are in data/
    ```
    train.sgml
    B1_training.sgml
    C1_training.sgml  
    SIGHAN15_CSC_A2_Training.sgml  
    SIGHAN15_CSC_B2_Training.sgml  
    SIGHAN15_CSC_TestInput.txt
    SIGHAN15_CSC_TestTruth.txt
    ```
6. run the following command to process the data
```shell
python main.py --mode preproc
```

## prepare bert checkpoint
1. download bert checkpoint (pytorch_model.bin) from [https://huggingface.co/bert-base-chinese/tree/main](https://huggingface.co/bert-base-chinese/tree/main) to checkpoint/

## run
1. run the following command to train the model.
```shell
python main.py --mode train
```
2. run the following command to test the model.
```shell
python main.py --mode test
```
3. you can use the following command to get any help for arguments.
```shell
python main.py --help
```


## references
1. [Spelling Error Correction with Soft-Masked BERT](https://arxiv.org/abs/2005.07421)
2. [http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html](http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html)
3. [https://github.com/wdimmy/Automatic-Corpus-Generation](https://github.com/wdimmy/Automatic-Corpus-Generation)
4. [transformers](https://huggingface.co/)
5. [https://github.com/sunnyqiny/Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check](https://github.com/sunnyqiny/Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check)