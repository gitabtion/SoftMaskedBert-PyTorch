# SoftMaskedBert-PyTorch
🙈 基于 huggingface/transformers 的SoftMaskedBert的非官方实现

[English](README_eng.md)|简体中文

## 环境准备
1. 安装 python 3.6+
2. 运行以下命令以安装必要的包.
```shell
pip install -r requirements.txt
```

## 数据准备
1. 从 [http://nlp.ee.ncu.edu.tw/resource/csc.html](http://nlp.ee.ncu.edu.tw/resource/csc.html)下载SIGHAN数据集
2. 解压上述数据集并将文件夹中所有 ''.sgml'' 文件复制至 data/ 目录
3. 复制 ''SIGHAN15_CSC_TestInput.txt'' 和 ''SIGHAN15_CSC_TestTruth.txt'' 至 data/ 目录
4. 下载 [https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml](https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml) 至 data/ 目录
5. 请确保以下文件在 data/ 中
    ```
    train.sgml
    B1_training.sgml
    C1_training.sgml  
    SIGHAN15_CSC_A2_Training.sgml  
    SIGHAN15_CSC_B2_Training.sgml  
    SIGHAN15_CSC_TestInput.txt
    SIGHAN15_CSC_TestTruth.txt
    ```
6. 运行以下命令进行数据预处理
```shell
python main.py --mode preproc
```

## 下载预训练权重
1.   从 [https://huggingface.co/bert-base-chinese/tree/main](https://huggingface.co/bert-base-chinese/tree/main) 下载BERT的预训练权重(pytorch_model.bin) 至 checkpoint/ 目录

## 训练及测试
1. 运行以下命令以训练模型。
```shell
python main.py --mode train
```
2. 运行以下命令以测试模型。
```shell
python main.py --mode test
```
3. 更多模型运行及训练参数请使用以下命令查看。
```shell
python main.py --help
```
```
  --hard_device HARD_DEVICE
                        硬件，cpu or cuda
  --gpu_index GPU_INDEX
                        gpu索引, one of [0,1,2,3]
  --load_checkpoint [LOAD_CHECKPOINT]
                        是否加载训练保存的权重, one of [t,f]
  --bert_checkpoint BERT_CHECKPOINT
  --model_save_path MODEL_SAVE_PATH
  --epochs EPOCHS       训练轮数
  --batch_size BATCH_SIZE
                        批大小
  --warmup_epochs WARMUP_EPOCHS
                        warmup轮数, 需小于训练轮数
  --lr LR               学习率
  --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                        梯度累加的batch数
  --mode MODE           代码运行模式，以此来控制训练测试或数据预处理，one of [train, test, preproc]
  --loss_weight LOSS_WEIGHT
                        论文中的lambda，即correction loss的权重
```

## 实验结果
### 字级
|component|p|r|f|
|:-:|:-:|:-:|:-:|
|Detection|0.8417|0.8274|0.8345|
|Correction|0.9487|0.8739|0.9106|
### 句级
|acc|p|r|f|
|:-:|:-:|:-:|:-:|
|0.8145|0.8674|0.7361|0.7964|

detection的表现差是因为欠拟合，该实验结果仅是在处理后的数据集上跑了10个epochs的结果，并没有像paper一样做大量的预训练。


## References
1. [Spelling Error Correction with Soft-Masked BERT](https://arxiv.org/abs/2005.07421)
2. [http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html](http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html)
3. [https://github.com/wdimmy/Automatic-Corpus-Generation](https://github.com/wdimmy/Automatic-Corpus-Generation)
4. [transformers](https://huggingface.co/)
5. [https://github.com/sunnyqiny/Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check](https://github.com/sunnyqiny/Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check)