# SoftMaskedBert-PyTorch
🙈 基于 huggingface/transformers 的SoftMaskedBert的非官方实现

[ENG_README](README.md)

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
3. 下载 [https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml](https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml) 至 data/ 目录
4. 运行以下命令进行数据预处理
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

## 参考文献
1. [Spelling Error Correction with Soft-Masked BERT](https://arxiv.org/abs/2005.07421)
2. [http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html](http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html)
3. [https://github.com/wdimmy/Automatic-Corpus-Generation](https://github.com/wdimmy/Automatic-Corpus-Generation)
4. [transformers](https://huggingface.co/)
5. [https://github.com/sunnyqiny/Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check](https://github.com/sunnyqiny/Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check)