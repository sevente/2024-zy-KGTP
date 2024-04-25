# 执行过程
Prompting: 生成json格式的instruction数据集. 
```
    python Prompting/instruction.py
```

train with LoRA and test with BLEU score 
- 目前batch size > 1 的情况下进行generate，生成的样本质量较差（generate_batch 暂时不能用）
```
    ./example.sh
```

