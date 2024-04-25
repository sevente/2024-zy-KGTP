# 执行过程

## 数据集
#### 采用Control Prefix项目提供的数据集
Prompting: 生成json格式的instruction数据集. 参考：/home/sdb/xx/path/LLM/alpacaLoraData2Text
可以直接拷贝参考工程的Dataset到本项目
```
    python Prompting/instruction.py
```

#### 采用mvp项目提供的数据集
```
    python prepare_data.py
```


evaluation with batched test dataset
```
    ./evaluation_batch.sh
```

evaluation with single sample from test dataset
```
    ./example.sh
```


test generate with user input:
```
    CUDA_VISIBLE_DEVICES=0, python generate.py \
    --base_model 'RUCAIBox/mvp-data-to-text' \
    --test_file "./Dataset/webnlg17Instruct/test_both0.json" \
    --save_path 'webnlg17-mvp-data-to-text_pred' \
    --turn_on_chat
```

