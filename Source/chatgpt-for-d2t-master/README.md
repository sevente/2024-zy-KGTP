# steps
- 在 https://35.aigcbest.top/ 中生成令牌使用3.5系列; (主站 https://api.aigcbest.top/)， 并把令牌拷贝到 evaluate_with_chatgpt.py 中的 Authorization
- 运行 evaluate_with_chatgpt.py, 通过设置 dataset_name 改变测试集； 最终输出prediction到 数据集对应的文件夹下的 test 文件
- 运行 metrics.py 得到 bleu score


# 问题记录
- gpt-3.5-turbo 在运行过程中因为频繁访问网站，导致卡顿 ---> 设置每隔多个samples之后，sleep，防止频繁访问导致api访问被block