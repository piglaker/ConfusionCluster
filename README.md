# ConfusionCluster

分析及CCCR Probe 指标对于ACL 2023 (Findings): [Investigating Glyph Phonetic Information for Chinese Spell Checking: What Works and What's Next](https://arxiv.org/abs/2212.04068)

训练纠错模型的代码见另一github仓库:[SpecialEdition](https://github.com/piglaker/SpecialEdition)
论文中的Probe实验一见github仓库:[]()

## Environment setting

python >= 3.7 \


`conda create -n ctcSE python=3.7` 
`conda activate ctcSE` \

```
pip install -r requirements.txt
```

## Quick Start

将main.py中路径设置为你要测试的纠错模型的路径。  

`bash run.sh`

⚠️注意最好该模型是huggingface对象输出能够返回logits用于分析，否则请参考本项目中对ReaLiSe的处理，提前运行一次inference将模型logits保存到文件再读入。



