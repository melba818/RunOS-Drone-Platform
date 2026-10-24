"""最小 timm 替身（**只实现 OSTrack 真正用到的那几个符号**）。

为什么不用真 timm：后端 venv 里没有 timm，而 `pip install timm` 有连带升级
torch / torchvision 的风险 —— 那会同时动到 ultralytics（YOLOE）这条线，
代价远超收益。OSTrack 从 timm 拿的都是**纯工具函数**（初始化、DropPath、Mlp、
位置编码插值、ImageNet 常量），实现是公开且稳定的，照抄一份即可。

这个包只在 `lmm.tracking.ostrack` 里把 `_vendor` 塞进 sys.path 之后才会被
`import timm` 命中；项目其它地方不会误用。
"""
