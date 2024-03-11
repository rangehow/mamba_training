#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/mamba-2.8b-hf',cache_dir='/data/ruanjh/mamba-2.8b-hf')