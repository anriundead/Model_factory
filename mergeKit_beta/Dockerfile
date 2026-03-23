# 在仓库根 ServiceEndFiles 执行：
#   docker build -f Workspaces/mergeKit_beta/Dockerfile -t mergekit-beta .
# 需要 GPU 运行时：docker run --gpus all -p 5000:5000 ...
#
# 说明：保持宿主机目录布局与代码一致——mergeKit_beta 与 modelmerge_visual 同为
# Workspaces 下兄弟目录，以便 run_vlm_search_bridge 默认 VLM_SEARCH_DIR 生效。
# 若本机无 modelmerge_visual，镜像内仅创建空目录；进化融合需挂载含 run_vlm_search.py 的目录
# 或设置环境变量 VLM_SEARCH_DIR。

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:/opt/conda/envs/mergenetic/bin:${PATH} \
    LOCAL_MODELS_PATH=/data/Models \
    MERGENETIC_PYTHON=/opt/conda/envs/mergenetic/bin/python

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl bzip2 git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# 仅先复制环境文件以利用构建缓存
COPY Workspaces/mergeKit_beta/environment.yml /tmp/environment.yml
# 与本机「已升级」栈冲突：pip 同期无法解析 mergenetic 旧约束与 HF 新栈；先从 yml 去掉易冲突行，create 后再升级（与 DEVELOPMENT.md 一致）
RUN sed -e '/lm-eval==0.4.11/d' \
        -e '/transformers==5.3.0/d' \
        -e '/huggingface-hub==1.5.0/d' \
        -e '/tokenizers==0.22.2/d' \
        -e '/virtualenv==20.35.4/d' \
        /tmp/environment.yml > /tmp/environment.docker.yml \
    && . /opt/conda/etc/profile.d/conda.sh \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda env create -n mergenetic -f /tmp/environment.docker.yml \
    && conda run -n mergenetic pip install --upgrade \
        'huggingface-hub==1.5.0' 'transformers==5.3.0' 'lm_eval[hf]==0.4.11' \
    && conda clean -afy \
    && rm -f /tmp/environment.yml /tmp/environment.docker.yml

# 与宿主机 ServiceEndFiles 对齐的路径
WORKDIR /app/ServiceEndFiles/Workspaces/mergeKit_beta

COPY Workspaces/mergeKit_beta /app/ServiceEndFiles/Workspaces/mergeKit_beta
COPY Packages /app/ServiceEndFiles/Packages

# 无 VLM 源码时保证默认路径存在；有源码时可在构建前把目录放到 Workspaces/modelmerge_visual 再取消下行注释并增加 COPY
RUN mkdir -p /app/ServiceEndFiles/Workspaces/modelmerge_visual/VLM_merge_total/VLM_merge

# 若存在 Workspaces/modelmerge_visual，构建前取消注释：
# COPY Workspaces/modelmerge_visual /app/ServiceEndFiles/Workspaces/modelmerge_visual

SHELL ["/bin/bash", "-lc"]
CMD ["./start_app.sh"]
