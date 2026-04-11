# 在仓库根 Workspaces 执行：
#   docker build -f mergeKit_beta/Dockerfile -t mergekit-beta .
# 需要 GPU 运行时：docker run --gpus all -p 5000:5000 ...
#
# 说明：保持目录布局与代码一致——mergeKit_beta 与 modelmerge_visual 同为
# Workspaces 下兄弟目录，以便 run_vlm_search_bridge 默认 VLM_SEARCH_DIR 生效。
# 若本机无 modelmerge_visual，镜像内仅创建空目录；进化融合需挂载含 run_vlm_search.py 的目录
# 或设置环境变量 VLM_SEARCH_DIR。
# Packages 不再在镜像构建期 COPY，改为运行时卷挂载到 /app/ServiceEndFiles/Packages。

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:/opt/conda/envs/mergenetic/bin:${PATH} \
    LOCAL_MODELS_PATH=/data/Models \
    MERGENETIC_PYTHON=/opt/conda/envs/mergenetic/bin/python

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.ustc.edu.cn/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.ustc.edu.cn/ubuntu|g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl bzip2 git \
    # vLLM/Triton/torch.compile 运行期可能需要本机 C 编译器（如编译 triton driver.c 小模块）
    build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# 仅先复制环境文件以利用构建缓存
COPY mergeKit_beta/environment.yml /tmp/environment.yml
# 与本机「已升级」栈冲突：pip 同期无法解析 mergenetic 旧约束与 HF 新栈；先从 yml 去掉易冲突行，create 后再升级（与 DEVELOPMENT.md 一致）
RUN sed -e '/lm-eval==0.4.11/d' \
        -e '/transformers==5.3.0/d' \
        -e '/huggingface-hub==1.5.0/d' \
        -e '/tokenizers==0.22.2/d' \
        -e '/mergekit==0.1.4/d' \
        -e '/virtualenv==20.35.4/d' \
        /tmp/environment.yml > /tmp/environment.docker.yml \
    && conda config --system --remove-key channels || true \
    && conda config --system --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge \
    && conda config --system --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main \
    && conda config --system --set show_channel_urls true \
    && . /opt/conda/etc/profile.d/conda.sh \
    && python -m pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple \
    && python -m pip config set install.trusted-host pypi.mirrors.ustc.edu.cn \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda env create -n mergenetic -f /tmp/environment.docker.yml \
    && conda run -n mergenetic pip install --upgrade \
        'huggingface-hub==1.5.0' 'transformers==5.3.0' 'lm_eval[hf]==0.4.11' \
    && conda run -n mergenetic pip install --no-deps 'mergekit>=0.1.4' \
    && conda clean -afy \
    && rm -f /tmp/environment.yml /tmp/environment.docker.yml

# 与宿主机路径对齐
WORKDIR /app/ServiceEndFiles/Workspaces/mergeKit_beta

COPY mergeKit_beta /app/ServiceEndFiles/Workspaces/mergeKit_beta
RUN mkdir -p /app/ServiceEndFiles/Packages

# 无 VLM 源码时保证默认路径存在；有源码时可在构建前把目录放到 modelmerge_visual 再取消下行注释并增加 COPY
RUN mkdir -p /app/ServiceEndFiles/Workspaces/modelmerge_visual/VLM_merge_total/VLM_merge

# 若存在 modelmerge_visual，构建前取消注释：
# COPY modelmerge_visual /app/ServiceEndFiles/Workspaces/modelmerge_visual

SHELL ["/bin/bash", "-lc"]
CMD ["./start_app.sh"]
