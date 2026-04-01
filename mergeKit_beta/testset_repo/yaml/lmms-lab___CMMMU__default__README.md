---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: type
    dtype: string
  - name: source_type
    dtype: string
  - name: source
    dtype: string
  - name: question
    dtype: string
  - name: option1
    dtype: string
  - name: option2
    dtype: string
  - name: option3
    dtype: string
  - name: option4
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: answer
    dtype: string
  - name: analysis
    dtype: string
  - name: distribution
    dtype: string
  - name: difficulty_level
    dtype: string
  - name: subcategory
    dtype: string
  - name: category
    dtype: string
  - name: subfield
    dtype: string
  - name: img_type
    dtype: string
  - name: image_1_filename
    dtype: string
  - name: image_2_filename
    dtype: string
  - name: image_3_filename
    dtype: string
  - name: image_4_filename
    dtype: string
  - name: image_5_filename
    dtype: string
  splits:
  - name: dev
    num_bytes: 13180933.0
    num_examples: 112
  - name: val
    num_bytes: 95817884.0
    num_examples: 900
  - name: test
    num_bytes: 3146080167.0
    num_examples: 11000
  download_size: 1297435382
  dataset_size: 3255078984.0
configs:
- config_name: default
  data_files:
  - split: dev
    path: data/dev-*
  - split: val
    path: data/val-*
  - split: test
    path: data/test-*
---


<p align="center" width="100%">
<img src="https://i.postimg.cc/g0QRgMVv/WX20240228-113337-2x.png"  width="100%" height="80%">
</p>

# Large-scale Multi-modality Models Evaluation Suite

> Accelerating the development of large-scale multi-modality models (LMMs) with `lmms-eval`

🏠 [Homepage](https://lmms-lab.github.io/) | 📚 [Documentation](docs/README.md) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab)

# This Dataset

This is a formatted version of [CMMMU](https://cmmmu-benchmark.github.io/). It is used in our `lmms-eval` pipeline to allow for one-click evaluations of large multi-modality models.

```
@article{zhang2024cmmmu,
        title={CMMMU: A Chinese Massive Multi-discipline Multimodal Understanding Benchmark},
        author={Ge, Zhang and Xinrun, Du and Bei, Chen and Yiming, Liang and Tongxu, Luo and Tianyu, Zheng and Kang, Zhu and Yuyang, Cheng and Chunpu, Xu and Shuyue, Guo and Haoran, Zhang and Xingwei, Qu and Junjie, Wang and Ruibin, Yuan and Yizhi, Li and Zekun, Wang and Yudong, Liu and Yu-Hsuan, Tsai and Fengji, Zhang and Chenghua, Lin and Wenhao, Huang and Wenhu, Chen and Jie, Fu},
        journal={arXiv preprint arXiv:2401.20847},
        year={2024},
      }
```
