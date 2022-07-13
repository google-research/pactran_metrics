# PACTran Metrics

This is the code repository for the paper: PACTran: PAC-Bayesian Metrics for Estimating the Transferability of Pretrained Models to Classification Tasks (`https://arxiv.org/abs/2203.05126`).

## Introduction

PACTran is a theoretically grounded family of metrics for pretrained model selection and transferability measurement. The family is derived from the optimal PAC-Bayesian bound under the transfer learning setting. It contains three metric instantiations: PACTran-Dirichlet, PACTran-Gamma, PACTran-Gaussian.
Our experiments showed that PACTran-Gaussian is a more consistent and effective transferability measure compared to other existing selection methods.

## How to use

In the following, we provide the instructions of using the metrics (while using the oxford_iiit_pet task in the Visual Task Adaptation Benchmark (VTAB) [1] and the Sup-100% checkpoint as an example).

- Prerequisites:
  - Tensorflow
  - Tensorflow-Hub
  - Tensorflow-datasets
  - Numpy
  - Scipy
  - Scikit-learn


- For feature prediction, run
```
python -m pactran_metrics.adapt_and_eval \
--hub_module 'https://tfhub.dev/vtab/sup-100/1'  \
--hub_module_signature default \
--finetune_layer default \
--work_dir /tmp/all_features/sup-100/oxford_iiit_pet/ \
--dataset 'oxford_iiit_pet' \
--batch_size 512 \
--batch_size_eval 512 \
--initial_learning_rate 0.001 \
--decay_steps 1500,3000,4500 \
--max_steps 1 \
--run_adaptation=False \
--run_evaluation=False \
--run_prediction=True \
--linear_eval=False
```

- For whole network finetuning, run
```
hub='https://tfhub.dev/vtab/sup-100/1'
model_name=sup-100
python -m pactran_metrics.adapt_and_eval \
--hub_module ${hub}  \
--hub_module_signature default \
--finetune_layer default \
--work_dir /tmp/all_models/${model_name} \
--dataset 'oxford_iiit_pet' \
--batch_size 512 \
--batch_size_eval 512 \
--initial_learning_rate 0.001 \
--decay_steps 1500,3000,4500 \
--max_steps 5000 \
--run_adaptation \
--use_anil False
```

- For top-layer only finetuning, run
```
python -m pactran_metrics.anil_classifier \
--hub_module="https://tfhub.dev/vtab/sup-100/1" \
--dataset="oxford_iiit_pet" \
--work_dir=/tmp/all_results \
--feature_dir=/tmp/all_features
```

- For metrics estimation, run
```
python -m pactran_metrics.compute_metrics \
--hub_module="https://tfhub.dev/vtab/sup-100/1" \
--dataset="oxford_iiit_pet" \
--work_dir=/tmp/all_results \
--feature_dir=/tmp/all_features \
--num_examples=2
```
- All available datasets (in `./data` folder):
  - "caltech101"
  - "oxford_flowers102"
  - "patch_camelyon"
  - "sun397"
  - "dmlab"
  - "cifar"
  - "ddsm"
  - "oxford_iiit_pet"
  - "smallnorb"
  
- All available model URLs:
  - "https://tfhub.dev/vtab/sup-100/1" \
  - "https://tfhub.dev/vtab/sup-rotation-100/1" \
  - "https://tfhub.dev/vtab/sup-exemplar-100/1" \
  - "https://tfhub.dev/vtab/semi-exemplar-10/1" \
  - "https://tfhub.dev/vtab/semi-rotation-10/1" \
  - "https://tfhub.dev/vtab/rotation/1" \
  - "https://tfhub.dev/vtab/exemplar/1" \
  - "https://tfhub.dev/vtab/relative-patch-location/1" \
  - "https://tfhub.dev/vtab/jigsaw/1" \
  - "https://tfhub.dev/vtab/uncond-biggan/1" \
  - "https://tfhub.dev/vtab/cond-biggan/1" \
  - "https://tfhub.dev/vtab/wae-mmd/1" \
  - "https://tfhub.dev/vtab/vae/1" \
  - "https://tfhub.dev/vtab/wae-ukl/1" \
  - "https://tfhub.dev/vtab/wae-gan/1"

## Reference:

[1] Zhai, X., Puigcerver, J., Kolesnikov, A., Ruyssen, P., Riquelme, C., Lucic, M., Djolonga,
J., Pinto, A.S., Neumann, M., Dosovitskiy, A., et al.: A large-scale study of
representation learning with the visual task adaptation benchmark. arXiv preprint
arXiv:1910.04867 (2019)
