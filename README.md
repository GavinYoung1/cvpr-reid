The CVPR2023 paper under mindspore to inference for CC-ReID

## Supported
- [x] Inference based on mindspore and GPU
- [x] Distributed training
- [ ] Convert pytorch pretraining weight to mindspore(align conv/pool pad)
- [ ] Quantitative training and reasoning (FP16, INT8)


## Environment
1. mindspore>=1.0.0, [INSTALL](https://www.mindspore.cn/install)
3. opencv-python, yacs, apex (optional)


## checklist
- [x] baseline: resnet50 + random sampler + CrossEntropy Loss
- [x] pair-wise loss
- [x] Distributed
- [ ] FP16
- [ ] convert weights from pytorch
- [ ] pretrain model

