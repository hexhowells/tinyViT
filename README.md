# tinyViT
Tiny implementation of vision transformers

---

Implementation of the original [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) paper. 

Trained on the [Imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) dataset. Dataset downloaded via `huggingface-cli download ILSVRC/imagenet-1k --repo-type dataset --revision refs/convert/parquet --local-dir /media/datasets/image-datasets/imagenet-1k --local-dir-use-symlinks False`

Training a ViT-small model on Imagenet-1k for 50 epochs yields an accuracy of 0.6. In order to reduce overfitting this repo also implements [Stochastic depth](https://arxiv.org/abs/1603.09382) and [Mixup](https://arxiv.org/abs/1710.09412) which allows the validation accuracy to match the training accuracy throughout training (with minimal impact to the training loss/acc)
