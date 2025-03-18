# Line_detect_2_cls

## Training With Multi-GPUs

```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py  --batch-size 4
```

## Training With Single GPU

```bash
python train.py
```


