# BlockDetector

## Training With Multi-GPUs

```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py  --batch-size 4
```

## Training With Single GPU

```bash
python train.py
```

## Result

<table align="center" style="border: 1px solid black;">
    <tr>
        <td align="center">
            <img src="../DOCME/4-1.png" alt="result1">
        </td>
        <td align="center">
            <img src="../DOCME/4-2.png" alt="result2">
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="../DOCME/5-1.png" alt="result3">
        </td>
        <td align="center">
            <img src="../DOCME/5-2.png" alt="result4">
        </td>
    </tr>
    <tr>
        <td align="center">
            Ground Truth
        </td>
        <td align="center">
            Predicted
        </td>
    </tr>
</table>
