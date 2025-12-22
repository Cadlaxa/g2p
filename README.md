# G2P model

## Training

### Setup

`k2` package only supports CUDA on Linux. It is recommended to use WSL on Windows or Linux.

```
pip install -r requirements.txt
```

### Train en_us arpabet model

- Place the `cmudict-0.7b` file in the `data` folder.
- Run:
    ```
    python train.py
    ```
- You can use Tensorboard to visualize logs in the `logs` folder.
    ```
    tensorboard --logdir=logs
    ```

### Train a different model

- Place your dictionary in the `data` folder. You may need to reformat your dictionary so that `SphinxDataset` can load it, or write your own dataset class.
- Create a file in `configs/symbols` for your symbols. The first 4 graphemes and phonemes must be `<unk>`, `<pad>`, `<bos>`, `<eos>`.
- Create a new config `cfg_<yours>.yaml` similar to `cfg_ds.yaml`. Besides the `symbols` override, any config can be overridden by following the same yaml structure, like `dataset.path` in `cfg_ds.yaml`.
- Run:
    ```
    python train.py --config-name=cfg_<yours>
    ```

### Sweep hyperparameters

Example:
```
python train.py --config-name=cfg_ds -m train.epochs=160,200,240 model.encoder.num_layers=4 model.decoder.num_layers=1,2
```

## Exporting oudep

In your config file, override the `export` section as needed. An example can be found in `cfg_ds.yaml`.
```
python export.py --ckpt=logs/xxx/checkpoints/xxx.ckpt
```

## The Model

This repository provides a G2P model, which is a modified RNN-T model.
- The default config is 3 encoder layers + 1 decoder layer, chosen to reduce file size. Scaling up, e.g., to 5+2 layers, could provide minor improvements, e.g., 0.1-0.2% PER or 0.5-0.7% WER.
- Since the losses do not fully align with error rates, the error rates will vary slightly between runs. You can run training multiple times and take the best result.
- Achieving the best error rate is not strictly necessary, as the model is only used for entries out of dictionary.

| | | PER (ins/del/sub) % | WER % |
|-|-|-|-|
| cmudict-0.7b | train | 2.71 (0.31/0.35/2.10) | 12.81 |
| | val | 8.56 (0.92/0.98/6.87) | 34.92 |
| cmudict-0.7b (stress removed) | train | 2.20 (0.30/0.34/1.60) | 11.17 |
| | val | 6.76 (0.91/0.97/5.08) | 30.27 |
| ds_cmudict-07b.txt | train | 2.12 (0.30/0.33/1.54) | 10.51 |
| | val | 7.19 (1.00/0.99/5.45) | 31.24 |
