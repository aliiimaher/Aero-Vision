# Changes in Ultralytics Repository

This document outlines the recent changes made to the **Ultralytics** repository, focusing on modifications related to training configuration and validation functionality.

## 1. **Changes in `default.yaml` (Configuration File)**

Two new parameters have been added to the **default.yaml** configuration file:

- **`plot_interval`**: This parameter defines the frequency (in epochs) at which training plots should be generated.
- **`val_interval`**: This parameter defines the frequency (in epochs) at which model validation should be performed.

Example of updated `default.yaml`:

```yaml
# default.yaml
train:
  plot_interval: 10  # Generate plots every 10 epochs
  val_interval: 5    # Perform validation every 5 epochs
```

These options can be customized in your training configuration to better control the frequency of plot generation and validation during training.

---

## 2. **Changes in the `trainer.py` (Training Loop)**

In the `trainer.py` file, modifications were made to the training loop to integrate the newly added configuration parameters:

### Added validation condition:
Validation will now occur based on the `val_interval` parameter, allowing validation to be performed periodically during training.

```python
# trainer.py
if (self.args.val and (epoch % self.args.val_interval == 0)) or final_epoch:
    self.metrics, self.fitness = self.validate()
```

### Added plot condition:
Training sample plots are now generated periodically based on the `plot_interval` parameter.

```python
# trainer.py
if (self.args.plots) and (epoch % self.args.plot_interval == 0) and (i == 0):
    self.plot_training_samples(batch, i)
```

---

## 3. **Changes in the `validator.py` (Validation Loop)**

The validation loop has been updated to handle plotting of validation results:

### Plot validation samples:
Validation sample plots will be generated during validation, controlled by the `plots` flag and limited to the first few batches.

```python
# validator.py
if self.args.plots or (batch_i < 5):
    if self.training:
        batch_i += trainer.epoch
    self.plot_val_samples(batch, batch_i)
    self.plot_predictions(batch, preds, batch_i)
```

### Commented-out option for plotting less frequently:
A condition was added to allow fewer plots to be generated during validation, which can be adjusted by commenting or uncommenting the relevant code line:

```python
# if self.args.plots and batch_i < 3:
```

---

## 4. **Summary of Changes**

- **New Configuration Parameters**:
  - `plot_interval`: Controls how often to generate plots during training (in epochs).
  - `val_interval`: Controls how often to perform validation during training (in epochs).
  
- **Code Changes**:
  - In `trainer.py`, added logic to trigger validation and plotting based on the intervals set in `default.yaml`.
  - In `validator.py`, added functionality to plot validation samples and predictions during validation, controlled by the `plots` flag.

---

### How to Use the New Features:
1. **Update `default.yaml`**: Ensure that the `plot_interval` and `val_interval` values are set according to your needs.
2. **Set the `--plots` flag**: To enable plotting, use the `--plots` flag during training.
3. **Adjust the intervals**: Modify `plot_interval` and `val_interval` to control the frequency of validation and plot generation.

Example command to run training with plots and validation intervals:

```bash
python train.py --cfg default.yaml --plots --val
```
