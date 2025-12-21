# UniExtreme

*Here is the code for "UniExtreme: A Universal Foundation Model for Extreme Weather Forecasting" KDD'2026*

## Dependencies

python=3.9.18, torch=2.5.1+cu118

## Data Construction

For data download and extreme annotation, refer to https://github.com/HuskyNian/HR-Extreme.

For data downsampling, use:
```bash
python data_downsample.py
```

To calculate the mean and std for data normalization, use:
```bash
python cal_mean_std.py
```

To calculate climatology, use:
```bash
python cal_climatology.py
```

## Model

```bash
# train
python pretrain_prompt.py --version freq_prompt_space --batch_size 12 --steplr
# test
python test_pretrain_prompt_new.py
```