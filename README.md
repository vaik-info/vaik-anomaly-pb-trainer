# vaik-anomaly-pb-trainer

Train anomaly pb model

## train_pb.py

### Usage

```shell
pip install -r requirements.txt
python train_pb.py --train_image_dir_path ~/.vaik-mnist-anomaly-dataset/train/good \
                --test_good_image_dir_path ~/.vaik-mnist-anomaly-dataset/valid/good \
                --test_anomaly_image_dir_path ~/.vaik-mnist-anomaly-dataset/valid/anomaly \
                --epoch_size mobile_net_v2_model \
                --epochs 10 \
                --step_size 1000 \
                --batch_size 32 \
                --image_height 224 \
                --image_width 224 \
                --latent_dim 16 \
                --test_max_sample 100 \
                --output_dir_path ~/.vaik_anomaly_pb_trainer/output        
```


- train_image_dir_path & test_good_image_dir_path & test_anomaly_image_dir_path' 

```shell
.
├── train
│   └── good
│       ├── 00000000.png
│       ├── 00000001.png
│       ├── 00000002.png
│       ├── 00000003.png
│       ├── 00000004.png
│       ├── 00000005.png
│       ├── 00000006.png
│       ├── 00000007.png
│       ├── 00000008.png
│       └── 00000009.png
└── valid
    ├── anomaly
    │   ├── 00000000.png
    │   ├── 00000001.png
    │   ├── 00000002.png
    │   ├── 00000003.png
    │   ├── 00000004.png
    │   ├── 00000005.png
    │   ├── 00000006.png
    │   ├── 00000007.png
    │   ├── 00000008.png
    │   └── 00000009.png
    └── good
        ├── 00000000.png
        ├── 00000001.png
        ├── 00000002.png
        ├── 00000003.png
        ├── 00000004.png
        ├── 00000005.png
        ├── 00000006.png
        ├── 00000007.png
        ├── 00000008.png
        └── 00000009.png
```


### Output

![mnist-anomaly-pb-trainer-1](https://user-images.githubusercontent.com/116471878/226260503-fff19d5d-c4f0-4685-bd6d-1abbeadcc4d6.png)

![mnist-anomaly-pb-trainer-2](https://user-images.githubusercontent.com/116471878/226260509-cdf9aff5-1dca-4f25-bbc0-fc4eb371158d.png)

-----

## Export ONNX

### Usage

```shell
python -m tf2onnx.convert --opset 10 \
--saved-model ~/.vaik_anomaly_pb_trainer/output/2023-03-20-11-43-13/epoch-0000_steps-1000_batch-32_loss-30.9484_val_loss-28.4088_val_anomaly_loss-52.6480_mse-0.0191-val_mse-0.0172_val_anomaly_mse-0.0396_val_auroc_mean-0.9699/all_model \
--output ~/.vaik_anomaly_pb_trainer/output/model.onnx
```