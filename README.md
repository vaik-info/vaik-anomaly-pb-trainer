```shell
python -m tf2onnx.convert --opset 10 \
--saved-model ~/.vaik_anomaly_pb_trainer/output/2023-03-20-11-43-13/epoch-0000_steps-1000_batch-32_loss-30.9484_val_loss-28.4088_val_anomaly_loss-52.6480_mse-0.0191-val_mse-0.0172_val_anomaly_mse-0.0396_val_auroc_mean-0.9699/all_model \
--output ~/.vaik_anomaly_pb_trainer/output/model.onnx
```