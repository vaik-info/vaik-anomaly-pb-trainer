```shell
python -m tf2onnx.convert --opset 10 \
--saved-model ~/.vaik_anomaly_pb_trainer/output/2023-03-20-11-17-44/epoch-0006_steps-1000_batch-32_loss-26.7802_val_loss-25.3057_val_anomaly_loss-53.0980_mse-0.0141-val_mse-0.0139_val_anomaly_mse-0.0374_val_auroc_mean-0.9595/all_model \
--output ~/.vaik_anomaly_pb_trainer/output/model.onnx
```