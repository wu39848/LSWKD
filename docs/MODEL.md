#### Training Teacher Model
Following PLA to train Teacher Model,pcseg/models/head/caption_head.py is our modified code,which accomplishes Point-discriminative Contrastive Learning.
### Teacher Model 
- semantic segmentation

    | Dataset | Partition | Path |
    |:---:|:---:|:---:|
    | ScanNet | B15/N4 | [ckpt](https://onedrive.live.com/?cid=240D624894A89ED0&id=240D624894A89ED0%21200&parId=240D624894A89ED0%21190&o=OneUp) |
    | ScanNet | B12/N7 | [ckpt](https://onedrive.live.com/?cid=240D624894A89ED0&id=240D624894A89ED0%21198&parId=240D624894A89ED0%21190&o=OneUp) |
    | ScanNet | B10/N9 | [ckpt](https://onedrive.live.com/?cid=240D624894A89ED0&id=240D624894A89ED0%21199&parId=240D624894A89ED0%21190&o=OneUp) |
    | S3DIS | B8/N4 | [ckpt](https://onedrive.live.com/?cid=240D624894A89ED0&id=240D624894A89ED0%21194&parId=240D624894A89ED0%21190&o=OneUp) |
    | S3DIS | B6/N6 | [ckpt](https://onedrive.live.com/?cid=240D624894A89ED0&id=240D624894A89ED0%21196&parId=240D624894A89ED0%21190&o=OneUp) |
#### Training Student Model

```bash
cd tools
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} ${PY_ARGS} --pretrained_model teacherpath
```

For instance,
- train B15/N4 semantic segmentation on ScanNet:
    ```bash
    cd tools
    sh scripts/dist_train.sh 8 --cfg_file cfgs/scannet_models/spconv_clip_base15_caption_adamw.yaml --extra_tag exp_tag --ckpt teacherpath


#### Inference

```bash
cd tools
sh scripts/dist_test.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --ckpt ${CKPT_PATH}
```

For instance,
- to test a B15/N4 model on ScanNet:
    ```bash
    cd tools
    sh scripts/dist_test.sh 8 --cfg_file cfgs/scannet_models/spconv_clip_base15_caption_adamw.yaml --ckpt output/scannet_models/spconv_clip_base15_caption/exp_tag/ckpt/checkpoint_ep128.pth
    ```

### Model Zoo
- semantic segmentation

    | Dataset | Partition | hIoU / mIoU(B) / mIoU(N) | Path |
    |:---:|:---:|:---:|:---:|
    | ScanNet | B15/N4 | 68.2 / 68.6 / 67.8 | [ckpt](https://onedrive.live.com/?cid=240D624894A89ED0&id=240D624894A89ED0%21193&parId=240D624894A89ED0%21190&o=OneUp) |
    | ScanNet | B12/N7 | 56.8 / 69.2 / 48.2 | [ckpt](https://onedrive.live.com/?cid=240D624894A89ED0&id=240D624894A89ED0%21195&parId=240D624894A89ED0%21190&o=OneUp) |
    | ScanNet | B10/N9 | 54.5 / 74.9 / 42.8 | [ckpt](https://onedrive.live.com/?cid=240D624894A89ED0&id=240D624894A89ED0%21191&parId=240D624894A89ED0%21190&o=OneUp) |
    | S3DIS | B8/N4 |  39.0 / 57.0 / 29.6 | [ckpt](https://onedrive.live.com/?cid=240D624894A89ED0&id=240D624894A89ED0%21194&parId=240D624894A89ED0%21190&o=OneUp) |
    | S3DIS | B6/N6 | 41.1 / 54.2 / 33.0 | [ckpt](https://onedrive.live.com/?cid=240D624894A89ED0&id=240D624894A89ED0%21192&parId=240D624894A89ED0%21190&o=OneUp) |


