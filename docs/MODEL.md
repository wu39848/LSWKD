#### Training Teacher Model
Following PLA's steps to train Teacher Model,pcseg/models/head/caption_head.py is our modified code,which accomplishes Point-discriminative Contrastive Learning.
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
    | ScanNet | B15/N4 | 68.2 / 68.6 / 67.8 | [ckpt]() |
    | ScanNet | B12/N7 | 56.8 / 69.2 / 48.2 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007346_connect_hku_hk/EVl7SdeUEPFAvrj2xnWSb-sBCOtWYyVOwBo6ggFb9x7dNA?e=feZaxH) |
    | ScanNet | B10/N9 | 54.5 / 74.9 / 42.8 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007346_connect_hku_hk/Ef0P_6XraDpCo0RRgOJ1wGQB-xOW7T6lecvVRi5P90Edbw?e=hqrP8X) |
    | S3DIS | B8/N4 |  39.0 / 57.0 / 29.6 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007346_connect_hku_hk/EYIW4SNX5B9Go_LKiim1KFEB_abYv0bDZMggE_6Ifjau0g?e=8BD0K3) |
    | S3DIS | B6/N6 | 41.1 / 54.2 / 33.0 | [ckpt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007346_connect_hku_hk/EeNYtkS3pmhAvc3Hxj7__SwB8SMzZdzmljRtCYuYG8NHcA?e=aC0aE2) |


