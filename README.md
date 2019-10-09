# MNER
PyTorch code for the paper "Adaptive Co-Attention Network for Named Entity Recognition in Tweets" published in AAAI 2018
The authors released their Keras implementation (https://github.com/jlfu/NERmultimodal)

### Dependencies:
```
pytorch (tested on version 0.3.0.post4, should work fine on newer 1.0.0 version as well)
torchcrf
tqdm
```

This model is slightly different from the model explained in the paper. Instead of coAttn, we use stackedAttn.

The perforamcne is similar to the CNN+BiLSTM+CRF (Ma and Hovy 2016) model as mentioned in the paper.

Data can be downloaded from http://pan.baidu.com/s/1boSlljL

Train a model using the following command:
```
python main.py --split_file /data/extDisk2/vvaibhav/vner/NERmultimodal/data --word2vec_model /data/extDisk2/vvaibhav/vner/glove.6B.200d.word2vec --image_features_dir /data/extDisk2/vvaibhav/vner/feat_bu/ --mode 0 --visual_feature_dimension 2048 --regions_in_image 36 --hidden_dimension 256 --hidden_dimension_char 50 --embedding_dimension 200 --embedding_dimension_char 30 --use_char_embedding 1 --batch_size 10  --use_filter_gate 1 --gamma 0.05 --dropout 0.2 --use_only_text 0 --lr 0.015
```

To evaluate a trained model on test set:
```
python main.py --split_file /data/extDisk2/vvaibhav/vner/NERmultimodal/data --image_features_dir /data/extDisk2/vvaibhav/vner/feat_bu/ --hidden_dimension 256 --hidden_dimension_char 50 --embedding_dimension 200 --embedding_dimension_char 30 --use_char_embedding 1 --batch_size 10 --visual_feature_dimension 2048 --regions_in_image 36 --use_filter_gate 1 --gamma 0.05 --dropout 0.2 --use_only_text 0 --lr 0.015 --mode 1 --model_file_name best_model_weights_13_0.934.t7
```

If you found this code useful in your research, please consider acknowledging the repo.
