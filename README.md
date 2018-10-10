# MNER
Multimodal Named Entity Recognition

Train a model using the following command:
```
python main.py --split_file /data/extDisk2/vvaibhav/vner/NERmultimodal/data --word2vec_model /data/extDisk2/vvaibhav/vner/glove.twitter.27B.200d.word2vec --image_features_dir /data/extDisk2/vvaibhav/vner/feat_bu/ --visual_feature_dimension 2048 --regions_in_image 36 --hidden_dimension 256 --hidden_dimension_char 50 --embedding_dimension 200 --embedding_dimension_char 30 --use_char_embedding 1 --batch_size 128 
```
