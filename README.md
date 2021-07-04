## UvA MSc thesis (Cees Kaandorp)

This is the repository that I will use during my thesis. It mostly includes works that are used for video summarization.

## Video Summarization Pipeline 

<img src="./figures/diagram.svg">


## Description

We use the preprocessed datasets provided by: https://github.com/KaiyangZhou/pytorch-vsumm-reinforce  
We fetch the input features for the video frames and the ground truth scores from the preprocessed datasets.   
These datasets are HDF5 files structured in the following way:


```
***********************************************************************************************************************************************
/key
    /features                 2D-array with shape (n_steps, feature-dimension)
    /gtscore                  1D-array with shape (n_steps), stores ground truth improtance score (used for training, e.g. regression loss)
    /user_summary             2D-array with shape (num_users, n_frames), each row is a binary vector (used for test)
    /change_points            2D-array with shape (num_segments, 2), each row stores indices of a segment
    /n_frame_per_seg          1D-array with shape (num_segments), indicates number of frames in each segment
    /n_frames                 number of frames in original video
    /picks                    posotions of subsampled frames in original video
    /n_steps                  number of subsampled frames
    /gtsummary                1D-array with shape (n_steps), ground truth summary provided by user (used for training, e.g. maximum likelihood)
    /video_name (optional)    original video name, only available for SumMe dataset
***********************************************************************************************************************************************
Note: OVP and YouTube only contain the first three keys, i.e. ['features', 'gtscore', 'gtsummary']
```


In addition to this, we utilize the data splitfiles provided by: https://github.com/ok1zjf/VASNet  
These files are used for n-fold cross validation, and are structured in the following way:


```
[
    {
        "test_keys": [
            "eccv16_dataset_tvsum_google_pool5/video_10",
            "eccv16_dataset_tvsum_google_pool5/video_20",
            "eccv16_dataset_tvsum_google_pool5/video_23",
            ...
        ],
        "train_keys": [
            "eccv16_dataset_tvsum_google_pool5/video_1",
            "eccv16_dataset_tvsum_google_pool5/video_11",
            "eccv16_dataset_tvsum_google_pool5/video_12",
            ...
```


In total we have 4 datasets: TVSum, SumMe,  OVP and YouTube
From which two different data setups can be used:
  * **Canonical**: The standard approach where we split one dataset(TVSum/SumMe) into train and test sets
  * **Augmented**: For a given dataset (TVSum/SumMe) we extract a test set, and join the rest with all the other datasets to use for training 

## Training

We can train by calling train.py 
