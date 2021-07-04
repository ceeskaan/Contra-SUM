import os
import math

import numpy as np
import torch

from google_drive_downloader import GoogleDriveDownloader as gdd
import h5py
import json


class TVSum(torch.utils.data.dataset.Dataset):
    def __init__(self, split, fold_id, augment):

        self.root = 'data/preprocessed'
        self.split = split
        self.augment = augment

        if not augment:
            # Load/Download preprocessed TVSum dataset
            self.check_integrity(['TVSum'])
            self.preprocessed = self.load_preprocessed(
                [f'{self.root}/TVSum/eccv16_dataset_tvsum_google_pool5.h5'])

        if augment:
            # Load/Download preprocessed TVSum, SumMe, OVP, and YouTube
            self.check_integrity(['TVSum', 'SumMe', 'augment_data'])
            self.preprocessed = self.load_preprocessed([f'{self.root}/TVSum/eccv16_dataset_tvsum_google_pool5.h5',
                                                        f'{self.root}/SumMe/eccv16_dataset_summe_google_pool5.h5',
                                                        f'{self.root}/augment_data/eccv16_dataset_ovp_google_pool5.h5',
                                                        f'{self.root}/augment_data/eccv16_dataset_youtube_google_pool5.h5'])

        self.dataset = self.load_data(fold_id)

    def load_data(self, fold_id):
        """
            Loads inputs, target scores and additional information from preprocessed datasets
        """

        if self.augment:
            with open(f'{self.root}/TVSum/tvsum_aug_splits.json', 'r') as sf:
                folds = json.load(sf)
        else:
            with open(f'{self.root}/TVSum/tvsum_splits.json', 'r') as sf:
                folds = json.load(sf)

        keys = folds[fold_id][f'{self.split}_keys']

        data = []
        for key in keys:
            data_name, data_key = key.split('/')
            info = self.preprocessed[data_name][data_key]
            
            to_dict = {key: info[key][...] for key in info}
            data.append((info['features'][...], info['gtscore'][...], to_dict, data_key))

        return data

    def load_preprocessed(self, datasets):
        """
        Loads all h5 datasets from the datasets list into a dictionary self.dataset
        referenced by their base filename
        :param datasets:  List of dataset filenames
        :return:
        """

        datasets_dict = {}
        for dataset in datasets:
            _, base_filename = os.path.split(dataset)
            base_filename, _ = os.path.splitext(base_filename)
            datasets_dict[base_filename] = h5py.File(dataset, 'r')

        return datasets_dict

    def check_integrity(self, dataset):
        for i in dataset:
            if os.path.isdir(f'./{self.root}/{i}') == False:
                print('data not found')
                self.download_data(i)
        return None

    def download_data(self, dataset):
        if dataset == 'TVSum':
            print('Downloading TVSum data...')
            gdd.download_file_from_google_drive(file_id='13zXvw9Xgkv1yyeH_9PzNet8rZn0tzQVZ',
                                                dest_path='./data/preprocessed/TVSum.zip',
                                                unzip=True, showsize=True)
        if dataset == 'SumMe':
            print('Downloading SumMe data...')
            gdd.download_file_from_google_drive(file_id='1cobxmLW8d4hk5ogQMD3ZGnrOJJmX2V7v',
                                                dest_path='./data/preprocessed/SumMe.zip',
                                                unzip=True, showsize=True)
        if dataset == 'augment_data':
            print('Downloading augmentation data...')
            gdd.download_file_from_google_drive(file_id='1k9t627bpf6_GNnUD9stF6DokCKbpddux',
                                                dest_path='./data/preprocessed/augment_data.zip',
                                                unzip=True, showsize=True)

        os.remove(f'./{self.root}/{dataset}.zip')

    def __getitem__(self, index):
        input, target, info, key = self.dataset[index]

        # Normalize target scores
        target -= target.min()
        target /= target.max()

        return np.array(input), target, info, key

    def __len__(self):
        return len(self.dataset)


class TVSum_Batch(TVSum):
    def __init__(self, split, fold_id, augment, video_length=400, padding_method='zero'):
        super(TVSum_Batch, self).__init__(split, fold_id, augment)
        self.padding_method = padding_method
        self.video_length = video_length

        self.padded_data = self.pad_data(self.dataset)
        self.data = self.get_clips(self.padded_data)
        
        
    def pad(self, features, to_length, method='zero'):
        amount = math.ceil((to_length - len(features))/2)
        if method == 'zero':
            pad = torch.nn.ConstantPad1d(amount, 0)
            padded_features = pad(features.T).T
        if method == 'replication':
            pad = torch.nn.ReplicationPad1d(amount)
            padded_features = pad(features.T.unsqueeze(0)).squeeze(0).T
        return padded_features

    def pad_data(self, preprocessed):
        mean = round(sum([len(i) for i,j,k,l in preprocessed]) / len(preprocessed))

        videos = []
        for video,_,_,_ in preprocessed:
            data = torch.tensor(video)

            if len(data) <= mean:
                videos.append(self.pad(data, mean, self.padding_method))
                
            if len(data) > mean:
                videos.append(data)
        
        return videos

    def get_clips(self, videos):
        clip_data = []
        for video in videos:
            for i in range(len(video)-self.video_length+1):
                clip_data.append(video[i:i+self.video_length])

        return clip_data

    def __getitem__(self, index):
        x = self.data[index]
        return x
    
    def __len__(self):
        return len(self.data)


class SumMe(torch.utils.data.dataset.Dataset):
    def __init__(self, split, fold_id, augment):

        self.root = 'data/preprocessed'
        self.split = split
        self.augment = augment

        if not augment:
            # Load/Download preprocessed SumMe dataset
            self.check_integrity(['SumMe'])
            self.preprocessed = self.load_preprocessed(
                [f'{self.root}/SumMe/eccv16_dataset_summe_google_pool5.h5'])

        if augment:
            # Load/Download preprocessed TVSum, SumMe, OVP, and YouTube
            self.check_integrity(['TVSum', 'SumMe', 'augment_data'])
            self.preprocessed = self.load_preprocessed([f'{self.root}/TVSum/eccv16_dataset_tvsum_google_pool5.h5',
                                                        f'{self.root}/SumMe/eccv16_dataset_summe_google_pool5.h5',
                                                        f'{self.root}/augment_data/eccv16_dataset_ovp_google_pool5.h5',
                                                        f'{self.root}/augment_data/eccv16_dataset_youtube_google_pool5.h5'])

        self.dataset = self.load_data(fold_id)

    def load_data(self, fold_id):
        """
            Loads inputs, target scores and additional information from preprocessed datasets
        """

        if self.augment:
            with open(f'{self.root}/SumMe/summe_aug_splits.json', 'r') as sf:
                folds = json.load(sf)
        else:
            with open(f'{self.root}/SumMe/summe_splits.json', 'r') as sf:
                folds = json.load(sf)

        keys = folds[fold_id][f'{self.split}_keys']

        data = []
        for key in keys:
            data_name, data_key = key.split('/')
            info = self.preprocessed[data_name][data_key]

            to_dict = {key: info[key][...] for key in list(info)[:9]}
            data.append((info['features'][...], info['gtscore'][...], to_dict, data_key))

        return data

    def load_preprocessed(self, datasets):
        """
        Loads all h5 datasets from the datasets list into a dictionary self.dataset
        referenced by their base filename
        :param datasets:  List of dataset filenames
        :return:
        """

        datasets_dict = {}
        for dataset in datasets:
            _, base_filename = os.path.split(dataset)
            base_filename, _ = os.path.splitext(base_filename)
            datasets_dict[base_filename] = h5py.File(dataset, 'r')

        return datasets_dict

    def check_integrity(self, dataset):
        for i in dataset:
            if os.path.isdir(f'./{self.root}/{i}') == False:
                self.download_data(i)
        return None

    def download_data(self, dataset):
        if dataset == 'TVSum':
            print('Downloading TVSum data...')
            gdd.download_file_from_google_drive(file_id='13zXvw9Xgkv1yyeH_9PzNet8rZn0tzQVZ',
                                                dest_path='./data/preprocessed/TVSum.zip',
                                                unzip=True, showsize=True)
        if dataset == 'SumMe':
            print('Downloading SumMe data...')
            gdd.download_file_from_google_drive(file_id='1cobxmLW8d4hk5ogQMD3ZGnrOJJmX2V7v',
                                                dest_path='./data/preprocessed/SumMe.zip',
                                                unzip=True, showsize=True)
        if dataset == 'augment_data':
            print('Downloading augmentation data...')
            gdd.download_file_from_google_drive(file_id='1k9t627bpf6_GNnUD9stF6DokCKbpddux',
                                                dest_path='./data/preprocessed/augment_data.zip',
                                                unzip=True, showsize=True)

        os.remove(f'./{self.root}/{dataset}.zip')

    def __getitem__(self, index):
        input, target, info, key = self.dataset[index]

        # Normalize target scores
        target -= target.min()
        target /= target.max()

        return np.array(input), target, info, key

    def __len__(self):
        return len(self.dataset)

class SumMe_Batch(SumMe):
        def __init__(self, split, fold_id, augment, video_length=200, padding_method='zero'):
            super(SumMe_Batch, self).__init__(split, fold_id, augment)
            self.padding_method = padding_method
            self.video_length = video_length

            self.padded_data = self.pad_data(self.dataset)
            self.data = self.get_clips(self.padded_data)
            
            
        def pad(self, features, to_length, method='zero'):
            amount = math.ceil((to_length - len(features))/2)
            if method == 'zero':
                pad = torch.nn.ConstantPad1d(amount, 0)
                padded_features = pad(features.T).T
            if method == 'replication':
                pad = torch.nn.ReplicationPad1d(amount)
                padded_features = pad(features.T.unsqueeze(0)).squeeze(0).T
            return padded_features

        def pad_data(self, preprocessed):
            mean = round(sum([len(i) for i,j,k,l in preprocessed]) / len(preprocessed))

            videos = []
            for video,_,_,_ in preprocessed:
                data = torch.tensor(video)

                if len(data) <= mean:
                    videos.append(self.pad(data, mean, self.padding_method))
                    
                if len(data) > mean:
                    videos.append(data)
            
            return videos

        def get_clips(self, videos):
            clip_data = []
            for video in videos:
                for i in range(len(video)-self.video_length+1):
                    clip_data.append(video[i:i+self.video_length])

            return clip_data

        def __getitem__(self, index):
            x = self.data[index]
            return x
        
        def __len__(self):
            return len(self.data)