import librosa
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import random
import ray
from typing import *
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import subprocess
import librosa
import soundfile as sf


@ray.remote
def fit_poly(y: np.array, degree: int=8) -> torch.Tensor:
    # fit degree polynomial to uniformly spread target values y
    
    x = np.linspace(0, 1, len(y))

    lambda_param = 0.001
    ridge_model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=lambda_param))

    ridge_model.fit(x[:, np.newaxis], y)
    ridge_coefficients = ridge_model.named_steps['ridge'].coef_
    ridge_coefficients[0] += ridge_model.named_steps['ridge'].intercept_
    
    return torch.from_numpy(ridge_coefficients)


@ray.remote
class DataProcessor:
    # this is an actor for processing clean data
    # performs audio feature extraction and blendshape curve fitting for sneeze data
    
    def __init__(self, model_name: str):

        self.wav2vec_ftr_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        self.blendshapes = ['EyeBlinkLeft', 'EyeLookDownLeft', 'EyeLookInLeft', 'EyeLookOutLeft',
            'EyeLookUpLeft', 'EyeSquintLeft', 'EyeWideLeft', 'EyeBlinkRight',
            'EyeLookDownRight', 'EyeLookInRight', 'EyeLookOutRight',
            'EyeLookUpRight', 'EyeSquintRight', 'EyeWideRight', 'JawForward',
            'JawRight', 'JawLeft', 'JawOpen', 'MouthClose', 'MouthFunnel',
            'MouthPucker', 'MouthRight', 'MouthLeft', 'MouthSmileLeft',
            'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight',
            'MouthDimpleLeft', 'MouthDimpleRight', 'MouthStretchLeft',
            'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper',
            'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft',
            'MouthPressRight', 'MouthLowerDownLeft', 'MouthLowerDownRight',
            'MouthUpperUpLeft', 'MouthUpperUpRight', 'BrowDownLeft',
            'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight',
            'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft',
            'NoseSneerRight', 'TongueOut', 'HeadYaw', 'HeadPitch', 'HeadRoll',
            'LeftEyeYaw', 'LeftEyePitch', 'LeftEyeRoll', 'RightEyeYaw',
            'RightEyePitch', 'RightEyeRoll']

    def parse_sneeze(self, audio_file: str, blendshape_file: str) -> Tuple[torch.Tensor, torch.Tensor]:
        # parse a sneeze example into an input, target vector pair
        
        # audio feature extraction
        input_audio, sample_rate = librosa.load(audio_file,  sr=16000)
        i = self.wav2vec_ftr_extractor(input_audio, return_tensors="pt", sampling_rate=sample_rate)
        with torch.no_grad():
            o = self.wav2vec_model(i.input_values)

        input = o.last_hidden_state.mean(dim=[0, 1])

        blendshape_df = pd.read_csv(blendshape_file)
        futures = [fit_poly.remote(np.array(blendshape_df[blendshape])) for blendshape in self.blendshapes]

        target = torch.cat(ray.get(futures), dim=0)

        return input, target


def extract_data(files: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    # read clean files, perform feature extraction and return inputs, targets

    model_name = "facebook/wav2vec2-large-xlsr-53"
    model_actor = DataProcessor.remote(model_name)

    futures = []
    inputs = []
    targets = []

    for idx in files:
        audio = f"/Users/sepehr/Desktop/Facial_Animation_Course/sneeze_demo/clean_sneeze_audios/audio{idx}.wav"
        bldshape = f"/Users/sepehr/Desktop/Facial_Animation_Course/sneeze_demo/clean_sneeze_blendshapes/blendshapes{idx}.csv"

        futures.append(model_actor.parse_sneeze.remote(audio, bldshape))

    results = ray.get(futures)

    for input, target in results:
        inputs.append(input)
        targets.append(target)

    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)

    return inputs, targets


def get_data(files: List[int], test_size: float=0.25, batch_size: int=8) -> Tuple[DataLoader, DataLoader]:
    # returns train and test Dataloaders
    inputs, targets = extract_data(files)

    # split and load data into DataLoaders
    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, 
                                   random_state=0,  
                                   test_size=test_size,  
                                   shuffle=True)

    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)

    train_dl = DataLoader(train_set, batch_size, shuffle=True)
    test_dl = DataLoader(test_set, batch_size, shuffle=True)

    return train_dl, test_dl


def extract_audio(idx: int) -> None:
    # extract .wav audio file from .mov video of sneezing

    input_video = f"/Users/sepehr/Desktop/Facial_Animation_Course/sneeze_demo/me/20231128_MySlate_{idx}/MySlate_{idx}_iPhone.mov"
    output_audio = f"/Users/sepehr/Desktop/Facial_Animation_Course/sneeze_demo/raw_sneeze_audios/audio{idx}.wav"
    subprocess.run(["ffmpeg", "-i", input_video, output_audio])


@ray.remote
def clip_data(idx: int) -> None:
    # detect time window in which sneeze takes place and clip audio and blenshape data to this interval
    # detect expulsion point of sneeze as timestamp of maximum audio intensity
   
    extract_audio(idx)

    audio_data, sr = librosa.load(f"/Users/sepehr/Desktop/Facial_Animation_Course/sneeze_demo/raw_sneeze_audios/audio{idx}.wav")

    amplitude = np.abs(audio_data)
    max_amplitude_index = np.argmax(amplitude)

    # this is identified as the expulsion point
    timestamp = max_amplitude_index / sr

    start_time = max(timestamp - 1.0, 0.0)
    end_time = min(timestamp + 1.0, len(amplitude)/sr)

    # clip audio
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    clipped_audio = audio_data[start_sample:end_sample]

    output_path = f"/Users/sepehr/Desktop/Facial_Animation_Course/sneeze_demo/clean_sneeze_audios/audio{idx}.wav"
    sf.write(output_path, clipped_audio, sr)

    # clip blendshapes
    start_sample = int(start_time * 60)
    end_sample = int(end_time * 60)

    blendshape_df = pd.read_csv(f"/Users/sepehr/Desktop/Facial_Animation_Course/sneeze_demo/me/20231128_MySlate_{idx}/MySlate_{idx}_iPhone.csv")
    clipped_blendshapes = blendshape_df[start_sample:end_sample].reset_index(drop=True)

    clipped_blendshapes.to_csv(f"/Users/sepehr/Desktop/Facial_Animation_Course/sneeze_demo/clean_sneeze_blendshapes/blendshapes{idx}.csv")