import os
import sys
import time
import librosa
import yaml
import joblib
import argparse

import soundfile as sf
import numpy as np

from pathlib import Path
from collections import defaultdict
from typing import Optional
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# WavLM
from nnet.WavLM import WavLM, WavLMConfig

# Xcodec2
from vq.codec_encoder import CodecEncoder_Transformer
from vq.codec_decoder_vocos import CodecDecoderVocos
from vq.module import SemanticEncoder
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from collections import OrderedDict

# Dataloader
from loader.datareader import DataReader
from loader.datareader_aec import DataReaderAEC
from loader.datareader_tse import DataReaderTSE

# LLaSE
from nnet.llase import LLM_AR as model

class Encodec():
    '''
    Load Xcodec2
    '''
    def __init__(self,device="cpu") -> None:
        self.device=device
        ckpt = "./ckpt/codec_ckpt/epoch=4-step=1400000.ckpt",
        # ckpt = '/home/bykang/codec_ckpt/epoch=4-step=1400000.ckpt'
        ckpt = torch.load(ckpt, map_location='cpu')
        state_dict = ckpt['state_dict']
        filtered_state_dict_codec = OrderedDict()
        filtered_state_dict_semantic_encoder = OrderedDict()
        filtered_state_dict_gen = OrderedDict()
        filtered_state_dict_fc_post_a = OrderedDict()
        filtered_state_dict_fc_prior = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('CodecEnc.'):
                new_key = key[len('CodecEnc.'):]
                filtered_state_dict_codec[new_key] = value
            elif key.startswith('generator.'):
                new_key = key[len('generator.'):]
                filtered_state_dict_gen[new_key] = value
            elif key.startswith('fc_post_a.'):
                new_key = key[len('fc_post_a.'):]
                filtered_state_dict_fc_post_a[new_key] = value
            elif key.startswith('SemanticEncoder_module.'):
                new_key = key[len('SemanticEncoder_module.'):]
                filtered_state_dict_semantic_encoder[new_key] = value
            elif key.startswith('fc_prior.'):
                new_key = key[len('fc_prior.'):]
                filtered_state_dict_fc_prior[new_key] = value
        
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(
            "./ckpt/codec_ckpt/hub/models--facebook--w2v-bert-2.0",
            # "/home/bykang/codec_ckpt/hub/models--facebook--w2v-bert-2.0/snapshots/da985ba0987f70aaeb84a80f2851cfac8c697a7b",
            output_hidden_states=True)
        self.semantic_model=self.semantic_model.eval().to(self.device)
        
        self.SemanticEncoder_module = SemanticEncoder(1024,1024,1024)
        self.SemanticEncoder_module.load_state_dict(filtered_state_dict_semantic_encoder)
        self.SemanticEncoder_module = self.SemanticEncoder_module.eval().to(self.device)

        self.encoder = CodecEncoder_Transformer()
        self.encoder.load_state_dict(filtered_state_dict_codec)
        self.encoder = self.encoder.eval().to(self.device)

        self.decoder = CodecDecoderVocos()
        self.decoder.load_state_dict(filtered_state_dict_gen)
        self.decoder = self.decoder.eval().to(self.device)

        self.fc_post_a = nn.Linear( 2048, 1024 )
        self.fc_post_a.load_state_dict(filtered_state_dict_fc_post_a)
        self.fc_post_a = self.fc_post_a.eval().to(self.device)

        self.fc_prior = nn.Linear( 2048, 2048 )
        self.fc_prior.load_state_dict(filtered_state_dict_fc_prior)
        self.fc_prior = self.fc_prior.eval().to(self.device)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "./ckpt/codec_ckpt/hub/models--facebook--w2v-bert-2.0")
            # "/home/bykang/codec_ckpt/hub/models--facebook--w2v-bert-2.0/snapshots/da985ba0987f70aaeb84a80f2851cfac8c697a7b")
    
    def get_feat(self, wav_batch, pad=None):

        if len(wav_batch.shape) != 2:
            return self.feature_extractor(F.pad(wav_batch, pad), sampling_rate=16000, return_tensors="pt") .data['input_features']
        
        padded_wavs = torch.stack([F.pad(wav, pad) for wav in wav_batch])
        batch_feats = []

        for wav in padded_wavs:
            feat = self.feature_extractor(
                wav,
                sampling_rate=16000,
                return_tensors="pt"
            ).data['input_features']

            batch_feats.append(feat)
        feat_batch = torch.concat(batch_feats, dim=0).to(self.device)
        return feat_batch 

    def get_embedding(self, wav_cpu):
        wav_cpu = wav_cpu.cpu()
        feat = self.get_feat(wav_cpu,pad=(160,160))
        feat = feat.to(self.device)

        if(len(wav_cpu.shape)==1):
            wav = wav_cpu.unsqueeze(0).to(self.device)
        else:
            wav = wav_cpu.to(self.device)

        wav = torch.nn.functional.pad(wav, (0, (200 - (wav.shape[1] % 200))))
        with torch.no_grad():
            vq_emb = self.encoder(wav.unsqueeze(1))
            vq_emb = vq_emb.transpose(1, 2) 

            if vq_emb.shape[2]!=feat.shape[1]:
                feat = self.get_feat(wav_cpu)
                feat = feat.to(self.device)

            semantic_target = self.semantic_model(feat[:,  :,:])
            semantic_target = semantic_target.hidden_states[16]
            semantic_target = semantic_target.transpose(1, 2)
            semantic_target = self.SemanticEncoder_module(semantic_target)

            vq_emb = torch.cat([semantic_target, vq_emb], dim=1)

        return vq_emb
    
    def emb2token(self, emb):
        emb.to(self.device)
        emb =  self.fc_prior(emb.transpose(1, 2)).transpose(1, 2)
        _, vq_code, _ = self.decoder(emb, vq=True)
        return vq_code

    def token2wav(self, vq_code):
        vq_code.to(self.device)
        vq_post_emb = self.decoder.quantizer.get_output_from_indices(vq_code.transpose(1, 2))
        vq_post_emb = vq_post_emb.transpose(1, 2)
        vq_post_emb = self.fc_post_a(vq_post_emb.transpose(1,2)).transpose(1,2)
        recon = self.decoder(vq_post_emb.transpose(1, 2), vq=False)[0].squeeze()
        # if write the wav, add .squeeze().detach().cpu().numpy()
        # if need gradient use the config right now
        return recon

class WavLM_feat(object):
    '''
    Load WavLM
    '''
    def __init__(self, device):
        self.wavlm = self._reload_wavLM_large(device=device)

    def __call__(self, wav):
        T = wav.shape[-1]
        wav = wav.reshape(-1, T)
        with torch.no_grad():
            feat = self.wavlm.extract_features(wav, output_layer=6, ret_layer_results=False)[0]
            B, T, D = feat.shape
            feat = torch.reshape(feat, (-1, D))

            return feat

    def _reload_wavLM_large(self, path="./ckpt/codec_ckpt/WavLM-Large.pt", device: Optional[torch.device] = None):
        cpt = torch.load(path, map_location="cpu")
        cfg = WavLMConfig(cpt['cfg'])
        wavLM = WavLM(cfg)
        wavLM.load_state_dict(cpt['model'])
        wavLM.eval()
        if device != None:
            wavLM = wavLM.to(device)
        for p in wavLM.parameters():
            p.requires_grad = False
        print('successful to reload wavLM', path)
        return wavLM 

def get_firstchannel_read(path, fs=16000):
    '''
    Get first channel of the wav
    '''
    wave_data, sr = sf.read(path)
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, orig_sr=sr, target_sr=fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    if len(wave_data.shape) > 1:
        wave_data = wave_data[:, 0]
    return wave_data

def load_obj(obj, device):
    '''
    Offload tensor object in obj to cuda device
    '''
    def cuda(obj):
        return obj.to(device) if isinstance(obj, torch.Tensor) else obj
    
    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)

def run(args):
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])    
    dist.init_process_group(args.backend, rank=WORLD_RANK, world_size=WORLD_SIZE)    
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device('cuda', LOCAL_RANK)
    print(f"[{os.getpid()}] using device: {device}", torch.cuda.current_device(), "local rank", LOCAL_RANK)

    with open(args.conf, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    # Dataloader
    if conf["task"]=="AEC":
        data_reader = DataReaderAEC(**conf["datareader"])
    elif conf["task"]=="TSE":
        data_reader = DataReaderTSE(**conf["datareader"])
    else:
        data_reader = DataReader(**conf["datareader"])

    # Load WavLM and XCodec2
    codec = Encodec(device)
    wavlm_feat = WavLM_feat(device)

    # Load LLaSE
    nnet = model(**conf["nnet_conf"])
    cpt_fname = Path(conf["test"]["checkpoint"])
    cpt = torch.load(cpt_fname, map_location="cpu")

    nnet = nnet.to(device)
    nnet = DistributedDataParallel(nnet, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True) 
    nnet.load_state_dict(cpt["model_state_dict"])
    nnet.eval()

    # Make sure the dir exists
    if conf["task"]=="AEC":
        if not os.path.exists(conf["save"]["feat_dir"]+"/mic"):
            os.makedirs(conf["save"]["feat_dir"]+"/mic")
        if not os.path.exists(conf["save"]["feat_dir"]+"/ref"):
            os.makedirs(conf["save"]["feat_dir"]+"/ref")
    elif conf["task"]=="TSE":
        if not os.path.exists(conf["save"]["feat_dir"]+"/mic"):
            os.makedirs(conf["save"]["feat_dir"]+"/mic")
        if not os.path.exists(conf["save"]["feat_dir"]+"/ref"):
            os.makedirs(conf["save"]["feat_dir"]+"/ref")
    else:
        if not os.path.exists(conf["save"]["feat_dir"]):
            os.makedirs(conf["save"]["feat_dir"])
            
    if not os.path.exists(conf["save"]["wav_dir"]):
        os.makedirs(conf["save"]["wav_dir"])      

    # Main of inference
    if_feat_too = conf["test"]["infer_feat_too"]

    origin_feat_dir = conf["save"]["feat_dir"]
    origin_wav_dir = conf["save"]["wav_dir"]
    
    last_feat_dir = origin_feat_dir
    last_wav_dir = origin_wav_dir

    for inference_time in range(conf["test"]["inference_time"]):
        # For multi-inference
        if inference_time > 0:
            feat_dir = origin_feat_dir + "inference" + str(inference_time) 
            wav_dir = origin_wav_dir + "inference" + str(inference_time) 
        else:
            feat_dir = origin_feat_dir
            wav_dir = origin_wav_dir
            
        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)
        if not os.path.exists(wav_dir):
            os.makedirs(wav_dir)

        with torch.no_grad():
            # Extract WavLM features
            if if_feat_too ==True or inference_time>0:
                for egs in tqdm(data_reader):
                    egs = load_obj(egs, device)
                    
                    if conf["task"]=="AEC" or conf["task"]=="TSE":
                        if inference_time > 0:
                            mic_path = last_wav_dir + '/' + egs["mic_name"] + ".wav"
                            egs["mic"] = torch.from_numpy(get_firstchannel_read(mic_path).astype(np.float32)).unsqueeze(0).to(device)
                        else:
                            egs["mic"]=egs["mic"].contiguous()                            
                        egs["ref"]=egs["ref"].contiguous()

                        feat_mic = wavlm_feat(egs["mic"])
                        out_mic = feat_mic.detach().squeeze(0).cpu().numpy()
                        
                        if not os.path.exists(os.path.join(feat_dir, "mic")):
                            os.makedirs(os.path.join(feat_dir, "mic"))    
                        np.save(os.path.join(feat_dir, "mic", egs["mic_name"]), out_mic)
                        
                        # For AEC and TSE, reference audio only need to extract feats at first time
                        if inference_time == 0:
                            feat_ref = wavlm_feat(egs["ref"])
                            out_ref = feat_ref.detach().squeeze(0).cpu().numpy()
                            np.save(os.path.join(origin_feat_dir, "ref", egs["ref_name"]), out_ref)

                        torch.cuda.empty_cache()

                    else:
                        if inference_time > 0:
                            mix_path = last_wav_dir + '/' + egs["name"] + ".wav"
                            egs["mix"] = torch.from_numpy(get_firstchannel_read(mix_path).astype(np.float32)).unsqueeze(0).to(device)
                        else:
                            egs["mix"]=egs["mix"].contiguous()
                        
                        feat = wavlm_feat(egs["mix"])
                        out = feat.detach().squeeze(0).cpu().numpy()
                        np.save(os.path.join(feat_dir, egs["name"]), out)
            
            # Predict the clean tokens and token2wav
            for egs in tqdm(data_reader):
                egs = load_obj(egs, device)
                sr = 16000
                
                if conf["task"] == "AEC":
                    # Get feat
                    feat_path_mic = os.path.join(feat_dir, "mic", egs["mic_name"]) + ".npy" 
                    feat_path_ref = os.path.join(origin_feat_dir, "ref", egs["ref_name"]) + ".npy"

                    feat_mic = torch.from_numpy(np.load(feat_path_mic)).unsqueeze(0)
                    feat_ref = torch.from_numpy(np.load(feat_path_ref)).unsqueeze(0)

                    # For multi-inference
                    if inference_time > 0:
                        est = nnet(feat_mic)
                    else:
                        est = nnet(feat_mic, feat_ref)

                    # Get tokens and token2wav
                    max, max_indices_1 = torch.max(est[1], dim=1)
                    recon_1 = codec.token2wav(max_indices_1.unsqueeze(0)).squeeze().detach().cpu().numpy()

                    # Save the wav
                    target_path = os.path.join(wav_dir, egs["mic_name"] + ".wav")
                    print(target_path)
                    sf.write(target_path , recon_1, sr)   
                    
                elif conf["task"] == "TSE" :
                    # Get feat
                    feat_path_mic = os.path.join(feat_dir, "mic", egs["mic_name"]) + ".npy" 
                    feat_path_ref = os.path.join(origin_feat_dir, "ref", egs["ref_name"]) + ".npy"

                    feat_mic = torch.from_numpy(np.load(feat_path_mic)).unsqueeze(0)
                    feat_ref = torch.from_numpy(np.load(feat_path_ref)).unsqueeze(0)

                    # Choose if keep the enroallment audio while multi-inference
                    if_keep_ref = True

                    if inference_time>0 and if_keep_ref== False:
                        est = nnet(feat_mic)
                    else:
                        est = nnet(feat_mic, feat_ref)
                    
                    # Get tokens and token2wav
                    max, max_indices_1 = torch.max(est[0], dim=1)
                    recon_1 = codec.token2wav(max_indices_1.unsqueeze(0)).squeeze().detach().cpu().numpy()

                    # Save the wav
                    target_path = os.path.join(wav_dir, egs["mic_name"] + ".wav")
                    print(target_path)
                    sf.write(target_path , recon_1, sr) 
                    
                elif conf["task"] == "PLC":
                    # Get feat
                    feat_path = os.path.join(feat_dir, egs["name"]) + ".npy" 
                    feat = torch.from_numpy(np.load(feat_path)).unsqueeze(0)

                    # Get tokens and token2wav
                    est = nnet(feat)
                    max, max_indices_1 = torch.max(est[1], dim=1)
                    recon_1 = codec.token2wav(max_indices_1.unsqueeze(0)).squeeze().detach().cpu().numpy()

                    # Save the wav
                    target_path = os.path.join(wav_dir, egs["name"] + ".wav")
                    print(target_path)
                    sf.write(target_path , recon_1, sr)
                    
                elif conf["task"] == "SS":
                    # Get feat
                    feat_path = os.path.join(feat_dir, egs["name"]) + ".npy" 
                    feat = torch.from_numpy(np.load(feat_path)).unsqueeze(0)
                    
                    # Separate the first speaker
                    est = nnet(feat)
                    max, max_indices_1 = torch.max(est[1], dim=1)
                    recon_1 = codec.token2wav(max_indices_1.unsqueeze(0)).squeeze().detach().cpu().numpy()

                    target_path_1 = os.path.join(wav_dir, egs["name"] + ".wav")
                    sf.write(target_path_1 , recon_1, sr)
                    
                    # Separate the second speaker, SS need at least 2 inference time in config
                    if inference_time > 0:
                        origin_feat_path = os.path.join(origin_feat_dir, egs["name"]) + ".npy"
                        origin_feat = torch.from_numpy(np.load(origin_feat_path)).unsqueeze(0)
                        
                        est2 = nnet(origin_feat, feat)
                        max, max_indices_2 = torch.max(est2[1], dim=1)
                        recon_2 = codec.token2wav(max_indices_2.unsqueeze(0)).squeeze().detach().cpu().numpy()
                    
                        if not os.path.exists(last_wav_dir + "s2"):
                            os.makedirs(last_wav_dir + "s2")
                    
                        target_path_2 = os.path.join(last_wav_dir + "s2", egs["name"] + ".wav")
                        sf.write(target_path_2 , recon_2, sr)
                    
                else:
                    # Get feat
                    feat_path = os.path.join(feat_dir, egs["name"]) + ".npy" 
                    feat = torch.from_numpy(np.load(feat_path)).unsqueeze(0)
                    
                    # Get tokens and token2wav
                    est = nnet(feat)
                    max, max_indices_1 = torch.max(est[1], dim=1)
                    recon_1 = codec.token2wav(max_indices_1.unsqueeze(0)).squeeze().detach().cpu().numpy()

                    # Save the wav
                    target_path = os.path.join(wav_dir, egs["name"] + ".wav")
                    print(target_path)
                    sf.write(target_path , recon_1, sr)        

        # For next inference
        last_feat_dir = feat_dir
        last_wav_dir = wav_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Command to test separation model in Pytorch",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-conf",
                        type=str,
                        required=True,
                        help="Yaml configuration file for training")
    parser.add_argument("--backend",
                        type=str,
                        default="nccl",
                        choices=["nccl", "gloo"])                          
    args = parser.parse_args()    
    # for nccl debug
    os.environ["NCCL_DEBUG"] = "INFO"
    run(args)