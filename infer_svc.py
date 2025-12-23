import os
import torch
from make_parser import make_parse_cmd
import random
import numpy as np
from logger.utils import load_config, save_config
import time
from utils.models.models_v2_beta import load_generator_mel_diff_ddsp_v5a, load_controlsvc_v1, load_facodec_mel
from torch.nn import functional as F
from utils.utils import repeat_expand, repeat_expand_2d
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from utils.vocoder import Vocoder
from utils.data_loader_beta import load_style_audio_data, get_file_list
import soundfile as sf
from tqdm import tqdm
from utils.data_preprocess_v2_beta import get_processed_file, load_facodec, load_f0_extractor, load_volume_extractor, load_content_encoder
from utils.ThreeD_Speaker.speakerlab.bin.get_spk_sim import build_model

def get_mel(audio, sr, mel_extractor):
    if audio.shape[0] != 1:
        audio = audio.unsqueeze(0)
    mel = mel_extractor.extract(audio, sr) #input shape: (1, duration*sr)
    mel = mel.squeeze()
    return mel

def get_batch_mel(audio, sr, mel_extractor):
    if audio.shape[0]>1:
        mel_list = [] # audio: (B, 1, T)
        for i in range(audio.shape[0]):
            mel = get_mel(audio[i], sr, mel_extractor)
            mel_list.append(mel)
        return torch.stack(mel_list) # (B, T, D)
    else:
        return get_mel(audio, sr, mel_extractor)

def batch_repeat_expand_2d(x, T):
    if x.shape[0]>1:
        x_list = []
        for i in range(x.shape[0]):
            x_ = repeat_expand_2d(x[i], T)
            x_list.append(x_)
        return torch.stack(x_list)
    else:
        return repeat_expand_2d(x, T)

def batch_repeat_expand(x, T):
    if x.shape[0]>1:
        x_list = []
        for i in range(x.shape[0]):
            x_ = repeat_expand(x[i], T)
            x_list.append(x_)
        return torch.stack(x_list)
    else:
        return repeat_expand(x, T)

def align_mel(mel, max_len):
    mel_len = mel.shape[-1]
    if mel_len < max_len:
        mel = F.pad(mel, (0, max_len - mel_len))
    elif mel_len > max_len:
        mel = mel[:, :max_len]
    return mel

def get_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def get_mean_f0(f0):
    f0 = f0[f0>0]
    if len(f0) == 0:
        return 0
    return f0.mean()

def get_std_f0(f0):
    f0 = f0[f0>0]
    if len(f0) == 0:
        return 0
    return f0.std()

def quantize_key(key):
    return round(key)

def get_adjust_key_f0(src_f0, tar_f0, quantize=False):
    src_mean = get_mean_f0(src_f0)
    tar_mean = get_mean_f0(tar_f0)
    if src_mean == 0 or tar_mean == 0:
        return 0
    f0_rate = tar_mean / src_mean
    # # 计算源和目标基频的中位数
    # src_median = np.median(src_f0)
    # tar_median = np.median(tar_f0)
    # f0_rate = tar_median / src_median
    key = 12 * np.log2(f0_rate)
    if quantize:
        quantized_key = quantize_key(key.item())
    return quantized_key


def get_adjust_spk(src_spk, tar_spk):
    src_mean = src_spk.mean()
    tar_mean = tar_spk.mean()
    src_std = src_spk.std()
    tar_std = tar_spk.std()

    if src_std == 0 or tar_std == 0:
        return None
    
    mean = tar_mean - src_mean
    std = tar_std / src_std
    transform_spk = src_spk * std + mean
    return transform_spk
    
def get_vc_audios(net_g, vocoder, tar_loader, src_loader, args):
    device = args.device
    shift_key = args.shift_key if args.shift_key is not None else 0
    adjust_f0 = args.adjust_f0 if args.adjust_f0 is not None else False
    
    spk_shot = args.spk_shot # wav num of tar spk for reference
    wav_g_list = []
    if spk_shot == 0:
        spk_shot = len(tar_loader)
        
    with torch.no_grad(): 
        spk_ave = None
        cnt = 0
        
        for _, data in enumerate(tar_loader):
            spk = data['spk']
            
            spk_id = None
            if args.use_spk_id:
                spk_id = data['spk_id']
                
            if spk_ave is None:
                spk_ave = spk
            else:
                spk_ave += spk 
                
            if adjust_f0:
                f0 = data['f0']
                if cnt == 0:
                    tar_f0 = f0
                else:
                    tar_f0 = torch.cat((tar_f0, f0), axis=1)
            
            if cnt == 0:
                tar_spk = spk
            else:
                tar_spk += spk 
            cnt += 1
            if cnt >= spk_shot:
                break

        spk_ave /= cnt    
        spk_ave = spk_ave.squeeze(1).to(device)           
        
        cnt = 0     
        for _, data in enumerate(src_loader):
            # spk = data['spk']
            f0 = data['f0']
            if cnt == 0:
                src_f0 = f0
            else:
                src_f0 = torch.cat((src_f0, f0), axis=1)
             
            cnt += 1
            
        for _, data in enumerate(src_loader):
            vq_post = data['vq_post'].to(device)
            vol = data['vol'].to(device)
            vq_post = vq_post.squeeze(1)
            f0 = data['f0'].to(device)
                
            if args.adjust_f0:
                shift_key = get_adjust_key_f0(src_f0, tar_f0, quantize=args.quantize_key)
                f0 = f0.to(device) * 2 ** (float(shift_key) / 12)
                print(f'Shift key is adjusted to: {shift_key}')
            
            elif args.shift_key is not None:
                f0 = f0.to(device) * 2 ** (float(shift_key) / 12)
                print(f'Shift key is adjusted to: {shift_key}')
                
            style = None
                
            mel_g = net_g(vq_post, f0, vol, spk_ave, spk_id, style, 
                          gt_spec=data['mel_44k'].permute(0, 2, 1).to(device), infer=True, infer_speedup=args.infer_speedup, method=args.infer_method, vocoder=vocoder)
            mel_g = mel_g.permute(0, 2, 1) #Diffusion output shape is [B, D, T] to [B, T, D]
            mel_g = mel_g.to(device)
            if args.vocoder == 'nsf-hifigan':
                wav_g = vocoder.infer(mel_g.permute(0, 2, 1), f0)
            else:
                wav_g = vocoder.infer(mel_g.permute(0, 2, 1))
            wav_g_list.append(wav_g)
    return wav_g

def vc_infer(net_g, vocoder, tar_loader, src_loader, args, bert_adaptor=None):
    device = args.device
    # log_dir = os.path.join(args.ckpt_dir, "logs", get_time())
    model_path = args.model_path
    shift_key = args.shift_key if args.shift_key is not None else 0
    adjust_f0 = args.adjust_f0 if args.adjust_f0 is not None else False
    
    src_spk_name = os.path.split(args.src_path)[-1]
    tar_spk_name = os.path.split(args.tar_path)[-1]
    
    spk_shot = args.spk_shot # wav num of tar spk for reference
    
    if spk_shot == 0:
        spk_shot = len(tar_loader)
        
    with torch.no_grad(): 
        spk_ave = None
        cnt = 0
        
        for _, data in enumerate(tar_loader):
            spk = data['spk']
            
            spk_id = None
            if args.use_spk_id:
                spk_id = data['spk_id']
                
            if spk_ave is None:
                spk_ave = spk
            else:
                spk_ave += spk 
                
            if adjust_f0:
                f0 = data['f0']
                if cnt == 0:
                    tar_f0 = f0
                else:
                    tar_f0 = torch.cat((tar_f0, f0), axis=1)
            
            if cnt == 0:
                tar_spk = spk
            else:
                tar_spk += spk 
            cnt += 1
            if cnt >= spk_shot:
                break

        spk_ave /= cnt    
        spk_ave = spk_ave.squeeze(1).to(device)       
        now_time = get_time()

        output_dir = None
    
        
        cnt = 0     
        for _, data in enumerate(src_loader):
            # spk = data['spk']
            f0 = data['f0']
            if cnt == 0:
                src_f0 = f0
            else:
                src_f0 = torch.cat((src_f0, f0), axis=1)
             
            cnt += 1
            
        for _, data in enumerate(src_loader):
            vq_post = data['vq_post'].to(device)
            file_name = data['name'][0].split('/')[-1]
            vol = data['vol'].to(device)
            vq_post = vq_post.squeeze(1)
            f0 = data['f0'].to(device)
                
            if args.adjust_f0:
                shift_key = get_adjust_key_f0(src_f0, tar_f0, quantize=args.quantize_key)
                f0 = f0.to(device) * 2 ** (float(shift_key) / 12)
                print(f'Shift key is adjusted to: {shift_key}')
            
            elif args.shift_key is not None:
                f0 = f0.to(device) * 2 ** (float(shift_key) / 12)
                print(f'Shift key is adjusted to: {shift_key}')
                
            style = None
            style_label = ''
                
            mel_g = net_g(vq_post, f0, vol, spk_ave, spk_id, style, 
                          gt_spec=data['mel_44k'].permute(0, 2, 1).to(device), infer=True, infer_speedup=args.infer_speedup, method=args.infer_method, vocoder=vocoder)
            mel_g = mel_g.permute(0, 2, 1) #Diffusion output shape is [B, D, T] to [B, T, D]
            mel_g = mel_g.to(device)
            if args.vocoder == 'nsf-hifigan':
                wav_g = vocoder.infer(mel_g.permute(0, 2, 1), f0)
            else:
                wav_g = vocoder.infer(mel_g.permute(0, 2, 1))
            if output_dir is None:
                ckpt_steps = os.path.basename(model_path).split('_')[0]
                output_dir = os.path.join(os.path.dirname(model_path), f'output_{ckpt_steps}_step', style_label, 
                                f'{src_spk_name}_to_{tar_spk_name}_{cnt}_shot_{shift_key}_key/{now_time}')
                os.makedirs(output_dir, exist_ok=True) 
            output_path = os.path.join(output_dir, f'{file_name}.wav')
            sf.write(output_path, wav_g.squeeze().cpu().numpy(), 44100)
        save_config(args.config, output_dir)
        print(' Output dir:', output_dir)
        
def vc_infer_with_wav(net_g, vocoder, args):
    
    device = args.device
    # log_dir = os.path.join(args.ckpt_dir, "logs", get_time())
    model_path = args.model_path
    # args.log_dir = log_dir
    shift_key = args.shift_key if args.shift_key is not None else 0
    adjust_f0 = args.adjust_f0 if args.adjust_f0 is not None else False

    src_spk_name = os.path.split(args.src_path)[-1]
    tar_spk_name = os.path.split(args.tar_path)[-1]
    
    tar_path = args.tar_path
    src_path = args.src_path
    
    spk_shot = args.spk_shot # wav num of tar spk for reference
    demo_num = args.demo_num
    
    fa_encoder = fa_decoder = content_encoder = spk_encoder = None
    if args.content_encoder is None or args.content_encoder == 'FACodec':
        fa_encoder, fa_decoder = load_facodec(device)
    else:
        try:
            content_encoder = load_content_encoder(args, device)
            spk_encoder = feature_extractor, embedding_model, _ = build_model()
        except Exception as e:
            print(f'\n[Error] Failed to load encoder. Error: {e}')

    f0_extractor = load_f0_extractor(args)
    volume_extractor = load_volume_extractor(args)
    
    tar_files = get_file_list(tar_path)
    src_files = get_file_list(src_path)
    
    if spk_shot > 0:
        tar_files = tar_files[:spk_shot]
    if demo_num > 0:
        src_files = src_files[:demo_num]
    if len(tar_files) == 0 or len(src_files) == 0:
        print('\n[Error] Failed to load datat.')
    if spk_shot == 0:
        spk_shot = len(tar_files)
        
    spk_ave = None
    cnt = 0
    sr = args.sample_rate
    encoder_sr = args.encoder_sr
    with torch.no_grad(): 
         for tar_file in tqdm(tar_files, desc=f'Extracted speaker embedding of {tar_spk_name}:'):
            tar_data = get_processed_file(tar_file, sr, encoder_sr, vocoder, volume_extractor, f0_extractor, fa_encoder, fa_decoder, content_encoder, spk_encoder, device=device, f0_interpolate_mode='part')
            if tar_data is None:
                print(f'{tar_file} is skipped.')
                continue
            spk = tar_data['spk']
            if spk_ave is None:
                spk_ave = spk
            else:
                spk_ave += spk
                
            if adjust_f0:
                # f0 = tar_data['f0']
                f0 = tar_data['f0_origin']
                if cnt == 0:
                    tar_f0 = f0
                else:
                    tar_f0 = np.concatenate((tar_f0, f0), axis=0)
            cnt += 1
            if cnt >= spk_shot:
                break
    spk_ave /= cnt
    spk_ave = spk_ave.squeeze().to(device)
    now_time = get_time()
    ckpt_steps = os.path.basename(model_path).split('_')[0]
    with torch.no_grad(): 
        # text = None
        for src_file in tqdm(src_files, desc=f'Infering {src_spk_name} to {tar_spk_name}:'):
            src_data = get_processed_file(src_file, sr, encoder_sr, vocoder, volume_extractor, f0_extractor, fa_encoder, fa_decoder, content_encoder, spk_encoder, device=device, f0_interpolate_mode='part')
            vq_post = src_data['vq_post'].unsqueeze(0).to(device)
            file_name = src_data['name']
            vol = src_data['vol'].unsqueeze(0).to(device)
            f0 = src_data['f0'].unsqueeze(0).to(device)
            # src_f0 = src_data['f0'].numpy()
            src_f0 = src_data['f0_origin']
            
            if args.adjust_f0:
                shift_key = get_adjust_key_f0(src_f0, tar_f0, quantize=args.quantize_key)
                f0 = f0.to(device) * 2 ** (float(shift_key) / 12)
                print(f'Shift key is adjusted to: {shift_key}')
            
            elif args.shift_key is not None:
                f0 = f0.to(device) * 2 ** (float(shift_key) / 12)
                print(f'Shift key is adjusted to: {shift_key}')

            mel_g = net_g(vq_post, f0, vol, spk_ave, 
                          gt_spec=None, infer=True, infer_speedup=args.infer_speedup, method=args.infer_method, vocoder=vocoder)
            mel_g = mel_g.permute(0, 2, 1)
            mel_g = mel_g.to(device)
            
            if args.vocoder == 'nsf-hifigan':
                wav_g = vocoder.infer(mel_g.permute(0, 2, 1), f0)
            else:
                wav_g = vocoder.infer(mel_g.permute(0, 2, 1))
            
            output_dir = os.path.join(os.path.dirname(model_path), f'output_{ckpt_steps}_step', f'{src_spk_name}_to_{tar_spk_name}_{cnt}_shot_{shift_key}_key/{now_time}')
            
            os.makedirs(output_dir, exist_ok=True) 
            output_path = os.path.join(output_dir, f'{file_name}.wav')
            sf.write(output_path, wav_g.squeeze().cpu().numpy(), 44100)
            
        save_config(args.config, output_dir)

def main():    
    cmd = make_parse_cmd().parse_args()
    args = load_config(cmd.config)
    args.config = cmd.config
    print(' > config:', cmd.config)
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    device = args.device

    if args.vocoder == 'nsf-hifigan' or args.vocoder is None:
        vocoder = Vocoder(vocoder_type='nsf-hifigan', 
                        vocoder_ckpt='utils/pretrain/nsf_hifigan/model', 
                        device=device)
        
    elif args.vocoder == 'bigvgan':
        vocoder = Vocoder(vocoder_type='bigvgan', 
                        vocoder_ckpt='nvidia/bigvgan_v2_44khz_128band_512x',
                        device=device)

    model_path = args.model_path
    model_name = args.model_name
    if model_name == 'facodec':
        net_g = load_facodec_mel(mode='infer', model_path=model_path, device=device)
    elif args.model_name == 'v1.5.3a':
        net_g = load_generator_mel_diff_ddsp_v5a(mode='infer', device=device, model_path=model_path, args=args)
    elif args.model_name == 'v2.0.0':
        net_g = load_controlsvc_v1(mode='infer', device=device, model_path=model_path, args=args)
    else:
        raise ValueError(f" [x] Unknown Model Name: {model_name}")
    
    net_g.eval()
    mode = args.infer_mode
    if mode == 'vc':
        tar_loader, src_loader = load_style_audio_data(args=args, mode='vc', fast_data_load=False)
        vc_infer(net_g, vocoder, tar_loader, src_loader, args) # 参考音色音频可以是单个样本，或多个样本（取平均？）
    elif mode == 'vc_with_wav':
        vc_infer_with_wav(net_g, vocoder, args)
    else:
        raise ValueError(f" [x] Unknown Mode: {mode}")
        
if __name__ == "__main__":
    main()
# python infer_svc.py --config configs/hq_svc_infer.yaml --model_path ckpt/250000_step_val_loss_0.50.pth --src_path demo/singing/ --tar_path demo/speech/ --spk_shot 5 --adjust_f0 True --demo_num 3