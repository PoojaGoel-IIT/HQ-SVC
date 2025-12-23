import os
import sys
import torch
import numpy as np
import gradio as gr
import soundfile as sf
import tempfile
import hashlib
import atexit
from importlib.metadata import version, PackageNotFoundError

# ================= 1. 增强型依赖追踪逻辑 =================
def save_used_dependencies():
    """
    导出当前运行环境下已加载的第三方库及其版本。
    """
    print("\n[System] 正在扫描内存中的依赖组件...")
    used_packages = set()
    
    # 映射表：导入名 -> PyPI 上的安装包名
    # 研三学生常用库映射
    mapping = {
        "yaml": "PyYAML",
        "cv2": "opencv-python",
        "sklearn": "scikit-learn",
        "skimage": "scikit-image",
        "faiss": "faiss-cpu",  # 或 faiss-gpu
        "gradio": "gradio",
        "torch": "torch",
        "numpy": "numpy",
        "soundfile": "soundfile",
        "librosa": "librosa",
        "scipy": "scipy"
    }

    # 遍历当前所有已加载模块
    for name in list(sys.modules.keys()):
        root_package = name.split('.')[0]
        
        # 排除内置模块
        if root_package in sys.builtin_module_names:
            continue
            
        final_name = mapping.get(root_package, root_package)
        used_packages.add(final_name)
    
    # 过滤掉本地文件夹模块和 pip 相关工具
    # 根据你的项目结构，排除 logger 和 utils
    excluded = {'pip', 'setuptools', 'wheel', 'pkg_resources', 'logger', 'utils', 'importlib'} 
    final_list = sorted(list(used_packages - excluded))

    output_path = 'used_requirements.txt'
    lines = []
    for pkg in final_list:
        try:
            # 获取版本号
            ver = version(pkg)
            lines.append(f"{pkg}=={ver}")
        except PackageNotFoundError:
            # 可能是本地库或者无法识别安装来源
            if pkg not in ['__main__', 'atexit', 'tempfile', 'hashlib']:
                lines.append(f"{pkg}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    msg = f"✨ 依赖清单已更新至: {os.path.abspath(output_path)}"
    print(msg)
    return msg

# 注册正常退出时的钩子
atexit.register(save_used_dependencies)

# ================= 2. 路径与模型加载逻辑 =================
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
utils_path = os.path.join(now_dir, 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

# 注意：这些导入需要确保你的目录结构正确
from logger.utils import load_config
from utils.models.models_v2_beta import load_hq_svc
from utils.vocoder import Vocoder
from utils.data_preprocessing import load_facodec, load_f0_extractor, load_volume_extractor, get_processed_file

# 全局变量缓存
NET_G = None
VOCODER = None
ARGS = None
PREPROCESSORS = {}
TARGET_CACHE = {"file_hash": None, "spk_ave": None, "all_tar_f0": None}

def initialize_models(config_path):
    global NET_G, VOCODER, ARGS, PREPROCESSORS
    ARGS = load_config(config_path)
    ARGS.config = config_path
    device = ARGS.device
    VOCODER = Vocoder(vocoder_type='nsf-hifigan', vocoder_ckpt='utils/pretrain/nsf_hifigan/model', device=device)
    NET_G = load_hq_svc(mode='infer', device=device, model_path=ARGS.model_path, args=ARGS)
    NET_G.eval()
    fa_encoder, fa_decoder = load_facodec(device)
    PREPROCESSORS = {
        "fa_encoder": fa_encoder, "fa_decoder": fa_decoder, 
        "f0_extractor": load_f0_extractor(ARGS), 
        "volume_extractor": load_volume_extractor(ARGS),
        "content_encoder": None, "spk_encoder": None
    }

# ================= 3. 推理逻辑 =================
def predict(source_audio, target_files, shift_key, adjust_f0):
    global TARGET_CACHE
    if source_audio is None: return "错误: 未检测到源音频", None
    sr, encoder_sr, device = ARGS.sample_rate, ARGS.encoder_sr, ARGS.device

    with torch.no_grad():
        is_reconstruction = (target_files is None or len(target_files) == 0)
        # 计算目标音频列表哈希以判断是否使用缓存
        current_hash = hashlib.md5("".join([f.name if hasattr(f, 'name') else f for f in (target_files or [])]).encode()).hexdigest()
        
        if is_reconstruction:
            t_data = get_processed_file(source_audio, sr, encoder_sr, VOCODER, PREPROCESSORS["volume_extractor"], PREPROCESSORS["f0_extractor"], PREPROCESSORS["fa_encoder"], PREPROCESSORS["fa_decoder"], None, None, device=device)
            spk_ave, all_tar_f0 = t_data['spk'].squeeze().to(device), t_data['f0_origin']
            status = "✨ 超分模式"
        elif TARGET_CACHE["file_hash"] == current_hash:
            spk_ave, all_tar_f0 = TARGET_CACHE["spk_ave"], TARGET_CACHE["all_tar_f0"]
            status = "🚀 缓存命中"
        else:
            spk_list, f0_list = [], []
            for f in target_files[:20]:
                t_data = get_processed_file(f.name if hasattr(f, 'name') else f, sr, encoder_sr, VOCODER, PREPROCESSORS["volume_extractor"], PREPROCESSORS["f0_extractor"], PREPROCESSORS["fa_encoder"], PREPROCESSORS["fa_decoder"], None, None, device=device)
                if t_data: spk_list.append(t_data['spk']); f0_list.append(t_data['f0_origin'])
            spk_ave = torch.stack(spk_list).mean(dim=0).squeeze().to(device)
            all_tar_f0 = np.concatenate(f0_list)
            TARGET_CACHE.update({"file_hash": current_hash, "spk_ave": spk_ave, "all_tar_f0": all_tar_f0})
            status = "✅ 音色提取完成"

        src_data = get_processed_file(source_audio, sr, encoder_sr, VOCODER, PREPROCESSORS["volume_extractor"], PREPROCESSORS["f0_extractor"], PREPROCESSORS["fa_encoder"], PREPROCESSORS["fa_decoder"], None, None, device=device)
        f0 = src_data['f0'].unsqueeze(0).to(device)
        
        if adjust_f0 and not is_reconstruction:
            shift_key = round(12 * np.log2(all_tar_f0[all_tar_f0>0].mean()/src_data['f0_origin'][src_data['f0_origin']>0].mean()))
        
        f0 = f0 * 2 ** (float(shift_key) / 12)
        mel_g = NET_G(src_data['vq_post'].unsqueeze(0).to(device), f0, src_data['vol'].unsqueeze(0).to(device), spk_ave, gt_spec=None, infer=True, infer_speedup=ARGS.infer_speedup, method=ARGS.infer_method, vocoder=VOCODER)
        wav_g = VOCODER.infer(mel_g, f0) if ARGS.vocoder == 'nsf-hifigan' else VOCODER.infer(mel_g)
        
        out_p = tempfile.mktemp(suffix=".wav")
        sf.write(out_p, wav_g.squeeze().cpu().numpy(), 44100)
        return f"{status} | 变调: {shift_key}", out_p

# ================= 4. UI 界面 =================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
:root { --font: 'Press Start 2P', cursive !important; }
* { font-family: 'Press Start 2P', cursive !important; border-radius: 0px !important; }
.gradio-container {
    background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)),
                url('https://img.moegirl.org.cn/common/d/d3/K-ON_key_visual_2.jpg');
    background-size: cover;
}
.gr-box, .gr-input, .gr-button { border: 4px solid #000 !important; box-shadow: 8px 8px 0px #000 !important; }
label, p, .time-info { color: #f36c18 !important; font-size: 10px !important; text-transform: uppercase; }
h1 { color: #FFFF00 !important; text-shadow: 4px 4px 0px #000 !important; text-align: center; }
button.primary { background-color: #ff69b4 !important; color: #fff !important; }
footer { display: none !important; }
"""

def build_ui():
    with gr.Blocks(css=custom_css, title="HQ-SVC Pixel Pro") as demo:
        gr.HTML("""
            <div style="display: flex; justify-content: center; margin: 20px 0;">
                <div style="border: 4px solid #000; box-shadow: 8px 8px 0px #000;">
                    <img src="file/images/kon-new.gif" style="max-width: 400px; width: 100%;">
                </div>
            </div>
        """)
        gr.Markdown("# 🎸 HQ-SVC: SINGING VOICE CONVERSION 🍰")
        
        with gr.Row():
            with gr.Column():
                src_audio = gr.Audio(label="STEP 1: SOURCE VOICE", type="filepath")
                tar_files = gr.File(label="STEP 2: TARGET REFERENCE", file_count="multiple")
                with gr.Row():
                    key_shift = gr.Number(label="PITCH SHIFT", value=0)
                    auto_f0 = gr.Checkbox(label="AUTO PITCH", value=False)
                run_btn = gr.Button("🎤 START CONVERSION!", variant="primary")
            
            with gr.Column():
                status_box = gr.Textbox(label="SYSTEM TERMINAL", interactive=False)
                result_audio = gr.Audio(label="OUTPUT (44.1kHz HQ)")

        run_btn.click(predict, [src_audio, tar_files, key_shift, auto_f0], [status_box, result_audio])
        
        # 底部管理按钮
        with gr.Row():
            export_btn = gr.Button("📦 导出依赖清单", variant="secondary")
            exit_btn = gr.Button("🚫 关闭系统", variant="stop")

        # 逻辑绑定
        export_btn.click(fn=save_used_dependencies, inputs=None, outputs=status_box)
        
        def safe_exit():
            save_used_dependencies()
            print("系统正在关闭...")
            sys.exit(0) # 触发 atexit 钩子

        exit_btn.click(fn=safe_exit, inputs=None, outputs=None)

    return demo

if __name__ == "__main__":
    # 确保配置文件路径正确
    config_p = "configs/hq_svc_infer.yaml"
    if os.path.exists(config_p):
        initialize_models(config_p)
    else:
        print(f"警告: 找不到配置文件 {config_p}，请检查路径。")
        
    demo = build_ui()
    
    print(">>> 界面启动成功。")
    print(">>> 提示：请进行至少一次转换推理，让系统加载动态依赖。")
    
    # allowed_paths 允许访问图片文件夹
    demo.launch(share=True, allowed_paths=[os.path.join(now_dir, "images")])