import numpy as np
import os
import pydub
from scipy.io import wavfile
from tqdm.auto import tqdm
from typing import List

from tts.utils.sequence_utils.ml_text import symbols
from tts.utils.synthesis_utils import supported_model_names
from tts.utils.synthesis_utils import vits, get_vits_synthesizer


_xinapse_tts_path = '/'.join(__file__.split('/')[:-1])
_default_output_dir = '/'.join(__file__.split('/')[:-2])
_default_output_dir = os.path.join(_default_output_dir, "audio")


def _get_vits_model():
    vits_info = {
        supported_model_names[0]: {
            'model_path': os.path.join(_xinapse_tts_path, "xinapse_tts/model_weights/vits-ailab_joodw_22k/G_449000.pth"),
            'hps_path': os.path.join(_xinapse_tts_path, "xinapse_tts/configs/ailab_joodw_22k.json"),
            'speaker_ids_path': os.path.join(_xinapse_tts_path, "xinapse_tts/configs/total_speaker_ids_ailab_joodw_normal.json"),
            'symbols': symbols, 
            'device': 0, 
            'use_cpu': True, # always use GPU
            'use_fine_tuned': False, 
            'use_cross_tokenizer': False, 
        }, 
        # supported_model_names[1]: {
        # }
    }
    return get_vits_synthesizer(vits, vits_info)

def _get_nix_model():
    # for future lightweight model inference
    raise NotImplementedError


def get_model(model_name:str="vits"):
    assert model_name in ("vits", ), "{model_name} is not implemented.".format(model_name=model_name)

    if model_name == "vits":
        model = _get_vits_model()
        model_type_names = supported_model_names # == ["ailab_22k"]
    else: # TODO: if model_name == "nix":
        model = None
        model_type_names = None
    
    return model


def load_wav_scipy(full_path):
    sr, wav = wavfile.read(full_path)
    return sr, wav 

def save_wav_scipy(wav, sr, path):
    wavfile.write(path, sr, wav.astype(np.int16))

def save_mp3_pydub(wav, sr, path):
    # original code from https://stackoverflow.com/a/66191603
    channels = 2 if (wav.ndim == 2 and wav.shape[1] == 2) else 1
    song = pydub.AudioSegment(np.int16(wav).tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(path, format="mp3", bitrate="320k")


def save_audio_to_file(output_dir, audio, sampling_rate, text, audio_ftype="wav"):
    # 현재 파일명 : 스크립트 내용
    fname = text.replace('.', '_').replace(',', '_').replace('?', '_').replace('!', '_').replace(' ', '_')
    
    # 파일 저장
    if audio_ftype == "mp3":
        save_wav_scipy(audio, sampling_rate, os.path.join(output_dir, "temp.wav"))
        wav = load_wav_scipy(os.path.join(output_dir, "temp.wav"))
        save_mp3_pydub(wav, sampling_rate, os.path.join(output_dir, fname+".mp3"))
    elif audio_ftype == "wav":
        save_wav_scipy(audio, sampling_rate, os.path.join(output_dir, fname+".wav"))
    else:
        raise NotImplementedError("audio_ftype must be either mp3 or wav, not {audio_ftype}.".format(audio_ftype=audio_ftype))


def inference(model, texts:List[str], speaker_name:str="ailab_joodw-neutral", output_dir:str=_default_output_dir):
    """목소리를 생성하는 코드입니다.

    - texts에 들어온 모든 문장을 하나의 wav 파일로 합쳐서 출력합니다.
    - 현재 speaker_name은 고정입니다.

    예시)
        texts = [
            "안녕하세요, 여러분?",
            "반갑습니다. 좋은 하루 되세요."
        ]
        output_dir = "audio/name/"

        model = get_model()
        inference(model, texts, output_dir=output_dir)
    
    예시2) 
        texts = [
            "안녕하세요,",
            "~님, ~님, ~님 * N",
            "반갑습니다. 좋은 하루 되세요."
        ]
        output_dir = "audio/name/"

        model = get_model()
        inference(model, texts, output_dir=output_dir)
    """
    assert speaker_name == "ailab_joodw-neutral", "speaker_name must be ailab_joodw-neutral"
    
    pairs = list()
    for model_type_name in supported_model_names:
        for text in texts:
            pairs.append((model_type_name, text))

    sec = 0.5 # 초단위로 공백을 입력하세요.
    T = int(sec * model.models[model_type_name]['hps'].data.sampling_rate)
    pad = np.zeros(T)

    audio = np.array([0])
    text_all = ''
    for idx, (model_type_name, text) in tqdm(enumerate(pairs), total=len(pairs)):
        audio = np.concatenate(
            [audio, model.synthesis(model_type_name, speaker_name, text), pad], axis=0)
        text_all += ' ' + text
    audio = audio[:-T]
    text_all = text_all.strip()
    
    save_audio_to_file(
        output_dir, 
        audio * model.models[model_type_name]['hps'].data.max_wav_value, 
        model.models[model_type_name]['hps'].data.sampling_rate,
        text_all, 
        audio_ftype="wav"
    )
    
    # 임시파일 제거
    if os.path.exists(os.path.join(output_dir, "temp.wav")):
        os.remove(os.path.join(output_dir, "temp.wav"))



if __name__ == '__main__':
    texts = [
              "반갑습니다"
	    ]
    model = get_model()
    inference(model, texts, output_dir = 'audio/name/')

    
