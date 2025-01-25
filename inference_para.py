# @ hwang258@jh.edu

# TODO figure out dependencies in inference_scale.py, edit_utils_en.py, models/ssr.py, data/tokenizer.py
# TODO Get to a minimal implementation
# TODO remove unnecessary imports
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["USER"] = "root" # TODO change this to your username

import shutil
import torch
import torchaudio
import numpy as np
import random
from argparse import Namespace
from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
from edit_utils_en import parse_edit_en
from inference_scale import inference_one_sample
import time
from tqdm import tqdm
import argparse
from models import ssr
import re
from num2words import num2words
import uuid
import json
import pathlib


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

def replace_numbers_with_words(sentence):
    sentence = re.sub(r'(\d+)', r' \1 ', sentence) # add spaces around numbers
    def replace_with_words(match):
        num = match.group(0)
        try:
            return num2words(num) # Convert numbers to words
        except:
            return num # In case num2words fails (unlikely with digits but just to be safe)
    return re.sub(r'\b\d+\b', replace_with_words, sentence) # Regular expression that matches numbers

def get_transcribe_state(segments):
    transcript = " ".join([segment["text"] for segment in segments])
    transcript = transcript[1:] if transcript[0] == " " else transcript
    return {
        "segments": segments,
        "transcript": transcript,
    }

def get_random_string():
    return "".join(str(uuid.uuid4()).split("-"))

def get_mask_interval(transcribe_state, word_span):
    seg_num = len(transcribe_state['segments'])
    data = []
    for i in range(seg_num):
        words = transcribe_state['segments'][i]['words']
        for item in words:
          data.append([item['start'], item['end'], item['word']])

    s, e = word_span[0], word_span[1]
    assert s <= e, f"s:{s}, e:{e}"
    assert s >= 0, f"s:{s}"
    assert e <= len(data), f"e:{e}"
    if e == 0: # start
        start = 0.
        end = float(data[0][0])
    elif s == len(data): # end
        start = float(data[-1][1])
        end = float(data[-1][1]) # don't know the end yet
    elif s == e: # insert
        start = float(data[s-1][1])
        end = float(data[s][0])
    else:
        start = float(data[s-1][1]) if s > 0 else float(data[s][0])
        end = float(data[e][0]) if e < len(data) else float(data[-1][1])

    return (start, end)

def parse_args():
    base_dir = pathlib.Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(description="inference speech editing")
    parser.add_argument("--sub_amount", type=float, default=0.12, help="if the performance is not good, try modify this span, not used for tts")
    parser.add_argument('--codec_audio_sr', type=int, default=16000)
    parser.add_argument('--codec_sr', type=int, default=50)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--temperature', type=int, default=1)
    parser.add_argument('--kvcache', type=int, default=1)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--stop_repetition', type=int, default=2, help="-1 means do not adjust prob of silence tokens. if there are long silence or unnaturally strecthed words, increase sample_batch_size to 2, 3 or even 4.")
    parser.add_argument('--sample_batch_size', type=int, default=1, help="what this will do to the model is that the model will run sample_batch_size examples of the same audio")
    parser.add_argument('--cfg_coef', type=float, default=1.5)
    parser.add_argument('--cfg_stride', type=int, default=5)
    parser.add_argument('--aug_text', action='store_true')
    parser.add_argument('--aug_context', action='store_true')
    parser.add_argument('--prompt_length', type=int, default=3, help='used for tts prompt, will automatically cut the prompt audio to this length')
    parser.add_argument('--model_path', type=str, default=f"{base_dir}/pretrained_models/English.pth")
    parser.add_argument('--codec_path', type=str, default=f"{base_dir}/pretrained_models/wmencodec.th")
    parser.add_argument('--orig_audio', type=str, default=None)
    parser.add_argument('--segments', type=str, required=True, help="JSON file containing word-level segments with timing information")
    parser.add_argument('--target_transcript', type=str, default=None)
    parser.add_argument('--temp_folder', type=str, default="/tmp")
    parser.add_argument('--output_dir', type=str, default=".")
    parser.add_argument('--savename', type=str, default=None)
    parser.add_argument('--starting_interval', type=float, required=True)
    parser.add_argument('--ending_interval', type=float, required=True)

    return parser.parse_args()

def main(args):
    seed_everything(args.seed)
        
    # Initialize models
    filepath = os.path.join(args.model_path)
    ckpt = torch.load(filepath, map_location="cpu")
    model = ssr.SSR_Speech(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    config = vars(model.args)
    phn2num = ckpt["phn2num"]
    model.to(device)
    model.eval()
    audio_tokenizer = AudioTokenizer(signature=args.codec_path)
    text_tokenizer = TextTokenizer(backend="espeak")
        
    start_time = time.time()
    # move the audio and transcript to temp folder
    os.makedirs(args.temp_folder, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.system(f"cp {args.orig_audio} {args.temp_folder}")
    filename = os.path.splitext(args.orig_audio.split("/")[-1])[0]
    audio_fn = f"{args.temp_folder}/{filename}.wav"

    segments = json.loads(args.segments)
    
    transcribe_state = get_transcribe_state(segments)
    orig_transcript = transcribe_state['transcript'].lower()
    target_transcript = args.target_transcript.lower()
        
    starting_intervals = [args.starting_interval]
    ending_intervals = [args.ending_interval]

    print("intervals: ", starting_intervals, ending_intervals)

    info = torchaudio.info(audio_fn)
    audio_dur = info.num_frames / info.sample_rate
    
    def combine_spans(spans, threshold=0.2):
        spans.sort(key=lambda x: x[0])
        combined_spans = []
        current_span = spans[0]

        for i in range(1, len(spans)):
            next_span = spans[i]
            if current_span[1] >= next_span[0] - threshold:
                current_span[1] = max(current_span[1], next_span[1])
            else:
                combined_spans.append(current_span)
                current_span = next_span
        combined_spans.append(current_span)
        return combined_spans
    
    morphed_span = [[max(start - args.sub_amount, 0), min(end + args.sub_amount, audio_dur)]
                    for start, end in zip(starting_intervals, ending_intervals)] # in seconds
    morphed_span = combine_spans(morphed_span, threshold=0.2)
    print("morphed_spans: ", morphed_span)
    save_morphed_span = f"{args.output_dir}/{args.savename}_mask.pt"
    torch.save(morphed_span, save_morphed_span)
    mask_interval = [[round(span[0]*args.codec_sr), round(span[1]*args.codec_sr)] for span in morphed_span]
    mask_interval = torch.LongTensor(mask_interval) # [M,2], M==1 for now

    decode_config = {'top_k': args.top_k, 'top_p': args.top_p, 'temperature': args.temperature, 'stop_repetition': args.stop_repetition, 'kvcache': args.kvcache, "codec_audio_sr": args.codec_audio_sr, "codec_sr": args.codec_sr}

    elapsed_time = time.time() - start_time

    print(f"Preparation time: {elapsed_time:.4f} s")

    for num in tqdm(range(args.sample_batch_size)):
        seed_everything(args.seed+num)
        new_audio = inference_one_sample(model, Namespace(**config), phn2num, text_tokenizer, audio_tokenizer, audio_fn, orig_transcript, target_transcript, mask_interval, args.cfg_coef, args.cfg_stride, args.aug_text, args.aug_context, device, decode_config)
        new_audio = new_audio[0].cpu()
        save_fn_new = f"{args.output_dir}/{args.savename}_new_seed{args.seed+num}.wav"
        torchaudio.save(save_fn_new, new_audio, args.codec_audio_sr)

    save_fn_orig = f"{args.output_dir}/{args.savename}_orig.wav"
    shutil.copyfile(audio_fn, save_fn_orig)
        
    elapsed_time = time.time() - start_time
    
    print(f"Running time: {elapsed_time:.4f} s")

if __name__ == "__main__":
    args = parse_args()
    main(args)
