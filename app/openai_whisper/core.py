import os
from io import StringIO, BytesIO
from threading import Lock
from typing import BinaryIO, Union
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel, PeftConfig
from safetensors.torch import safe_open
from whisper.utils import WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON, ResultWriter

model_path = "./100_/" 
processor_path = "openai/whisper-large-v2"

peft_config = PeftConfig.from_pretrained(model_path)

base_model = WhisperForConditionalGeneration.from_pretrained(processor_path)

model = PeftModel(base_model, peft_config)
print(model.peft_config)

processor = WhisperProcessor.from_pretrained(processor_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model_lock = Lock()

def transcribe(
        audio,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        vad_filter: Union[bool, None],
        word_timestamps: Union[bool, None],
        output,
):
    generation_kwargs = {}

    if language:
        model.config.language = language
    if task:
        model.config.task = task

    if initial_prompt:
        generation_kwargs["prompt"] = initial_prompt

    generation_kwargs["return_dict_in_generate"] = True

    if word_timestamps:
        generation_kwargs["return_timestamps"] = "word"
        generation_kwargs["use_cache"] = False 
    else:
        pass

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with model_lock:
        outputs = model.generate(input_features, **generation_kwargs)

    if word_timestamps:
        if hasattr(outputs, 'timestamps') and outputs.timestamps is not None:
            transcription = processor.decode(outputs.sequences[0], skip_special_tokens=True)
            word_offsets_list = outputs.timestamps[0] 

            result = {'text': transcription, 'segments': []}
            for i, word_info in enumerate(word_offsets_list):
                start_time = word_info['start']
                end_time = word_info['end']
                word = word_info['word']
                result['segments'].append({
                    'id': i,
                    'seek': 0,
                    'start': start_time,
                    'end': end_time,
                    'text': word,
                    'tokens': [],
                    'temperature': 0.0,
                    'avg_logprob': 0.0,
                    'compression_ratio': 0.0,
                    'no_speech_prob': 0.0,
                })
        else:
            print("Nie udało się uzyskać znaczników czasowych słów.")
            transcription = processor.decode(outputs.sequences[0], skip_special_tokens=True)
            result = {
                'text': transcription,
                'segments': [{
                    'id': 0,
                    'seek': 0,
                    'start': 0.0,
                    'end': 0.0,
                    'text': transcription,
                    'tokens': [],
                    'temperature': 0.0,
                    'avg_logprob': 0.0,
                    'compression_ratio': 0.0,
                    'no_speech_prob': 0.0,
                }]
            }
    else:
        transcription = processor.decode(outputs.sequences[0], skip_special_tokens=True)
        result = {
            'text': transcription,
            'segments': [{
                'id': 0,
                'seek': 0,
                'start': 0.0,
                'end': 0.0,
                'text': transcription,
                'tokens': [],
                'temperature': 0.0,
                'avg_logprob': 0.0,
                'compression_ratio': 0.0,
                'no_speech_prob': 0.0,
            }]
        }

    output_file = StringIO()
    write_result(result, output_file, output)
    output_file.seek(0)

    return output_file

def language_detection(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with model_lock:
        predicted_ids = model.generate(input_features, max_length=1)
    predicted_tokens = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    detected_language = predicted_tokens[0]

    return detected_language

def write_result(result, file: BinaryIO, output: Union[str, None]):
    options = {
        'max_line_width': 1000,
        'max_line_count': 10,
        'highlight_words': False
    }

    if output == "srt":
        WriteSRT(ResultWriter).write_result(result, file=file, options=options)
    elif output == "vtt":
        WriteVTT(ResultWriter).write_result(result, file=file, options=options)
    elif output == "tsv":
        WriteTSV(ResultWriter).write_result(result, file=file, options=options)
    elif output == "json":
        WriteJSON(ResultWriter).write_result(result, file=file, options=options)
    elif output == "txt":
        WriteTXT(ResultWriter).write_result(result, file=file, options=options)
    else:
        file.write(result['text'])