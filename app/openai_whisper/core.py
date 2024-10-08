import os
from io import StringIO, BytesIO
from threading import Lock
from typing import BinaryIO, Union, List, Dict
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from whisper.utils import WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON, ResultWriter

model_path = "Michal0607/Whisper-v2-tuned" 
processor_path = "openai/whisper-large-v2"

peft_config = PeftConfig.from_pretrained(model_path)
quant_config = BitsAndBytesConfig(load_in_8bit=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    base_model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=quant_config
    )
else:
    base_model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path
    )

model = PeftModel.from_pretrained(base_model, model_path)

processor = WhisperProcessor.from_pretrained(processor_path)

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
    if language:
        model.config.language = language
    if task:
        model.config.task = task

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )

    generation_kwargs = {}

    if word_timestamps:
        generation_kwargs["return_timestamps"] = "word"
        generation_kwargs["chunk_length_s"] = 30.0  

    asr_result = asr_pipeline(
        audio,
        **generation_kwargs
    )

    text = asr_result['text']
    segments = []
    
    if word_timestamps:
        word_timestamps_list = asr_result.get('chunks', [])
        current_segment = {
            'id': 0,
            'seek': 0,
            'start': word_timestamps_list[0]['timestamp'][0],
            'end': word_timestamps_list[0]['timestamp'][1],
            'text': '',
            'tokens': [],
            'no_speech_prob': 0.0, 
            'words': []
        }

        for i, word_info in enumerate(word_timestamps_list):
            word = word_info['text']
            start_time = word_info['timestamp'][0]
            end_time = word_info['timestamp'][1]

            if start_time == end_time:
                continue
            
            if end_time - current_segment['start'] > 5.0:  
                current_segment['tokens'] = processor.tokenizer.encode(current_segment['text'].strip())
                segments.append(current_segment)
                current_segment = {
                    'id': current_segment['id'] + 1,
                    'seek': 0,
                    'start': start_time,
                    'end': end_time,
                    'text': '',
                    'tokens': [],
                    'no_speech_prob': 0.0,
                    'words': []
                }

            current_segment['text'] += word + ' '
            current_segment['end'] = end_time
            current_segment['words'].append({
                'word': word,
                'start': start_time,
                'end': end_time,
                'probability': 0.0
            })

        if current_segment['text'].strip():
            current_segment['tokens'] = processor.tokenizer.encode(current_segment['text'].strip())
        segments.append(current_segment)
    
    else:
        segments = [{
            'id': 0,
            'seek': 0,
            'start': 0.0,
            'end': 0.0,
            'text': text,
            'no_speech_prob': 0.0,
        }]

    result = {
        'text': text,
        'segments': segments,
        'language': language if language else "pl"
    }

    output_file = StringIO()
    write_result(result, output_file, output)
    output_file.seek(0)

    return output_file

def language_detection(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with model_lock:
        predicted_ids = model.generate(input_features=input_features, max_length=1)
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
