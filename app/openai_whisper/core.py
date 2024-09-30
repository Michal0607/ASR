import os
from io import StringIO, BytesIO
from threading import Lock
from typing import BinaryIO, Union
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    GenerationConfig,
    AutomaticSpeechRecognitionPipeline,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig
from safetensors.torch import safe_open
from whisper.utils import WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON, ResultWriter

# Ścieżka do Twojego dotrenowanego modelu
model_path = "Michal0607/Whisper-v2-tuned" 
processor_path = "openai/whisper-large-v2"

# Ładowanie konfiguracji PEFT
peft_config = PeftConfig.from_pretrained(model_path)
quant_config = BitsAndBytesConfig(load_in_8bit=True)

# Ładowanie bazowego modelu Whisper z parametrami autora
base_model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path,
    quantization_config=quant_config,            # Użycie 8-bitowej wersji modelu dla oszczędności pamięci
    device_map="auto"             # Automatyczne mapowanie urządzeń
)

# Ładowanie modelu PEFT
model = PeftModel.from_pretrained(base_model, model_path)

# Ładowanie procesora Whisper
processor = WhisperProcessor.from_pretrained(processor_path)

# Ustawienie urządzenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Blokada dla wątków
model_lock = Lock()

# Inicjalizacja pipeliny
pipe = AutomaticSpeechRecognitionPipeline(
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor
)

def transcribe(
        audio,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        vad_filter: Union[bool, None],
        word_timestamps: Union[bool, None],
        output,
):
    try:
        if language:
            pipe.model.config.language = language
        if task:
            pipe.model.config.task = task

        generation_kwargs = {}

        if initial_prompt:
            prompt_ids = processor.tokenizer.encode(initial_prompt, add_special_tokens=False)
            generation_kwargs["decoder_prompt_ids"] = prompt_ids

        if word_timestamps:
            generation_kwargs["return_timestamps"] = "word"
            generation_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(language=language, task=task)

        # Generowanie transkrypcji za pomocą pipeliny
        result = pipe(audio, **generation_kwargs)

        text = result.get('text', "")
        if word_timestamps:
            word_timestamps_list = result.get('chunks', [])

            result = {'text': text, 'segments': []}
            for i, word_info in enumerate(word_timestamps_list):
                start_time = word_info.get('timestamp', [0.0, 0.0])[0]
                end_time = word_info.get('timestamp', [0.0, 0.0])[1]
                word = word_info.get('text', "")
                result['segments'].append({
                    'id': i,
                    'seek': 0,
                    'start': start_time,
                    'end': end_time,
                    'text': word,
                })
        else:
            result = {
                'text': text,
                'segments': [{
                    'id': 0,
                    'seek': 0,
                    'start': 0.0,
                    'end': 0.0,
                    'text': text,
                }]
            }

        output_file = StringIO()
        write_result(result, output_file, output)
        output_file.seek(0)

        return output_file

    except Exception as e:
        # Logowanie błędu (możesz dostosować sposób logowania)
        print(f"Błąd podczas transkrypcji: {e}")
        return None

def language_detection(audio):
    try:
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with model_lock:
            predicted_ids = model.generate(input_features, attention_mask=attention_mask, max_length=1)
        detected_language = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

        return detected_language

    except Exception as e:
        print(f"Błąd podczas wykrywania języka: {e}")
        return None

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
