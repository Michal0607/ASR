import os
from io import StringIO, BytesIO
from threading import Lock
from typing import BinaryIO, Union
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel, PeftConfig
from safetensors.torch import safe_open
from whisper.utils import WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON, ResultWriter

# Ścieżki do Twojego dotrenowanego modelu oraz procesora
model_path = "./100_/"  # Zmień tę ścieżkę na rzeczywistą
processor_path = "openai/whisper-large-v2"

# Załaduj PeftConfig
peft_config = PeftConfig.from_pretrained(model_path)

# Załaduj bazowy model Whisper
base_model = WhisperForConditionalGeneration.from_pretrained(processor_path)

# Załaduj wagi LoRA na bazowy model z PeftConfig
model = PeftModel(base_model, peft_config)
print(model.peft_config)

# Załaduj procesor
processor = WhisperProcessor.from_pretrained(processor_path)

# Przenieś model na GPU, jeśli jest dostępne
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
    # Argumenty generowania
    generation_kwargs = {}

    # Ustawienie języka i zadania
    if language:
        model.config.language = language
    if task:
        model.config.task = task

    if initial_prompt:
        generation_kwargs["prompt"] = initial_prompt

    # Zawsze ustawiamy return_dict_in_generate na True
    generation_kwargs["return_dict_in_generate"] = True

    # Ustawienia dla znaczników czasowych słów
    if word_timestamps:
        generation_kwargs["return_timestamps"] = "word"
        generation_kwargs["use_cache"] = False  # Zalecane dla dokładniejszych znaczników
    else:
        # Domyślne ustawienia
        pass

    # Przetworzenie audio na tensor przy użyciu procesora
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    # Generowanie transkrypcji przy użyciu modelu
    with model_lock:
        outputs = model.generate(input_features, **generation_kwargs)

    if word_timestamps:
        # Sprawdzenie, czy 'timestamps' istnieje w outputs
        if hasattr(outputs, 'timestamps') and outputs.timestamps is not None:
            # Uzyskanie transkrypcji
            transcription = processor.decode(outputs.sequences[0], skip_special_tokens=True)
            word_offsets_list = outputs.timestamps[0]  # Zakładamy batch_size=1

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
            # Jeśli timestamps nie są dostępne
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
        # Dekodowanie bez znaczników czasowych
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

    # Zapis wyników do pliku
    output_file = StringIO()
    write_result(result, output_file, output)
    output_file.seek(0)

    return output_file

def language_detection(audio):
    # Przetwarzanie audio przy użyciu procesora
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    # Generowanie prognozy języka
    with model_lock:
        predicted_ids = model.generate(input_features, max_length=1)
    predicted_tokens = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    detected_language = predicted_tokens[0]

    return detected_language

# Funkcja do zapisu wyników
def write_result(result, file: BinaryIO, output: Union[str, None]):
    options = {
        'max_line_width': 1000,
        'max_line_count': 10,
        'highlight_words': False
    }

    # Formatowanie i zapisywanie wyników w zależności od formatu wyjściowego
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
        # Domyślnie zapisuje zwykły tekst, jeśli nie wybrano poprawnego formatu
        file.write(result['text'])