import random
import logging

from jiwer import compute_measures, wer

import evaluate

from dataset_src.text_normalizer.preprocess_text import preprocess_text_asr

# Ten instructions, random select one for each sample
st_instructions = [
    "Listen to the speech clip and translate it into English.",
    "Play the speech recording and translate it to English.",
    "Hear the audio clip and convert it to English.",
    "Listen to the speech and translate it into English.",
    "Play the recorded speech and provide a translation in English.",
    "Hear the speech audio and translate it into English.",
    "Listen to the speech audio and translate it to English.",
    "Play the speech clip and translate it into English.",
    "Listen to the speech recording and translate it into English.",
    "Hear the audio clip and translate it to English.",
    "Listen to the speech and provide a translation in English.",
    "Play the audio of the speech and translate it into English.",
    "Hear the speech and convert it to English.",
    "Listen to the speech audio clip and translate it to English.",
    "Play the speech recording and translate it into English.",
    "Listen to the speech clip and provide a translation in English.",
    "Hear the recorded speech and translate it to English.",
    "Play the audio speech and translate it into English.",
    "Listen to the speech and translate it to English.",
    "Hear the audio of the speech and translate it into English."
]

class fleurs_ko_en_test_dataset(object):

    def __init__(self, raw_data, raw_data_en, raw_data_ko, number_of_samples):

        if number_of_samples != -1:
            #raw_data = raw_data.shuffle(seed=42)
            raw_data = raw_data.select(range(number_of_samples))
            raw_data_en = raw_data_en.select(range(number_of_samples))
            raw_data_ko = raw_data_ko.select(range(number_of_samples))
        
        self.raw_data = raw_data
        self.raw_data_en = raw_data_en
        self.raw_data_ko = raw_data_ko

        self.prompt   = st_instructions
        logging.info('Number of samples: {}'.format(len(self.raw_data)))


    def prepare_model_input(self):

        input_data = []
        for sample_en, sample_ko in zip( self.raw_data_en, self.raw_data_ko):
            audio       = sample_ko['audio']
            reference   = sample_en['transcription']
            instruction = random.choice(self.prompt)

            input_data.append({
                                "audio"      : audio,
                                "instruction": instruction,
                                "reference"  : reference,
                                "task_type"  : "ST-KO-EN" # ST-EN-ZH to ST-KO-EN changed
                                })

        logging.info('\n=  =  =  Dataset Sample  =  =  =')
        logging.info(random.sample(input_data, 1)[0])
        logging.info('=  =  =  =  =  =  =  =  =  =  =  =\n')

        return input_data


    def format_model_predictions(self, input_data, model_predictions):

        data_with_model_predictions = []
        for sample in input_data:
            new_sample = sample.copy()
            del new_sample["audio"]
            new_sample['model_prediction'] = model_predictions.pop(0)
            data_with_model_predictions.append(new_sample)
        return data_with_model_predictions


    def compute_score(self, data_with_model_predictions, metrics=None):

        if metrics != 'bleu':
            raise ValueError(f"Unsupported metric: {metrics}. Supported metrics: 'bleu' for ST")
       
        predictions = []
        references  = []
        for item in data_with_model_predictions:
            # model_prediction = preprocess_text_asr(item["model_prediction"])
            # answer           = preprocess_text_asr(item["answer"])
            model_prediction = item["model_prediction"]
            answer           = item["reference"]

            if len(model_prediction) == 0: model_prediction = "empty"
            if len(answer) == 0: answer = "empty"

            predictions.append(model_prediction)
            references.append(answer)

        sacrebleu = evaluate.load("sacrebleu")
        # Updated to flores101 tokenizer (Thanks Chenyang Lv)
        # results = sacrebleu.compute(predictions=predictions, references=references, tokenize='13a')
        results = sacrebleu.compute(predictions=predictions, references=references, tokenize='flores101')

        return {"bleu": results['score']}