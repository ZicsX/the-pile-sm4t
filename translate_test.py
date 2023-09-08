import torch
import concurrent.futures
from seamless_communication.models.inference import Translator
from datasets import load_dataset, Dataset

# Initialize a Translator object with a multitask model, vocoder on the GPU.
translator = Translator("seamlessM4T_large", "vocoder_36langs", torch.device("cuda:0"), torch.bfloat16)

def translate(text):
    """Translate English text to Hindi."""
    try:
        translated_text, _, _ = translator.predict(text, "t2tt", 'hin', src_lang='eng')
        return translated_text
    except Exception as e:
        print(f"An error occurred while translating: {e}")
        return None

# Load the dataset
dataset = load_dataset(
    path="./parquet_files/Enron_Emails",
    split='train[:1000]'
)

# Create an empty list to store translated data
translated_dataset = []

# Process translations in parallel using concurrent.futures.ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(translate, [datapoint['text'] for datapoint in dataset]))

# # Pair the translated text with its corresponding metadata and domain
# for i, (result, datapoint) in enumerate(zip(results, dataset)):
#     if result:
#         translated_dataset.append({
#             'text': result,
#             'meta': datapoint['meta'],
#             'domain': datapoint['domain']
#         })

#     if i % 100 == 0:
#         print(f"Processed {i} datapoints...")

# # Create a new Dataset from the translated data
# translated_dataset = Dataset.from_dict({
#     'text': [item['text'] for item in translated_dataset],
#     'meta': [item['meta'] for item in translated_dataset],
#     'domain': [item['domain'] for item in translated_dataset]
# })

# Filter out None results and update the translated_texts list
translated_texts = [result for result in results if result is not None]

# Create a new Dataset from the translated data
translated_dataset = Dataset.from_dict({'text': translated_texts})

# Save the dataset to disk
translated_dataset.save_to_disk("./translated_enron_emails")

print("Translation complete and dataset saved.")
