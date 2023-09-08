import time
import torch
import gc
import pandas as pd
from seamless_communication.models.inference import Translator
from datasets import load_dataset
from prettytable import PrettyTable

MAX_SEQ_LEN = 1024  # Define the maximum sequence length

def run_benchmark(model_name, dtype, dataset):
    try:
        translator = Translator(model_name, "vocoder_36langs", torch.device("cuda"), dtype)
        print("LOADED")
    except Exception as e:
        print(f"Failed to initialize the model: {e}")
        return None, None

    elapsed_times = []
    throughputs = []
    translated_data = {'Original': [], 'Translated': []}

    num_samples = 0
    for item in iter(dataset):
        if num_samples >= 100: break
        text = item['text']
        
        # Truncate text if it exceeds the maximum sequence length
        text = text[:MAX_SEQ_LEN]

        start_time = time.time()
        result = translator.predict(text, "t2tt", 'hin', src_lang='eng')[0]
        end_time = time.time()

        elapsed_time = end_time - start_time
        throughput = 1 / elapsed_time

        elapsed_times.append(elapsed_time)
        throughputs.append(throughput)
        translated_data['Original'].append(text)
        translated_data['Translated'].append(result)
        num_samples += 1

        # Clear GPU and RAM immediately
        del result
        torch.cuda.empty_cache()

    # Create DataFrame and save to CSV
    df = pd.DataFrame(translated_data)
    csv_filename = f"translated_results_{model_name}_{dtype}.csv"
    df.to_csv(csv_filename, index=False)

    avg_elapsed_time = sum(elapsed_times) / num_samples
    avg_throughput = sum(throughputs) / num_samples

    # Clear GPU and RAM
    del translator, translated_data, df
    gc.collect()
    torch.cuda.empty_cache()
    return avg_elapsed_time, avg_throughput

# Load the dataset as a stream
dataset = load_dataset(path="parquet_files/Enron_Emails", split='train', streaming=True)

# Table to hold benchmark results
table = PrettyTable()
table.field_names = ["Model", "Data Type", "Avg Time (s)", "Avg Throughput (datapoints/s)"]

model_list = ["seamlessM4T_large", "seamlessM4T_medium"]
dtype_list = [
    torch.float32,
    torch.float16,
    torch.bfloat16
]

for model in model_list:
    for dtype in dtype_list:
        avg_time, avg_throughput = run_benchmark(model, dtype, dataset)
        if avg_time is not None and avg_throughput is not None:
            table.add_row([model, dtype, avg_time, avg_throughput])
            gc.collect()
            torch.cuda.empty_cache()

# Save the benchmark results to a txt file
with open("benchmark_results.txt", "w") as f:
    f.write(str(table))

# Also print the table
print(table)
