from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('cnn_dailymail', '3.0.0')