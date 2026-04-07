from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained("model")

print("Tokenizer saved successfully")