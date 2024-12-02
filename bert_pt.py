from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode text
text = "Here is an example sentence."
encoded_input = tokenizer(text, return_tensors='pt')

# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

# Perform inference
with torch.no_grad():
    output = model(**encoded_input)

# Extract the embeddings
embeddings = output.last_hidden_state

print(embeddings)