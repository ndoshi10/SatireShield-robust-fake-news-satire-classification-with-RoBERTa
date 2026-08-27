def predict_text(text):
    model.eval()
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=1).item()
    return label_encoder.inverse_transform([pred])[0]

# Example
print(predict_text("The economy is growing rapidly in the last quarter."))