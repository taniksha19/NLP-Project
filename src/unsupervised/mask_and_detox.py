import json
import re

# Load toxic word list
with open("/content/toxic_words_by_language.json", "r") as f:
    toxic_dict = json.load(f)

# For now, only English
english_toxic_words = set([w.lower() for w in toxic_dict.get("en", [])])

print("Loaded English toxic words:", len(english_toxic_words))

def mask_toxic_words(text, toxic_words):
    tokens = text.split()
    masked_tokens = []

    for tok in tokens:
        # lowercase version without punctuation
        clean = re.sub(r"[^\w']", "", tok.lower())

        if clean in toxic_words:
            masked = f"<mask>"
            masked_tokens.append(masked)
        else:
            masked_tokens.append(tok)

    return " ".join(masked_tokens)
