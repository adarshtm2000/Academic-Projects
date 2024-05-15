from flask import Flask, render_template, request
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from deep_translator import GoogleTranslator
import spacy

def translate_text(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def translate(malayalam_text):
    try:
        translated_text = translate_text(malayalam_text)
        return translated_text
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return error_message
    
def remove_starting_and(completion):

    if completion.startswith("and"):

        completion = completion[len("and"):].lstrip()  # Remove "and" from the beginning and strip leading whitespace

    return completion

app = Flask(__name__)

model = AutoModelForSeq2SeqLM.from_pretrained('model', max_length=1024)
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        malayalam_text = request.form.get('malayalam_text', '')

        if not malayalam_text:  # If no text is provided, render the template without generating text
            return render_template('index.html')
        
       # print("Malayalam Text:", malayalam_text)  # Before translation
        
        prompt = translate(malayalam_text)

       # print("Translated Text:", prompt)  # After translation
        
        nlp_ner = spacy.load("model-best")

        input_ids = tokenizer.encode(prompt, return_tensors='pt', max_length=4096, truncation=False, padding=True)

        #print("Input IDs:", input_ids)  # After tokenization

        output_ids = model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(
            max_new_tokens=500,
            repetition_penalty=1.2,  # Adjust as needed
            do_sample=True,
            temperature=1.0,  # Adjust to a lower value for higher accuracy
            top_k=50,  # Increase if necessary
            top_p=0.9  # Adjust as needed
        )
)
        completion = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        #print("Output IDs:", output_ids)  # After generation
       # print("Completion:", completion)  # After generation

       # print("Generated completion before removal:", completion)  # Debugging print statement

    # Remove prefix-suffix overlap, including 'and' at the beginning
        completion = remove_starting_and(completion)

       # print("Generated completion after removal:", completion)  # Debugging print statement

        #print(completion)

        return render_template('index.html', prompt=True, completion=completion)
    
    else: 
        # Render index template for query submission
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
