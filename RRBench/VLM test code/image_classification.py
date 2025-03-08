import os
import base64
import json
from tqdm import tqdm
import pandas as pd
from openai import OpenAI

class ImageClassifier:
    def __init__(self, api_key, base_url, model_name):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def clean_json_response(self, response_text):
        clean_text = response_text.replace('```json', '').replace('```', '').strip()
        return clean_text

    def classify_image(self, image_path):
        base64_image = self.encode_image(image_path)
        
        prompt = """Act as an expert in computational photography and generative AI. Analyze the given image metadata and visual characteristics to classify its origin as either real-world captured or AI-generated.

Your analysis should:
- Examine technical artifacts (unnatural textures, perfect symmetry, atypical shadow patterns)
- Check for common GAN/diffusion model fingerprints
- Evaluate biological plausibility (eyes, hair, skin textures)
- Identify hyperrealistic elements vs physical-world imperfections

Format response as JSON: { "classification": "AI-generated/Real"}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert image analyzer that responds only in JSON format."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            response_content = response.choices[0].message.content
            try:
                clean_content = self.clean_json_response(response_content)
                json_response = json.loads(clean_content)
                classification = json_response.get('classification', '').lower()
                return 'real' if 'real' in classification else 'fake'
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Problematic content: {response_content}")
                return None
                
        except Exception as e:
            print(f"Classification error: {str(e)}")
            return None

def calculate_metrics(predictions, true_labels):
    total = len(predictions)
    if total == 0:
        return {'acc': 0, 'real_acc': 0, 'fake_acc': 0}
        
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    
    real_predictions = [p for p, t in zip(predictions, true_labels) if t == 'real']
    real_labels = [t for t in true_labels if t == 'real']
    real_correct = sum(p == 'real' for p in real_predictions)
    
    fake_predictions = [p for p, t in zip(predictions, true_labels) if t == 'fake']
    fake_labels = [t for t in true_labels if t == 'fake']
    fake_correct = sum(p == 'fake' for p in fake_predictions)
    
    return {
        'acc': correct / total,
        'real_acc': real_correct / len(real_labels) if real_labels else 0,
        'fake_acc': fake_correct / len(fake_labels) if fake_labels else 0
    }

def evaluate_model(classifier, test_folder):
    results = {
        'original': {'predictions': [], 'true_labels': []},
        'transfer': {'predictions': [], 'true_labels': []},
        'redigital': {'predictions': [], 'true_labels': []}
    }
    
    for condition in results.keys():
        print(f"\nProcessing {condition} condition...")
        
        real_folder = os.path.join(test_folder, f"{condition}_real_images")
        if os.path.exists(real_folder):
            for img in tqdm(os.listdir(real_folder), desc=f"{condition} real images"):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    pred = classifier.classify_image(os.path.join(real_folder, img))
                    if pred:
                        results[condition]['predictions'].append(pred)
                        results[condition]['true_labels'].append('real')
        
        ai_folder = os.path.join(test_folder, f"{condition}_ai_images")
        if os.path.exists(ai_folder):
            for img in tqdm(os.listdir(ai_folder), desc=f"{condition} AI images"):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    pred = classifier.classify_image(os.path.join(ai_folder, img))
                    if pred:
                        results[condition]['predictions'].append(pred)
                        results[condition]['true_labels'].append('fake')
    
    metrics = {}
    for condition in results:
        metrics[condition] = calculate_metrics(
            results[condition]['predictions'],
            results[condition]['true_labels']
        )
    
    return metrics

def main():
    api_key = "YOUR_API_KEY"
    base_url = "YOUR_API_ENDPOINT"
    test_folder = "path/to/your/test/folder"
    
    models = [
        "model-1",
        "model-2",
        "model-3"
    ]

    all_results = {}
    
    for model in models:
        print(f"\n\nEvaluating model: {model}")
        print("="*50)
        
        classifier = ImageClassifier(api_key, base_url, model)
        metrics = evaluate_model(classifier, test_folder)
        all_results[model] = metrics
        
        for condition, scores in metrics.items():
            print(f"\n{condition.upper()} condition results:")
            print(f"Overall accuracy (acc): {scores['acc']:.4f}")
            print(f"Real image accuracy (real_acc): {scores['real_acc']:.4f}")
            print(f"AI image accuracy (fake_acc): {scores['fake_acc']:.4f}")
    
    rows = []
    for model in all_results:
        for condition, metrics in all_results[model].items():
            row = {
                'model': model,
                'condition': condition,
                'acc': metrics['acc'],
                'real_acc': metrics['real_acc'],
                'fake_acc': metrics['fake_acc']
            }
            rows.append(row)
    
    results_df = pd.DataFrame(rows)
    results_df.to_csv('classification_results.csv', index=False)
    print("\nDetailed results saved to classification_results.csv")

if __name__ == "__main__":
    main() 