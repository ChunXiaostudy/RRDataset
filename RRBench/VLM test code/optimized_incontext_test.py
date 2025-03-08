import os
import base64
import json
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from PIL import Image
import io

class OptimizedInContextClassifier:
    def __init__(self, api_key, base_url, model_name):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.example_images = {}
        
    def optimize_image(self, image_path, max_size=800):
        """优化图片大小"""
        with Image.open(image_path) as img:
            # 保持宽高比缩放
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # 转换为JPEG格式并压缩
            buffer = io.BytesIO()
            img.convert('RGB').save(buffer, format='JPEG', quality=85, optimize=True)
            return buffer.getvalue()
    
    def encode_image(self, image_path, optimize=True):
        """将图片转换为base64编码，可选择是否优化"""
        if optimize:
            image_data = self.optimize_image(image_path)
            return base64.b64encode(image_data).decode('utf-8')
        else:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

    def load_example_images(self, base_path):
        """加载并优化示例图片"""
        example_paths = {
            'transfer_real': os.path.join(base_path, "transfer_real_images", "transfer_real_000001.png"),
            'transfer_fake': os.path.join(base_path, "test", "transfer_ai_images", "transfer_Medical_&_Public_Health_000837.png"),
            'redigital_real': os.path.join(base_path, "test", "redigital_real_images", "redigital_real_001809.jpg"),
            'redigital_fake': os.path.join(base_path, "test", "redigital_ai_images", "redigital_Culture_&_Religion_000459.jpg")
        }
        
        for key, path in example_paths.items():
            if os.path.exists(path):
                self.example_images[key] = self.encode_image(path, optimize=True)
            else:
                print(f"警告：示例图片不存在: {path}")

    def get_base_messages(self):
        """获取基础消息结构"""
        base_messages = [
            {
                "role": "system",
                "content": "You are an expert image analyzer that responds only in JSON format."
            }
        ]
        
        # 添加示例图片
        if self.example_images:
            example_contents = []
            examples_text = {
                'transfer_real': "Example 1 - Real transmitted image with natural textures and imperfections",
                'transfer_fake': "Example 2 - AI-generated transmitted image with typical GAN artifacts",
                'redigital_real': "Example 3 - Real re-digitized image with authentic details",
                'redigital_fake': "Example 4 - AI-generated re-digitized image with synthesis patterns"
            }
            
            for key, base64_data in self.example_images.items():
                example_contents.extend([
                    {"type": "text", "text": examples_text[key]},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}}
                ])
            
            if example_contents:
                base_messages.append({
                    "role": "user",
                    "content": example_contents
                })
        
        return base_messages

    def classify_image(self, image_path):
        """分类单张图片"""
        base64_image = self.encode_image(image_path, optimize=True)
        
        # 使用原始详细的prompt
        prompt = """Act as a forensic image analyst specializing in origin classification. Analyze images through their intrinsic visual patterns while disregarding transmission/redigitization artifacts. Focus on fundamental generation traces rather than secondary distortions.

Contextual Examples:
1. [Transmitted Image] Compressed JPEG with blocking artifacts, but shows consistent micro-textures in hair strands and natural skin pore variation → Real
2. [Transmitted Image] Lossy webP image with chromatic aberration, yet reveals perfect fractal patterns in background and asymmetric eyelash duplication → AI-generated
3. [Re-digitized Image] A scanned copy with a moiré pattern, but the content and details of the picture are real → Real 
4. [Re-digitized Image] Photographed print showing lens glare, yet contains Diffusion-typical floating specks and impossible light intersections → AI-generated

Analysis Protocol:
1. Primary Focus Areas:
   - Microscopic texture coherence (brush strokes/sensor noise patterns)
   - High-frequency detail preservation beyond compression/scanning
   - Biological imperfection consistency (asymmetric irises, skin translucency)
   - Physical light interaction validity (shadow falloff, subsurface scattering)

2. Artifact Discounting:
   - Ignore format-specific compression patterns (JPEG blocking, WEBP smearing)
   - Disregard scanning artifacts (dust particles, Newton rings)
   - Overlook resampling distortions (aliasing, interpolation errors)

3. Decisive Indicators:
   - Generator fingerprints in frequency domain (FFT quadrant patterns)
   - Anatomical plausibility under 3x digital magnification
   - Material property consistency (metallic reflections, cloth draping)

Output JSON: { "classification": "AI-generated/Real" }"""

        messages = self.get_base_messages()
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            
            response_content = response.choices[0].message.content
            try:
                # 清理和解析响应
                clean_content = response_content.replace('```json', '').replace('```', '').strip()
                json_response = json.loads(clean_content)
                classification = json_response.get('classification', '').lower()
                return 'real' if 'real' in classification else 'fake'
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                print(f"问题内容: {response_content}")
                return None
                
        except Exception as e:
            print(f"分类错误: {str(e)}")
            return None

def evaluate_model(classifier, test_folder):
    """评估模型性能"""
    results = {
        'original': {'predictions': [], 'true_labels': []},
        'transfer': {'predictions': [], 'true_labels': []},
        'redigital': {'predictions': [], 'true_labels': []}
    }
    
    for condition in results.keys():
        print(f"\n处理 {condition} 条件的图片...")
        
        # 处理真实图片
        real_folder = os.path.join(test_folder, f"{condition}_real_images")
        if os.path.exists(real_folder):
            for img in tqdm(os.listdir(real_folder), desc=f"{condition} 真实图片"):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    pred = classifier.classify_image(os.path.join(real_folder, img))
                    if pred:
                        results[condition]['predictions'].append(pred)
                        results[condition]['true_labels'].append('real')
        
        # 处理AI生成图片
        ai_folder = os.path.join(test_folder, f"{condition}_ai_images")
        if os.path.exists(ai_folder):
            for img in tqdm(os.listdir(ai_folder), desc=f"{condition} AI生成图片"):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    pred = classifier.classify_image(os.path.join(ai_folder, img))
                    if pred:
                        results[condition]['predictions'].append(pred)
                        results[condition]['true_labels'].append('fake')
    
    # 计算指标
    metrics = {}
    for condition in results:
        total = len(results[condition]['predictions'])
        if total > 0:
            correct = sum(p == t for p, t in zip(results[condition]['predictions'], results[condition]['true_labels']))
            real_correct = sum(p == t and t == 'real' for p, t in zip(results[condition]['predictions'], results[condition]['true_labels']))
            real_total = sum(t == 'real' for t in results[condition]['true_labels'])
            fake_correct = sum(p == t and t == 'fake' for p, t in zip(results[condition]['predictions'], results[condition]['true_labels']))
            fake_total = sum(t == 'fake' for t in results[condition]['true_labels'])
            
            metrics[condition] = {
                'acc': correct / total,
                'real_acc': real_correct / real_total if real_total > 0 else 0,
                'fake_acc': fake_correct / fake_total if fake_total > 0 else 0
            }
        else:
            metrics[condition] = {'acc': 0, 'real_acc': 0, 'fake_acc': 0}
    
    return metrics, results

def main():
    # 配置参数
    api_key = "sk-zk2d04aa9450b2c0db94490b3e53179ea5a67c3088a3dc91"
    base_url = "https://api.zhizengzeng.com/v1"
    base_path = r"E:\ImageNet\RRDataset\RRDataset_final"
    test_folder = os.path.join(base_path, "test")
    
    models = [
        "glm-4v-plus",
        "grok-2-vision",
        "Qwen2.5-VL-72B",
        "gemini-1.5-pro"
    ]
    
    for model in models:
        print(f"\n\n评估模型: {model}")
        print("="*50)
        
        # 初始化分类器
        classifier = OptimizedInContextClassifier(api_key, base_url, model)
        classifier.load_example_images(base_path)
        
        # 评估模型
        metrics, results = evaluate_model(classifier, test_folder)
        
        # 打印结果
        for condition, scores in metrics.items():
            print(f"\n{condition.upper()} 条件下的结果:")
            print(f"总体准确率 (acc): {scores['acc']:.4f}")
            print(f"真实图片准确率 (real_acc): {scores['real_acc']:.4f}")
            print(f"AI生成图片准确率 (fake_acc): {scores['fake_acc']:.4f}")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'results_{model}_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump({
                'model': model,
                'metrics': metrics,
                'results': results
            }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main() 