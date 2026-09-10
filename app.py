import os
import io
import base64
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session
from PIL import Image, ImageColor
import numpy as np
import torch
import cv2
from transformers import pipeline

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Explicitly load .env from the current working directory
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "harvestiq_codoleox16_secret_key_2026")

print("🌾 [HarvestIQ by CodoLeoX16] Loading Vision Model...")
# Override this with a local trained model path or a valid Hugging Face model ID.
MODEL_ID = os.getenv("VISION_MODEL_ID", "wambugu71/crop_leaf_diseases_vit")
viz_pipe = pipeline("image-classification", model=MODEL_ID)
viz_model = viz_pipe.model
if hasattr(viz_model, "set_attn_implementation"):
    viz_model.set_attn_implementation("eager")
viz_model.eval()
print("✅ Vision Model Loaded.")

SUPPORTED_CROPS = ("Corn", "Potato", "Rice", "Wheat")
MIN_CONFIDENCE = 0.50

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "google").strip().lower()
    if provider == "groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            print("⚠️ Warning: GROQ_API_KEY not found in environment variables.")
            return None
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.7,
            groq_api_key=api_key
        )

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        print("⚠️ Warning: GOOGLE_API_KEY not found in environment variables.")
        return None
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
        temperature=0.7,
        google_api_key=api_key
    )

def parse_prediction(label):
    clean_label = label.replace("___", " ").replace("_", " ").strip()
    detected_crop = "Unidentified"
    detected_disease = clean_label

    for crop in SUPPORTED_CROPS:
        if crop.lower() in clean_label.lower():
            detected_crop = crop
            idx = clean_label.lower().find(crop.lower())
            disease_part = clean_label[idx + len(crop):].strip()
            if disease_part:
                detected_disease = disease_part
            break

    if "healthy" in clean_label.lower():
        detected_disease = "Healthy Leaf"

    return detected_crop, detected_disease.title()

def create_attention_overlay(image):
    """Create a coarse ViT attention map; this is an estimate, not a lesion mask."""
    try:
        processor = viz_pipe.image_processor
        inputs = processor(images=image, return_tensors="pt")
        device = next(viz_model.parameters()).device
        inputs = {name: value.to(device) for name, value in inputs.items()}

        with torch.no_grad():
            outputs = viz_model(**inputs, output_attentions=True)

        attentions = getattr(outputs, "attentions", None)
        if not attentions:
            return None

        # Roll attention through the ViT blocks so the CLS token receives a spatial map.
        rollout = torch.eye(attentions[0].shape[-1], device=device).unsqueeze(0)
        for attention in attentions:
            averaged = attention.mean(dim=1)
            averaged = averaged + torch.eye(averaged.shape[-1], device=device)
            averaged = averaged / averaged.sum(dim=-1, keepdim=True)
            rollout = averaged @ rollout

        patch_attention = rollout[0, 0, 1:]
        side = int(patch_attention.numel() ** 0.5)
        heatmap = patch_attention.reshape(side, side).cpu().numpy()
        heatmap -= heatmap.min()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(image.size, Image.Resampling.BILINEAR)
        heatmap_array = np.asarray(heatmap_image)
        threshold = max(150, int(np.percentile(heatmap_array, 88)))
        mask = np.uint8(heatmap_array >= threshold) * 255
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        marked = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR).copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minimum_area = image.width * image.height * 0.003
        for contour in contours:
            if cv2.contourArea(contour) < minimum_area:
                continue
            cv2.drawContours(marked, [contour], -1, (40, 40, 239), 4)
            x, y, width, height = cv2.boundingRect(contour)
            cv2.rectangle(marked, (x, y), (x + width, y + height), (40, 40, 239), 3)

        overlay = Image.fromarray(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))

        output = io.BytesIO()
        overlay.save(output, format="JPEG", quality=88)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        print(f"Attention rollout failed: {str(e)}")
        return None

def quota_fallback(disease, crop):
    return f"""⚠️ **AI advisory temporarily unavailable**

Gemini's free API quota has been reached. The visual result is still available, but this treatment guidance is a general first-aid checklist for **{crop} - {disease}** and should be confirmed by a local agronomist.

**Immediate steps**
1. Isolate visibly affected leaves and dispose of them away from the field; do not compost diseased material.
2. Keep foliage dry when possible. Water at the soil line, improve airflow, and avoid working among wet plants.
3. Check nearby plants every day for spreading spots, rust, mildew, or wilting.
4. Do not apply a chemical product until its label confirms it is approved for **{crop}** and this disease in your country. Follow the label, protective-equipment, harvest-interval, and dosage instructions.

The AI advisory will be available again when the Gemini quota resets or billing/quota limits are increased."""

def get_expert_remedy(disease, crop, llm, language, weather):
    template = """
    You are an expert agriculturalist and plant pathologist representing HarvestIQ (developed by CodoLeoX16).
    The user has uploaded a photo of a {crop} leaf. 
    Our vision diagnostic pipeline has detected: "{disease}".
    Current Local Weather: {weather}
    
    Please provide:
    1. **Visual Confirmation**: What this disease looks like and its immediate risk.
    2. **Actionable Treatment Plan**: Clear, step-by-step remedy (include organic/chemical options). Mention weather impacts (e.g., rain wash-off, humidity).
    3. **Preventative Protocols**: Simple practices to stop recurrence.
    
    CRITICAL INSTRUCTION: You must translate your entire response into {language}. Keep the tone clear, encouraging, and easy for a farmer to execute.
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm 
    
    try:
        response = chain.invoke({
            "crop": crop, 
            "disease": disease, 
            "language": language, 
            "weather": weather
        })
        
        content = response.content
        if isinstance(content, list):
            return content[0].get('text', '')
        return str(content)
        
    except Exception as e:
        print(f"LLM Error in analyze: {str(e)}")
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e) or "quota" in str(e).lower():
            return quota_fallback(disease, crop)
        return "⚠️ The AI advisory is temporarily unavailable. Please try again shortly."

@app.route('/')
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No leaf image uploaded"}), 400
        
    file = request.files['file']
    language = request.form.get('language', 'English')
    weather = request.form.get('weather', 'Unknown')
    
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400
    
    results = viz_pipe(image, top_k=3)
    top = results[0]
    final_crop, final_disease = parse_prediction(top['label'])
    confidence = float(top['score'])

    if confidence < MIN_CONFIDENCE or final_crop == "Unidentified" or "background" in top['label'].lower():
        final_crop = "Unidentified"
        final_disease = "Non-Supported Specimen"
        remedy = "⚠️ **HarvestIQ Notice**: The model could not identify an agricultural specimen with sufficient confidence. Please provide a clear, close-up photograph of a plant leaf under balanced lighting."
        attention_overlay = None
    else:
        attention_overlay = create_attention_overlay(image)
        llm = get_llm()
        if not llm:
            return jsonify({"error": "Backend Error: GOOGLE_API_KEY is missing from environment!"}), 500

        if "Healthy" in final_disease:
            remedy = f"✅ The **{final_crop}** leaf appears in optimal condition with no major active lesions detected. Continue regular irrigation and nutrient monitoring."
        else:
            remedy = get_expert_remedy(final_disease, final_crop, llm, language, weather)
    
    return jsonify({
        "crop": final_crop,
        "disease": final_disease,
        "confidence": confidence,
        "remedy": remedy,
        "is_healthy": "Healthy" in final_disease,
        "attention_overlay": attention_overlay,
        "supported_crops": list(SUPPORTED_CROPS)
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    user_query = data.get('message')
    language = data.get('language', 'English')
    weather = data.get('weather', 'Unknown')
    
    if not user_query:
        return jsonify({"response": "Please enter a message."}), 400

    llm = get_llm()
    if not llm:
        return jsonify({"response": "⚠️ Backend Error: GOOGLE_API_KEY is missing from environment!"}), 500
    
    lc_history = []
    for msg in session.get('chat_history', []):
        if msg['role'] == 'user':
            lc_history.append(HumanMessage(content=msg['content']))
        else:
            lc_history.append(AIMessage(content=msg['content']))
            
    system_prompt = """
    You are HarvestBot, an AI Agricultural Expert built for HarvestIQ by Team CodoLeoX16.
    Current Local Weather of the farmer: {weather}. Consider this context for crop safety and irrigation.
    Deliver direct, practical, and highly accessible answers for farmers.
    CRITICAL: You must reply entirely in {language}.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm 
    
    try:
        response = chain.invoke({
            "history": lc_history,
            "input": user_query,
            "language": language,
            "weather": weather
        })
        
        content = response.content
        if isinstance(content, list):
            response_text = content[0].get('text', '')
        else:
            response_text = str(content)
            
    except Exception as e:
        print(f"LLM Error in chat: {str(e)}")
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e) or "quota" in str(e).lower():
            return jsonify({"response": "⚠️ Gemini's free API quota has been reached. Please try again after the quota resets or increase the project's API quota."}), 429
        return jsonify({"response": "⚠️ The AI assistant is temporarily unavailable. Please try again shortly."}), 503
    
    history = session.get('chat_history', [])
    history.append({'role': 'user', 'content': user_query})
    history.append({'role': 'assistant', 'content': response_text})
    session['chat_history'] = history
    session.modified = True
    
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)
