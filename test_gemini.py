"""測試 Gemini API 是否能跑"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")

if not api_key:
    print("❌ 找不到 GEMINI_API_KEY,檢查 .env")
    exit()

print(f"使用 model: {model_name}")
print(f"API key 前 8 碼: {api_key[:8]}...")

genai.configure(api_key=api_key)

try:
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        "用一句話告訴我:長榮 (2603) 是不是景氣循環股?"
    )
    print("\n✅ Gemini API 連線成功!")
    print(f"\n回應:\n{response.text}")
    
    # 顯示用量(讓你心裡有底)
    if hasattr(response, 'usage_metadata'):
        print(f"\n📊 Token 用量:")
        print(f"  輸入: {response.usage_metadata.prompt_token_count}")
        print(f"  輸出: {response.usage_metadata.candidates_token_count}")
        print(f"  總計: {response.usage_metadata.total_token_count}")
        
except Exception as e:
    print(f"❌ 失敗: {e}")