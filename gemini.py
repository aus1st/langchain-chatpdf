import google.generativeai as genai
import os
from dotenv import load_dotenv

def main():

    load_dotenv()
    genai.configure(api_key=os.getenv('API_KEY'))

    model= genai.GenerativeModel('gemini-pro')
    response = model.generate_content('Say hi')

    print(response.text)

main()