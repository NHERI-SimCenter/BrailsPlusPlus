import json
import glob
import pandas as pd
import csv
import requests
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def prompt_and_save_caption(image_path, prompt, api_key, model_arch, output_file = None, max_tokens = 300):
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model_arch,
        "messages": [
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
        ],
        "max_tokens": max_tokens
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if(output_file != None):
        with open(output_file, 'w') as f:
            json.dump(response.json(), f)
    return response.json()

def find_json(folder_path):
    if not folder_path.endswith('/'):
        folder_path += '/'
    image_paths = glob.glob(folder_path + '*.json', recursive=True)
    return image_paths

def load_predictions_from_json(json_contents, img_keys, options):
    if(type(json_contents)!=list):
        json_contents = [json_contents]
    if(type(img_keys)!=list):
        img_keys = [img_keys]
    image_preds = {}

    for content, img_key in zip(json_contents, img_keys):
        #image_name = j_file.split('/')[-1].replace('_caption.json','.png')
        if 'choices' in content:
            content = content['choices'][0]['message']['content']
            content = content[:80] #get first 60 characters to acquire main prediction message
            pred = 'none'
            for option in options:
                if(option in content):
                    pred = option
            image_preds[img_key] = pred
        else:
            image_preds[img_key] = 'none'
            
        if(image_preds[img_key] == 'none'):
            print(f'pred = none for key = {img_key}, response -> {content}')
        
    return image_preds


if __name__ == '__main__':
    pass
