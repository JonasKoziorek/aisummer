import base64
import ollama
from httpx import DigestAuth
from pydantic import BaseModel
from ollama import Client
from enum import Enum
from typing import Optional
from httpx import DigestAuth
import json
import time



client = ollama.Client(
            host="https://ollama.kky.zcu.cz",
            auth=DigestAuth(
                "niryo", 
                "ahPh7JohThe3kie3eeng"
            ),
        )




class AnswerLLM(BaseModel):
    response: str
    seal_number: int
    note: Optional[str] = None

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



def call_openai_api(
        client, 
        image_path, 
        command,
        prompt
    ):

    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
                "role": "developer", 
                "content": prompt
            },
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": command
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                    }
                    }
                ]
            }
        ],
        response_format=AnswerLLM,
    )

    return completion.choices[0].message.parsed

def call_kky_ollama_api(
        client, 
        image_path, 
        command,
        prompt,
        model='gemma3:12b'
    ):

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": prompt
            },
            {
                "role": "user",
                "content": command,
                'images': [image_path]
            },
        ],
        format=AnswerLLM.model_json_schema(),        
    )
    return response.message.content

def call_openai_api_on_fly(
        client, 
        image, 
        command,
        prompt
    ):

    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
                "role": "developer", 
                "content": prompt
            },
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": command
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    }
                    }
                ]
            }
        ],
        response_format=AnswerLLM,
    )

    return completion.choices[0].message.parsed

def call_kky_ollama_api_on_fly(
        client, 
        image, 
        command,
        prompt,
        model='gemma3:12b'
    ):

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": prompt
            },
            {
                "role": "user",
                "content": command,
                'images': [image]
            },
        ],
        format=AnswerLLM.model_json_schema(),        
    )
    return response.message.content

def call_llm(image_path, command, prompt, openai=True, openai_api_key=None,
            kky_ollama_uname=None, kky_ollama_password=None, kky_ollama_server=None):

    answer_dict = {}

    if openai:
        # OpenAI API
        client = OpenAI(api_key=openai_api_key)

        answer = call_openai_api(
            client,
            image_path=image_path,
            command=command,
            prompt=prompt
        )

        answer_dict = answer.__dict__
    else:
        # KKY ollama server
        client = Client(
            host=kky_ollama_server,
            auth=DigestAuth(
                kky_ollama_uname, 
                kky_ollama_password
            ),
        )

        answer = call_kky_ollama_api(
            client,
            image_path=image_path,
            command=command,
            prompt=prompt
        )
        answer_dict = json.loads(answer)

    return answer_dict

def call_llm_on_fly(image, command, prompt, openai=True, openai_api_key=None,
            kky_ollama_uname=None, kky_ollama_password=None, kky_ollama_server=None): 

    answer_dict = {}

    if openai:
        # OpenAI API
        client = OpenAI(api_key=openai_api_key)

        answer = call_openai_api_on_fly(
            client,
            image=image,
            command=command,
            prompt=prompt
        )

        answer_dict = answer.__dict__
    else:
        # KKY ollama server
        client = Client(
            host=kky_ollama_server,
            auth=DigestAuth(
                kky_ollama_uname, 
                kky_ollama_password
            ),
        )

        answer = call_kky_ollama_api_on_fly(
            client,
            image=image,
            command=command,
            prompt=prompt
        )
        answer_dict = json.loads(answer)

    return answer_dict

start = time.perf_counter()
answer = call_kky_ollama_api(
    client=client,
    image_path=r"D:\AiSimmer2026\images\ai_summer_school_dataset\train\00016.png",
    command="""Read the label in this image. Return the result strictly in this format: BRAND: <text> NUMBER: <digits only, no spaces> Read the digits left to right, character by character. Do not guess or round to a "nice" number — report exactly what is printed.""",
    prompt="""You are a precise OCR system specialized in reading embossed or printed text on macro photographs of industrial labels and component plates. You read characters exactly as they visually appear, without correcting them to "plausible" or "expected" values. You ignore background noise such as scratches, tool marks, screws, reflections, and glare. When a character is ambiguous, you report your best-guess reading and note the alternative in parentheses. You never add explanations, scene descriptions, or extra commentary — you output only the requested fields, nothing else.""",
    model='gemma3:12b'
)
elapsed = time.perf_counter() - start
print(f"Inference time: {elapsed:.3f}s")



print(answer)