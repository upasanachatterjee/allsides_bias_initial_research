from groq import Groq
import os
import json
from dotenv import load_dotenv
from utils import load_json

load_dotenv()  # Load variables from .env into the environment

api_key = os.getenv("GROQ_API_KEY")


def get_groq_client():
    client = Groq(api_key=api_key)
    return client


def run_groq(client, prompt, max_tokens, system_prompt, response_format=False):
    format = {"type": "json_object"} if response_format else None
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=1,
        max_completion_tokens=max_tokens,
        top_p=1,
        stream=False,
        stop=None,
        response_format=format,
    )
    return completion.choices[0].message.content


def generate_text_groq(idx, path, groq_client):
    json_data = load_json(path)
    word_count = len(json_data["original_text"].split())

    print("generating for id = ", idx, "word count = ", word_count)

    system_prompt = "Continue the following text, keeping the same tone"

    output = run_groq(
        groq_client, json_data["original_text"], word_count * 2, system_prompt
    )
    print("generated sequence of length = ", len("".join(output).split()))
    return output


def generate_bias_groq(prompt, client):
    system_prompt = ("classify the following text with one of the following biases:"
    + "left, center, right. Do this in the following json format {'bias' : 'text'}")
    output = run_groq(client, prompt, 1024, system_prompt, True)
    return json.loads(output)["bias"]


def generate_topic_groq(client, loaded_json):
    system_prompt = ("Are the following two texts on roughly the same topic?"
                     + "Use the following JSON format: {'same_topic': boolean}")
    prompt = json.dumps(
        {
            "text_1": loaded_json["generated_text"],
            "text_2": loaded_json["original_text"],
        }
    )
    output = run_groq(client, prompt, 1024, system_prompt, True)
    return json.loads(output)["same_topic"]


def generate_bias_groq_with_samples(prompt, sample, client):
    system_prompt = (
        "classify the following text with one of the following biases: "
        + "left, center, right. Do this in the following json format {'bias' : 'text'}. "
        + "Here are some samples "
        + sample
    )
    output = run_groq(client, prompt, 1024, system_prompt, True)
    return json.loads(output)["bias"]
