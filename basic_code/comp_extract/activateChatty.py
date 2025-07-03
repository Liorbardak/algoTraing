"""
ChatGPT API Integration Module

This module provides an interface to interact with ChatGPT API for natural language
processing tasks. It's used for analyzing text data, particularly in the context
of earnings calls and financial documents.

Key Features:
- ChatGPT API integration
- Text analysis and processing
- Sentiment analysis
- Natural language understanding
- Response formatting

Author: talhadaski
Last updated: 2024-03-29
"""

# import openai
from openai import OpenAI
import json
import os
import re
class ActivateChatty:
    def __init__(self):

        API_KEY_CHAT = "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"
        self.client = OpenAI(
            api_key=API_KEY_CHAT
        )

    def activate(self, user_prompt_file, data, model = "gpt-4o-mini"):
        # Define the user prompt with the earnings call text and the question
        # Open the file in read mode
        with open(user_prompt_file, 'r', encoding='utf-8') as file:
            # Read the contents of the file
            prompt_detection = file.read()
        user_prompt = prompt_detection.format(text=data)
        # Make the API call to generate a completion
        responseChat = self.client.chat.completions.create(
            model=model,  # Specify GPT-4 model
            # model = "gpt-4o-mini",

            messages=[
                {"role": "system", "content": "You are a financial analysis assistant."},
                {"role": "user", "content": user_prompt}
            ],
        )
        json_string = responseChat.choices[0].message.content.strip('```json\n').strip('\n```')
        # Remove invalid control characters using regex
        cleaned_json_string = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_string)

        # Adding missing commas between JSON objects if any
        cleaned_json_string = re.sub(r'}\s*{', '},{', cleaned_json_string)

        # Decode the cleaned JSON string
        try:
            output = json.loads(cleaned_json_string)
            print("JSON decoded successfully!")
        except json.JSONDecodeError as e:
            output = []
            print(f"Error decoding JSON: {e}")



        return output

    # Function to write the answers to a JSON file
    def write_to_json_file_total(self, data, directory, prefix):
        filename = os.path.join(directory, f'{prefix}.json')

        # Load the JSON string
        # Write the dictionary to a JSON file
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

    # Function to write the answers to a JSON file
    def write_to_json_file(self, data, directory, prefix,year, quarter):
        filename = os.path.join(directory, f'{prefix}_{str(year)}_{str(quarter)}_results.json')

        # Load the JSON string
        # Write the dictionary to a JSON file
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

    # Function to write the answers to a JSON file with _new suffix
    def write_to_json_file_new(self, data, directory, prefix, year, quarter):
        filename = os.path.join(directory, f'{prefix}_{str(year)}_{str(quarter)}_results_new.json')

        # Load the JSON string
        # Write the dictionary to a JSON file
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

    # Function to write the total answers to a JSON file with _new suffix
    def write_to_json_file_total_new(self, data, directory, prefix):
        filename = os.path.join(directory, f'{prefix}_new.json')

        # Load the JSON string
        # Write the dictionary to a JSON file
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

    def read_from_json_file( self,ticker,x, i):
        filename = f"{ticker}/validatedParsed_{x}_{i}_results.json"
        with open(filename, "r") as file:
            temp = json.load(file)
        if isinstance(temp, dict)  and temp.get('compliments') is not None:
            temp = temp['compliments']
        return temp

