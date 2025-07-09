import json
import os
from activateChatty import ActivateChatty

# Find the most recent results folder
results_base_path = '../../data/results/'
results_folders = [f for f in os.listdir(results_base_path) if f.startswith('results_')]
results_folders.sort(reverse=True)
latest_results_folder = results_folders[0]
results_path = os.path.join(results_base_path, latest_results_folder)

print(f"Using results folder: {latest_results_folder}")

# Load the detected compliments
detected_file = os.path.join(results_path, 'ADMA_detected_compliments_before_validation.json')
with open(detected_file, 'r', encoding='utf-8') as f:
    all_quarter_compliments = json.load(f)

print(f"Loaded {len(all_quarter_compliments)} quarters of data")

# Prepare input (same structure as input)
validation_input = {}
for quarter_key, quarter_data in all_quarter_compliments.items():
    validation_input[quarter_key] = {
        'year': quarter_data['year'],
        'quarter': quarter_data['quarter'],
        'date': quarter_data['date'],
        'compliments': quarter_data['compliments']
    }

# Convert to JSON string
batch_validation_input_str = json.dumps(validation_input, ensure_ascii=False, indent=2)
print(f"Input prepared, length: {len(batch_validation_input_str)} characters")

print("Making API call...")

# Make API call
chatty = ActivateChatty()
response = chatty.activate('../../data/prompts/validation.txt', batch_validation_input_str, 'gpt-4o')

print(f"API Response received!")
print(f"Response type: {type(response)}")

if isinstance(response, dict):
    print(f"Response has {len(response)} top-level keys")
    print(f"Response keys: {list(response.keys())}")
    
    # Show raw response (first 1000 chars)
    response_str = json.dumps(response, indent=2, ensure_ascii=False)
    print(f"\nRaw response (first 1000 chars):")
    print(response_str[:1000])
    
    # Write full response to file
    output_file = os.path.join(results_path, 'validation_api_response.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(response, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Full API response saved to: {output_file}")
    
elif isinstance(response, list):
    print(f"Response is a list with {len(response)} items")
    if len(response) > 0:
        print(f"First item: {response[0]}")
    
    # Write full response to file
    output_file = os.path.join(results_path, 'validation_api_response.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(response, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Full API response saved to: {output_file}")
    
else:
    print(f"Unexpected response type")
    print(f"Response: {response}")
    
    # Write response to file anyway
    output_file = os.path.join(results_path, 'validation_api_response.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(str(response))
    print(f"\n✓ Response saved to: {output_file}")

# Also save the input that was sent to the API
input_file = os.path.join(results_path, 'validation_api_input.json')
with open(input_file, 'w', encoding='utf-8') as f:
    json.dump(validation_input, f, ensure_ascii=False, indent=2)
print(f"✓ Input data saved to: {input_file}")

print(f"\nFiles saved in: {results_path}")
print("- validation_api_input.json (what we sent)")
print("- validation_api_response.json (what we received)") 