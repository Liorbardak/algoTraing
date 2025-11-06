import json
import re


def validate_and_clean_json(input_file, output_file=None):
    """
    Validate JSON and optionally clean up analyst names
    """

    # Read the file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("✓ JSON is valid!")
    except json.JSONDecodeError as e:
        print(f"✗ JSON is invalid: {e}")
        return None
    except FileNotFoundError:
        print(f"✗ File not found: {input_file}")
        return None

    # Name normalization mapping
    name_corrections = {
        "Kristen Kleska": "Kristen Kluska",
        "Kristin Kluxa": "Kristen Kluska",
        "Kristen Kolska": "Kristen Kluska",
        "Christian Kleska": "Kristen Kluska",  # Assuming this is the same person
        "Elliot Wilber": "Elliot Wilbur",
        "Anthony Patron": "Anthony Petroni",
        "Anthony Petro": "Anthony Petroni",
        "Gary Nockman": "Gary Nachman",
        "Gary Matchman": "Gary Nachman"
    }

    # Clean up the data
    cleaned_data = {}
    changes_made = []

    for quarter, quarter_data in data.items():
        cleaned_data[quarter] = quarter_data.copy()

        for i, compliment in enumerate(quarter_data.get('compliments', [])):
            original_name = compliment.get('analyst_name', '')

            # Check if name needs correction
            if original_name in name_corrections:
                corrected_name = name_corrections[original_name]
                cleaned_data[quarter]['compliments'][i]['analyst_name'] = corrected_name
                changes_made.append(f"{quarter}: '{original_name}' → '{corrected_name}'")

            # Clean up quoted compliments (remove extra quotes if present)
            quote = compliment.get('quoted_compliment', '')
            if quote.startswith('\"') and quote.endswith('\"') and quote.count('\"') == 2:
                cleaned_quote = quote[1:-1]  # Remove outer quotes
                cleaned_data[quarter]['compliments'][i]['quoted_compliment'] = cleaned_quote

    # Report changes
    if changes_made:
        print(f"\n📝 Made {len(changes_made)} name corrections:")
        for change in changes_made:
            print(f"  - {change}")
    else:
        print("\n✓ No name corrections needed")

    # Save cleaned data
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
        print(f"\n💾 Cleaned data saved to: {output_file}")

    return cleaned_data


def analyze_json_structure(data):
    """
    Analyze the structure of the JSON data
    """
    print("\n📊 JSON Structure Analysis:")
    print(f"  - Total quarters: {len(data)}")

    total_compliments = 0
    analysts = set()

    for quarter, quarter_data in data.items():
        compliments = quarter_data.get('compliments', [])
        total_compliments += len(compliments)

        for compliment in compliments:
            analysts.add(compliment.get('analyst_name', ''))

    print(f"  - Total compliments: {total_compliments}")
    print(f"  - Unique analysts: {len(analysts)}")
    print(f"  - Analyst names: {sorted(analysts)}")


def main():
    # File paths
    input_file = "C:/Users/dadab\projects/algotrading/data/results/results_20250709_221606/ADMA_detected_compliments_before_validation.json"
    output_file = "ADMA_detected_compliments_cleaned.json"

    # Validate and clean
    cleaned_data = validate_and_clean_json(input_file, output_file)

    if cleaned_data:
        analyze_json_structure(cleaned_data)

        # Verify the output is valid JSON
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                json.load(f)
            print(f"\n✅ Output file '{output_file}' is valid JSON!")
        except:
            print(f"\n❌ Error validating output file")


if __name__ == "__main__":
    main()