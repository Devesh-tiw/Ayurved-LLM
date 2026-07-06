import csv
from deep_translator import GoogleTranslator
import time

print("Starting the translation engine... This might take a few minutes!")

# Initialize the translator (Hindi to English)
translator = GoogleTranslator(source='hi', target='en')

# Open the original Hindi file and create a new English file
with open('total_ayurveda_database.csv', 'r', encoding='utf-8') as infile, \
     open('english_ayurveda_database.csv', 'w', encoding='utf-8', newline='') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Read and write the Header row
    headers = next(reader)
    writer.writerow(headers)
    
    row_count = 0
    
    # Loop through every row in the database
    for row in reader:
        new_row = []
        for cell in row:
            text = str(cell).strip()
            # If the cell has text, translate it!
            if text:
                try:
                    translated_text = translator.translate(text)
                    new_row.append(translated_text)
                except Exception as e:
                    # If translation fails, just keep the original text
                    new_row.append(text)
            else:
                new_row.append("") # Keep empty cells empty
                
        writer.writerow(new_row)
        row_count += 1
        
        # Print an update every 10 rows so you know it's working
        if row_count % 10 == 0:
            print(f"✅ Successfully translated {row_count} rows...")
            time.sleep(0.5) # A tiny pause so we don't get blocked by Google

print("🎉 Translation Complete! Saved as 'english_ayurveda_database.csv'")