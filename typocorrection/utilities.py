from PyPDF2 import PdfReader
from transformers import BertTokenizerFast, BertForTokenClassification
import torch
import random
import numpy as np
import pdfplumber, re
from autocorrection.autocorrect_main import process_article

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# set_seed(8)  # Use a fixed seed for reproducibility

def extract_text_from_pdf(file_path):
    try:
        all_text = ""
        print(f"Opening PDF file: {file_path}")  # Debug statement
        with pdfplumber.open(file_path) as reader:
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                print(f"Page {page_num} text: {text}")  # Debug extracted text per page
                if text:
                    text = re.sub(r'^\d+\s*', '', text, flags=re.MULTILINE)
                    text = text.replace('\n', '').replace(' ', '')
                    all_text += text
        print("Final extracted text:", all_text)  # Debug final text
        return all_text if all_text else None  # Ensure None if no text
    except Exception as e:
        print(f"Error extracting PDF: {e}")  # Debug any extraction errors
        return None

# Autocorrect text using the model
def autocorrect_text(input_text):
    model_path = "D:/NCU/FirstSemester/LegalAI/django/legalAI/autocorrection/step-10200_f1-76.67.bin"
    pretrained_model_path = "bert-base-chinese"
    prompt_length = 1

    print("Debug: Input text type:", type(input_text))  # Debug input type
    print("Debug: Input text preview:", input_text[:100])  # Show first 100 characters

    try:
        corrected_text = process_article(input_text, model_path, pretrained_model_path, prompt_length)
        print("Debug: Corrected text:", corrected_text)  # Debug output
        return corrected_text
    except Exception as e:
        print("Error in autocorrect_text:", e)
        return f"Error: {str(e)}"

def normalize_text(text):
    """
    Normalize text to replace special characters like ㈠, ㈢ with placeholders or equivalents.
    """
    replacements = {
        '㈠': '(1)',
        '㈡': '(2)',
        '㈢': '(3)',
        '㈣': '(4)',
        '㈤': '(5)',
        # Add more mappings as needed
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text

def find_errors(original, corrected):
    """
    Compare original and corrected text, accounting for single-to-multiple character corrections.
    Handles alignment issues when characters in the original text are corrected into multiple characters.
    """
    errors = []
    corrected_index = 0

    for i in range(len(original)):
        if corrected_index >= len(corrected):  # If corrected text runs out
            errors.append((i, original[i]))
            continue

        # Handle single-to-multiple character corrections
        if original[i] == corrected[corrected_index]:
            corrected_index += 1  # Move to the next corrected character
        elif original[i] in "㈠㈡㈢㈣㈤⒈⒉⒊⒋⒌…":  # Skip special characters in the original text
            continue
        elif corrected[corrected_index:].startswith(original[i]):  # Match multi-char correction
            corrected_index += len(original[i])  # Skip all matching chars in corrected
        else:
            errors.append((i, original[i]))  # Record the error
            # Adjust for multi-char corrections by moving corrected_index forward carefully
            corrected_index += 1

    return errors

def highlight_errors(original_text, errors):
    """
    Highlight specific erroneous characters in the original text.
    :param original_text: The original text.
    :param errors: List of (index, character) tuples.
    :return: Text with errors wrapped in <span> tags.
    """
    highlighted_text = ""
    for i, char in enumerate(original_text):
        # If this character's index is in errors, wrap it in <span>
        if i in [error[0] for error in errors]:
            highlighted_text += f'<span class="error">{char}</span>'
        else:
            highlighted_text += char
    return highlighted_text



