from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import PDFUploadForm
from .utilities import extract_text_from_pdf, autocorrect_text, find_errors, highlight_errors
import os

def upload_and_process_pdf(request):
    corrected_text = None
    highlighted_text = None
    form = PDFUploadForm()

    if request.method == "POST":
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            fs = FileSystemStorage(location='/tmp')
            file_path = fs.save(uploaded_file.name, uploaded_file)
            full_file_path = fs.path(file_path)

            try:
                # Step 1: Extract text
                original_text = extract_text_from_pdf(full_file_path)

                if original_text:
                    # Step 2: Correct the text
                    corrected_text = autocorrect_text(original_text)

                    # Step 3: Find and highlight errors
                    errors = find_errors(original_text, corrected_text)
                    highlighted_text = highlight_errors(original_text, errors)
                else:
                    corrected_text = "No text could be extracted from the PDF file."

            except Exception as e:
                corrected_text = f"An error occurred: {str(e)}"

            finally:
                os.remove(full_file_path)

    return render(
        request,
        "typocorrection/upload_and_process.html",
        {
            "form": form,
            "corrected_text": corrected_text,
            "highlighted_text": highlighted_text,
        }
    )