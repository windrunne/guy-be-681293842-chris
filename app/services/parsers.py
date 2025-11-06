from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table
from docx.text.paragraph import Paragraph
import pdfplumber
from typing import List, Optional
import base64
import time
import fitz
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.core.config import settings
from app.core.logging_config import get_logger
from app.utils.dependencies import get_openai_client

logger = get_logger(__name__)

class DocumentParser:
    """
    Parse various document formats and extract:
    - Text from paragraphs
    - Tables with structured data
    - Images (with descriptions using OpenAI Vision or basic extraction)
    - Charts/Graphs (extracted as text descriptions where possible)
    """
    
    def parse_docx(self, file_path: str, progress_callback=None) -> str:
        """
        Parse DOCX file and extract:
        - Text from paragraphs
        - Tables with structured data
        - Images with descriptions (if OpenAI Vision available)
        - Chart/graph references
        """
        parse_start = time.time()
        
        try:
            # Load document
            if progress_callback:
                progress_callback("parsing", "Loading DOCX document...", 10)
            load_start = time.time()
            doc = Document(file_path)
            load_duration = time.time() - load_start
            
            content_parts = []
            paragraph_count = 0
            table_count = 0
            total_elements = len(doc.element.body) if hasattr(doc.element, 'body') else 0
            
            # Process document elements in order (maintains structure)
            if progress_callback:
                progress_callback("parsing", f"Processing {total_elements} document elements (paragraphs, tables)...", 12)
            element_start = time.time()
            
            for idx, element in enumerate(doc.element.body):
                # Check if it's a paragraph
                if isinstance(element, CT_P):
                    paragraph = Paragraph(element, doc)
                    text = paragraph.text.strip()
                    if text:
                        content_parts.append(text)
                        paragraph_count += 1
                
                # Check if it's a table
                elif isinstance(element, CT_Tbl):
                    table = Table(element, doc)
                    table_data = self._extract_table_data(table)
                    if table_data:
                        content_parts.append(f"\n[TABLE]\n{table_data}\n[/TABLE]")
                        table_count += 1
                
                # Update progress every 10 elements
                if progress_callback and (idx + 1) % 10 == 0:
                    progress = 12 + int((idx + 1) / total_elements * 5) if total_elements > 0 else 12
                    progress_callback("parsing", f"Processed {idx + 1}/{total_elements} elements ({paragraph_count} paragraphs, {table_count} tables)...", progress)
            
            element_duration = time.time() - element_start
            
            # Extract images from document
            if progress_callback:
                progress_callback("parsing", "Extracting images from document...", 18)
            image_start = time.time()
            image_descriptions = self._extract_images_from_docx(doc, progress_callback)
            image_duration = time.time() - image_start
            if image_descriptions:
                content_parts.append("\n[IMAGES]\n" + "\n".join(image_descriptions) + "\n[/IMAGES]")
            
            parse_duration = time.time() - parse_start
            content_length = len("\n\n".join(content_parts))
            
            return "\n\n".join(content_parts)
            
        except Exception as e:
            parse_duration = time.time() - parse_start
            raise Exception(f"Error parsing DOCX: {str(e)}")
    
    def _extract_table_data(self, table: Table) -> str:
        """Extract structured data from table"""
        table_data = []
        for row in table.rows:
            row_data = []
            for cell in row.cells:
                cell_text = cell.text.strip()
                if cell_text:
                    row_data.append(cell_text)
            if row_data:
                table_data.append(" | ".join(row_data))
        return "\n".join(table_data)
    
    def _extract_images_from_docx(self, doc: DocxDocument, progress_callback=None) -> List[str]:
        """Extract images from DOCX and generate descriptions in parallel"""
        image_descriptions = []
        
        try:            
            # First, collect all images
            if progress_callback:
                progress_callback("parsing", "Scanning document for images...", 18)
            image_data_list = []
            image_count = 0
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    image_count += 1
                    try:
                        image_part = rel.target_part
                        image_bytes = image_part.blob
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                        image_data_list.append((image_count, image_base64))
                    except Exception as e:
                        image_descriptions.append(f"Image {image_count}: [Error extracting image: {str(e)}]")
            
            if not image_data_list:
                if progress_callback:
                    progress_callback("parsing", "No images found in document", 19)
                return image_descriptions
            
            if progress_callback:
                progress_callback("parsing", f"Found {len(image_data_list)} images, analyzing with AI...", 18)
            
            # Function to process a single image
            def process_image(image_num, image_base64):
                image_start = time.time()
                try:
                    client = get_openai_client()
                    response = client.chat.completions.create(
                        model=settings.OPENAI_VISION_MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Describe this image in detail, including any text, charts, graphs, or data visualizations. Focus on extracting all readable information."
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_base64}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=settings.OPENAI_VISION_MAX_TOKENS
                    )
                    description = response.choices[0].message.content
                    image_duration = time.time() - image_start
                    return image_num, f"Image {image_num}: {description}", None
                except Exception as e:
                    image_duration = time.time() - image_start
                    return image_num, f"Image {image_num}: [Error processing image: {str(e)}]", str(e)
            
            # Process images in parallel
            max_workers = min(len(image_data_list), settings.DOCUMENT_IMAGE_EXTRACTION_PARALLEL_WORKERS)
            
            parallel_start = time.time()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all image processing tasks
                future_to_image = {
                    executor.submit(process_image, image_num, image_base64): image_num
                    for image_num, image_base64 in image_data_list
                }
                
                # Collect results as they complete
                image_results = {}
                completed_count = 0
                for future in as_completed(future_to_image):
                    image_num = future_to_image[future]
                    try:
                        result_image_num, description, error = future.result()
                        image_results[result_image_num] = description
                        completed_count += 1
                        if progress_callback:
                            progress = 18 + int((completed_count / len(image_data_list)) * 1)
                            progress_callback("parsing", f"Analyzed image {completed_count}/{len(image_data_list)} with AI...", progress)
                    except Exception as e:
                        image_results[image_num] = f"Image {image_num}: [Error: {str(e)}]"
                        completed_count += 1
            
            parallel_duration = time.time() - parallel_start
            
            # Combine results in order
            for image_num in sorted(image_results.keys()):
                image_descriptions.append(image_results[image_num])
            
        except Exception as e:
            if image_count > 0:
                image_descriptions.append(f"[{image_count} image(s) found but could not be processed: {str(e)}]")
        
        return image_descriptions
    
    def parse_pdf(self, file_path: str, progress_callback=None) -> str:
        """
        Parse PDF file and extract:
        - Text from pages
        - Tables with structured data
        - Images with descriptions (if OpenAI Vision available)
        - Chart/graph data where possible
        """
        parse_start = time.time()
        
        try:
            # Open PDF
            if progress_callback:
                progress_callback("parsing", "Opening PDF file...", 10)
            open_start = time.time()
            pdf = pdfplumber.open(file_path)
            open_duration = time.time() - open_start
            
            content_parts = []
            total_pages = len(pdf.pages)
            total_tables = 0
            total_images = 0
            
            if progress_callback:
                progress_callback("parsing", f"Processing {total_pages} pages...", 12)
            
            page_start = time.time()
            for page_num, page in enumerate(pdf.pages):
                if progress_callback:
                    progress = 12 + int((page_num / total_pages) * 5) if total_pages > 0 else 12
                    progress_callback("parsing", f"Processing page {page_num + 1}/{total_pages}...", progress)
                page_content = []
                
                # Extract text
                text_start = time.time()
                text = page.extract_text()
                text_duration = time.time() - text_start
                if text:
                    page_content.append(text)
                
                # Extract tables
                table_start = time.time()
                tables = page.extract_tables()
                table_duration = time.time() - table_start
                for table_num, table in enumerate(tables):
                    if table:
                        table_text = []
                        for row in table:
                            if row:
                                row_text = " | ".join([
                                    str(cell) if cell else "" 
                                    for cell in row
                                ])
                                table_text.append(row_text)
                        
                        if table_text:
                            page_content.append(
                                f"\n[TABLE {table_num + 1}]\n" + 
                                "\n".join(table_text) + 
                                "\n[/TABLE]"
                            )
                            total_tables += 1
                
                
                # Extract images from page
                image_start = time.time()
                try:
                    images = page.images
                    if images:
                        if progress_callback:
                            progress_callback("parsing", f"Extracting images from page {page_num + 1}/{total_pages}...", 17)
                        image_descriptions = self._extract_images_from_pdf_page(pdf, page_num, images, progress_callback)
                        if image_descriptions:
                            page_content.append("\n[IMAGES]\n" + "\n".join(image_descriptions) + "\n[/IMAGES]")
                            total_images += len(image_descriptions)
                except Exception as img_error:
                    # Image extraction is optional, continue if it fails
                    logger.debug(f"Page {page_num + 1}: Image extraction skipped: {str(img_error)}")
                
                image_duration = time.time() - image_start
                
                if page_content:
                    content_parts.append(f"\n--- Page {page_num + 1} ---\n" + "\n".join(page_content))
            
            page_duration = time.time() - page_start
            pdf.close()
            
            parse_duration = time.time() - parse_start
            content_length = len("\n\n".join(content_parts))
            
            return "\n\n".join(content_parts)
            
        except Exception as e:
            parse_duration = time.time() - parse_start
            raise Exception(f"Error parsing PDF: {str(e)}")
    
    def _extract_images_from_pdf_page(self, pdf, page_num: int, images: List, progress_callback=None) -> List[str]:
        """Extract and describe images from PDF page using PyMuPDF or pdfplumber in parallel"""
        image_descriptions = []
        
        try:
            # First, collect all images from the page
            image_data_list = []
        
            try:
                doc = fitz.open(pdf.path if hasattr(pdf, 'path') else pdf)
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_num, img in enumerate(image_list):
                    try:
                        # Get image bytes
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image.get('ext', 'png')
                        
                        # Convert to base64
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                        image_data_list.append((img_num + 1, image_base64, image_ext))
                    except Exception as e:
                        image_descriptions.append(f"Image {img_num + 1} (Page {page_num + 1}): [Error extracting: {str(e)}]")
                
                doc.close()
                
                if image_data_list:
                    # Process images in parallel
                    if progress_callback:
                        progress_callback("parsing", f"Analyzing {len(image_data_list)} images from page {page_num + 1} with AI...", 18)
                    
                    # Function to process a single image
                    def process_pdf_image(img_num, image_base64, image_ext, page_num):
                        image_start = time.time()
                        try:
                            client = get_openai_client()
                            response = client.chat.completions.create(
                                model=settings.OPENAI_VISION_MODEL,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": "Describe this image in detail, including any text, charts, graphs, data visualizations, or numerical data. Extract all readable information including axes labels, data points, and trends."
                                            },
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/{image_ext};base64,{image_base64}"
                                                }
                                            }
                                        ]
                                    }
                                ],
                                max_tokens=settings.OPENAI_VISION_MAX_TOKENS
                            )
                            description = response.choices[0].message.content
                            image_duration = time.time() - image_start
                            return img_num, f"Image {img_num} (Page {page_num + 1}): {description}", None
                        except Exception as e:
                            image_duration = time.time() - image_start
                            return img_num, f"Image {img_num} (Page {page_num + 1}): [Error processing: {str(e)}]", str(e)
                    
                    # Process images in parallel
                    max_workers = min(len(image_data_list), settings.DOCUMENT_IMAGE_EXTRACTION_PARALLEL_WORKERS)
                    
                    parallel_start = time.time()
                    completed_count = 0
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all image processing tasks
                        future_to_image = {
                            executor.submit(process_pdf_image, img_num, image_base64, image_ext, page_num): img_num
                            for img_num, image_base64, image_ext in image_data_list
                        }
                        
                        # Collect results as they complete
                        image_results = {}
                        for future in as_completed(future_to_image):
                            img_num = future_to_image[future]
                            try:
                                result_img_num, description, error = future.result()
                                image_results[result_img_num] = description
                                completed_count += 1
                                if progress_callback:
                                    progress = 18 + int((completed_count / len(image_data_list)) * 1)
                                    progress_callback("parsing", f"Analyzed image {completed_count}/{len(image_data_list)} from page {page_num + 1}...", progress)
                            except Exception as e:
                                image_results[img_num] = f"Image {img_num} (Page {page_num + 1}): [Error: {str(e)}]"
                                completed_count += 1
                    
                    parallel_duration = time.time() - parallel_start
                    
                    # Combine results in order
                    for img_num in sorted(image_results.keys()):
                        image_descriptions.append(image_results[img_num])
                    
                    return image_descriptions
            except Exception as e:
                # Fallback if PyMuPDF extraction fails
                raise Exception(f"Error extracting images from PDF page {page_num + 1}: {str(e)}")
            
            # Fallback: Use pdfplumber image coordinates (less accurate)
            for img_num, img_info in enumerate(images):
                image_descriptions.append(
                    f"Image {img_num + 1} on page {page_num + 1}: "
                    f"[Image detected at coordinates: x0={img_info.get('x0', 'N/A')}, y0={img_info.get('y0', 'N/A')}]"
                )
        
        except Exception as e:
            image_descriptions.append(f"[Image extraction error on page {page_num + 1}: {str(e)}]")
        
        return image_descriptions
    
    def parse_txt(self, file_path: str, progress_callback=None) -> str:
        """Parse TXT file"""
        try:
            if progress_callback:
                progress_callback("parsing", "Reading TXT file...", 10)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            content_length = len(content)
            
            if progress_callback:
                progress_callback("parsing", f"Read {content_length:,} characters from TXT file", 20)
            
            return content
        except Exception as e:
            raise Exception(f"Error parsing TXT: {str(e)}")
