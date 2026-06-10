#for image processing
import io
import base64

# standard library imports for type hints and path handling
from pathlib import Path
from typing import List, Dict, Any

#docling libraries import :
#This defines the supported file formats that docling can ingest
#acts as a standard data type
from docling.datamodel.base_models import InputFormat
#this lets us fine tune the granular parsing behaviours of the document so that whatever the kind of file is we esend them to te correct pipeline
from docling.datamodel.pipeline_options import PdfPipelineOptions
#DocumentConverter is the main structural execution engine of the Docling library
from docling.document_converter import DocumentConverter, PdfFormatOption
# the PdfPipelineOptions class used to toggle on native picture parsing and scale up optical resolution used to activate or deactivate advanced multi-modal extractions

# docling element type imports for isinstance checks
# Note: older docling versions used HeadingItem - your version uses SectionHeaderItem
from docling.datamodel.document import TextItem, SectionHeaderItem, TableItem, PictureItem

# langchain message type for the vision model prompt
from langchain_core.messages import HumanMessage


class MultimodalStructuralChunker:
    """Orchestrates the full multimodal chunking pipeline:
    loads a PDF via Docling, splits it by section headings,
    extracts tables as markdown, and summarises images via a vision LLM."""

    def __init__(self, vision_model):
        # vision_model: any LangChain-compatible chat model that accepts image_url content
        self.vision_model = vision_model

        #configure docling to activate native picture parsing and high-res optical rendering
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True

        #wire the PDF pipeline options into the converter so every PDF goes through the right path
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def process_document(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Loads a local PDF file, runs layout analysis via Docling, and deconstructs it into structural parent chunks."""
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"Target document not found at: {pdf_path}")
        print(f"initialising structural layout parsing for {pdf_file.name}")

        #execute the local file type conversion pipeline to get the document object tree
        conversion_result = self.converter.convert(pdf_file)
        doc = conversion_result.document

        #initialise the tracking variables for building our chunks
        #an empty list that'll contain the chunks in the form of dictionaries
        chunks = []
        # the basic dictionaries format
        current_chunk = {
            "title": "Introduction/Root",
            "text_buffer": [],
            "tables": [],
            "images": []
        }
        #this acts as a sliding memory window that holds a rolling buffer of the last few paragraph text parsed by the engine
        recent_text = []

        """When docling extracts a diagram or a graph from a PDF, the image itself usually doesn't 
        have text labels or a full explanation attached to it.
        by keeping a rolling history of the last 3 paragraphs in recent_text, we capture that critical surrounding context"""

        print("processing document items and analyzing layout structures...")

        for element, _level in doc.iterate_items():

            #1. track and buffer regular paragraphs:
            if isinstance(element, TextItem):
                #the element object of atomic pieces that we get
                current_chunk["text_buffer"].append(element.text)
                recent_text.append(element.text)
                #if the recent_text buffer is full then we input the new one and remove the oldest one
                if len(recent_text) > 3:
                    recent_text.pop(0)

            elif isinstance(element, SectionHeaderItem):
                if current_chunk["text_buffer"] or current_chunk["tables"] or current_chunk["images"]:
                    chunks.append(current_chunk)
                #if a new heading is found we create a new chunk
                current_chunk = {
                    "title": element.text,
                    "text_buffer": [],
                    "tables": [],
                    "images": []
                }

            elif isinstance(element, TableItem):
                current_chunk["tables"].append(element.export_to_markdown())

            elif isinstance(element, PictureItem):
                print("Found diagram under heading")

                b64_str = self._to_base64(element.get_image(doc))
                caption = element.caption_text(doc=doc) or "No Explicit caption"
                context_window = "\n".join(recent_text)
                #add the context of the image with the image itself
                #give a prompt to the vision model with the image to create a detailed summary of what it is
                prompt = HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": f"Describe this diagram. Anchored context from document text: \n{context_window}\nCaption: {caption}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_str}"}
                        }
                    ]
                )
                summary = self.vision_model.invoke([prompt]).content
                current_chunk["text_buffer"].append(f"\n[Visual Diagram Analysis]: {summary}\n")

        # append the final chunk after the loop ends (was incorrectly inside the loop before)
        if current_chunk["text_buffer"] or current_chunk["tables"] or current_chunk["images"]:
            chunks.append(current_chunk)
        return self._finalize_chunks(chunks)

    def _to_base64(self, pil_image) -> str:
        """converts the local PIL image object to a Base64 string"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _finalize_chunks(self, raw_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """harmonize text buffers, vision summaries, and markdown tables into a unified context string block ready for downstream indexing"""
        finalized_payloads = []
        for idx, rc in enumerate(raw_chunks):
            combined_text = "\n".join(rc["text_buffer"])
            combined_tables = "\n\n".join(rc["tables"])

            full_context = f"Section Title: {rc['title']}\n\n{combined_text}\n\n{combined_tables}"
            finalized_payloads.append({
                "page_content": full_context.strip(),
                "metadata": {
                    "title": rc["title"],
                    "chunk_id": f"doc_chunk_{idx}"
                }
            })
        return finalized_payloads