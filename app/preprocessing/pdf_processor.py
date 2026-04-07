"""
pdf_processor.py — PDF document preprocessor
Reads a PDF file into structured page data, including:
  1. Extracts text from each page using pypdf
  2. Cleans noise (page number lines, header lines, soft hyphens, extra whitespace)
  3. Identifies chapter structure (excludes TOC entries, only recognizes real headings in body)
  4. Extracts images from each page using PyMuPDF, associates captions, saves to disk
  5. Outputs a page list with metadata for the downstream Chunker

Changelog:
  - Chapter regex excludes TOC entries (lines containing ．．．)
  - Added removal of "number + chapter name" header lines (e.g., "94 第二篇胃肠功能紊乱")
  - Section title regex tightened (avoids misidentifying numeric body sentences as section titles)
  - Low content density pages (short text pages) marked as is_toc/is_low_content
  - Added multimodal image extraction (PyMuPDF + caption matching)
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

from pypdf import PdfReader

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Regex: chapter/section recognition
# ─────────────────────────────────────────────────────────────────────────────

CHAPTER_RE = re.compile(
    r'^(第\s*[一二三四五六七八九十百\d]+\s*[篇章部])\s+([^\d．…\s][^．…]{2,30})\s*$',
    re.MULTILINE,
)

SECTION_NUM_RE = re.compile(
    r'^(\d{1,3}[\.、])\s*([\u4e00-\u9fff]{2,20}[病症炎癌瘤综合征疾病功能紊乱障碍损伤感染中毒]?)\s*$',
    re.MULTILINE,
)

PAGE_NUM_RE = re.compile(r'^\s*\d{1,4}\s*$', re.MULTILINE)
TOC_LINE_RE = re.compile(r'[．…]{3,}')

RUNNING_HEADER_RE = re.compile(
    r'^(\d{1,4})\s+(第[一二三四五六七八九十百\d]+[篇章部])\s*(.{2,20}?)\s*$',
    re.MULTILINE,
)

HEADER_FOOTER_RE = re.compile(
    r'(默克|诊疗手册|Merck|MSD|版权|©|\bwww\..+\b)',
    re.IGNORECASE,
)

SOFT_HYPHEN_RE = re.compile(r'-\n')
MULTI_BLANK_RE = re.compile(r'\n{3,}')
TOC_DENSITY_RE = re.compile(r'[．…]{3,}')

# ─────────────────────────────────────────────────────────────────────────────
# Regex: figure caption recognition
# Matches "图4-2 可调型胃锥手术" / "图 4.2 胃束带" / "图1 示意图" etc.
# ─────────────────────────────────────────────────────────────────────────────

FIGURE_CAPTION_RE = re.compile(
    r'图\s*(\d+(?:[.\-–—]\d+)?)'   # "图" + numeric code (may contain hyphen/dot)
    r'\s*'                           # optional whitespace
    r'([^\n。]{0,60})',              # up to 60 chars of caption text (no newline or period)
)


class PDFProcessor:
    """
    Processor that parses a PDF file into structured page data.

    Usage:
        processor = PDFProcessor(pdf_path)
        pages = processor.extract()  # Returns List[Dict]
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    def extract(self) -> List[Dict[str, Any]]:
        """
        Main entry: reads PDF, extracts and cleans text page by page, returns page data list.

        Returns:
            List[Dict], each dict contains:
              - page_number (int)  : original page number (starting from 1)
              - text (str)         : cleaned body text
              - chapter (str)      : current page's "chapter" title (tracked across pages)
              - section_title (str): current page's section title (tracked across pages)
              - is_toc (bool)      : whether this is a TOC page (TOC pages are not indexed)
              - raw_text (str)     : original text (for debugging)
              - images (List[Dict]): image info list extracted from this page
        """
        logger.info(f"Starting PDF parsing: {self.pdf_path.name}")
        reader = PdfReader(str(self.pdf_path))
        total_pages = len(reader.pages)
        logger.info(f"Total pages: {total_pages}")

        pages: List[Dict[str, Any]] = []
        current_chapter = ""
        current_section = ""

        for page_idx, page in enumerate(reader.pages):
            page_number = page_idx + 1

            # ── Step 1: Extract raw text ──────────────────────────────────────
            raw_text = page.extract_text() or ""

            # ── Step 2: Check if this is a TOC page ───────────────────────────
            is_toc = self._is_toc_page(raw_text, page_number)

            # ── Step 3: Extract chapter info (before cleaning) ────────────────
            if not is_toc:
                header_match = RUNNING_HEADER_RE.search(raw_text)
                if header_match:
                    pian_part = header_match.group(2).strip()
                    name_part = header_match.group(3).strip()
                    if name_part:
                        current_chapter = f"{pian_part} {name_part}"

            # ── Step 4: Clean text ────────────────────────────────────────────
            cleaned = self._clean_text(raw_text)

            # ── Step 5: Identify section titles ───────────────────────────────
            if not is_toc:
                section_match = SECTION_NUM_RE.search(cleaned)
                if section_match:
                    current_section = section_match.group(2).strip()

            # ── Step 6: Store ─────────────────────────────────────────────────
            pages.append({
                "page_number": page_number,
                "text": cleaned,
                "raw_text": raw_text,
                "chapter": current_chapter,
                "section_title": current_section,
                "is_toc": is_toc,
                "images": [],  # Image info will be populated in _extract_images
            })

            if page_number % 200 == 0:
                logger.info(f"  Processed {page_number}/{total_pages} pages")

        valid = sum(1 for p in pages if not p["is_toc"] and len(p["text"]) > 30)
        logger.info(f"Text parsing complete, total pages: {len(pages)}, valid body pages: {valid}")

        # ── Step 7: Image extraction (PyMuPDF) ────────────────────────────────
        self._extract_images(pages)

        return pages

    # ─────────────────────────────────────────────────────────────────────
    # Image extraction
    # ─────────────────────────────────────────────────────────────────────

    def _extract_images(self, pages: List[Dict[str, Any]]) -> None:
        """
        Extracts images from each page using PyMuPDF, associates captions, saves to disk.
        Results are written to the "images" key of each page dict.

        Image save path: {pic_dir}/{pdf_stem}/page_{n}_img_{i}.png
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF not installed, skipping image extraction. Run: pip install pymupdf")
            return

        from app.config import settings

        pic_dir = Path(settings.pic_dir) / self.pdf_path.stem
        pic_dir.mkdir(parents=True, exist_ok=True)

        try:
            fitz_doc = fitz.open(str(self.pdf_path))
        except Exception as e:
            logger.error(f"PyMuPDF failed to open PDF: {e}")
            return

        total_images = 0

        try:
            for page_data in pages:
                if page_data.get("is_toc"):
                    continue

                page_idx = page_data["page_number"] - 1
                if page_idx >= len(fitz_doc):
                    continue

                fitz_page = fitz_doc[page_idx]
                img_list = fitz_page.get_images(full=True)

                saved_images = []
                img_counter = 0

                for img_info in img_list:
                    xref = img_info[0]
                    try:
                        pix = fitz.Pixmap(fitz_doc, xref)

                        # Skip decorative images smaller than 80px in width or height
                        if pix.width < 80 or pix.height < 80:
                            pix = None
                            continue

                        # Convert non-RGB color spaces to RGB
                        if pix.n > 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        img_filename = f"page_{page_data['page_number']}_img_{img_counter}.png"
                        img_path = pic_dir / img_filename
                        pix.save(str(img_path))
                        pix = None

                        # Associate caption
                        caption, figure_number = self._find_figure_caption(
                            page_data["text"], img_counter
                        )

                        # If no caption, build fallback description from page section info
                        if not caption:
                            section = page_data.get("section_title", "")
                            caption = f"Figure (page {page_data['page_number']}" + (f", {section}" if section else "") + ")"

                        saved_images.append({
                            "image_path": str(img_path),
                            "caption": caption,
                            "figure_number": figure_number,
                            "page_number": page_data["page_number"],
                        })
                        img_counter += 1
                        total_images += 1

                    except Exception as e:
                        logger.debug(
                            f"Image extraction failed page={page_data['page_number']} xref={xref}: {e}"
                        )

                page_data["images"] = saved_images

        finally:
            fitz_doc.close()

        logger.info(f"Image extraction complete: {total_images} images → {pic_dir}")

    def _find_figure_caption(self, text: str, img_idx: int) -> Tuple[str, str]:
        """
        Finds the caption for the img_idx-th figure in the page text.

        Returns:
            (full_caption, figure_number)
            e.g., ("图4-2 可调型胃锥手术", "4-2") or ("", "")
        """
        matches = list(FIGURE_CAPTION_RE.finditer(text))
        if not matches:
            return "", ""

        # Get caption at img_idx; fall back to last one if out of bounds
        match = matches[img_idx] if img_idx < len(matches) else matches[-1]

        # Normalize hyphen format: –/— → -
        figure_number = re.sub(r'[–—.]', '-', match.group(1))
        caption_suffix = match.group(2).strip()

        full_caption = f"图{figure_number}"
        if caption_suffix:
            full_caption += f" {caption_suffix}"

        return full_caption, figure_number

    # ─────────────────────────────────────────────────────────────────────
    # Private text processing methods
    # ─────────────────────────────────────────────────────────────────────

    def _is_toc_page(self, raw: str, page_number: int) -> bool:
        dot_matches = len(TOC_DENSITY_RE.findall(raw))
        if dot_matches >= 5 and page_number <= 30:
            return True
        if dot_matches >= 15:
            return True
        return False

    def _clean_text(self, raw: str) -> str:
        # 1. Remove isolated page number lines
        text = PAGE_NUM_RE.sub('', raw)

        # 2. Remove "number + chapter name" header lines
        lines = text.split('\n')
        filtered = []
        for line in lines:
            stripped = line.strip()
            if RUNNING_HEADER_RE.match(stripped):
                continue
            if len(stripped) < 35 and HEADER_FOOTER_RE.search(stripped):
                continue
            filtered.append(line)
        text = '\n'.join(filtered)

        # 3. Fix soft hyphens
        text = SOFT_HYPHEN_RE.sub('', text)

        # 4. Compress excessive blank lines
        text = MULTI_BLANK_RE.sub('\n\n', text)

        # 5. Strip leading/trailing whitespace from each line
        text = '\n'.join(line.strip() for line in text.split('\n'))

        return text.strip()

    def get_pdf_metadata(self) -> Dict[str, str]:
        """Returns basic metadata of the PDF file"""
        reader = PdfReader(str(self.pdf_path))
        meta = reader.metadata or {}
        return {
            "title": meta.get("/Title", self.pdf_path.stem),
            "author": meta.get("/Author", ""),
            "pages": str(len(reader.pages)),
            "source": self.pdf_path.name,
        }
