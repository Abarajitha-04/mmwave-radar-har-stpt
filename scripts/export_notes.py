"""Export slide titles + speaker notes from the built deck into a stand-alone document (DOCX and PDF)."""
import sys, re, subprocess, os
from pptx import Presentation
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

src = sys.argv[1] if len(sys.argv) > 1 else 'presentation/Team10_mmWave_HAR.pptx'
out = 'presentation/Team10_Speaker_Notes.docx'
prs = Presentation(src)
doc = Document()
for s in doc.sections:
    s.page_height, s.page_width = Cm(29.7), Cm(21.0)
    s.left_margin = s.right_margin = Cm(2.2); s.top_margin = s.bottom_margin = Cm(2.0)
st = doc.styles['Normal']; st.font.name = 'Times New Roman'; st.font.size = Pt(12)
st.paragraph_format.line_spacing = 1.15; st.paragraph_format.space_after = Pt(4)
for h in ['Heading 1', 'Heading 2']:
    doc.styles[h].font.name = 'Times New Roman'; doc.styles[h].font.color.rgb = RGBColor(0x12, 0x35, 0x5B)

t = doc.add_heading('Speaker Notes', 0); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = doc.add_paragraph('Privacy-Preserving Human Activity Recognition from mmWave Radar Point Clouds using Lightweight Point Transformers\n'
                      'Team 10 · 23I006 Deep Learning · PSG College of Technology'); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph('Presenters: Abarajitha S K (23I301), G Aswath Rupan (23I321), Indu Vadana M D (23I323), Murugan S (23I338). '
                  'Each slide lists its presenter, what to say, and likely examiner questions with prepared answers. '
                  f'Deck: {os.path.basename(src)}, {len(prs.slides)} slides; about 1.5 minutes per slide.')
n_notes = 0
for i, sl in enumerate(prs.slides, 1):
    title = ''
    for sh in sl.shapes:
        if sh.is_placeholder and sh.placeholder_format.type is not None and 'TITLE' in str(sh.placeholder_format.type) and sh.has_text_frame:
            title = sh.text_frame.text.strip(); break
    if not title:
        texts = [sh.text_frame.text.strip() for sh in sl.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
        title = texts[0] if texts else f'Slide {i}'
    doc.add_heading(f'Slide {i}: {title}', 1)
    notes = sl.notes_slide.notes_text_frame.text.strip() if sl.has_notes_slide else ''
    if notes: n_notes += 1
    for block in notes.split('\n'):
        b = block.strip()
        if not b: continue
        para = doc.add_paragraph()
        if b.startswith('Presenter:'):
            r = para.add_run(b); r.bold = True; r.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)
        elif b == 'Possible examiner questions:':
            r = para.add_run(b); r.bold = True
        elif b.startswith('Q: '):
            r = para.add_run('Q: '); r.bold = True; para.add_run(b[3:]).italic = True
        elif b.startswith('A: '):
            r = para.add_run('A: '); r.bold = True; para.add_run(b[3:]); para.paragraph_format.left_indent = Cm(0.6)
        else:
            para.add_run(b)
doc.save(out)
subprocess.run(['python3', os.environ.get('PPTX_SKILL', '.') + '/scripts/office/soffice.py', '--headless', '--convert-to', 'pdf', '--outdir', 'presentation', out],
               capture_output=True)
print('slides', len(prs.slides), 'with notes', n_notes, '->', out)
