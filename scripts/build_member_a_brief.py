"""Build the two-page Chinese Member A brief and its editable HTML companion.

Requires reportlab, pypdf and a Chinese TrueType font (Microsoft YaHei on Windows).
Edit reports/member_a_brief/brief_content.json, then run this script from any cwd.
Override the default font paths with --font and --bold-font on other platforms.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'reports/member_a_brief/brief_content.json'
OUT = ROOT / 'output/pdf/HyCal_Member_A_Research_Progress_CN.pdf'
SOURCE = DATA.parent
W, H = A4
M, CW = 47, A4[0] - 94
NAVY = colors.HexColor('#18305b')
BLUE = colors.HexColor('#214785')
TEXT = colors.HexColor('#17243b')
MUTED = colors.HexColor('#566783')
BORDER = colors.HexColor('#c6d3eb')
PALE = colors.HexColor('#f2f5fb')
GREEN = colors.HexColor('#087a48')
ORANGE = colors.HexColor('#94600a')


def esc(s):
    return html.escape(str(s))


class Brief:
    def __init__(self, path):
        self.c = canvas.Canvas(str(path), pagesize=A4, pageCompression=1, invariant=1)
        self.c.setTitle('HyCal Member A | 花卉识别与跨数据集验证进度简报')
        self.c.setAuthor('HyCal Member A')
        self.y = H - 46
        self.ends = []

    def para(self, text, x=M, width=CW, size=9.3, leading=14.0,
             color=TEXT, bold=False, gap=0, align=TA_LEFT, draw=True, y=None):
        style = ParagraphStyle('p', fontName='CNB' if bold else 'CN', fontSize=size,
                               leading=leading, textColor=color, alignment=align,
                               wordWrap='CJK', splitLongWords=True)
        p = Paragraph(text, style)
        _, height = p.wrap(width, 1000)
        top = self.y if y is None else y
        if draw:
            p.drawOn(self.c, x, top - height)
        if y is None:
            self.y -= height + gap
        return height

    def rule(self, gap=10):
        self.c.setStrokeColor(BORDER)
        self.c.setLineWidth(.6)
        self.c.line(M, self.y, W-M, self.y)
        self.y -= gap

    def section(self, n, title):
        self.c.setFillColor(BLUE)
        self.c.rect(M, self.y-14, 2.6, 15.4, fill=1, stroke=0)
        self.para(f'{n}  |  {esc(title)}', x=M+10, width=CW-10, size=12.1,
                  leading=17, color=BLUE, bold=True, gap=6)

    def bullet(self, name, value, size=9.3, leading=14.0, gap=4):
        self.para(f'<b>{esc(name)}：</b>{esc(value)}', size=size, leading=leading, gap=gap)

    def box(self, paragraphs, fill=PALE, stroke=BORDER, pad=10, gap=10):
        height = sum(self.para(t, width=CW-2*pad, draw=False, y=0, **kw)
                     for t, kw in paragraphs) + 2*pad
        self.c.setFillColor(fill)
        self.c.setStrokeColor(stroke)
        self.c.roundRect(M, self.y-height, CW, height, 6, fill=1, stroke=1)
        top = self.y-pad
        for text, kw in paragraphs:
            ph = self.para(text, x=M+pad, width=CW-2*pad, y=top, **kw)
            top -= ph
        self.y -= height+gap

    def table(self, headers, rows, widths, green_cols=(), x=M, width=CW,
              size=9.1, leading=13.0, pad=6, y=None, gap=7):
        widths = [width*k/sum(widths) for k in widths]
        data = []
        for ri, row in enumerate([headers, *rows]):
            cells = []
            for j, s in enumerate(row):
                col = NAVY if ri == 0 else GREEN if j in green_cols else TEXT
                style = ParagraphStyle('cell', fontName='CNB' if ri == 0 else 'CN',
                                       fontSize=size, leading=leading, wordWrap='CJK',
                                       textColor=col, alignment=TA_LEFT if j == 0 else TA_CENTER)
                cells.append(Paragraph(esc(s), style))
            data.append(cells)
        t = Table(data, colWidths=widths, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e6edf8')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#fafbfe'), colors.white]),
            ('LINEBELOW', (0, 0), (-1, -1), .45, BORDER),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), pad),
            ('BOTTOMPADDING', (0, 0), (-1, -1), pad),
            ('LEFTPADDING', (0, 0), (-1, -1), 7),
            ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ]))
        _, height = t.wrap(width, H)
        top = self.y if y is None else y
        t.drawOn(self.c, x, top-height)
        if y is None:
            self.y -= height+gap
        return height

    def footer(self, page):
        assert self.y >= 45, f'Page {page} content collides with footer: y={self.y:.2f}'
        self.ends.append(round(self.y, 3))
        self.c.setFont('CN', 7.5)
        self.c.setFillColor(MUTED)
        self.c.drawString(M, 27, 'HyCal · Member A · Flowers102 / DTD / Aircraft')
        self.c.drawRightString(W-M, 27, f'{page} / 2')
        self.c.showPage()
        self.y = H-46


def html_source(d):
    def table(h, rows, cls=''):
        return '<table class="'+cls+'"><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in h)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'
    def section(n, title): return f'<h2>{n} <span>|</span> {title}</h2>'
    def bullets(items): return '<ul>'+''.join('<li><strong>'+esc(k)+'：</strong>'+esc(v)+'</li>' for k,v in items)+'</ul>'
    def p(key, cls=''): return '<p class="'+cls+'">'+esc(d[key])+'</p>'
    def foot(page): return '<footer><span>HyCal · Member A · Flowers102 / DTD / Aircraft</span><span>'+str(page)+' / 2</span></footer>'
    p1 = '<div class="eyebrow">HYCAL RESEARCH · MEMBER A</div><h1>'+esc(d['title'])+'</h1><div class="subtitle">'+esc(d['subtitle'])+' | '+esc(d['date'])+'</div><div class="status"><b>'+esc(d['badge'])+'</b><span>供小组内部交流与三域整合</span></div><hr>'+p('intro')
    p1 += '<div class="cards">'+''.join('<div><strong>'+esc(a)+'</strong><span>'+esc(b)+'</span><small>'+esc(c)+'</small></div>' for a,b,c in d['cards'])+'</div>'
    p1 += section('01','已完成的工作')+bullets(d['work'])+section('02','为什么共享统计有效（简单解释）')+p('mechanism')+'<div class="callout">'+p('mechanism_callout')+p('protocol_callout','note')+'</div>'
    p1 += section('03','三个单域的核心结果（类宏平均）')+table(**{'h':d['core_table']['headers'],'rows':d['core_table']['rows']})+p('core_note','note')+foot(1)
    p2 = '<div class="eyebrow">KEY FINDINGS & NEXT STEPS</div><h1>研究发现与后续合作</h1><div class="subtitle">把 Flowers 主实验与跨数据集证据带入小组的统一方法讨论</div><hr>'
    p2 += section('04','提升来自哪里？强基线与消融比较')+p('comparison_intro')+table(d['comparison_table']['headers'],d['comparison_table']['rows'])+p('comparison_note','note')+table(d['extra_table']['headers'],d['extra_table']['rows'])+p('extra_note','note')
    p2 += section('05','类别改善与需要保留的负面结果')+p('findings_intro')+'<div class="cols"><div><h3>平均改善较多的例子</h3>'+table(['类别','相对 H2'],d['positive'])+'</div><div><h3>需要关注的下降</h3>'+table(['类别','相对 H2'],d['negative'])+'</div></div>'+p('findings_note','note')
    p2 += section('06','研究局限')+bullets(d['limitations'])+section('07','接下来需要组员协作的内容')+table(['负责人','下一步'],d['cooperation'])+'<div class="callout takeaway">'+p('takeaway')+'</div>'+p('record','note')+foot(2)
    css = '''@page{size:A4;margin:0}*{box-sizing:border-box}body{margin:0;background:#eef1f6;color:#17243b;font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;font-size:9.1pt;line-height:1.5}.page{width:210mm;min-height:297mm;background:white;margin:12mm auto;padding:15.5mm 16.6mm 16mm;position:relative;page-break-after:always}.page:last-child{page-break-after:auto}.eyebrow{font-size:8pt;letter-spacing:1.6pt;color:#214785;font-weight:bold}h1{font-size:21pt;line-height:1.3;color:#18305b;margin:5pt 0 6pt}.subtitle,.note{color:#566783}.subtitle{font-size:8.4pt}.status{display:flex;gap:12pt;align-items:center;margin:11pt 0}.status b{background:#e6f4eb;border:1px solid #a4d6b7;color:#087a48;padding:4pt 10pt;border-radius:20pt}.status span{font-size:8.4pt;color:#566783}hr{border:0;border-top:1px solid #c6d3eb;margin:8pt 0}p{margin:5pt 0 7pt}.cards{display:flex;gap:10pt;margin:11pt 0}.cards>div{flex:1;border:1px solid #c6d3eb;border-radius:7pt;padding:9pt 12pt}.cards strong{font-size:21pt;color:#214785}.cards span,.cards small{display:block;color:#566783;font-size:8pt}h2{border-left:3px solid #214785;padding-left:8pt;font-size:12pt;color:#214785;margin:12pt 0 6pt;line-height:1.25}h2 span{color:#c6d3eb;margin:0 6pt}h3{font-size:9.5pt;margin:2pt 0 5pt}ul{padding-left:0;list-style:none;margin:0 0 7pt}li{margin-bottom:4pt}table{border-collapse:collapse;width:100%;font-size:8.8pt;margin:5pt 0 6pt}th,td{border-bottom:1px solid #c6d3eb;padding:5pt 7pt;text-align:center}th:first-child,td:first-child{text-align:left}th{background:#e6edf8;color:#18305b}tr:nth-child(odd) td{background:#fafbfe}.note{font-size:7.7pt;line-height:1.45}.callout{background:#f2f5fb;border:1px solid #c6d3eb;border-radius:6pt;padding:7pt 10pt;margin:8pt 0}.callout p{margin:0 0 4pt}.callout p:last-child{margin:0}.cols{display:flex;gap:18pt}.cols>div{flex:1}.takeaway{font-size:8.6pt}footer{position:absolute;left:16.6mm;right:16.6mm;bottom:9mm;display:flex;justify-content:space-between;font-size:7.5pt;color:#566783}.page:nth-of-type(2){padding-bottom:14mm;font-size:8.6pt;line-height:1.43}.page:nth-of-type(2) h2{margin:9pt 0 5pt}.page:nth-of-type(2) th,.page:nth-of-type(2) td{padding:3pt 7pt}.page:nth-of-type(2) .note{font-size:7.5pt;line-height:1.4}@media print{body{background:white}.page{margin:0;min-height:297mm;box-shadow:none}}'''
    return '<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>HyCal Member A 研究进度简报</title><style>'+css+'</style></head><body><section class="page">'+p1+'</section><section class="page">'+p2+'</section></body></html>'


def build(args):
    d = json.loads(DATA.read_text(encoding='utf-8'))
    for name, path in [('CN', args.font), ('CNB', args.bold_font)]:
        if not Path(path).is_file(): raise FileNotFoundError(f'Chinese font missing: {path}; pass --font/--bold-font')
        pdfmetrics.registerFont(TTFont(name, path, subfontIndex=0))
    pdfmetrics.registerFontFamily('CN', normal='CN', bold='CNB', italic='CN', boldItalic='CNB')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    b = Brief(OUT)
    b.para('HYCAL RESEARCH · MEMBER A', size=8.1, leading=12, color=BLUE, bold=True, gap=4)
    b.para(esc(d['title']), size=20.0, leading=27, bold=True, color=NAVY, gap=5)
    b.para(esc(d['subtitle'])+' | '+esc(d['date']), size=8.15, leading=12, color=MUTED, gap=12)
    b.c.setFillColor(colors.HexColor('#e5f4eb'));b.c.setStrokeColor(colors.HexColor('#a8d8b9'))
    b.c.roundRect(M,b.y-19,147,22,11,fill=1,stroke=1)
    b.para(esc(d['badge']),x=M+11,width=130,size=8.6,leading=13,color=GREEN,bold=True,y=b.y-2)
    b.para('供小组内部交流与三域整合',x=M+160,width=250,size=8.5,leading=13,color=MUTED,y=b.y-2)
    b.y-=29;b.rule(10)
    b.para(esc(d['intro']),size=9.4,leading=14.4,gap=11)
    cw=(CW-18)/3
    for i,(value,label,note) in enumerate(d['cards']):
        x=M+i*(cw+9)
        b.c.setStrokeColor(BORDER);b.c.setFillColor(colors.white)
        b.c.roundRect(x,b.y-62,cw,62,7,fill=1,stroke=1)
        b.para(esc(value),x=x+11,width=cw-22,size=19.5,leading=24,color=BLUE,bold=True,y=b.y-8)
        b.para(esc(label),x=x+11,width=cw-22,size=8.1,leading=12,color=MUTED,y=b.y-34)
        b.para(esc(note),x=x+11,width=cw-22,size=7.5,leading=10.8,color=MUTED,y=b.y-47)
    b.y-=74
    b.section('01','已完成的工作')
    for k,v in d['work']: b.bullet(k,v,size=9.3,leading=14,gap=4)
    b.y-=5;b.section('02','为什么共享统计有效（简单解释）')
    b.para(esc(d['mechanism']),size=9.3,leading=14,gap=8)
    b.box([(esc(d['mechanism_callout']),dict(size=9,leading=13)),
           (esc(d['protocol_callout']),dict(size=8.25,leading=12.2,color=MUTED))],pad=9,gap=12)
    b.section('03','三个单域的核心结果（类宏平均）')
    b.table(d['core_table']['headers'],d['core_table']['rows'],[1.4,1,1.1,1.1,1.15],green_cols=(3,4),pad=6)
    b.para(esc(d['core_note']),size=7.8,leading=11.4,color=MUTED)
    b.footer(1)

    b.para('KEY FINDINGS & NEXT STEPS',size=8.1,leading=12,color=BLUE,bold=True,gap=4)
    b.para('研究发现与后续合作',size=20.8,leading=28,color=NAVY,bold=True,gap=4)
    b.para('把 Flowers 主实验与跨数据集证据带入小组的统一方法讨论',size=8.7,leading=12,color=MUTED,gap=9)
    b.rule(10);b.section('04','提升来自哪里？强基线与消融比较')
    b.para(esc(d['comparison_intro']),size=8.9,leading=13,gap=5)
    b.table(d['comparison_table']['headers'],d['comparison_table']['rows'],[1.35,1,1,1.15,1.15],green_cols=(4,),pad=3.5,size=8.8,leading=12)
    b.para(esc(d['comparison_note']),size=7.8,leading=11.3,color=MUTED,gap=7)
    b.table(d['extra_table']['headers'],d['extra_table']['rows'],[2.8,1,1,1.1],green_cols=(2,3),pad=3.5,size=8.5,leading=12)
    b.para(esc(d['extra_note']),size=7.8,leading=11.3,color=MUTED,gap=9)
    b.section('05','类别改善与需要保留的负面结果')
    b.para(esc(d['findings_intro']),size=8.9,leading=13,gap=5)
    half=(CW-16)/2;ytop=b.y
    for x,title,rows,green in [(M,'平均改善较多的例子',d['positive'],(1,)),(M+half+16,'需要关注的下降',d['negative'],())]:
        b.para(title,x=x,width=half,size=9.1,leading=14,bold=True,color=NAVY,y=ytop)
        height=b.table(['类别','相对 H2'],rows,[1.8,1],green_cols=green,x=x,width=half,size=8.5,leading=12,pad=3.5,y=ytop-17)
    b.y=ytop-17-height-6
    b.para(esc(d['findings_note']),size=7.8,leading=11.3,color=MUTED,gap=9)
    b.section('06','研究局限')
    for k,v in d['limitations']: b.bullet(k,v,size=8.65,leading=12.5,gap=3)
    b.y-=5;b.section('07','接下来需要组员协作的内容')
    b.table(['负责人','下一步'],d['cooperation'],[1.0,4.5],size=8.6,leading=12,pad=3.5,gap=8)
    b.box([(esc(d['takeaway']),dict(size=8.4,leading=12.5))],pad=8,gap=7)
    b.para(esc(d['record']),size=7.2,leading=10.3,color=MUTED)
    b.footer(2)
    b.c.save()
    (SOURCE/'member_a_brief.html').write_text(html_source(d),encoding='utf-8')
    reader=PdfReader(OUT)
    assert len(reader.pages)==2
    for i,page in enumerate(reader.pages,1):
        assert abs(float(page.mediabox.width)-W)<.02 and abs(float(page.mediabox.height)-H)<.02
        assert page.extract_text().strip(),f'Page {i} has no extractable text'
    proof={'pages':2,'format':'A4 portrait','content_bottom_y_pt':b.ends,
           'pdf_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),
           'content_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
           'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in d['sources']},
           'font_sources':[str(args.font),str(args.bold_font)],
           'note':'Visual QA of rendered PDF is required in addition to this structural check.'}
    (SOURCE/'build_verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'pdf':str(OUT),'pages':2,'content_bottom_y_pt':b.ends},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--font',default='C:/Windows/Fonts/msyh.ttc')
    p.add_argument('--bold-font',default='C:/Windows/Fonts/msyhbd.ttc')
    build(p.parse_args())
