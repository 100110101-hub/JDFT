import sys, re, urllib.request, urllib.parse, html

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'

def ddg(q):
    url = 'https://lite.duckduckgo.com/lite/?q=' + urllib.parse.quote(q)
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=60) as f:
        raw = f.read().decode('utf-8', 'replace')
    print('#' * 70)
    print('QUERY:', q)
    # rows: link + result-snippet
    links = re.findall(r'<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', raw, re.S)
    snips = re.findall(r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>', raw, re.S)
    def clean(s):
        return html.unescape(re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', s))).strip()
    if not links:
        print('  (no results)')
    for i, (u, t) in enumerate(links[:8]):
        print(' ', i + 1, clean(t)[:100])
        print('    ', u[:160])
        if i < len(snips):
            print('    >', clean(snips[i])[:300])

for q in sys.argv[1:]:
    try:
        ddg(q)
    except Exception as e:
        print('FAIL', q, e)
