import sys, re, urllib.request, urllib.parse, html

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'

def bing(q, n=8):
    url = 'https://www.bing.com/search?q=' + urllib.parse.quote(q) + '&format=rss&count=15'
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    raw = urllib.request.urlopen(req, timeout=60).read().decode('utf-8', 'replace')
    items = re.findall(r'<item>(.*?)</item>', raw, re.S)
    print('#' * 70)
    print('QUERY:', q)
    for it in items[:n]:
        t = re.search(r'<title>(.*?)</title>', it, re.S)
        l = re.search(r'<link>(.*?)</link>', it, re.S)
        d = re.search(r'<description>(.*?)</description>', it, re.S)
        clean = lambda s: html.unescape(re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', s))).strip()
        print('-', clean(t.group(1))[:110] if t else '')
        print('  ', clean(l.group(1))[:150] if l else '')
        if d:
            print('  >', clean(d.group(1))[:260])

for q in sys.argv[1:]:
    try:
        bing(q)
    except Exception as e:
        print('FAIL', q, e)
