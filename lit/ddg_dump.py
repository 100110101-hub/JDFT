import sys, urllib.request, urllib.parse

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
q = sys.argv[1]
url = 'https://lite.duckduckgo.com/lite/?q=' + urllib.parse.quote(q)
req = urllib.request.Request(url, headers={'User-Agent': UA})
raw = urllib.request.urlopen(req, timeout=60).read().decode('utf-8', 'replace')
open(r'.\lit\ddg_raw.html', 'w', encoding='utf-8').write(raw)
print(len(raw))
print(raw[:4000])
