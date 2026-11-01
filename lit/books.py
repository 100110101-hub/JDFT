import sys, json, urllib.request, urllib.parse

def books(q, n=8):
    url = ('https://www.googleapis.com/books/v1/volumes?q=' + urllib.parse.quote(q)
           + '&maxResults=%d' % n)
    req = urllib.request.Request(url, headers={'User-Agent': 'lit-check/1.0'})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    print('#' * 70)
    print('QUERY:', q, '| total:', d.get('totalItems'))
    for it in d.get('items', []):
        v = it.get('volumeInfo', {})
        snip = (it.get('searchInfo') or {}).get('textSnippet')
        print('-', (v.get('title') or '')[:80], '|', ', '.join(v.get('authors') or [])[:40],
              '|', v.get('publishedDate'))
        if snip:
            print('   SNIP:', snip[:400])

for q in sys.argv[1:]:
    try:
        books(q)
    except Exception as e:
        print('FAIL', q, e)
