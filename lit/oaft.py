import sys, json, urllib.request, urllib.parse, time

MAIL = 'mailto=lit-check@example.org'

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'lit-check/1.0 (' + MAIL + ')'})
    with urllib.request.urlopen(req, timeout=90) as f:
        return json.load(f)

for q in sys.argv[1:]:
    url = ('https://api.openalex.org/works?filter=fulltext.search:' + urllib.parse.quote(q)
           + '&per-page=8&' + MAIL)
    print('#' * 70)
    print('QUERY:', q)
    try:
        d = get(url)
    except Exception as e:
        print('  FAIL', e)
        time.sleep(3)
        continue
    print('  count:', d.get('meta', {}).get('count'))
    for w in d.get('results', []):
        oa = w.get('open_access') or {}
        print('  -', w.get('publication_year'), '|', (w.get('title') or '')[:90])
        print('    doi:', (w.get('doi') or '').replace('https://doi.org/', ''), '| oa:', oa.get('is_oa'), oa.get('oa_url'))
    time.sleep(1.5)
