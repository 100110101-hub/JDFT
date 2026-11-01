import sys, json, urllib.request, urllib.parse

def s2(doi):
    url = ('https://api.semanticscholar.org/graph/v1/paper/DOI:' + doi
           + '?fields=title,abstract,year,venue,externalIds,openAccessPdf')
    req = urllib.request.Request(url, headers={'User-Agent': 'lit-check/1.0'})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=60))
    except Exception as e:
        print('FAIL', doi, e)
        return
    print('=' * 70)
    print('DOI:', doi, '|', d.get('year'), '|', d.get('venue'))
    print('TITLE:', d.get('title'))
    print('OA PDF:', (d.get('openAccessPdf') or {}).get('url'))
    print('ABS:', (d.get('abstract') or '(none)')[:1600])

for doi in sys.argv[1:]:
    s2(doi)
