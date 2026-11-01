import sys, json, urllib.request, urllib.parse

def search(q, n=15):
    url = ('https://www.ebi.ac.uk/europepmc/webservices/rest/search?query='
           + urllib.parse.quote(q) + '&format=json&pageSize=%d' % n)
    with urllib.request.urlopen(url, timeout=60) as f:
        d = json.load(f)
    print('QUERY:', q, '| hits:', d.get('hitCount'))
    for r in d.get('resultList', {}).get('result', []):
        print(' ', r.get('pmid'), r.get('pmcid'), '|', (r.get('title') or '')[:110],
              '|', r.get('journalTitle'), r.get('pubYear'))
    print()

for q in sys.argv[1:]:
    search(q)
