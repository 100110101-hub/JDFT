import sys, json, urllib.request, urllib.parse

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'lit-check/1.0 (mailto:lit-check@example.org)'})
    with urllib.request.urlopen(req, timeout=90) as f:
        return json.load(f)

q = sys.argv[1]
url = ('https://api.crossref.org/works?query.bibliographic=' + urllib.parse.quote(q)
       + '&rows=6&select=DOI,title,container-title,volume,page,issued,author')
d = get(url)
print('QUERY:', q)
for it in d['message']['items']:
    au = '; '.join((a.get('family', '') + ' ' + a.get('given', '')).strip() for a in it.get('author', [])[:4])
    print('-', (it.get('title') or [''])[0][:120])
    print('   ', (it.get('container-title') or [''])[0], 'vol', it.get('volume'), 'p', it.get('page'),
          '|', (it.get('issued', {}).get('date-parts') or [['?']])[0][0], '| DOI:', it.get('DOI'))
    print('    authors:', au)
