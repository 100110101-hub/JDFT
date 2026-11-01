import sys, json, urllib.request, urllib.parse

def get(url):
    with urllib.request.urlopen(url, timeout=90) as f:
        return json.load(f)

def show(d, show_abstract=True):
    print('-' * 70)
    print('TITLE:', d.get('title'))
    print('YEAR:', d.get('publication_year'), '| DOI:', d.get('doi'))
    loc = d.get('primary_location') or {}
    src = loc.get('source') or {}
    print('VENUE:', src.get('display_name'), '| OA:', d.get('open_access', {}).get('is_oa'),
          '| OA_URL:', d.get('open_access', {}).get('oa_url'))
    if show_abstract:
        inv = d.get('abstract_inverted_index')
        if inv:
            pos = {}
            for w, idxs in inv.items():
                for i in idxs:
                    pos[i] = w
            print('ABS:', ' '.join(pos[i] for i in sorted(pos))[:1400])
        else:
            print('ABS: (none)')

mode = sys.argv[1]
if mode == 'title':
    q = sys.argv[2]
    url = ('https://api.openalex.org/works?search=' + urllib.parse.quote(q)
           + '&per-page=5')
    for d in get(url).get('results', []):
        show(d)
elif mode == 'fulltext':
    q = sys.argv[2]
    url = ('https://api.openalex.org/works?search=' + urllib.parse.quote(q)
           + '&per-page=10&filter=has_abstract:true')
    for d in get(url).get('results', []):
        show(d, show_abstract=False)
elif mode == 'doi':
    for doi in sys.argv[2:]:
        try:
            show(get('https://api.openalex.org/works/doi:' + doi))
        except Exception as e:
            print('FAIL', doi, e)
