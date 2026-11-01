import sys, json, urllib.request

def abstract_for(doi):
    url = 'https://api.openalex.org/works/doi:' + doi
    try:
        with urllib.request.urlopen(url, timeout=60) as f:
            d = json.load(f)
    except Exception as e:
        print('DOI', doi, 'FAILED', e)
        return
    print('=' * 80)
    print('DOI:', doi)
    print('TITLE:', d.get('title'))
    print('YEAR:', d.get('publication_year'), '| VENUE:', (d.get('primary_location') or {}).get('source', {}) and ((d.get('primary_location') or {}).get('source') or {}).get('display_name'))
    inv = d.get('abstract_inverted_index')
    if inv:
        pos = {}
        for w, idxs in inv.items():
            for i in idxs:
                pos[i] = w
        print('ABSTRACT:', ' '.join(pos[i] for i in sorted(pos)))
    else:
        print('ABSTRACT: (none deposited)')

for doi in sys.argv[1:]:
    abstract_for(doi)
