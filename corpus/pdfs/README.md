# PDF corpus

Drop public-domain or permissively-licensed security PDFs into this
directory before running the ingest. The pipeline picks them up
automatically (see `ingest/pdf_loader.py`) and emits one Document per
page.

Suggested seed set (all redistributable):

- NIST SP 800-53 Rev 5 (Security and Privacy Controls)
- NIST SP 800-61 Rev 2 (Computer Security Incident Handling Guide)
- NIST SP 800-115 (Technical Guide to Information Security Testing)
- MITRE ATT&CK whitepapers (publicly downloadable)
- CISA Cybersecurity Performance Goals (CPG) PDFs
- ENISA Threat Landscape annual reports (open access)

Every PDF you add becomes addressable in the RAG corpus as
`<filename>:p<page-number>`. That's also how the answer cites it,
so name the files something the cite would read sensibly:
`nist-sp-800-53r5.pdf`, not `download (3).pdf`.

This directory is intentionally checked in, but with no PDFs by
default — populating it is an operator-side step so we don't ship
copies of large public documents inside a small research repo.
