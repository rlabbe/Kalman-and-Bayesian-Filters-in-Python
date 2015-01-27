"""
IPython Notebook magic command for automatically numbering sections of a notebook "interactively"

Use:
    %install_ext https://raw.github.com/dpsanders/ipython_extensions/master/section_numbering/secnum.py

once to install, and then

    %load_ext secnum
    %secnum

to get automatic section numbering, which is updated when any cell changes type.
(This could, of course, be more efficient.)

This is based, completely and shamelessly, on MinRK's `nbtoc` magic
<https://github.com/minrk/ipython_extensions>

Thanks also to:
- Clayton Davis, for a great explanation of HTML, CSS and JavaScript
- Brian Granger, for a crucial hint about extracting a JQuery object from a notebook cell object
- Fernando Perez, for an inspiring talk at SciPy 2013 (and, of course, for creating IPython)

"""

import io
import os
import urllib2

from IPython.display import display_html, display_javascript

here = os.path.abspath(os.path.dirname(__file__))
secnum_js = ""

def download(fname, redownload=False):
    """download a file
    
    if redownload=False, the file will not be downloaded if it already exists.
    """
    dest = os.path.join(here, fname)
    if os.path.exists(dest) and not redownload:
        return
    url = 'https://raw.github.com/dpsanders/ipython_extensions/master/section_numbering/' + fname
    print("Downloading %s to %s" % (url, dest))
    
    filein  = urllib2.urlopen(url)
    fileout = open(dest, "wb")
    chunk = filein.read(1024)
    while chunk:
        fileout.write(chunk)
        chunk = filein.read(1024)
    filein.close()
    fileout.close()

def load_file(fname, redownload=False):
    """load global variable from a file"""
    download(fname, redownload)
    with io.open(os.path.join(here, fname)) as f:
        globals()[fname.replace('.', '_')] = f.read()


load_file('secnum.js')



def secnum(line):
    """add a table of contents to an IPython Notebook"""
    display_javascript(secnum_js, raw=True)


def update_secnum(line):
    """download the latest version of the nbtoc extension from GitHub"""
    download('secnum.py', True)
    download('secnum.js', True)
    get_ipython().extension_manager.reload_extension("secnum")
    
def load_ipython_extension(ip):
    ip.magics_manager.register_function(secnum)
    ip.magics_manager.register_function(update_secnum)

