from IPython.display import display, FileLink, HTML
import json
import base64
import zipfile
import glob

#!tar chvfz example.tar.gz ~/work/dsm0en/module-00/chapter-02/ *
#tar -czf attempt.tar.gz chapter-02
#!zip -r example.zip ~/work/dsm0en/module-00
#!zip -r chapter-02.zip chapter-02/
#!Tar results.zip  *



#local_file = FileLink('./notebook.zip', result_html_prefix="Click here to download: ", )
#display(local_file)


#import os 
#os.system("zip -r results1.zip . -i *.csv *.pdf *.ipynb *.gif")

#import subprocess
#subprocess.Popen("zip -r results2.zip . -i *.csv *.pdf *.ipynb *.gif")

with zipfile.ZipFile('all-files.zip', 'w') as f:
    for file in glob.glob('*'):
        f.write(file)

encoded = base64.b64encode(json.dumps('notebook.tar.gz').encode('utf-8')).decode('utf-8')
HTML(f'<a href="data:application/json;base64,{encoded}" download="all-files.zip">all-files.zip download</a>')

encoded = base64.b64encode(json.dumps('notebook.tar.gz').encode('utf-8')).decode('utf-8')
HTML(f'<a href="data:application/json;base64,{encoded}" download="notebook.tar.gz">notebook.tar.gz download</a>')


