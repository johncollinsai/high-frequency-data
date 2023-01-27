FROM python:slim

WORKDIR /home/posts/high-frequency-data

COPY requirements.txt requirements.txt 
RUN python -m venv venv
RUN venv/bin/pip install -r requirements.txt

# I use boot.sh rather than ENTRYPOINT in the Dockerfile because the exec command
# in my boot.sh does not work here in Dockerfile in ENTRYPOINT. Don't know why
COPY hfdata.ipynb hfdata.py boot.sh ./ 
# COPY ALL.csv DEXJPUS.csv bigF.csv ./
RUN chmod a+x boot.sh
ENV PORT 8080

# I follow https://github.com/photonics-project/notebooks/blob/main/Dockerfile
# but put jimustafa's ENTRYPOINT into boot.sh
ENTRYPOINT ["./boot.sh"]



