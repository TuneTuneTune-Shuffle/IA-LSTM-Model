FROM python
WORKDIR /ia-lstm-model
COPY . /ia-lstm-model
CMD ["python", "analyzeSongs.py"]