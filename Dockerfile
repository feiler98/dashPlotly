FROM feiler98/pyomics_fedora

RUN mkdir -p /scratch/tmp/feiler/dashPlotly
WORKDIR /scratch/tmp/feiler/dashPlotly

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python3", "/scratch/tmp/feiler/dashPlotly/utility_lib/cnv_plot.py"]

